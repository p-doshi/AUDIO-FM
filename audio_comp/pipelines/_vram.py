"""Progressive GPU-VRAM claimer -- grab whatever memory is free right now
without blocking, keep opportunistically grabbing more in the background
up to a cap, then release it all into THIS process's caching allocator
right before the real (model) allocations, so a co-tenant process on a
contended GPU can't squeeze the run out mid-way and OOM it.

Lifted from the restricted-domain variant of this pipeline (built there 2026-08-23 for
wav2vec2_conformer under heavy GPU contention) into a standalone module
with no confidential-pipeline imports, so any pipeline script can use it.
Only needs torch + threading. See that file's journal entries for the
design rationale (non-blocking partial grab, background growth, release-
into-allocator-not-OS).
"""
from __future__ import annotations

import threading
import time

import torch


class GreedyVRAMClaimer:
    """`.start()` does one synchronous grab of whatever is free right now
    and returns immediately (does NOT block to reach `max_gb`); a daemon
    thread then keeps claiming newly-freed memory up to `max_gb` (or
    unbounded if None). `.wait_until_target(timeout)` optionally blocks
    until the cap is claimed. `.release()` frees every claimed chunk back
    into this process's caching allocator (NOT to the OS -- do not call
    torch.cuda.empty_cache() afterwards or the reservation is lost), so
    subsequent model allocations reuse that pool instead of racing other
    processes for fresh device memory."""

    def __init__(self, device: int = 0, poll_interval: float = 1.0, max_gb: float | None = None, min_chunk_gb: float = 0.02):
        self.device = device
        self.poll_interval = poll_interval
        self.max_bytes = int(max_gb * (1024 ** 3)) if max_gb is not None else None
        self.min_chunk_bytes = int(min_chunk_gb * (1024 ** 3))
        self._chunks: list[torch.Tensor] = []
        self._claimed_bytes = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_diag_print = 0.0

    def start(self) -> "GreedyVRAMClaimer":
        self._try_claim_available()
        self._thread.start()
        return self

    def _diag(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_diag_print >= 10:
            print(msg, flush=True)
            self._last_diag_print = now

    def _try_claim_available(self) -> None:
        free_bytes, _total = torch.cuda.mem_get_info(self.device)
        claimable = free_bytes
        if self.max_bytes is not None:
            claimable = min(claimable, self.max_bytes - self._claimed_bytes)
        if claimable < self.min_chunk_bytes:
            self._diag(f"[vram-claim] GPU{self.device}: only {free_bytes / 1024**2:.1f} MiB free -- skipping poll.")
            return
        try:
            chunk = torch.empty(claimable, dtype=torch.uint8, device=f"cuda:{self.device}")
        except torch.cuda.OutOfMemoryError:
            self._diag(f"[vram-claim] GPU{self.device}: {claimable / 1024**2:.1f} MiB reported free but alloc "
                       f"failed (lost a race / fragmented) -- retrying next poll.")
            return
        with self._lock:
            self._chunks.append(chunk)
            self._claimed_bytes += claimable
        print(f"[vram-claim] +{claimable / 1024**3:.2f} GB (total {self._claimed_bytes / 1024**3:.2f} GB)", flush=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._try_claim_available()
            self._stop_event.wait(self.poll_interval)

    def claimed_gb(self) -> float:
        with self._lock:
            return self._claimed_bytes / 1024 ** 3

    def wait_until_target(self, timeout: float | None = None) -> None:
        if self.max_bytes is None:
            raise ValueError("wait_until_target() needs a max_gb target")
        start = time.monotonic()
        last_print = 0.0
        while self.claimed_gb() * 1024 ** 3 < self.max_bytes:
            if self._stop_event.is_set():
                return
            elapsed = time.monotonic() - start
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(
                    f"[vram-claim] gave up after {elapsed:.0f}s with {self.claimed_gb():.2f}/"
                    f"{self.max_bytes / 1024**3:.2f} GB on GPU{self.device}."
                )
            if elapsed - last_print >= 30:
                print(f"[vram-claim] waiting: {self.claimed_gb():.2f}/{self.max_bytes / 1024**3:.2f} GB "
                      f"({elapsed:.0f}s)...", flush=True)
                last_print = elapsed
            time.sleep(self.poll_interval)
        print(f"[vram-claim] target reached: {self.claimed_gb():.2f} GB on GPU{self.device}.", flush=True)

    def release(self) -> float:
        self._stop_event.set()
        self._thread.join(timeout=5)
        with self._lock:
            claimed_gb = self._claimed_bytes / 1024 ** 3
            self._chunks.clear()
            self._claimed_bytes = 0
        print(f"[vram-claim] released {claimed_gb:.2f} GB into this process's allocator.", flush=True)
        return claimed_gb
