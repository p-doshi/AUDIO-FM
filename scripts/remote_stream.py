"""
Streaming access to acoustic buoy data on bigdata6.research.cs.dal.ca.

Purpose: give analysis code (spectrograms, DEMON, PNCC/GFCC, energy/PSD,
EBM training, sanity-check plots, ...) a way to read buoy recordings
directly off the remote host, block by block, without ever materializing
a local copy of the dataset. Nothing this module reads is written to
local disk; callers get numpy arrays / bytes in memory only.

CONFIDENTIALITY (read before assuming more than this provides): this
targets a real, confidential, labelled vessel-acoustic dataset. See the
project's saved memory (`confidential_vessel_data`, `remote_stream_access`)
for the full standing rules; the short version: never persist labels or
per-file details anywhere (git, journal, chat output) beyond "confidential
data was used, N vessel classes" — stream only, never copy audio locally,
delete local label/index files once the job using them is done.

**This script is meant to be run by the data owner, in their own
terminal, not invoked by an assistant/agent's tool-calling loop** — the
`getpass` password prompt and the real audio blocks/labels are not meant
to pass through any intermediary's context. Code that builds on
`RemoteStream` should be handed to the user to run themselves via the CLI
command below, not executed on their behalf.

PRIVACY NOTE (read before assuming more than this provides):
This directory is proprietary and access should stay minimal. What this
module does on the client side to help with that:
  - Uses the SFTP subsystem only (paramiko), never a remote shell/exec.
    There is no bash history, no remote `ls`/`cat`/`find` command trail,
    because no shell is ever invoked on the remote end.
  - Never writes a local copy of any remote file.
  - Never logs specific filenames anywhere by default (only file *counts*
    are printed unless verbose=True is explicitly requested).
  - Prompts for the password interactively (getpass) — it is never
    hardcoded, passed as a CLI arg, or put in an env var, so it can't
    leak via shell history or `ps aux` on either machine.
What this module CANNOT do: guarantee invisibility from the server
owner. sshd (and auditd, if enabled) is configured and controlled by
the remote host's admin, not by this client. If they turn on verbose
SFTP logging, they can see which files were opened, regardless of what
this script does. If true confidentiality against the directory owner
is a hard requirement, that has to be resolved at the server/infra
level (ask what sshd/auditd logging is enabled) — this script only
minimizes the footprint that is actually under our control.

REMOTE_BASE_DIR is deliberately read from an environment variable rather
than hardcoded here, even though this file has no credentials in it —
the exact remote path ties to a specific station/collaboration and is
treated as sensitive-adjacent by default (see remote_stream_access
memory) unless/until confirmed otherwise. Set it yourself, don't commit
it:
    export AUDIO_COMP_REMOTE_BASE_DIR="<the real path>"

Usage:
    from remote_stream import RemoteStream

    with RemoteStream() as remote:
        names = remote.list()                      # filenames only
        for block in remote.stream_audio(names[0]):
            ...                                      # numpy array, no disk write

Dependencies (not auto-installed — install yourself before use):
    pip install --user paramiko soundfile numpy
"""

from __future__ import annotations

import contextlib
import fnmatch
import getpass
import os
import sys

import paramiko

REMOTE_HOST = "bigdata6.research.cs.dal.ca"
REMOTE_USER = "parth"
REMOTE_BASE_DIR = os.environ.get("AUDIO_COMP_REMOTE_BASE_DIR")

_KNOWN_HOSTS = os.path.expanduser("~/.ssh/known_hosts")


def _verify_host_key(client: paramiko.SSHClient, hostname: str) -> None:
    """Trust-on-first-use with an explicit fingerprint prompt, instead of
    silently auto-accepting unknown host keys."""
    if os.path.exists(_KNOWN_HOSTS):
        client.load_host_keys(_KNOWN_HOSTS)

    host_keys = client.get_host_keys()
    if host_keys.lookup(hostname):
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        return

    # No entry yet for this host — ask the user to confirm, same as a
    # normal `ssh` first-connection prompt, rather than trusting blindly.
    transport_probe = paramiko.Transport((hostname, 22))
    try:
        transport_probe.start_client()
        key = transport_probe.get_remote_server_key()
    finally:
        transport_probe.close()

    fingerprint = key.get_fingerprint().hex(":")
    print(f"The authenticity of host '{hostname}' can't be established.")
    print(f"{key.get_name()} key fingerprint is {fingerprint}.")
    answer = input("Are you sure you want to continue connecting (yes/no)? ")
    if answer.strip().lower() not in ("yes", "y"):
        raise RuntimeError("Host key not confirmed by user — aborting connection.")

    host_keys.add(hostname, key.get_name(), key)
    os.makedirs(os.path.dirname(_KNOWN_HOSTS), exist_ok=True)
    host_keys.save(_KNOWN_HOSTS)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())


class RemoteStream:
    """Streaming handle onto REMOTE_BASE_DIR. Open via `with RemoteStream() as r:`."""

    def __init__(self, base_dir: str | None = None, verbose: bool = False):
        # base_dir is only required by the relative-path methods (list(),
        # open()) -- callers using open_absolute() exclusively (e.g. an
        # external manifest that already stores absolute remote paths)
        # don't need it, so this doesn't raise until one of those methods
        # is actually called without it.
        self.base_dir = base_dir or REMOTE_BASE_DIR
        self.verbose = verbose
        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def __enter__(self) -> "RemoteStream":
        password = getpass.getpass(f"Password for {REMOTE_USER}@{REMOTE_HOST}: ")

        client = paramiko.SSHClient()
        _verify_host_key(client, REMOTE_HOST)

        client.connect(
            REMOTE_HOST,
            username=REMOTE_USER,
            password=password,
            look_for_keys=False,
            allow_agent=False,
        )
        del password

        self._client = client
        self._sftp = client.open_sftp()
        if self.verbose:
            print(f"Connected to {REMOTE_HOST}.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sftp is not None:
            self._sftp.close()
        if self._client is not None:
            self._client.close()
        if self.verbose:
            print("Disconnected.")

    def _require_base_dir(self) -> str:
        if not self.base_dir:
            raise RuntimeError(
                "No base_dir given and AUDIO_COMP_REMOTE_BASE_DIR is not set — "
                "export it yourself, it is not committed to this repo. (Not needed "
                "if you only use open_absolute() with already-absolute paths.)"
            )
        return self.base_dir

    def _remote_path(self, name: str) -> str:
        return f"{self._require_base_dir()}/{name}"

    def list(self, pattern: str = "*") -> list[str]:
        """List filenames in the target directory (SFTP LISTDIR only, no shell)."""
        assert self._sftp is not None, "use inside a `with RemoteStream() as r:` block"
        names = self._sftp.listdir(self._require_base_dir())
        return sorted(n for n in names if fnmatch.fnmatch(n, pattern))

    @contextlib.contextmanager
    def open(self, name: str, mode: str = "rb"):
        """Raw streaming file handle — reads come from the remote host on
        demand, nothing is buffered to local disk."""
        assert self._sftp is not None, "use inside a `with RemoteStream() as r:` block"
        f = self._sftp.open(self._remote_path(name), mode)
        f.set_pipelined(True)  # prefetch ahead for sequential streaming reads
        try:
            yield f
        finally:
            f.close()

    @contextlib.contextmanager
    def open_absolute(self, path: str, mode: str = "rb"):
        """Same as `open`, but for a path that is already absolute on the
        remote host (e.g. a flac_path column from an external manifest),
        rather than one relative to `base_dir`."""
        assert self._sftp is not None, "use inside a `with RemoteStream() as r:` block"
        f = self._sftp.open(path, mode)
        f.set_pipelined(True)
        try:
            yield f
        finally:
            f.close()

    def stream_audio(self, name: str, block_size: int = 65536, dtype: str = "float32"):
        """Yield successive numpy blocks of an audio file streamed straight
        off the remote host (via `soundfile`), never touching local disk."""
        import soundfile as sf

        with self.open(name) as f:
            with sf.SoundFile(f) as audio:
                yield from audio.blocks(blocksize=block_size, dtype=dtype)

    def read_bytes(self, name: str) -> bytes:
        """Read a small non-audio file (e.g. instrument log/xml) fully into
        memory — still never written to local disk."""
        with self.open(name) as f:
            return f.read()


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    with RemoteStream(verbose=verbose) as remote:
        files = remote.list()
        print(f"Connected. {len(files)} entries visible in target directory.")
        if verbose:
            for n in files:
                print(f"  {n}")
