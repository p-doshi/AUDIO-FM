"""Work around xares==0.1.3's download_zenodo_record always using Zenodo's
bulk "files-archive" endpoint, which Zenodo rejects outright for any
record whose total size exceeds 300MB (confirmed: FMA ~8GB, UrbanSound8K
~7.1GB — both of this project's overlapping X-ARES tasks hit this; likely
affects most substantial audio-benchmark records, not just ours).

Downloads each file individually via Zenodo's per-file API instead
(https://zenodo.org/api/records/{id}/files/{filename}/content, no size
cap), then places the same marker files download_zenodo_record checks for
("{zenodo_id}.zip" and ".unzipped" in the task's env dir) so a normal
`python -m xares.run` invocation recognizes the task as already
downloaded and skips straight past its own (broken, for these sizes)
stage 0.

Usage:
    python -m xares_eval.download_zenodo_task <task_name> <zenodo_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ENV_ROOT = Path("env")


def download_zenodo_task(task_name: str, zenodo_id: str) -> None:
    target_dir = ENV_ROOT / task_name
    target_dir.mkdir(parents=True, exist_ok=True)

    record = requests.get(f"https://zenodo.org/api/records/{zenodo_id}", timeout=30).json()
    files = record["files"]
    print(f"{task_name}: {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB total")

    for f in files:
        dest = target_dir / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  {f['key']}: already present ({f['size'] / 1e6:.1f} MB), skipping")
            continue
        url = f["links"]["self"]
        print(f"  downloading {f['key']} ({f['size'] / 1e6:.1f} MB) ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    out.write(chunk)

    (target_dir / f"{zenodo_id}.zip").touch()
    (target_dir / ".unzipped").touch()
    print(f"{task_name}: ready.")


if __name__ == "__main__":
    download_zenodo_task(sys.argv[1], sys.argv[2])
