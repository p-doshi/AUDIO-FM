"""Layer-wise frozen embeddings: one mean-pooled vector per clip for every block of the encoder.

Same clips, order, preprocessing and adapter as the cached final-layer embeddings
(<EMB>/<dataset>__<model>.npz); the adapter's own forward pass is run unchanged and every block's output is captured
with forward hooks. Blocks = the longest nn.ModuleList that fires during the forward pass (transformer layers);
models without one (CNNs) fall back to the top-level children that fire, in call order. Pooling: mean over every axis
except batch and feature (time for (B, T, D) or (T, B, D); frequency and time for (B, C, F, T)).
Usage: python layerwise_extract.py --dataset <d> --model <m> --emb <cached embedding dir> --out <dir>
Output: OUT/<dataset>__<model>.npz with layer_<i> arrays (n, d_i) float16, clip_ids, labels, block_path,
and final_check = Spearman between the last captured layer's RDM and the adapter output's RDM on 500 clips.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
from audio_comp.models import get_model_class  # noqa: E402
from audio_comp.pipelines.finetune_data_geometry import BATCH_SIZE, MODEL_BATCH_SIZE_OVERRIDES  # noqa: E402
from audio_comp.pipelines.generic_lora_trainer import read_batch_skip_bad  # noqa: E402
from audio_comp.pipelines.rsa_cka_finetuned import CATEGORY_LOADERS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
outp = f"{a.out}/{a.dataset}__{a.model}.npz"
if os.path.exists(outp):
    print("exists", outp); sys.exit(0)

z = np.load(f"{a.emb}/{a.dataset}__{a.model}.npz", allow_pickle=True)
ids = [str(c) for c in z["clip_ids"]]
all_clips, _, _, _, native_sr, max_dur = CATEGORY_LOADERS[a.dataset]()
by_path = {str(c["path"]): c for c in all_clips}
clips = [by_path[i] for i in ids if i in by_path]
if len(clips) != len(ids):  # cached ids written under another root: match on the file stem (unique outside MIMII)
    by_stem = {os.path.splitext(os.path.basename(str(c["path"])))[0]: c for c in all_clips}
    clips = [by_stem[os.path.splitext(os.path.basename(i))[0]] for i in ids if os.path.splitext(os.path.basename(i))[0] in by_stem]
print(a.dataset, a.model, "clips", len(clips), "of", len(ids), flush=True)
assert len(clips) == len(ids), "clip ids do not match the loader paths"
if os.environ.get("LIMIT"):
    clips, ids = clips[:int(os.environ["LIMIT"])], ids[:int(os.environ["LIMIT"])]
    z = {"embeddings": z["embeddings"][:len(ids)], "labels": z["labels"][:len(ids)]}

dev = "cuda" if torch.cuda.is_available() else "cpu"
ad = get_model_class(a.model)(device=dev)
ad.load()
roots = {k: v for k, v in vars(ad).items() if isinstance(v, torch.nn.Module)}
mlists = []
for rk, root in roots.items():
    for name, mod in root.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
            mlists.append((f"{rk}.{name}", mod))
print("candidate ModuleLists:", [(n, len(m)) for n, m in mlists], flush=True)

store, fired = {}, set()


def pool(t, B):
    if isinstance(t, (tuple, list)):
        t = t[0]
    if not torch.is_tensor(t) or t.dim() < 2:
        return None
    t = t.detach().float()
    if t.dim() == 2:
        return t if t.shape[0] == B else None
    if t.dim() == 3:
        if t.shape[0] == B:
            return t.mean(1)
        if t.shape[1] == B:
            return t.mean(0)
        return None
    if t.shape[0] == B:
        return t.mean(dim=tuple(range(2, t.dim())))
    return None


def hook(key, B_ref):
    def f(_m, _i, o):
        fired.add(key)
        p = pool(o, B_ref[0])
        if p is not None:
            store.setdefault(key, []).append(p.cpu().numpy().astype(np.float16))
    return f


B_ref = [0]
bs = MODEL_BATCH_SIZE_OVERRIDES.get(a.model, BATCH_SIZE)
handles = []
for n, ml in mlists:
    for i, blk in enumerate(ml):
        handles.append(blk.register_forward_hook(hook(f"{n}[{i}]", B_ref)))
for rk, root in roots.items():
    for cn, ch in root.named_children():
        handles.append(ch.register_forward_hook(hook(f"top:{rk}.{cn}", B_ref)))

# probe pass on one batch to decide which blocks to keep
w, kept = read_batch_skip_bad(clips[:bs], native_sr, max_dur)
B_ref[0] = len(w)
final0 = ad.embed_batch(w, native_sr)
fired_lists = [(n, ml) for n, ml in mlists if all(f"{n}[{i}]" in store for i in range(len(ml)))]
# convolutional waveform front-ends (e.g. SEW's 13 conv layers) are not encoder blocks; keep them only if nothing else fired
non_conv = [(n, ml) for n, ml in fired_lists if "feature_extractor" not in n and "conv_layers" not in n]
fired_lists = non_conv or fired_lists
if fired_lists:
    n, ml = max(fired_lists, key=lambda x: len(x[1]))
    keys = [f"{n}[{i}]" for i in range(len(ml))]
    block_path = n
else:
    keys = [k for k in store if k.startswith("top:")]
    block_path = "top-level children"
print("blocks:", block_path, len(keys), flush=True)
store.clear()

finals = []
for start in range(0, len(clips), bs):
    batch = clips[start:start + bs]
    w, kept = read_batch_skip_bad(batch, native_sr, max_dur)
    assert len(kept) == len(batch), "a clip failed to decode that decoded for the cached embeddings"
    B_ref[0] = len(w)
    with torch.no_grad():
        finals.append(np.asarray(ad.embed_batch(w, native_sr), dtype=np.float32))
    for k in list(store):
        if k not in keys:
            store.pop(k)
    if dev == "cuda":
        torch.cuda.empty_cache()
    if (start // bs) % 50 == 0:
        print(f"  {start + len(batch)}/{len(clips)}", flush=True)
for h in handles:
    h.remove()

layers = {f"layer_{i}": np.concatenate(store[k]) for i, k in enumerate(keys)}
fin = np.concatenate(finals)
m = min(500, len(fin), len(z["embeddings"]))
chk = spearmanr(pdist(fin[:m], "correlation"), pdist(layers[f"layer_{len(keys) - 1}"][:m].astype(np.float32), "correlation"))[0]
cached = spearmanr(pdist(fin[:m], "correlation"), pdist(z["embeddings"][:m].astype(np.float32), "correlation"))[0]
print("final_check (last block vs adapter output):", chk, " adapter output vs cached:", cached, flush=True)
if os.path.exists(outp):
    print("exists (written by the other array)", outp); sys.exit(0)
tmp = f"{outp[:-4]}.{os.environ.get('SLURM_JOB_ID', 'x')}.tmp.npz"
np.savez(tmp, clip_ids=np.array(ids), labels=z["labels"], block_path=block_path, n_layers=len(keys),
         final_check=chk, cached_check=cached, **layers)
os.replace(tmp, outp)
print("saved", outp, {k: v.shape for k, v in list(layers.items())[:2]}, flush=True)
