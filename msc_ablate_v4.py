"""
MSC v4 — Focused Ablation (2 variants)
======================================
  1. v4 Full (baseline)
  2. v4 without Dirichlet + graph entropy
  3. v4 with A_sym (symmetrized adjacency, i.e., v3 behavior)

Uses frozen backbone + Logistic Regression probe (5-fold CV).
Lightweight: 800 samples, 2 clips/video, batch_size=8.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import random
import argparse
import gc

import decord
decord.bridge.set_bridge('torch')

from msc_model import MSCDetector


# ============================================================
# 1. Data loading (lightweight: 2 clips per video)
# ============================================================
def collect_test_files(root_dir, sample_limit=800):
    """Collect balanced test files."""
    root = Path(root_dir)
    real_files, fake_files = [], []
    for mp4 in root.rglob('*.mp4'):
        if mp4.name.startswith('.') or mp4.stat().st_size < 102400:
            continue
        p = str(mp4).lower()
        fake_kw = ['sora', 'cogvideo', 'hunyuanvideo', 'fakevideo']
        real_kw = ['kinetics', 'realvideo', 'voxceleb']
        is_fake = any(k in p for k in fake_kw)
        is_real = any(k in p for k in real_kw)
        if is_fake == is_real:
            continue
        if is_fake:
            fake_files.append(str(mp4))
        else:
            real_files.append(str(mp4))

    random.seed(42)
    random.shuffle(real_files)
    random.shuffle(fake_files)

    n = min(len(real_files), len(fake_files), sample_limit // 2)
    files = [(f, 0) for f in real_files[:n]] + [(f, 1) for f in fake_files[:n]]
    random.shuffle(files)
    return files


def load_clips(video_path, num_clips=2, clip_len=32, spatial_size=224):
    """Load K uniformly-spaced clips from a video."""
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        if total <= clip_len:
            starts = [0] * num_clips
        else:
            step = (total - clip_len) / max(1, num_clips - 1)
            starts = [int(i * step) for i in range(num_clips)]

        clips = []
        for s in starts:
            idx = list(range(s, min(total, s + clip_len)))
            while len(idx) < clip_len:
                idx.append(idx[-1])
            frames = vr.get_batch(idx).float() / 255.0
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]
            C, T_f, H, W = frames.shape
            frames = frames.permute(1, 0, 2, 3)  # [T, C, H, W]
            frames = F.interpolate(frames, size=(spatial_size, spatial_size), mode='bilinear')
            frames = frames.permute(1, 0, 2, 3)  # [C, T, H, W]
            clips.append(frames)
        return torch.stack(clips, dim=0), True
    except Exception:
        return None, False


# ============================================================
# 2. Variant definitions
# ============================================================
VARIANTS = [
    {
        'name': 'v4_Full',
        'desc': 'v4 full model (baseline)',
    },
    {
        'name': 'v4_NoGraphPhysics',
        'desc': 'v4 without Dirichlet energy + graph entropy',
        'zero_graph_physics': True,
    },
    {
        'name': 'v4_A_sym',
        'desc': 'v4 with symmetrized A (A_sym instead of A_tilde)',
        'use_sym': True,
    },
]


# ============================================================
# 3. Monkey-patch helpers (applied per-variant, no side effects)
# ============================================================
def patch_model(model, cfg):
    """Apply variant-specific patches to a fresh model instance."""

    if cfg.get('zero_graph_physics'):
        # Zero out Dirichlet energy and graph entropy features in classifier input
        # These are at indices [256] and [257] in the cls_input vector
        orig_forward = model.forward

        def patched_forward(video, audio_mel=None, mask_ratio=0.0, return_all=False):
            out = orig_forward(video, audio_mel, mask_ratio, return_all)
            if return_all:
                # Zero out dirichlet (256) and entropy (257) in features
                feats = out['features']
                feats[:, 256] = 0.0
                feats[:, 257] = 0.0
                out['features'] = feats
            else:
                feats = out['features']
                feats[:, 256] = 0.0
                feats[:, 257] = 0.0
                out['features'] = feats
            return out

        model.forward = patched_forward

    if cfg.get('use_sym'):
        # Symmetrize A_tilde before GCN: A_sym = (A + A^T) / 2
        orig_forward = model.forward

        def patched_forward(video, audio_mel=None, mask_ratio=0.0, return_all=False):
            # We need to intercept after DualDecoder but before GCN
            # Simplest approach: patch gcn to symmetrize its input
            pass
            return orig_forward(video, audio_mel, mask_ratio, return_all)

        # Actually, patch SimpleGCN.forward to symmetrize A
        orig_gcn_forward = model.gcn.forward

        def patched_gcn_forward(x, A):
            A_sym = (A + A.transpose(1, 2)) / 2.0
            # Re-normalize rows after symmetrization
            A_sym = A_sym / (A_sym.sum(dim=-1, keepdim=True) + 1e-8)
            return orig_gcn_forward(x, A_sym)

        model.gcn.forward = patched_gcn_forward


# ============================================================
# 4. Main ablation runner
# ============================================================
def run_ablation(model_path, data_root, sample_limit=800, num_clips=2, clip_len=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    print(f"[Samples] {sample_limit}, [Clips] {num_clips}")

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    raw = ckpt.get('model_state_dict', ckpt)
    clean = {}
    for k, v in raw.items():
        clean[k.replace('_orig_mod.', '')] = v
    # Filter size-mismatched keys (if checkpoint from different arch)
    ref = MSCDetector(use_audio=False).state_dict()
    clean_filt = {}
    skipped = []
    for k, v in clean.items():
        if k in ref and v.shape == ref[k].shape:
            clean_filt[k] = v
        else:
            skipped.append(k)
    if skipped:
        print(f"[Skip] {len(skipped)} mismatched keys: {skipped[:3]}...")

    # Collect test files
    test_files = collect_test_files(data_root, sample_limit)
    print(f"[Files] {len(test_files)} total")

    # Load all clips as list of tensors (don't concat to avoid OOM)
    all_clips = []
    all_labels = []
    failed = 0
    for path, label in tqdm(test_files, desc="Loading clips"):
        clips, ok = load_clips(path, num_clips, clip_len)
        if ok:
            all_clips.append(clips)        # each: [K, C, T, H, W]
            all_labels.append(label)
        else:
            failed += 1
    if failed:
        print(f"[Warn] {failed} videos failed to load")

    n_per_video = [c.shape[0] for c in all_clips]
    y_all = np.array(all_labels)
    n_total_clips = sum(n_per_video)
    print(f"[Data] {n_total_clips} clips from {len(all_clips)} videos")

    # ── Run each variant ──
    results = []
    for vi, cfg in enumerate(VARIANTS):
        name = cfg['name']
        desc = cfg['desc']
        print(f"\n{'='*55}")
        print(f"[{vi+1}/{len(VARIANTS)}] {name}: {desc}")

        # Fresh model per variant
        model = MSCDetector(use_audio=False)
        model.load_state_dict(clean_filt, strict=False)
        model = model.to(device)
        model.eval()
        patch_model(model, cfg)

        # Extract features in chunks to avoid OOM (30GB if all loaded at once)
        all_feats_list = []  # per-video feature arrays
        chunk_size = 50      # process 50 videos (~100 clips, ~2.5GB) at a time
        B = 8
        with torch.no_grad():
            for chunk_start in tqdm(range(0, len(all_clips), chunk_size),
                                     desc=f"  Extracting", leave=False):
                chunk_end = min(chunk_start + chunk_size, len(all_clips))
                # Collect clips for this chunk
                chunk_clips = []
                chunk_n = []
                for i in range(chunk_start, chunk_end):
                    chunk_clips.append(all_clips[i])
                    chunk_n.append(n_per_video[i])
                chunk_cat = torch.cat(chunk_clips, dim=0)  # [sum(n), C, T, H, W]

                chunk_feats = []
                for i in range(0, len(chunk_cat), B):
                    sub = chunk_cat[i:i + B].to(device)
                    with torch.amp.autocast(device_type=device.type):
                        out = model(sub, mask_ratio=0.0, return_all=False)
                    chunk_feats.append(out['features'].cpu())

                chunk_feats = torch.cat(chunk_feats, dim=0)  # [sum(n), dim]

                # Split back to per-video features via top-k pooling
                clip_start = 0
                for n in chunk_n:
                    video_feats = chunk_feats[clip_start:clip_start + n]
                    k = min(3, n)
                    topk_vals, _ = torch.topk(video_feats, k, dim=0)
                    all_feats_list.append(topk_vals.mean(dim=0).numpy())
                    clip_start += n

                del chunk_cat, chunk_feats

        X = np.stack(all_feats_list)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 5-fold CV with Logistic Regression
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, aucs, aps = [], [], []
        for tr, va in skf.split(X, y_all):
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
            clf.fit(X[tr], y_all[tr])
            y_pred = clf.predict(X[va])
            y_prob = clf.predict_proba(X[va])[:, 1]
            accs.append(accuracy_score(y_all[va], y_pred))
            aucs.append(roc_auc_score(y_all[va], y_prob))
            aps.append(average_precision_score(y_all[va], y_prob))

        r = {
            'name': name, 'desc': desc,
            'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
            'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
            'ap_mean': np.mean(aps),
        }
        results.append(r)
        print(f"  ACC: {r['acc_mean']:.4f} ± {r['acc_std']:.4f}  |  "
              f"AUC: {r['auc_mean']:.4f} ± {r['auc_std']:.4f}  |  "
              f"AP: {r['ap_mean']:.4f}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  MSC v4 Ablation Results")
    print(f"{'='*55}")
    baseline = results[0]
    for r in results:
        d_auc = r['auc_mean'] - baseline['auc_mean']
        sign = '+' if d_auc >= 0 else ''
        print(f"  {r['name']:<22s} | ACC: {r['acc_mean']:.4f} ± {r['acc_std']:.4f} | "
              f"AUC: {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
              f"ΔAUC: {sign}{d_auc:.4f}")
        print(f"    {r['desc']}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='msc_checkpoints/best_stage2.pth')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp')
    parser.add_argument('--sample_limit', type=int, default=800)
    parser.add_argument('--num_clips', type=int, default=2)
    args = parser.parse_args()

    run_ablation(args.model, args.data_root, args.sample_limit, args.num_clips)
