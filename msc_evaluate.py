"""
MSC Framework — Multi-Clip Evaluation
Top-3 Mean Pooling, reports AUC, ACC, per-source breakdown, MSC/S_vn stats.
"""

import torch
import decord
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import torch.nn.functional as F
import random
import argparse
from collections import defaultdict

from msc_model import MSCDetector

decord.bridge.set_bridge('torch')


def load_multiclip(video_path, num_clips=8, clip_len=32, spatial_size=224):
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
            frames = frames.permute(3, 0, 1, 2)          # [C, T, H, W]
            # Treat T as batch dim for 4D bilinear interpolation
            C, T_f, H, W = frames.shape
            frames = frames.permute(1, 0, 2, 3)          # [T, C, H, W]
            frames = F.interpolate(
                frames, size=(spatial_size, spatial_size), mode='bilinear'
            )
            frames = frames.permute(1, 0, 2, 3)          # [C, T, H, W]
            clips.append(frames)
        return torch.stack(clips, dim=0), True
    except Exception:
        return None, False


def get_source(p):
    p = p.lower()
    if 'sora' in p: return 'Sora'
    if 'cogvideo' in p: return 'CogVideo'
    if 'hunyuanvideo' in p: return 'HunyuanVideo'
    if 'kinetics' in p: return 'Kinetics(Real)'
    return 'Other'


@torch.no_grad()
def evaluate(model_path, data_root, sample_limit=2000, num_clips=8,
             clip_len=32, output_csv=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    raw = ckpt.get('model_state_dict', ckpt)
    clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

    # Detect audio mode from classifier weight shape
    use_audio = False
    for k, v in clean.items():
        if 'classifier.0.weight' in k:
            use_audio = (v.shape[1] == 260)  # 256 + 4 features
            break
    print(f"[Mode] {'Audio-Visual' if use_audio else 'Pure Video'}")

    model = MSCDetector(use_audio=use_audio)
    model.load_state_dict(clean, strict=False)
    model = model.to(device)
    model.eval()

    # Collect test files
    all_f = list(Path(data_root).rglob('*.mp4'))
    random.seed(42)
    random.shuffle(all_f)
    test = []
    for f in all_f:
        if f.name.startswith('.') or f.stat().st_size < 102400:
            continue
        p = str(f).lower()
        fake_kw = ['sora', 'cogvideo', 'hunyuanvideo', 'fakevideo']
        real_kw = ['kinetics', 'realvideo', 'voxceleb']
        is_fake = any(k in p for k in fake_kw)
        is_real = any(k in p for k in real_kw)
        if is_fake != is_real:
            test.append((str(f), 1 if is_fake else 0))
        if len(test) >= sample_limit:
            break

    print(f"[Samples] {len(test)}")
    print(f"[Inference] {num_clips} clips × {clip_len}frames...")

    results = []
    failed = 0
    partial_csv = "msc_eval_partial.csv"
    for vi, (path, true_label) in enumerate(tqdm(test, desc="Eval")):
        batch, ok = load_multiclip(path, num_clips, clip_len)
        if not ok:
            failed += 1
            continue
        try:
            batch = batch.to(device)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(batch, mask_ratio=0.0, return_all=False)
                logits = outputs['logits'].squeeze(-1)

            probs = torch.sigmoid(logits).cpu().numpy()
            sorted_p = np.sort(probs.flatten())
            final_prob = np.mean(sorted_p[-3:]) if len(sorted_p) >= 3 else np.mean(sorted_p)
            pred = 1 if final_prob > 0.5 else 0

            msc_val = outputs['msc_lh'].mean().item()
            svn_val = outputs.get('S_vn', torch.zeros(1)).mean().item()

            results.append({
                'Path': path,
                'Source': get_source(path),
                'True': true_label,
                'Pred': pred,
                'Prob': final_prob,
                'MSC': msc_val,
                'S_vn': svn_val,
                'OK': true_label == pred,
            })
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n  [Warn] {path.split('/')[-1][:60]}: {e}")

        # Save partial results every 100 videos
        if (vi + 1) % 100 == 0 and results:
            pd.DataFrame(results).to_csv(partial_csv, index=False)

    if failed:
        print(f"\n[Note] {failed} videos skipped")

    df = pd.DataFrame(results)
    yt, yp, yprob = df['True'].values, df['Pred'].values, df['Prob'].values

    print("\n" + "=" * 55)
    print("  MSC Merged Framework — Evaluation")
    print("=" * 55)

    print("\n[1] Overall")
    print(classification_report(yt, yp, target_names=['Real', 'Fake']))

    try:
        auc = roc_auc_score(yt, yprob)
        ap = average_precision_score(yt, yprob)
        print(f"AUC: {auc:.4f}  |  AP: {ap:.4f}")
    except Exception:
        auc = None

    cm = confusion_matrix(yt, yp)
    if cm.shape == (2, 2):
        print(f"FP(Real→Fake): {cm[0][1]}/{cm[0].sum()} ({cm[0][1]/cm[0].sum()*100:.1f}%)")
        print(f"FN(Fake→Real): {cm[1][0]}/{cm[1].sum()} ({cm[1][0]/cm[1].sum()*100:.1f}%)")

    print(f"Accuracy: {(yt==yp).mean()*100:.2f}%")

    print("\n[2] Per Source")
    for src in sorted(df['Source'].unique()):
        sub = df[df['Source'] == src]
        acc = sub['OK'].mean() * 100
        r = sub[sub['True'] == 0]
        f = sub[sub['True'] == 1]
        msc_gap = (r['MSC'].mean() - f['MSC'].mean()) if len(r) and len(f) else 0
        print(f"  {src:<18} Acc:{acc:.1f}% n={len(sub):<4} "
              f"MSC_R:{r['MSC'].mean():.3f} MSC_F:{f['MSC'].mean():.3f} Gap:{msc_gap:.3f}")

    out = output_csv or "msc_eval_results.csv"
    df.to_csv(out, index=False)
    print(f"\n[Saved] {out}")
    return df, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='msc_checkpoints/best_stage2.pth')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp')
    parser.add_argument('--sample_limit', type=int, default=2000)
    parser.add_argument('--num_clips', type=int, default=8)
    parser.add_argument('--clip_len', type=int, default=32)
    parser.add_argument('--output_csv', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model, args.data_root, args.sample_limit,
             args.num_clips, args.clip_len, args.output_csv)
