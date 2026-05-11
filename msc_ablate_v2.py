"""Minimal ablation runner — verified working pattern."""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from tqdm import tqdm
from msc_ablate import *
from msc_model import MSCDetector
import gc

apply_all_patches()
device = torch.device("cuda")

# Load checkpoint
ckpt = torch.load("msc_checkpoints/best_stage2.pth", map_location=device, weights_only=False)
raw = ckpt.get("model_state_dict", ckpt)
clean = {k.replace("_orig_mod.", ""): v for k, v in raw.items()}

# Load clips
print("[*] Loading clips...")
test_files = collect_test_files("/root/autodl-tmp", 800)
all_clips, _, all_labels = load_all_clips(test_files, num_clips=2, clip_len=32)
print(f"[*] Loaded {len(all_clips)} clips")

all_cat = torch.cat([c for c in all_clips], dim=0)
print(f"[*] Cat shape: {all_cat.shape}")

# Run ablation variants
results = []
for vi, config in enumerate(ABLATION_CONFIGS):
    name = config["name"]
    desc = config.get("desc", "")
    group = config.get("group", "")
    abl_cfg = {k: v for k, v in config.items() if k not in ("name", "desc", "group")}

    # Setup model
    model = MSCDetector(use_audio=False)
    model.load_state_dict(clean, strict=False)
    model = model.to(device)
    model.ablation = abl_cfg
    model.dual_decoder.ablation = abl_cfg
    model.msc_probe_lh.ablation = abl_cfg
    model.msc_probe_lh.cross_12.ablation = abl_cfg
    model.msc_probe_lh.cross_21.ablation = abl_cfg

    if abl_cfg.get("pool") == "gap":
        model.low_encoder_gap = torch.nn.Sequential(
            torch.nn.Linear(3, 256), torch.nn.LayerNorm(256),
            torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.LayerNorm(256),
        ).to(device)
        model.high_encoder_gap = torch.nn.Sequential(
            torch.nn.Linear(21, 256), torch.nn.LayerNorm(256),
            torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.LayerNorm(256),
        ).to(device)

    model.eval()

    # Extract features
    all_feats_list = []
    B = 16
    with torch.no_grad():
        for i in range(0, len(all_cat), B):
            sub = all_cat[i:i + B].to(device)
            out = model(sub, mask_ratio=0.0, return_all=False)
            all_feats_list.append(out["features"].cpu())

    all_feats = torch.cat(all_feats_list, dim=0)

    X_list = []
    clip_start = 0
    for n in [c.shape[0] for c in all_clips]:
        video_feats = all_feats[clip_start:clip_start + n]
        k = min(3, n)
        topk_vals, _ = torch.topk(video_feats, k, dim=0)
        X_list.append(topk_vals.mean(dim=0).numpy())
        clip_start += n

    X = np.stack(X_list)
    y = all_labels[:len(X)]

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Probe evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, aps = [], [], []
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[va])
        y_prob = clf.predict_proba(X[va])[:, 1]
        accs.append(accuracy_score(y[va], y_pred))
        aucs.append(roc_auc_score(y[va], y_prob))
        aps.append(average_precision_score(y[va], y_prob))

    r = {
        "name": name, "desc": desc, "group": group,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "ap_mean": np.mean(aps), "ap_std": np.std(aps),
    }
    results.append(r)
    print(f"  [{vi+1}/{len(ABLATION_CONFIGS)}] {name:<22s} | ACC: {r['acc_mean']:.4f}+-{r['acc_std']:.4f} | AUC: {r['auc_mean']:.4f}+-{r['auc_std']:.4f} | AP: {r['ap_mean']:.4f}")

# Summary
print("\n" + "=" * 85)
print("  ABLATION RESULTS")
print("=" * 85)
results.sort(key=lambda r: (r["group"], -r["auc_mean"]))
prev_group = None
for r in results:
    if r["group"] != prev_group:
        print(f"\n  -- {r['group']} --")
        prev_group = r["group"]
    print(f"  {r['name']:<22s} ACC={r['acc_mean']:.4f} AUC={r['auc_mean']:.4f} AP={r['ap_mean']:.4f}")

print("\nDone!")
