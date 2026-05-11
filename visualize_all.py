"""
MSC Framework — Complete Visualization Suite (Memory-Safe)
===========================================================
Generates all method visualizations in batches.
Each figure runs independently to avoid OOM.
"""

import torch, numpy as np, random, gc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

torch.manual_seed(42); np.random.seed(42); random.seed(42)
device = torch.device('cuda')
SAVE = Path("visualizations"); SAVE.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'figure.dpi':150,
                      'savefig.dpi':200,'savefig.bbox':'tight','font.family':'serif'})

# ── Helper: load model once ──
def load_model():
    from msc_model import MSCDetector
    ckpt = torch.load("msc_checkpoints_purevideo_v4/best_stage2.pth",
                       map_location=device, weights_only=False)
    raw = ckpt.get('model_state_dict', ckpt)
    clean = {k.replace('_orig_mod.',''): v for k, v in raw.items()}
    model = MSCDetector(use_audio=False)
    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    return model

# ── Helper: collect sample paths (no tensors, memory safe) ──
def get_sample_paths():
    """Return (real_paths, fake_paths) lists. Excludes FakeAVCeleb."""
    from pathlib import Path
    root = Path("/root/autodl-tmp")
    real, fake = [], []
    for mp4 in root.rglob('*.mp4'):
        name = mp4.name
        if name.startswith('.'): continue
        try:
            if mp4.stat().st_size < 102400: continue
        except: continue
        p = str(mp4).lower()
        # Exclude FakeAVCeleb
        if 'fakeavceleb' in p: continue
        is_fake = any(k in p for k in ['sora','cogvideo','hunyuanvideo','fakevideo'])
        is_real = any(k in p for k in ['kinetics','realvideo','voxceleb'])
        if is_fake == is_real: continue
        (real if is_real else fake).append(str(mp4))
    random.shuffle(real); random.shuffle(fake)
    return real, fake

# ── Helper: load one video as tensor ──
import decord; decord.bridge.set_bridge('torch')
def load_video(path, clip_len=32, spatial_size=224):
    vr = decord.VideoReader(path, ctx=decord.cpu(0), width=spatial_size, height=spatial_size)
    total = len(vr)
    start = max(0, (total - clip_len) // 2)
    indices = list(range(start, min(total, start + clip_len)))
    while len(indices) < clip_len: indices.append(indices[-1])
    frames = vr.get_batch(indices).float() / 255.0
    return frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]

# ============================================================
print("=== Collecting sample paths ===")
real_paths, fake_paths = get_sample_paths()
print(f"Real: {len(real_paths)}, Fake: {len(fake_paths)}")

model = load_model()

# ============================================================
# Fig 1: DWT 8-subband (real vs fake)
# ============================================================
print("Fig 1: DWT...")
real_v = load_video(real_paths[0]).to(device)
fake_v = load_video(fake_paths[0]).to(device)
with torch.no_grad():
    r_dwt = model.dwt_3d(real_v)[0]  # [24,16,112,112]
    f_dwt = model.dwt_3d(fake_v)[0]

names = ['LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH']
fig, axes = plt.subplots(2, 8, figsize=(20, 5))
for i in range(8):
    for row, dwt, label in [(0, r_dwt, 'Real'), (1, f_dwt, 'Fake')]:
        r = dwt[i, 0].cpu().numpy()
        g = dwt[i+8, 0].cpu().numpy()
        b = dwt[i+16, 0].cpu().numpy()
        rgb = np.stack([r, g, b], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        axes[row, i].imshow(rgb)
        axes[row, i].set_title(names[i] if row == 0 else '', fontsize=10)
        axes[row, i].axis('off')
axes[0,0].set_ylabel('Real', fontsize=13, fontweight='bold')
axes[1,0].set_ylabel('Fake', fontsize=13, fontweight='bold')
fig.suptitle('3D Haar DWT — 8 Subband Decomposition', fontsize=15, y=1.01)
plt.tight_layout(); fig.savefig(SAVE/'Fig1_DWT_Subbands.png'); plt.close()
print("  OK"); gc.collect()

# ============================================================
# Fig 2: A_tilde adjacency + degree distribution (1 video)
# ============================================================
print("Fig 2: Adjacency matrix...")
with torch.no_grad():
    out = model(real_v, mask_ratio=0.0, return_all=True)
A = out['A_tilde'][0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
A_ds = A.reshape(16, 64, 16, 64).mean(axis=(1, 3))
im = axes[0].imshow(A_ds, cmap='hot', aspect='auto', vmin=0, vmax=A_ds.max())
axes[0].set_title(f'A_tilde (64x64 avg pool)\n{A[A>0].size/A.size*100:.1f}% non-zero'); plt.colorbar(im, ax=axes[0])

A_z = A[:128, :128]
im2 = axes[1].imshow(A_z, cmap='hot', aspect='auto', vmin=0, vmax=A_z.max())
axes[1].set_title('A_tilde [0:128, 0:128]'); plt.colorbar(im2, ax=axes[1])

# Edge weight rank plot (histogram fails: A is row-stochastic, degrees all = 1.0)
rng = np.random.RandomState(42)
for ni in range(5):
    idx = rng.choice(1024)
    w = sorted(A[idx][A[idx] > 0], reverse=True)
    axes[2].plot(range(1, len(w)+1), w, '.-',
                 label=f'Node {idx}', markersize=4, linewidth=1.2)
axes[2].set_xlabel('Neighbor rank')
axes[2].set_ylabel('Edge weight')
axes[2].set_title('Top-K Edge Weights (5 nodes)')
axes[2].legend(fontsize=7)

fig.suptitle('Path A — Adjacency Matrix & Top-K Sparsification', fontsize=15)
plt.tight_layout(); fig.savefig(SAVE/'Fig2_Adjacency_TopK.png'); plt.close()
print("  OK"); del out; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Fig 3: Joint Laplacian + eigenvalue spectrum
# ============================================================
print("Fig 3: Joint Laplacian...")
with torch.no_grad():
    out = model(real_v, mask_ratio=0.0, return_all=True)
    A_j = out['A_joint_lh'][0].cpu().numpy()
    msc_v = out['msc_lh'][0].item()
    svn_v = out['S_vn'][0].item()
D_vec = A_j.sum(axis=1); L_j = np.diag(D_vec) - A_j; ev = np.linalg.eigvalsh(L_j)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
im = axes[0].imshow(A_j, cmap='coolwarm', aspect='auto', vmin=-0.05, vmax=1.0)
axes[0].set_title('A_joint (32x32)\nLow(0-15) + High(16-31)')
axes[0].axhline(15.5, color='k', lw=0.5); axes[0].axvline(15.5, color='k', lw=0.5); plt.colorbar(im, ax=axes[0])

cols = ['red' if i <= 1 else 'steelblue' for i in range(32)]
axes[1].bar(range(32), ev, color=cols)
axes[1].set_xlabel('Index'); axes[1].set_ylabel('λ')
axes[1].set_title(f'Eigenvalues: λ₀={ev[0]:.2e}, λ₁(MSC)={ev[1]:.4f}')

A_cross = A_j[:16, 16:]
im3 = axes[2].imshow(A_cross, cmap='coolwarm', aspect='auto')
axes[2].set_title('A_cross (Low→High, 16x16)'); plt.colorbar(im3, ax=axes[2])

fig.suptitle('Path B — Joint Laplacian & MSC Probe', fontsize=15)
plt.tight_layout(); fig.savefig(SAVE/'Fig3_Joint_Laplacian.png'); plt.close()
print("  OK"); del out; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Fig 4: MSC gap histogram (100 samples each)
# ============================================================
print("Fig 4: MSC distribution...")
msc_r, msc_f = [], []
with torch.no_grad():
    for i in tqdm(range(100), desc="  MSC", leave=False):
        vr = load_video(real_paths[i]).to(device)
        vf = load_video(fake_paths[i]).to(device)
        msc_r.append(model(vr, mask_ratio=0.0, return_all=False)['msc_lh'].item())
        msc_f.append(model(vf, mask_ratio=0.0, return_all=False)['msc_lh'].item())
        del vr, vf
    torch.cuda.empty_cache()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(msc_r, bins=25, alpha=0.6, label='Real', color='steelblue', density=True)
ax.hist(msc_f, bins=25, alpha=0.6, label='Fake', color='coral', density=True)
ax.axvline(np.mean(msc_r), color='steelblue', linestyle='--', lw=2)
ax.axvline(np.mean(msc_f), color='coral', linestyle='--', lw=2)
ax.set_xlabel('MSC (λ₁)'); ax.set_ylabel('Density')
ax.set_title(f'MSC Distribution (N=100)\nReal μ={np.mean(msc_r):.4f}, Fake μ={np.mean(msc_f):.4f}, Gap={np.mean(msc_r)-np.mean(msc_f):.4f}')
ax.legend()
plt.tight_layout(); fig.savefig(SAVE/'Fig4_MSC_Gap.png'); plt.close()
print("  OK"); gc.collect()

# ============================================================
# Fig 5: t-SNE of features (N=200, memory-safe batches)
# ============================================================
print("Fig 5: t-SNE...")
from sklearn.manifold import TSNE
N = 200
feats, labels = [], []
with torch.no_grad():
    for i in tqdm(range(N//2), desc="  Real feats", leave=False):
        v = load_video(real_paths[i]).to(device)
        feats.append(model(v, mask_ratio=0.0, return_all=False)['features'][0].cpu().numpy()); labels.append(0); del v
    for i in tqdm(range(N//2), desc="  Fake feats", leave=False):
        v = load_video(fake_paths[i]).to(device)
        feats.append(model(v, mask_ratio=0.0, return_all=False)['features'][0].cpu().numpy()); labels.append(1); del v
    torch.cuda.empty_cache()

X = np.stack(feats); y = np.array(labels)
Xg, Xm = X[:, :256], X  # h_graph only, full features

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, title in [(axes[0], TSNE(2, perplexity=30, random_state=42, max_iter=800).fit_transform(Xg),
                           't-SNE of h_graph [256-dim]'),
                          (axes[1], TSNE(2, perplexity=30, random_state=42, max_iter=800).fit_transform(Xm),
                           't-SNE of Full Features [260-dim]')]:
    for lbl, c, nm in [(0, 'steelblue', 'Real'), (1, 'coral', 'Fake')]:
        m = y == lbl
        ax.scatter(data[m, 0], data[m, 1], c=c, label=nm, alpha=0.6, s=12)
    ax.set_title(title, fontsize=11); ax.legend()
fig.suptitle('Feature Space Visualization', fontsize=15)
plt.tight_layout(); fig.savefig(SAVE/'Fig5_Feature_tSNE.png'); plt.close()
print("  OK"); del X, feats; gc.collect()

# ============================================================
# Fig 6: Dual Routing comparison
# ============================================================
print("Fig 6: Dual routing...")
orig_fwd = model.dual_decoder.forward
@torch.no_grad()
def trace_fwd(self, x):
    B, N, D = x.shape
    sc = self.scale_sem ** 0.5
    q_s = self.q_sem(x) * sc; k_s = self.k_sem(x) * sc
    A_sem = torch.bmm(q_s, k_s.transpose(1, 2))
    u = self.w_i(x); v = self.w_j(x)
    u_r = self.rope(u); v_r = self.rope(v)
    A_kin = torch.bmm(u_r, v_r.transpose(1,2)) * self.scale_kin
    logits = self.alpha * A_sem + (1-self.alpha) * A_kin
    A_hat = torch.softmax(logits.float().clamp(-100,100), dim=-1).to(x.dtype)
    tv, ti = torch.topk(A_hat, self.k_sparse, dim=-1)
    A_s = torch.zeros_like(A_hat).scatter_(-1, ti, tv)
    A_s = A_s / (A_s.sum(dim=-1, keepdim=True) + 1e-8)
    self._sem = A_sem; self._kin = A_kin; self._fused = A_s; self._alp = self.alpha.item()
    return torch.nan_to_num(A_s, nan=0.0)

model.dual_decoder.forward = trace_fwd.__get__(model.dual_decoder)
with torch.no_grad():
    X_main, _ = model.dsft(real_v)
    _ = model.dual_decoder(X_main)
    sem = model.dual_decoder._sem[0].cpu().numpy()
    kin = model.dual_decoder._kin[0].cpu().numpy()
    fused = model.dual_decoder._fused[0].cpu().numpy()
    alpha = model.dual_decoder._alp
model.dual_decoder.forward = orig_fwd

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
crop = slice(0, 128)
for ax, mat, title in [
    (axes[0], sem[crop,crop], f'Semantic Routing\n(content similarity)'),
    (axes[1], kin[crop,crop], f'Kinematic Routing\n(3D-RoPE position-aware)'),
    (axes[2], fused[crop,crop], f'Fused A_tilde (α={alpha:.2f})\n(Top-K=30, directed)'),
]:
    im = ax.imshow(mat, cmap='hot', aspect='auto', vmin=0, vmax=np.percentile(mat, 99))
    ax.set_title(title, fontsize=11); plt.colorbar(im, ax=ax)
fig.suptitle('DualDecoder — Semantic vs Kinematic Routing', fontsize=14)
plt.tight_layout(); fig.savefig(SAVE/'Fig6_Dual_Routing.png'); plt.close()
print("  OK"); gc.collect(); torch.cuda.empty_cache()

# ============================================================
# Fig 7: Confusion matrix
# ============================================================
print("Fig 7: Confusion matrix...")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true, y_pred = [], []
with torch.no_grad():
    for i in tqdm(range(min(300, len(real_paths))), desc="  Eval", leave=False):
        vr = load_video(real_paths[i]).to(device)
        prob = torch.sigmoid(model(vr, mask_ratio=0.0, return_all=False)['logits']).item()
        y_true.append(0); y_pred.append(1 if prob > 0.5 else 0); del vr
    for i in tqdm(range(min(300, len(fake_paths))), desc="  Eval", leave=False):
        vf = load_video(fake_paths[i]).to(device)
        prob = torch.sigmoid(model(vf, mask_ratio=0.0, return_all=False)['logits']).item()
        y_true.append(1); y_pred.append(1 if prob > 0.5 else 0); del vf
    torch.cuda.empty_cache()

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['Real','Fake']).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix (N={len(y_true)})')
plt.tight_layout(); fig.savefig(SAVE/'Fig7_Confusion.png'); plt.close()
print("  OK")

# ============================================================
# Fig 8: Ablation waterfall
# ============================================================
print("Fig 8: Ablation waterfall...")
data = [
    ('Full Model (baseline)', 0.8537, 0.0),
    ('Top-K → Full graph', 0.8006, -5.31),
    ('GCN → Mean pool', 0.8066, -4.71),
    ('3D-RoPE removed', 0.8261, -2.76),
    ('Directed → Symmetric A', 0.7855, -2.10),
    ('Kinematic removed', 0.8355, -1.82),
    ('Semantic removed', 0.8551, +0.14),
    ('Dirichlet+Entropy removed', 0.8068, +0.00),
    ('Path B only', 0.5705, -28.32),
]
fig, ax = plt.subplots(figsize=(12, 7))
deltas = [d[2] for d in data]
colors = ['#2196F3' if d >= 0 else '#F44336' for d in deltas]
bars = ax.barh(range(len(data)), deltas, color=colors, height=0.6)
ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data])
ax.set_xlabel('ΔAUC (%)'); ax.set_title('Ablation Study — Component Importance')
ax.axvline(0, color='black', lw=1)
for bar, delta in zip(bars, deltas):
    x_pos = bar.get_width() + (0.5 if delta >= 0 else -0.5)
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{delta:+.2f}%',
            ha='left' if delta >= 0 else 'right', va='center', fontsize=10)
plt.tight_layout(); fig.savefig(SAVE/'Fig8_Ablation_Waterfall.png'); plt.close()
print("  OK")

# ============================================================
# Fig 9: Classifier weight importance
# ============================================================
print("Fig 9: Feature importance...")
w1 = model.classifier[0].weight.data.cpu().numpy()
fi = np.abs(w1).sum(axis=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(260), fi, color='steelblue', width=0.8)
axes[0].axvline(255.5, color='red', linestyle='--', label='h_graph(256) boundary')
axes[0].set_xlabel('Feature dim'); axes[0].set_ylabel('Sum |W₁|'); axes[0].legend()
axes[1].bar(range(256, 260), fi[256:], color=['coral','orange','green','purple'])
axes[1].set_xticks(range(256, 260))
axes[1].set_xticklabels(['Dirichlet','GraphEntropy','MSC_lh','S_vn'], fontsize=9)
axes[1].set_ylabel('Sum |W₁|'); axes[1].set_title('Last 4 Features (Path B + Physics)')
fig.suptitle('Classifier Weight Analysis', fontsize=14)
plt.tight_layout(); fig.savefig(SAVE/'Fig9_Feature_Importance.png'); plt.close()
print("  OK")

print(f"\n=== All 9 figures saved to {SAVE.resolve()}/ ===")
print("""Figures:
  Fig1_DWT_Subbands.png       — DWT 8-subband decomposition (real vs fake)
  Fig2_Adjacency_TopK.png     — A_tilde heatmap + degree distribution
  Fig3_Joint_Laplacian.png    — A_joint + eigenvalue spectrum
  Fig4_MSC_Gap.png            — MSC distribution histogram (100 samples)
  Fig5_Feature_tSNE.png       — t-SNE of h_graph + full features
  Fig6_Dual_Routing.png       — Semantic vs kinematic adjacency matrices
  Fig7_Confusion.png          — Confusion matrix
  Fig8_Ablation_Waterfall.png — Ablation waterfall chart
  Fig9_Feature_Importance.png — Classifier weight analysis
""")
