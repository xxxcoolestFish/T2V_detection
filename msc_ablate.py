"""
MSC Framework — Comprehensive Ablation Study
============================================
Post-hoc ablation: frozen backbone + per-variant probe (Logistic Regression).
Tests 18 variants across 7 dimensions.

Usage:
  python msc_ablate.py --model msc_checkpoints/best_stage2.pth --data_root /root/autodl-tmp
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
import json
import sys
from collections import defaultdict
from copy import deepcopy

import decord
decord.bridge.set_bridge('torch')

from msc_model import MSCDetector, DualDecoder, CrossStreamAttention, JointMSCProbe


# ============================================================
# 0. Ablation Configurations
# ============================================================
ABLATION_CONFIGS = [
    # ── Cross-path ──
    {'name': 'Full',                'desc': 'Full model (Path A + Path B)',        'group': 'Cross-path'},
    {'name': 'Path_A_only',         'desc': 'Path A only (no MSC probe)',           'group': 'Cross-path',  'path_b': False},
    {'name': 'Path_B_only',         'desc': 'Path B only (MSC probe, no backbone)', 'group': 'Cross-path',  'path_a': False},

    # ── CSA & temporal context ──
    {'name': 'No_CSA',              'desc': 'No Cross-Stream Attn (cosine sim)',    'group': 'CSA/tau',     'csa': False},
    {'name': 'Tau_0',               'desc': 'tau=0 (no temporal context)',          'group': 'CSA/tau',     'tau': 0},
    {'name': 'Tau_1',               'desc': 'tau=1',                                'group': 'CSA/tau',     'tau': 1},
    {'name': 'Tau_4',               'desc': 'tau=4',                                'group': 'CSA/tau',     'tau': 4},
    {'name': 'Tau_inf',             'desc': 'tau=inf (no temporal mask)',           'group': 'CSA/tau',     'tau': -1},

    # ── Features ──
    {'name': 'No_S_vn',             'desc': 'No von Neumann entropy (MSC only)',    'group': 'Features',    'use_svn': False},
    {'name': 'Low_stream_only',     'desc': 'S_low stream only (no S_high)',        'group': 'Stream',      'stream': 'low'},
    {'name': 'High_stream_only',    'desc': 'S_high stream only (no S_low)',        'group': 'Stream',      'stream': 'high'},

    # ── Pool method ──
    {'name': 'Pool_GAP',            'desc': 'GAP instead of SpatialGridPool 4x4',   'group': 'Pool',        'pool': 'gap'},

    # ── DualDecoder ──
    {'name': 'No_Kinematic',        'desc': 'No kinematic decoder branch',          'group': 'Decoder',     'kinematic': False},
    {'name': 'No_Semantic',         'desc': 'No semantic decoder branch',           'group': 'Decoder',     'semantic': False},
    {'name': 'No_3D_RoPE',          'desc': 'No 3D-RoPE in kinematic branch',       'group': 'Decoder',     'rope_3d': False},

    # ── GCN ──
    {'name': 'Mean_Pool_no_GCN',    'desc': 'Mean pool instead of GCN layers',      'group': 'GCN',         'gcn': False},

    # ── Top-K sparsity ──
    {'name': 'TopK_10',             'desc': 'Top-K=10 (sparser graph)',             'group': 'Top-K',       'top_k': 10},
    {'name': 'TopK_50',             'desc': 'Top-K=50 (denser graph)',              'group': 'Top-K',       'top_k': 50},
    {'name': 'TopK_full',           'desc': 'Full graph (no Top-K sparsification)', 'group': 'Top-K',       'top_k': -1},
]


# ============================================================
# 1. Model Patching (zero changes to msc_model.py)
# ============================================================
_ORIG_METHODS = {}

def _save_orig(cls, method_name):
    """Save original method before patching."""
    key = f"{cls.__name__}.{method_name}"
    if key not in _ORIG_METHODS:
        _ORIG_METHODS[key] = getattr(cls, method_name)


def _abl(cfg, key, default):
    """Safely get ablation config value."""
    if cfg is None:
        return default
    return cfg.get(key, default)


def _patch_dual_decoder():
    """Patch DualDecoder.forward to support kinematic/semantic/rope_3d/top_k ablation."""
    _save_orig(DualDecoder, 'forward')

    def patched_forward(self, x):
        abl = getattr(self, 'ablation', None) or {}
        B, N, D = x.shape

        # -- Semantic branch --
        sem_on = _abl(abl, 'semantic', True)
        kin_on = _abl(abl, 'kinematic', True)
        rope_on = _abl(abl, 'rope_3d', True)
        k_sparse = _abl(abl, 'top_k', self.k_sparse)
        if k_sparse == -1:
            k_sparse = N

        logits = 0.0
        total_weight = 0.0

        if sem_on:
            q_s = self.q_sem(x) * self.scale_sem
            k_s = self.k_sem(x) * self.scale_sem
            logits = logits + self.alpha * torch.bmm(q_s, k_s.transpose(1, 2))
            total_weight = total_weight + self.alpha

        if kin_on:
            u = self.w_i(x)
            v = self.w_j(x)
            if rope_on:
                u = self.rope(u)
                v = self.rope(v)
            A_kin = torch.bmm(u, v.transpose(1, 2)) * self.scale_kin
            kin_weight = 1.0 - self.alpha
            logits = logits + kin_weight * A_kin
            total_weight = total_weight + kin_weight

        # Renormalize if a branch is disabled
        if total_weight > 0 and total_weight != 1.0:
            logits = logits / total_weight

        logits = logits.float().clamp(-100, 100)
        A_hat = F.softmax(logits, dim=-1).to(x.dtype)

        topk_vals, topk_idx = torch.topk(A_hat, min(k_sparse, N), dim=-1)
        A_sparse = torch.zeros_like(A_hat).scatter_(-1, topk_idx, topk_vals)
        A_sparse = A_sparse / (A_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.nan_to_num(A_sparse, nan=0.0)

    DualDecoder.forward = patched_forward


def _patch_cross_stream_attn():
    """Patch CrossStreamAttention.forward to support csa/tau ablation."""
    _save_orig(CrossStreamAttention, 'forward')

    def patched_forward(self, x_src, x_tgt):
        abl = getattr(self, 'ablation', None) or {}
        csa_on = _abl(abl, 'csa', True)
        tau = _abl(abl, 'tau', None)
        if tau is None:
            tau = self.tau

        B, N, _ = x_src.shape
        q = self.q_proj(x_src) * self.scale
        k = self.k_proj(x_tgt) * self.scale
        scores = torch.bmm(q, k.transpose(1, 2))

        if not csa_on:
            # Pure cosine similarity (no temporal mask)
            A = F.softmax(scores.float().clamp(-100, 100), dim=-1).to(x_src.dtype)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            return A

        # CSA with temporal mask
        idx = torch.arange(N, device=x_src.device)
        if tau < 0:
            # No temporal mask
            mask = torch.ones(N, N, device=x_src.device).float()
        else:
            mask = (torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) <= tau).float()
        mask = mask.unsqueeze(0)

        masked_scores = scores.float() + (1.0 - mask) * (-1e9)
        masked_scores = masked_scores.clamp(-100, 100)
        A = F.softmax(masked_scores, dim=-1).to(x_src.dtype)
        A = A * mask.to(x_src.dtype)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        return A

    CrossStreamAttention.forward = patched_forward


def _patch_joint_msc_probe():
    """Patch JointMSCProbe.forward to support use_svn ablation and pass ablation to CSA."""
    _save_orig(JointMSCProbe, 'forward')

    def patched_forward(self, x1, x2):
        abl = getattr(self, 'ablation', None) or {}
        B, N, D = x1.shape

        A12 = self.cross_12(x1, x2)
        A21 = self.cross_21(x2, x1)
        A_cross = (A12 + A21.transpose(1, 2)) / 2.0

        eye = torch.eye(N, device=x1.device, dtype=x1.dtype).unsqueeze(0).expand(B, -1, -1)
        top = torch.cat([eye, A_cross], dim=-1)
        bottom = torch.cat([A_cross.transpose(1, 2), eye], dim=-1)
        A_joint = torch.cat([top, bottom], dim=1)

        D_vec = A_joint.sum(dim=-1)
        L_joint = torch.diag_embed(D_vec) - A_joint

        diag_noise = torch.rand(B, 2 * N, device=L_joint.device, dtype=torch.float32) * 1e-4
        L_safe = L_joint.float() + torch.diag_embed(diag_noise)

        eigenvalues = torch.linalg.eigvalsh(L_safe)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        lambda2 = eigenvalues[:, 1]

        use_svn = _abl(abl, 'use_svn', True)
        if use_svn:
            eps = 1e-8
            sum_ev = eigenvalues.sum(dim=-1, keepdim=True) + eps
            lambda_norm = eigenvalues / sum_ev
            S_vn = -torch.sum(lambda_norm * torch.log2(lambda_norm + eps), dim=-1)
        else:
            S_vn = torch.zeros(B, device=lambda2.device, dtype=lambda2.dtype)

        return lambda2.to(x1.dtype), S_vn.to(x1.dtype), A_joint

    JointMSCProbe.forward = patched_forward


def _patch_msc_detector():
    """Patch MSCDetector.forward to support path_a/path_b/gcn ablation."""
    _save_orig(MSCDetector, 'forward')
    _save_orig(MSCDetector, '_path_b_features')

    def patched_path_b_features(self, dwt_out):
        abl = getattr(self, 'ablation', None) or {}
        stream_mode = _abl(abl, 'stream', 'both')
        pool_mode = _abl(abl, 'pool', 'grid')

        idx_low = [0, 8, 16]
        idx_high = [i for i in range(24) if i not in idx_low]

        if stream_mode == 'low':
            # Use S_low for both streams (MSC becomes intra-low coherence)
            S_low = dwt_out[:, idx_low]
            if pool_mode == 'gap':
                F_low = S_low.mean(dim=[-1, -2]).transpose(1, 2).contiguous()
            else:
                F_low = self.grid_pool(S_low)
            X_low = getattr(self, 'low_encoder_gap', self.low_encoder)(F_low)
            return X_low, X_low  # both streams identical → MSC ≈ 0

        elif stream_mode == 'high':
            S_high = dwt_out[:, idx_high]
            if pool_mode == 'gap':
                F_high = S_high.mean(dim=[-1, -2]).transpose(1, 2).contiguous()
            else:
                F_high = self.grid_pool(S_high)
            X_high = getattr(self, 'high_encoder_gap', self.high_encoder)(F_high)
            return X_high, X_high

        else:  # both
            S_low = dwt_out[:, idx_low]
            S_high = dwt_out[:, idx_high]

            if pool_mode == 'gap':
                F_low = S_low.mean(dim=[-1, -2]).transpose(1, 2).contiguous()
                F_high = S_high.mean(dim=[-1, -2]).transpose(1, 2).contiguous()
            else:
                F_low = self.grid_pool(S_low)
                F_high = self.grid_pool(S_high)

            X_low = getattr(self, 'low_encoder_gap', self.low_encoder)(F_low)
            X_high = getattr(self, 'high_encoder_gap', self.high_encoder)(F_high)
            return X_low, X_high

    def patched_forward(self, video, audio_mel=None, mask_ratio=0.0, return_all=False):
        abl = getattr(self, 'ablation', None) or {}
        path_a_on = _abl(abl, 'path_a', True)
        path_b_on = _abl(abl, 'path_b', True)
        gcn_on = _abl(abl, 'gcn', True)
        use_audio = self.use_audio

        B = video.shape[0]
        device = video.device

        # Shared DWT (always needed for either path)
        X_main, dwt_out = self.dsft(video)
        N_patches = X_main.shape[1]

        # ── Path A ──
        if path_a_on:
            if mask_ratio > 0.0 and self.training:
                mask = torch.rand(B, N_patches, device=device) < mask_ratio
                X_main_m = torch.where(
                    mask.unsqueeze(-1),
                    self.mask_token.expand(B, N_patches, self.embed_dim), X_main
                )
            else:
                X_main_m = X_main
                mask = torch.zeros(B, N_patches, dtype=torch.bool, device=device)

            A_tilde = self.dual_decoder(X_main_m)
            A_sym = (A_tilde + A_tilde.transpose(1, 2)) / 2.0
            L_main = torch.diag_embed(A_sym.sum(dim=-1)) - A_sym

            if gcn_on:
                h_graph = self.gcn(X_main_m, A_sym)
            else:
                # Mean pool: use GCN linear layers without message passing
                h1 = F.gelu(self.gcn.gcn_1(X_main_m))
                h2 = F.gelu(self.gcn.gcn_2(h1))
                h_graph = h2.mean(dim=1)
        else:
            h_graph = torch.zeros(B, self.hidden_dim, device=device)
            X_main_m = X_main
            mask = torch.zeros(B, N_patches, dtype=torch.bool, device=device)
            A_tilde = torch.zeros(B, N_patches, N_patches, device=device)
            L_main = torch.zeros(B, N_patches, N_patches, device=device)

        # ── Path B ──
        if path_b_on:
            X_low, X_high = patched_path_b_features(self, dwt_out)
            N_stream = X_low.shape[1]

            if mask_ratio > 0.0 and self.training:
                mask_low_b = torch.rand(B, N_stream, device=device) < mask_ratio
                mask_high_b = torch.rand(B, N_stream, device=device) < mask_ratio
                X_low_m = torch.where(
                    mask_low_b.unsqueeze(-1),
                    self.mask_token_b.expand(B, N_stream, 256), X_low
                )
                X_high_m = torch.where(
                    mask_high_b.unsqueeze(-1),
                    self.mask_token_b.expand(B, N_stream, 256), X_high
                )
            else:
                X_low_m, X_high_m = X_low, X_high
                mask_low_b = torch.zeros(B, N_stream, dtype=torch.bool, device=device)
                mask_high_b = torch.zeros(B, N_stream, dtype=torch.bool, device=device)

            msc_lh, S_vn, A_joint_lh = self.msc_probe_lh(X_low_m, X_high_m)
        else:
            msc_lh = torch.zeros(B, device=device)
            S_vn = torch.zeros(B, device=device)
            A_joint_lh = torch.zeros(B, 32, 32, device=device)
            X_low = torch.zeros(B, 16, 256, device=device)
            X_high = torch.zeros(B, 16, 256, device=device)
            X_low_m, X_high_m = X_low, X_high
            mask_low_b = torch.zeros(B, 16, dtype=torch.bool, device=device)
            mask_high_b = torch.zeros(B, 16, dtype=torch.bool, device=device)

        # ── Audio (only if use_audio) ──
        msc_la = torch.zeros(B, device=device, dtype=h_graph.dtype)
        msc_ha = torch.zeros(B, device=device, dtype=h_graph.dtype)
        if use_audio and audio_mel is not None and audio_mel.dim() == 3 and path_b_on:
            if self.training and self.p_audio_drop > 0:
                drop_mask = torch.rand(B, device=device) > self.p_audio_drop
                audio_active = drop_mask.float().to(X_low.dtype)
            else:
                audio_active = torch.ones(B, device=device, dtype=X_low.dtype)

            X_audio_raw = audio_mel.permute(0, 2, 1).contiguous()
            if X_audio_raw.shape[1] != X_low.shape[1]:
                X_audio_raw = F.interpolate(
                    X_audio_raw.transpose(1, 2), size=X_low.shape[1],
                    mode='linear', align_corners=False
                ).transpose(1, 2)
            X_audio = self.audio_encoder(X_audio_raw)

            msc_la_full, _, _ = self.msc_probe_la(X_low_m, X_audio)
            msc_ha_full, _, _ = self.msc_probe_ha(X_high_m, X_audio)
            msc_la = msc_la_full * audio_active
            msc_ha = msc_ha_full * audio_active

        # ── Build feature vector (pre-classifier) ──
        if use_audio:
            features = torch.cat([
                h_graph, msc_lh.unsqueeze(-1), msc_la.unsqueeze(-1),
                msc_ha.unsqueeze(-1), S_vn.unsqueeze(-1)
            ], dim=-1)
        else:
            features = torch.cat([
                h_graph, msc_lh.unsqueeze(-1), S_vn.unsqueeze(-1)
            ], dim=-1)

        logits = self.classifier(features)

        if return_all:
            return {
                'logits': logits, 'features': features,
                'h_graph': h_graph, 'msc_lh': msc_lh, 'S_vn': S_vn,
                'X_main_orig': X_main, 'X_main_m': X_main_m, 'mask': mask,
                'A_tilde': A_tilde, 'L_main': L_main,
                'X_low_orig': X_low, 'X_high_orig': X_high,
                'X_low_m': X_low_m, 'X_high_m': X_high_m,
                'mask_low_b': mask_low_b, 'mask_high_b': mask_high_b,
                'A_joint_lh': A_joint_lh,
            }
        return {'logits': logits, 'features': features, 'msc_lh': msc_lh, 'S_vn': S_vn}

    MSCDetector.forward = patched_forward
    MSCDetector._path_b_features = patched_path_b_features


def apply_all_patches():
    """Apply all ablation patches. Call once before creating any models."""
    _patch_dual_decoder()
    _patch_cross_stream_attn()
    _patch_joint_msc_probe()
    _patch_msc_detector()


# ============================================================
# 2. Data Loading
# ============================================================
def get_source(p):
    p = p.lower()
    if 'sora' in p: return 'Sora'
    if 'cogvideo' in p: return 'CogVideo'
    if 'hunyuanvideo' in p: return 'HunyuanVideo'
    if 'kinetics' in p: return 'Kinetics(Real)'
    return 'Other'


def load_multiclip(video_path, num_clips=4, clip_len=32, spatial_size=224):
    """Load num_clips clips from video, each clip_len frames."""
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        if total <= clip_len:
            starts = [0] * num_clips
        else:
            step = max(1, (total - clip_len) // max(1, num_clips - 1))
            starts = [i * step for i in range(num_clips)]

        clips = []
        for s in starts:
            idx = list(range(s, min(total, s + clip_len)))
            while len(idx) < clip_len:
                idx.append(idx[-1])
            frames = vr.get_batch(idx).float() / 255.0        # [T, H, W, C]
            frames = frames.permute(3, 0, 1, 2)               # [C, T, H, W]
            # Treat T as batch dim for 4D interpolation
            C, T, H, W = frames.shape
            frames = frames.permute(1, 0, 2, 3)               # [T, C, H, W]
            frames = F.interpolate(frames, size=(spatial_size, spatial_size), mode='bilinear')
            frames = frames.permute(1, 0, 2, 3)               # [C, T, H, W]
            clips.append(frames)
        return torch.stack(clips, dim=0), True
    except Exception:
        return None, False


def collect_test_files(data_root, sample_limit=2000):
    """Collect test files with labels."""
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
    return test


# ============================================================
# 3. Feature Extraction & Probe Evaluation
# ============================================================
def load_all_clips(test_files, num_clips=2, clip_len=32, spatial_size=224):
    """Load clips from all test files once, return cached tensors + labels.
    Returns:
        all_clips: list of [num_clips, C, T, H, W] tensors
        clip_counts: list of clip counts per video
        labels: np.array of labels
    """
    all_clips = []
    clip_counts = []
    labels_list = []
    fail_count = 0

    for path, true_label in tqdm(test_files, desc="[Load] Cache clips"):
        try:
            batch, ok = load_multiclip(path, num_clips, clip_len, spatial_size)
            expected_shape = (num_clips, 3, clip_len, spatial_size, spatial_size)
            if ok and batch is not None and batch.shape == expected_shape:
                all_clips.append(batch)
                clip_counts.append(batch.shape[0])
                labels_list.append(true_label)
            else:
                fail_count += 1
        except Exception:
            fail_count += 1

    if fail_count:
        print(f"      Warning: {fail_count} videos failed to load")
    if not all_clips:
        raise RuntimeError("No videos loaded successfully! Check data_root path.")

    return all_clips, clip_counts, np.array(labels_list)


@torch.no_grad()
def extract_features_cached(model, all_clips, device, batch_size=16):
    """Extract pre-classifier features from cached clips through model.
    Uses batched concatenation (same as verified debug script).
    Returns: features [N_samples, D]
    """
    model.eval()

    all_clips_cat = torch.cat([c for c in all_clips], dim=0)  # keep on CPU
    total_clips = all_clips_cat.shape[0]

    all_feats_list = []
    for i in range(0, total_clips, batch_size):
        sub = all_clips_cat[i:i + batch_size].to(device)
        out = model(sub, mask_ratio=0.0, return_all=False)
        all_feats_list.append(out['features'].cpu())

    all_feats = torch.cat(all_feats_list, dim=0)

    # Top-K mean pooling per video (K = min(3, n))
    all_features = []
    clip_start = 0
    for n in [c.shape[0] for c in all_clips]:
        video_feats = all_feats[clip_start:clip_start + n]
        k = min(3, n)
        topk_vals, _ = torch.topk(video_feats, k, dim=0)
        all_features.append(topk_vals.mean(dim=0).numpy())
        clip_start += n

    return np.stack(all_features)


def evaluate_probe(X, y, n_folds=5, seed=42):
    """Evaluate with LogisticRegression + StratifiedKFold.
    Returns: dict with acc, auc, ap (mean ± std)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aucs, aps = [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=seed)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_va)
        y_prob = clf.predict_proba(X_va)[:, 1]

        accs.append(accuracy_score(y_va, y_pred))
        try:
            aucs.append(roc_auc_score(y_va, y_prob))
            aps.append(average_precision_score(y_va, y_prob))
        except ValueError:
            aucs.append(0.5)
            aps.append(0.5)

    return {
        'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'ap_mean': np.mean(aps),   'ap_std': np.std(aps),
    }


# ============================================================
# 4. Main Ablation Runner
# ============================================================
def run_ablation(model_path, data_root, sample_limit=800, num_clips=2,
                 clip_len=32, batch_size=16, n_folds=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Device] {device}")
    print(f"[Mode] Post-hoc ablation with LogisticRegression probes")
    print(f"[Data] sample_limit={sample_limit}, num_clips={num_clips}, {n_folds}-fold CV")

    # ── Apply patches ──
    print("\n[Setup] Applying ablation patches...")
    apply_all_patches()

    # ── Load checkpoint ──
    print(f"[Setup] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    raw = ckpt.get('model_state_dict', ckpt)
    clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}

    use_audio = False
    for k, v in clean.items():
        if 'classifier.0.weight' in k:
            use_audio = (v.shape[1] >= 260)
            break
    print(f"[Setup] Audio mode: {use_audio}")

    # ── Collect test files ──
    print(f"[Setup] Scanning test files from: {data_root}")
    test_files = collect_test_files(data_root, sample_limit)
    print(f"[Setup] Found {len(test_files)} test samples")

    # ── LOAD ALL CLIPS ONCE (major bottleneck, do only once) ──
    print(f"[Setup] Loading all clips (once)...")
    all_clips, clip_counts, all_labels = load_all_clips(test_files, num_clips, clip_len)
    num_videos = len(all_clips)
    print(f"[Setup] Loaded {num_videos} videos, {sum(clip_counts)} total clips")

    # ── Run each ablation variant ──
    results = []

    for i, config in enumerate(ABLATION_CONFIGS):
        name = config['name']
        desc = config['desc']
        group = config['group']
        abl_cfg = {k: v for k, v in config.items() if k not in ('name', 'desc', 'group')}

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(ABLATION_CONFIGS)}] {name}: {desc}")
        print(f"      Ablation: {abl_cfg}")

        # Create fresh model and load weights
        model = MSCDetector(use_audio=use_audio)
        model.load_state_dict(clean, strict=False)
        model = model.to(device)
        model.ablation = abl_cfg

        # Propagate ablation to sub-modules
        model.dual_decoder.ablation = abl_cfg
        model.msc_probe_lh.ablation = abl_cfg
        model.msc_probe_lh.cross_12.ablation = abl_cfg
        model.msc_probe_lh.cross_21.ablation = abl_cfg
        if use_audio:
            model.msc_probe_la.ablation = abl_cfg
            model.msc_probe_la.cross_12.ablation = abl_cfg
            model.msc_probe_la.cross_21.ablation = abl_cfg
            model.msc_probe_ha.ablation = abl_cfg
            model.msc_probe_ha.cross_12.ablation = abl_cfg
            model.msc_probe_ha.cross_21.ablation = abl_cfg

        # Handle GAP pool: create temporary encoders with correct input dims
        if abl_cfg.get('pool') == 'gap':
            model.low_encoder_gap = torch.nn.Sequential(
                torch.nn.Linear(3, 256), torch.nn.LayerNorm(256),
                torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.LayerNorm(256),
            ).to(device)
            model.high_encoder_gap = torch.nn.Sequential(
                torch.nn.Linear(21, 256), torch.nn.LayerNorm(256),
                torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.LayerNorm(256),
            ).to(device)

        # Extract features using cached clips
        X = extract_features_cached(model, all_clips, device, batch_size)
        del model
        torch.cuda.empty_cache()

        y = all_labels[:len(X)]  # align labels with features

        print(f"      Features: {X.shape}, Labels: real={int((y==0).sum())} fake={int((y==1).sum())}")

        # Evaluate with probe
        metrics = evaluate_probe(X, y, n_folds=n_folds, seed=42)
        metrics['name'] = name
        metrics['desc'] = desc
        metrics['group'] = group
        metrics['abl_cfg'] = abl_cfg
        metrics['feat_dim'] = X.shape[1]
        results.append(metrics)

        print(f"      -> ACC: {metrics['acc_mean']:.4f} ± {metrics['acc_std']:.4f}"
              f" | AUC: {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}"
              f" | AP: {metrics['ap_mean']:.4f} ± {metrics['ap_std']:.4f}")

    # ── Print Comparison ──
    print("\n" + "=" * 90)
    print("  Comprehensive Ablation Comparison")
    print("=" * 90)

    # Sort by group then AUC
    results.sort(key=lambda r: (r['group'], -r['auc_mean']))

    prev_group = None
    for r in results:
        if r['group'] != prev_group:
            print(f"\n── {r['group']} ──")
            prev_group = r['group']
        delta_auc = (r['auc_mean'] - results[0]['auc_mean']) * 100 if results else 0
        print(f"  {r['name']:<22s} | ACC: {r['acc_mean']:.3f}±{r['acc_std']:.3f}"
              f" | AUC: {r['auc_mean']:.4f}±{r['auc_std']:.4f}"
              f" | AP: {r['ap_mean']:.4f}±{r['ap_std']:.4f}"
              f" | Dim: {r['feat_dim']}d"
              f" | ΔAUC: {delta_auc:+.1f}%")

    # ── Save results ──
    output_file = Path(data_root) / 'msc_ablation_results.json'
    with open(output_file, 'w') as f:
        # Convert numpy values for JSON
        json_results = []
        for r in results:
            jr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k, v in r.items()}
            json_results.append(jr)
        json.dump(json_results, f, indent=2)
    print(f"\n[Saved] {output_file}")

    return results


# ============================================================
# 5. Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSC Framework Comprehensive Ablation")
    parser.add_argument('--model', type=str, default='msc_checkpoints/best_stage2.pth')
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp')
    parser.add_argument('--sample_limit', type=int, default=800)
    parser.add_argument('--num_clips', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    run_ablation(args.model, args.data_root, args.sample_limit,
                 args.num_clips, args.batch_size, args.n_folds)
