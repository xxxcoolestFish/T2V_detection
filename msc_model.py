"""
MSC Framework v3 — Merged Architecture
======================================
Path A (old DSFT+STARNet backbone, proven):
  Haar3D_DWT → DSFT patch_proj → 1024 nodes [B,1024,384]
  → DualDecoder (semantic+kinematic+3D-RoPE) → Top-K → A_tilde
  → 2-layer GCN → h_graph [B,256]

Path B (new MSC probe, theory-backed):
  Same DWT → split S_low(LLL) + S_high(7 subbands)
  → SpatialGridPool(4×4) → MLP → 16 nodes/stream [B,16,256]
  → CrossStreamAttention (temporal mask τ=2) → A_cross
  → L_joint → λ₂=MSC, S_vn=vonNeumann entropy

Classifier:
  [h_graph; Dirichlet; graph_entropy; MSC_lh; (MSC_la; MSC_ha; if audio); S_vn] → MLP → logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Haar 3D-DWT (unchanged from old code)
# ============================================================
class Haar3D_DWT(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 8
        self.dwt_conv = nn.Conv3d(
            in_channels, self.out_channels,
            kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels
        )
        w = torch.zeros(8, 1, 2, 2, 2)
        w[0] = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        w[1] = torch.tensor([[[1,-1], [1,-1]], [[1,-1], [1,-1]]])
        w[2] = torch.tensor([[[1, 1], [-1,-1]], [[1, 1], [-1,-1]]])
        w[3] = torch.tensor([[[1,-1], [-1, 1]], [[1,-1], [-1, 1]]])
        w[4] = torch.tensor([[[1, 1], [1, 1]], [[-1,-1], [-1,-1]]])
        w[5] = torch.tensor([[[1,-1], [1,-1]], [[-1, 1], [-1, 1]]])
        w[6] = torch.tensor([[[1, 1], [-1,-1]], [[-1,-1], [1, 1]]])
        w[7] = torch.tensor([[[1,-1], [-1, 1]], [[-1, 1], [1,-1]]])
        w = w / 2.828427
        self.dwt_conv.weight = nn.Parameter(
            w.repeat(in_channels, 1, 1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        return self.dwt_conv(x)  # [B, 24, T/2, H/2, W/2]


# ============================================================
# 2. Spatial Grid Pooling (replaces GAP for richer Path B features)
# ============================================================
class SpatialGridPool(nn.Module):
    """Divide spatial dims into grid_size×grid_size cells, average-pool each.
    Input:  [B, C, T, H, W]
    Output: [B, T, C × grid_size²]
    """
    def __init__(self, grid_size=4):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, x):
        B, C, T, H, W = x.shape
        g = self.grid_size
        # Adaptive average pool to [B, C, T, g, g]
        x = F.adaptive_avg_pool3d(x, (T, g, g))
        # Reshape: [B, C, T, g, g] → [B, T, C, g, g] → [B, T, C*g*g]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, T, C * g * g)
        return x


# ============================================================
# 3. Stream Encoder (Path B node feature extractor)
# ============================================================
class StreamEncoder(nn.Module):
    """MLP projection for Path B stream features."""
    def __init__(self, in_dim, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)  # [B, N, in_dim] → [B, N, embed_dim]


# ============================================================
# 4. DSFT (Path A feature tokenizer, unchanged from old)
# ============================================================
class DSFT(nn.Module):
    def __init__(self, patch_size=(1, 14, 14), embed_dim=384, grid_size=(16, 8, 8)):
        super().__init__()
        self.dwt = Haar3D_DWT(in_channels=3)
        self.patch_proj = nn.Conv3d(24, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, v):
        freq = self.dwt(v)                                    # [B,24,T/2,H/2,W/2]
        tokens = self.patch_proj(freq)                        # [B,D,T',H',W']
        B, D, T_, H_, W_ = tokens.shape
        N = T_ * H_ * W_
        x = tokens.view(B, D, N).transpose(1, 2).contiguous() # [B, N, D]
        x = x + self.pos_embed
        return x, freq  # return DWT output too, for Path B to reuse


# ============================================================
# 5. 3D-RoPE (from old STARNet, for Path A kinematic routing)
# ============================================================
class RoPE3D(nn.Module):
    def __init__(self, dim=64, grid_size=(16, 8, 8), base=10000):
        super().__init__()
        T, H, W = grid_size
        dt = dim // 3; dt += dt % 2
        dh = dt
        dw = dim - dt - dh

        def get_1d_freqs(length, d):
            positions = torch.arange(length, dtype=torch.float32)
            div_term = torch.exp(torch.arange(0, d, 2, dtype=torch.float32)
                                 * -(math.log(base) / d))
            freqs = torch.einsum('i,j->ij', positions, div_term)
            return torch.cat((freqs, freqs), dim=-1)

        freqs_t = get_1d_freqs(T, dt).view(T, 1, 1, dt).expand(T, H, W, dt)
        freqs_h = get_1d_freqs(H, dh).view(1, H, 1, dh).expand(T, H, W, dh)
        freqs_w = get_1d_freqs(W, dw).view(1, 1, W, dw).expand(T, H, W, dw)
        freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).reshape(-1, dim)
        self.register_buffer("cos_val", freqs.cos().unsqueeze(0))
        self.register_buffer("sin_val", freqs.sin().unsqueeze(0))

    def forward(self, x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * self.cos_val[:, :x.shape[1]] + x_rot * self.sin_val[:, :x.shape[1]]


# ============================================================
# 6. Dual Decoder (STARNet core, for Path A intra-graph edges)
# ============================================================
class DualDecoder(nn.Module):
    """Semantic + Kinematic dual-routing with Top-K sparsification."""
    def __init__(self, embed_dim=384, hidden_dim=256, attn_dim=64, k_sparse=30,
                 grid_size=(16, 8, 8)):
        super().__init__()
        self.k_sparse = k_sparse
        self.scale_sem = hidden_dim ** -0.5  # final scaling = d**-0.5 after q@k^T

        # Semantic routing (old STARNet pattern)
        self.q_sem = nn.Linear(embed_dim, hidden_dim)
        self.k_sem = nn.Linear(embed_dim, hidden_dim)

        # Kinematic routing (3D-RoPE + dot-product)
        self.attn_dim = attn_dim
        self.scale_kin = attn_dim ** -0.5
        self.w_i = nn.Linear(embed_dim, attn_dim)
        self.w_j = nn.Linear(embed_dim, attn_dim)
        self.rope = RoPE3D(dim=attn_dim, grid_size=grid_size)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, N, D = x.shape

        # Semantic branch (matching old STARNet: q,k scaled by d**-0.25 each, so product = d**-0.5)
        scale_sqrt = self.scale_sem ** 0.5  # = d**-0.25
        q_s = self.q_sem(x) * scale_sqrt
        k_s = self.k_sem(x) * scale_sqrt
        A_sem = torch.bmm(q_s, k_s.transpose(1, 2))

        # Kinematic branch
        u = self.w_i(x)
        v = self.w_j(x)
        u_rope = self.rope(u)
        v_rope = self.rope(v)
        A_kin = torch.bmm(u_rope, v_rope.transpose(1, 2)) * self.scale_kin

        # Fusion + Top-K sparsification
        logits = self.alpha * A_sem + (1.0 - self.alpha) * A_kin
        logits = logits.float().clamp(-100, 100)
        A_hat = F.softmax(logits, dim=-1).to(x.dtype)

        topk_vals, topk_idx = torch.topk(A_hat, self.k_sparse, dim=-1)
        A_sparse = torch.zeros_like(A_hat).scatter_(-1, topk_idx, topk_vals)
        A_sparse = A_sparse / (A_sparse.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.nan_to_num(A_sparse, nan=0.0)


# ============================================================
# 7. Cross-Stream Attention (Path B, temporal mask)
# ============================================================
class CrossStreamAttention(nn.Module):
    """Cross-stream attention with temporal proximity mask |i-j| ≤ τ."""
    def __init__(self, embed_dim=256, head_dim=64, tau=2):
        super().__init__()
        self.tau = tau
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)

    def forward(self, x_src, x_tgt):
        B, N, _ = x_src.shape
        q = self.q_proj(x_src) * self.scale
        k = self.k_proj(x_tgt) * self.scale
        scores = torch.bmm(q, k.transpose(1, 2))

        idx = torch.arange(N, device=x_src.device)
        mask = (torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) <= self.tau).float()
        mask = mask.unsqueeze(0)

        masked_scores = scores.float() + (1.0 - mask) * (-1e9)
        masked_scores = masked_scores.clamp(-100, 100)
        A = F.softmax(masked_scores, dim=-1).to(x_src.dtype)
        A = A * mask.to(x_src.dtype)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        return A


# ============================================================
# 8. Joint MSC Probe (Path B core: build L_joint, compute λ₂, S_vn)
# ============================================================
class JointMSCProbe(nn.Module):
    """Build joint Laplacian for two streams and extract spectral features."""
    def __init__(self, embed_dim=256, head_dim=64, tau=2):
        super().__init__()
        self.cross_12 = CrossStreamAttention(embed_dim, head_dim, tau)
        self.cross_21 = CrossStreamAttention(embed_dim, head_dim, tau)

    def forward(self, x1, x2):
        """x1, x2: [B, N, D] → msc: [B], S_vn: [B]"""
        B, N, D = x1.shape

        # Cross-stream adjacency (symmetrized)
        A12 = self.cross_12(x1, x2)
        A21 = self.cross_21(x2, x1)
        A_cross = (A12 + A21.transpose(1, 2)) / 2.0

        # Joint adjacency: [[I, A_cross], [A_cross^T, I]]
        # (no intra-stream edges in Path B — keep it focused on cross-stream)
        eye = torch.eye(N, device=x1.device, dtype=x1.dtype).unsqueeze(0).expand(B, -1, -1)
        top = torch.cat([eye, A_cross], dim=-1)
        bottom = torch.cat([A_cross.transpose(1, 2), eye], dim=-1)
        A_joint = torch.cat([top, bottom], dim=1)  # [B, 2N, 2N]

        # Joint Laplacian
        D_vec = A_joint.sum(dim=-1)
        L_joint = torch.diag_embed(D_vec) - A_joint

        # Numerical stabilization for eigvalsh
        diag_noise = torch.rand(B, 2 * N, device=L_joint.device, dtype=torch.float32) * 1e-4
        L_safe = L_joint.float() + torch.diag_embed(diag_noise)

        eigenvalues = torch.linalg.eigvalsh(L_safe)      # [B, 2N]
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        lambda2 = eigenvalues[:, 1]                       # MSC = Fiedler value

        # von Neumann entropy: S_vn = -Σ λ̂_i log₂(λ̂_i), λ̂_i = λ_i / Σλ_j
        eps = 1e-8
        sum_ev = eigenvalues.sum(dim=-1, keepdim=True) + eps
        lambda_norm = eigenvalues / sum_ev
        S_vn = -torch.sum(lambda_norm * torch.log2(lambda_norm + eps), dim=-1)

        return lambda2.to(x1.dtype), S_vn.to(x1.dtype), A_joint


# ============================================================
# 9. Simplified GCN (Path A graph convolution, no PGAM)
# ============================================================
class SimpleGCN(nn.Module):
    """2-layer GCN, outputs graph-level embedding."""
    def __init__(self, embed_dim=384, hidden_dim=256):
        super().__init__()
        self.gcn_1 = nn.Linear(embed_dim, hidden_dim)
        self.gcn_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, A):
        """x: [B,N,D], A: [B,N,N] → h_graph: [B,hidden_dim]"""
        msg1 = torch.bmm(A, x)
        h1 = F.gelu(self.gcn_1(msg1))
        msg2 = torch.bmm(A, h1)
        h2 = F.gelu(self.gcn_2(msg2))
        return h2.mean(dim=1)  # global mean pooling


# ============================================================
# 10. Full MSC Detector
# ============================================================
class MSCDetector(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=256, attn_dim=64, k_sparse=30,
                 tau=2, use_audio=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_audio = use_audio

        # ── Shared DWT ──
        self.dwt_3d = Haar3D_DWT(in_channels=3)

        # ── Path A: Old backbone ──
        grid_size = (16, 8, 8)
        self.dsft = DSFT(embed_dim=embed_dim, grid_size=grid_size)
        self.dual_decoder = DualDecoder(
            embed_dim=embed_dim, hidden_dim=hidden_dim, attn_dim=attn_dim,
            k_sparse=k_sparse, grid_size=grid_size
        )
        self.gcn = SimpleGCN(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=.02)

        # ── Path B: Cross-stream MSC probe ──
        self.grid_pool = SpatialGridPool(grid_size=4)
        self.low_encoder = StreamEncoder(in_dim=3 * 16, embed_dim=256)    # 48 → 256
        self.high_encoder = StreamEncoder(in_dim=21 * 16, embed_dim=256)  # 336 → 256
        self.msc_probe_lh = JointMSCProbe(embed_dim=256, head_dim=64, tau=tau)

        if use_audio:
            self.audio_encoder = StreamEncoder(in_dim=80, embed_dim=256)
            self.msc_probe_la = JointMSCProbe(embed_dim=256, head_dim=64, tau=tau)
            self.msc_probe_ha = JointMSCProbe(embed_dim=256, head_dim=64, tau=tau)

        # Path B mask token (for Stage 1 pre-training)
        self.mask_token_b = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.mask_token_b, std=.02)

        # ── Classifier ──
        # Pure video: [h_graph(256); Dirichlet(1); Entropy(1); MSC_lh(1); S_vn(1)] → 260
        # Audio-visual: + [MSC_la(1); MSC_ha(1)] → 262
        cls_in = hidden_dim + 2 + (4 if use_audio else 2)  # h_graph + Dirichlet + entropy + MSC + S_vn
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        # ── Audio dropout probability ──
        self.p_audio_drop = 0.3

    def _path_b_features(self, dwt_out):
        """Extract S_low and S_high features from DWT output.
        dwt_out: [B, 24, T/2, H/2, W/2]
        Returns: X_low [B, N, 256], X_high [B, N, 256]
        """
        # Channel layout: LLL at [0,8,16], rest are 21 high channels
        idx_low = [0, 8, 16]
        idx_high = [i for i in range(24) if i not in idx_low]
        S_low = dwt_out[:, idx_low]    # [B, 3, T', H', W']
        S_high = dwt_out[:, idx_high]  # [B, 21, T', H', W']

        # Spatial grid pooling → [B, N, C*grid²]
        F_low = self.grid_pool(S_low)     # [B, N, 48]
        F_high = self.grid_pool(S_high)   # [B, N, 336]

        # MLP projection
        X_low = self.low_encoder(F_low)     # [B, N, 256]
        X_high = self.high_encoder(F_high)  # [B, N, 256]
        return X_low, X_high

    def forward(self, video, audio_mel=None, mask_ratio=0.0, return_all=False):
        """
        Args:
            video: [B, 3, T, H, W]
            audio_mel: [B, 80, T_mel] or None
            mask_ratio: 0.0 for inference/Stage2, >0 for Stage1
            return_all: return intermediates for Stage1 loss
        """
        B = video.shape[0]
        device = video.device

        # ================================================================
        # Shared: 3D-DWT (called once, shared between Path A and Path B)
        # ================================================================
        # Path A: DSFT internally calls DWT and patch_proj
        X_main, dwt_out = self.dsft(video)  # X_main: [B, 1024, 384]
        N_patches = X_main.shape[1]           # 1024

        # ================================================================
        # Path A: Masking + Dual Decoder + GCN
        # ================================================================
        if mask_ratio > 0.0 and self.training:
            mask = torch.rand(B, N_patches, device=device) < mask_ratio
            X_main_m = torch.where(
                mask.unsqueeze(-1), self.mask_token.expand(B, N_patches, self.embed_dim), X_main
            )
        else:
            X_main_m = X_main
            mask = torch.zeros(B, N_patches, dtype=torch.bool, device=device)

        A_tilde = self.dual_decoder(X_main_m)  # [B, 1024, 1024] directed, row-stochastic
        # Laplacian features from directed graph
        D_vec = A_tilde.sum(dim=-1)  # row degrees
        L_main = torch.diag_embed(D_vec) - A_tilde
        # Dirichlet energy: tr(H1^T @ L @ H1) approximated via mean degree variance
        # Use directed A_tilde in GCN (matches old model = better performance)
        h_graph = self.gcn(X_main_m, A_tilde)     # [B, 256]

        # ================================================================
        # Path B: Cross-stream MSC probe
        # ================================================================
        # Get S_low/S_high features from DWT (reusing dwt_out)
        X_low, X_high = self._path_b_features(dwt_out)  # [B, N, 256] each, N=16
        N_stream = X_low.shape[1]

        # Optional masking for Stage 1 pre-training
        if mask_ratio > 0.0 and self.training:
            mask_low_b = torch.rand(B, N_stream, device=device) < mask_ratio
            mask_high_b = torch.rand(B, N_stream, device=device) < mask_ratio
            X_low_m = torch.where(
                mask_low_b.unsqueeze(-1), self.mask_token_b.expand(B, N_stream, 256), X_low
            )
            X_high_m = torch.where(
                mask_high_b.unsqueeze(-1), self.mask_token_b.expand(B, N_stream, 256), X_high
            )
        else:
            X_low_m, X_high_m = X_low, X_high
            mask_low_b = torch.zeros(B, N_stream, dtype=torch.bool, device=device)
            mask_high_b = torch.zeros(B, N_stream, dtype=torch.bool, device=device)

        # Compute MSC(low, high) and S_vn
        msc_lh, S_vn, A_joint_lh = self.msc_probe_lh(X_low_m, X_high_m)

        # Audio streams (if applicable)
        msc_la = torch.zeros(B, device=device, dtype=X_low.dtype)
        msc_ha = torch.zeros(B, device=device, dtype=X_low.dtype)

        if self.use_audio and audio_mel is not None and audio_mel.dim() == 3:
            # Audio dropout during training
            if self.training and self.p_audio_drop > 0:
                drop_mask = torch.rand(B, device=device) > self.p_audio_drop
                audio_active = drop_mask.float().to(X_low.dtype)
            else:
                audio_active = torch.ones(B, device=device, dtype=X_low.dtype)

            # Reshape mel: [B, 80, T_mel] → [B, T_mel, 80] → interpolate → [B, N, 80] → encode
            X_audio_raw = audio_mel.permute(0, 2, 1).contiguous()  # [B, T_mel, 80]
            if X_audio_raw.shape[1] != N_stream:
                X_audio_raw = F.interpolate(
                    X_audio_raw.transpose(1, 2), size=N_stream,
                    mode='linear', align_corners=False
                ).transpose(1, 2)
            X_audio = self.audio_encoder(X_audio_raw)  # [B, N_stream, 256]

            # MSC(low, audio) and MSC(high, audio)
            msc_la_full, _, _ = self.msc_probe_la(X_low_m, X_audio)
            msc_ha_full, _, _ = self.msc_probe_ha(X_high_m, X_audio)
            msc_la = msc_la_full * audio_active
            msc_ha = msc_ha_full * audio_active

        # ================================================================
        # Classifier (with Path A graph physics features)
        # ================================================================
        # Dirichlet energy: tr(H^T L H) approximation via L's trace per sample
        dirichlet = D_vec.mean(dim=-1)  # [B] mean degree = trace(L)/N, graph connectivity proxy
        
        # Graph entropy: eigenvalue-free proxy via degree distribution entropy
        with torch.no_grad():
            D_norm = D_vec / (D_vec.sum(dim=-1, keepdim=True) + 1e-8)
            graph_entropy = -torch.sum(D_norm * torch.log(D_norm + 1e-8), dim=-1)  # [B]
        
        if self.use_audio:
            cls_input = torch.cat([h_graph, dirichlet.unsqueeze(-1), graph_entropy.unsqueeze(-1),
                                   msc_lh.unsqueeze(-1), msc_la.unsqueeze(-1),
                                   msc_ha.unsqueeze(-1), S_vn.unsqueeze(-1)], dim=-1)
        else:
            cls_input = torch.cat([h_graph, dirichlet.unsqueeze(-1), graph_entropy.unsqueeze(-1),
                                   msc_lh.unsqueeze(-1), S_vn.unsqueeze(-1)], dim=-1)

        logits = self.classifier(cls_input)  # [B, 1]

        if return_all:
            return {
                'logits': logits,
                'features': cls_input,
                # Path A
                'X_main_orig': X_main,
                'X_main_m': X_main_m,
                'mask': mask,
                'A_tilde': A_tilde,
                'L_main': L_main,
                'h_graph': h_graph,
                # Path B
                'X_low_orig': X_low,
                'X_high_orig': X_high,
                'X_low_m': X_low_m,
                'X_high_m': X_high_m,
                'mask_low_b': mask_low_b,
                'mask_high_b': mask_high_b,
                'A_joint_lh': A_joint_lh,
                'msc_lh': msc_lh,
                'S_vn': S_vn,
            }

        return {
            'logits': logits,
            'features': cls_input,
            'msc_lh': msc_lh,
            'S_vn': S_vn,
        }
