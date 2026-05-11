# MSC: Multi-Stream Spectral Coherence for AI-Generated Video Detection

Official implementation of the MSC framework — a **dual-path video forgery detection method** that grounds detection in provable physical consistency rather than data-distribution-dependent artifacts.

## Overview

Modern T2V diffusion models (Sora, CogVideo, HunyuanVideo) can synthesize highly realistic videos. MSC detects these AI-generated videos by asking a fundamentally different question: not "does this look fake?" but **"does this obey physical causality?"**

The framework transforms physical-consistency verification into mathematically provable propositions via **spectral graph theory**:

- **Path A** — Frequency-domain Graph Backbone: builds a directed graph over DWT frequency tokens and extracts discriminative graph-level features via GCN.
- **Path B** — Cross-Stream Spectral Coherence Probe: computes the Fiedler value (algebraic connectivity) of the joint Laplacian across frequency streams — a physically interpretable coherence metric.
- **Audio-Visual Extension**: adds an independent audio modality to break the DWT linear dependency in pure-video settings.
- **TPC Extension** (theory): Temporal Phase Coherence via nonlinear phase analysis for audio-absent scenarios.

## Architecture

```
Video [B,3,T,H,W]
  │
  └── 3D Haar DWT (shared, zero-parameter) ──→ 8 sub-bands (24ch)
        │
        ├── Path A: DSFT (1024 nodes) → DualDecoder (semantic + kinematic, 3D-RoPE)
        │     → Top-K sparsification → 2-layer GCN → h_graph [B,256]
        │
        └── Path B: Band split → S_low (LLL) + S_high (7 sub-bands)
              → GridPool + MLP → CSA (τ=2) → L_joint → MSC=λ₂
                                                          │
        Classifier ← [h_graph; Dirichlet; Entropy; MSC; S_vn; (MSC_la; MSC_ha)] → logit
```

## Key Results

| Setting | AUC | Params |
|---------|-----|--------|
| Pure Video (Path A only) | **0.84** | ~3.2M |
| Audio-Visual (Path A+B+Audio) | **0.85** | ~3.2M |

Compared to baselines like FakeSTormer (AUC 0.89, CLIP ViT-L/14, 300M+ params), MSC achieves competitive performance with **~1% of the parameter count** and with **provable cross-model generalization guarantees** rooted in physical law.

## Theorem System

| Theorem | Statement | Tool |
|---------|-----------|------|
| **1** | MSC ≥ 0 (non-negativity) | Gershgorin + Courant-Fischer |
| **2** | Independent generation → E[MSC] → 0 | Johnson-Lindenstrauss + Weyl |
| **3** | Physical causality → MSC > 0 | Rayleigh quotient variation |
| **Corollary 1** | Cross-modal distinguishability (audio-visual) | |
| **Corollary 2** | Pure-video non-distinguishability (DWT linear dependency) | |

Corollary 2 drives the design of both the audio-visual extension and the TPC extension.

## Installation

```bash
# Clone
git clone https://github.com/xxxcoolestFish/msc.git
cd msc/msc_src

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-capable GPU (recommended, 24GB VRAM for batch_size=64)
- decord (video decoding)
- scikit-learn, numpy, pandas, tqdm

For audio-visual training, additionally:
```bash
pip install av  # PyAV for audio stream extraction
```

## Usage

### Training

```bash
# Pure video mode (no audio)
python msc_train.py \
  --data_root /path/to/videos \
  --data_type purevideo \
  --batch_size 64 \
  --epochs_s1 5 \
  --epochs_s2 20

# Audio-visual mode (requires FakeAVCeleb_v1.2 structure)
python msc_train.py \
  --data_root /path/to/FakeAVCeleb_v1.2 \
  --data_type fakeavceleb \
  --batch_size 64 \
  --fake_ratio 3.0
```

**Expected data structure** for `purevideo`:
```
/path/to/videos/
├── kinetics/          (or any folder with 'kinetics' / 'realvideo' in name)
│   ├── video1.mp4
│   └── ...
├── sora/              (or any folder with 'sora' / 'cogvideo' / 'hunyuanvideo')
│   ├── fake1.mp4
│   └── ...
```

For `fakeavceleb`, the expected structure follows FakeAVCeleb_v1.2:
```
FakeAVCeleb_v1.2/
├── RealVideo-RealAudio/
├── RealVideo-FakeAudio/
├── FakeVideo-RealAudio/
└── FakeVideo-FakeAudio/
```

**Two-stage training**:
- Stage 1 (5 epochs): Masked reconstruction on **real videos only** — model learns intrinsic frequency-spatial-temporal patterns.
- Stage 2 (20 epochs): BCE fine-tuning on real+fake — model learns the decision boundary.

Checkpoints are saved to `msc_checkpoints/` (auto-resumes from `latest.pth`).

### Evaluation

```bash
python msc_evaluate.py \
  --model msc_checkpoints/best_stage2.pth \
  --data_root /path/to/test/videos \
  --sample_limit 2000
```

Reports AUC, accuracy, per-source breakdown, and MSC gap statistics.

### Ablation Study

```bash
python msc_ablate.py \
  --model msc_checkpoints/best_stage2.pth \
  --data_root /path/to/videos \
  --sample_limit 800
```

Runs post-hoc ablation across 18 variants (cross-path, CSA/tau, decoder, GCN, Top-K, etc.) using frozen backbone + Logistic Regression probes.

### Visualization

```bash
# Generate all paper figures (DWT subbands, adjacency, Laplacian, t-SNE, etc.)
python visualize_all.py
```

Output images are saved to `visualizations/` directory.

### ROC Analysis

```bash
# Analyze ROC curve from eval CSV
python analyze_roc.py
```

## File Structure

```
msc_src/
├── msc_model.py         # Core model: Haar3D_DWT, DSFT, DualDecoder, RoPE3D,
│                        #   CrossStreamAttention, JointMSCProbe, GCN, MSCDetector
├── msc_train.py          # Two-stage training + MSCVideoDataset + MSCAudioVideoDataset
├── msc_evaluate.py       # Multi-clip inference with Top-3 mean pooling
├── msc_ablate.py         # Comprehensive ablation (18 variants, post-hoc LR probes)
├── msc_ablate_v2.py      # Minimal ablation runner (batch-processed)
├── msc_ablate_v4.py      # Focused ablation: v4 Full / No Graph Physics / A_sym
├── visualize_all.py      # Paper figure generation (Figs 1-9)
├── analyze_roc.py        # ROC curve analysis from eval CSV
├── run_train_v4.sh       # Shell script to launch training
├── requirements.txt
├── .gitignore
└── README.md
```


## License

This project is released for academic research purposes.
