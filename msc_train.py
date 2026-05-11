"""
MSC Framework — Two-Stage Training (Merged Architecture)
Stage 1: Masked reconstruction on real videos only
         (Path A: 1024 nodes, Path B: 16x2 nodes)
Stage 2: BCE fine-tuning with real+fake, MSC+S_vn probes active

Supports:
  --data_type purevideo   (Kinetics + Sora/CogVideo/HunyuanVideo)
  --data_type fakeavceleb (FakeAVCeleb_v1.2, audio-visual, 4 categories)
"""

import os
import random
import logging
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
import decord
from tqdm import tqdm

from msc_model import MSCDetector

decord.bridge.set_bridge('torch')
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("msc_training.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 0. Mel Spectrogram (pure PyTorch, no torchaudio needed)
# ============================================================
def mel_filterbank(n_mels=80, n_fft=512, sr=16000, f_min=0, f_max=8000):
    """Build triangular Mel filterbank matrix [n_mels, n_fft//2+1]."""
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bin_indices[i], bin_indices[i+1], bin_indices[i+2]
        for j in range(left, center):
            filters[i, j] = (j - left) / max(1, center - left)
        for j in range(center, min(right, n_freqs)):
            filters[i, j] = (right - j) / max(1, right - center)
    return torch.from_numpy(filters)


def compute_mel(audio_np, sr=16000, n_mels=80, n_fft=512, hop=160,
                target_frames=16, mel_fb=None):
    """Convert raw audio numpy array to Mel spectrogram [n_mels, target_frames].

    Args:
        audio_np: 1D numpy array of audio samples
        sr: sample rate
        n_mels: number of Mel bands
        n_fft: FFT size
        hop: hop length
        target_frames: interpolate to this many time frames
        mel_fb: pre-computed Mel filterbank (created if None)
    Returns:
        torch.Tensor [n_mels, target_frames]
    """
    if mel_fb is None:
        mel_fb = mel_filterbank(n_mels, n_fft, sr)

    audio_t = torch.from_numpy(audio_np).float()
    if audio_t.dim() > 1:
        audio_t = audio_t.mean(dim=0)  # stereo -> mono

    window = torch.hann_window(n_fft)
    stft = torch.stft(audio_t, n_fft=n_fft, hop_length=hop, window=window,
                       return_complex=True, center=True)
    mag = stft.abs()
    mel_spec = torch.log(mel_fb.to(mag.device) @ mag + 1e-6)

    # Interpolate to target_frames
    mel_spec = mel_spec.unsqueeze(0)  # [1, n_mels, T]
    mel_interp = F.interpolate(mel_spec, size=target_frames,
                                mode='linear', align_corners=False)
    return mel_interp.squeeze(0)  # [n_mels, target_frames]


# ============================================================
# 1. Datasets
# ============================================================
class MSCVideoDataset(Dataset):
    """Pure video dataset: Kinetics + Sora/CogVideo/HunyuanVideo."""
    def __init__(self, root_dir="/root/autodl-tmp", clip_len=32, spatial_size=224,
                 split='train', split_ratio=0.9, stage=1):
        super().__init__()
        self.root_dir = Path(root_dir).resolve()
        self.clip_len = clip_len
        self.spatial_size = spatial_size
        self.split = split
        self.stage = stage
        self.video_samples = []
        self._build_index(split_ratio)

    def _build_index(self, split_ratio):
        real_samples, fake_samples = [], []
        for mp4_file in self.root_dir.rglob('*.mp4'):
            if mp4_file.name.startswith('.'):
                continue
            try:
                if mp4_file.stat().st_size < 102400:
                    continue
            except OSError:
                continue

            p = str(mp4_file).lower()
            is_fake = any(k in p for k in ['sora', 'cogvideo', 'hunyuanvideo', 'fakevideo'])
            is_real = any(k in p for k in ['kinetics', 'realvideo', 'voxceleb'])
            if is_fake == is_real:
                continue
            label = 1 if is_fake else 0
            if is_real:
                real_samples.append((str(mp4_file), label))
            else:
                fake_samples.append((str(mp4_file), label))

        rng = random.Random(42)
        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)

        def split_list(lst, ratio):
            n = int(len(lst) * ratio)
            return lst[:n], lst[n:]

        real_train, real_val = split_list(real_samples, split_ratio)
        fake_train, fake_val = split_list(fake_samples, split_ratio)

        if self.split == 'train':
            self.video_samples = real_train + (fake_train if self.stage >= 2 else [])
        else:
            self.video_samples = real_val + (fake_val if self.stage >= 2 else [])

        rng.shuffle(self.video_samples)
        n_real = sum(1 for s in self.video_samples if s[1] == 0)
        n_fake = sum(1 for s in self.video_samples if s[1] == 1)
        mode = 'PureVideo' if self.stage <= 1 or not fake_samples else 'Real+Fake'
        logger.info(f"Dataset [S{self.stage} {self.split}]: {len(self.video_samples)} samples "
                    f"(real={n_real}, fake={n_fake}, mode={mode})")

    def __len__(self):
        return len(self.video_samples)

    def __getitem__(self, idx):
        video_path, label = self.video_samples[idx]
        try:
            vr = decord.VideoReader(
                video_path, ctx=decord.cpu(0),
                width=self.spatial_size, height=self.spatial_size
            )
            total_frames = len(vr)

            if self.split == 'train':
                start = random.randint(0, max(0, total_frames - self.clip_len))
            else:
                start = max(0, (total_frames - self.clip_len) // 2)

            indices = list(range(start, min(total_frames, start + self.clip_len)))
            while len(indices) < self.clip_len:
                indices.append(indices[-1])

            frames = vr.get_batch(indices).float() / 255.0
            frames = frames.permute(3, 0, 1, 2)
            return frames, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


class MSCAudioVideoDataset(Dataset):
    """Audio-visual dataset for FakeAVCeleb_v1.2.

    Directory structure:
        FakeAVCeleb_v1.2/
            RealVideo-RealAudio/    -> label 0 (real)
            RealVideo-FakeAudio/    -> label 1 (fake)
            FakeVideo-RealAudio/    -> label 1 (fake)
            FakeVideo-FakeAudio/    -> label 1 (fake)
    """
    def __init__(self, root_dir="/root/autodl-tmp/FakeAVCeleb_v1.2",
                 clip_len=32, spatial_size=224, split='train',
                 split_ratio=0.9, stage=1, fake_ratio=3.0):
        """
        Args:
            root_dir: path to FakeAVCeleb_v1.2
            clip_len: number of frames per clip
            spatial_size: resize video to (spatial_size, spatial_size)
            split: 'train' or 'val'
            split_ratio: train/val split ratio
            stage: 1 (real only) or 2 (real+fake)
            fake_ratio: ratio of fake:real samples (default 3.0 -> 3x more fake)
        """
        super().__init__()
        self.root_dir = Path(root_dir).resolve()
        self.clip_len = clip_len
        self.spatial_size = spatial_size
        self.split = split
        self.stage = stage
        self.fake_ratio = fake_ratio
        self.video_samples = []

        # Pre-compute Mel filterbank once
        self.mel_fb = mel_filterbank()

        # Try to import av (PyAV)
        try:
            import av as _av
            self._av = _av
        except ImportError:
            raise ImportError("PyAV is required for audio-visual training. "
                              "Install with: pip install av")

        self._build_index(split_ratio)

    def _build_index(self, split_ratio):
        cat_dirs = {
            'RealVideo-RealAudio': 0,
            'RealVideo-FakeAudio': 1,
            'FakeVideo-RealAudio': 1,
            'FakeVideo-FakeAudio': 1,
        }

        real_samples, fake_samples = [], []
        for cat_name, label in cat_dirs.items():
            cat_path = self.root_dir / cat_name
            if not cat_path.exists():
                continue
            for mp4_file in cat_path.rglob('*.mp4'):
                if mp4_file.name.startswith('.'):
                    continue
                try:
                    if mp4_file.stat().st_size < 10240:
                        continue
                except OSError:
                    continue
                if label == 0:
                    real_samples.append((str(mp4_file), label))
                else:
                    fake_samples.append((str(mp4_file), label))

        rng = random.Random(42)
        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)

        # Balance: limit fake to fake_ratio * n_real
        n_real = len(real_samples)
        n_fake_max = int(n_real * self.fake_ratio)
        if len(fake_samples) > n_fake_max:
            fake_samples = fake_samples[:n_fake_max]
            logger.info(f"  Subsampled fake: {n_fake_max}/{len(fake_samples) + n_fake_max} "
                        f"(ratio {self.fake_ratio}:1)")

        def split_list(lst, ratio):
            n = int(len(lst) * ratio)
            return lst[:n], lst[n:]

        real_train, real_val = split_list(real_samples, split_ratio)
        fake_train, fake_val = split_list(fake_samples, split_ratio)

        if self.split == 'train':
            self.video_samples = real_train + (fake_train if self.stage >= 2 else [])
        else:
            self.video_samples = real_val + (fake_val if self.stage >= 2 else [])

        rng.shuffle(self.video_samples)
        n_r = sum(1 for s in self.video_samples if s[1] == 0)
        n_f = sum(1 for s in self.video_samples if s[1] == 1)
        mode = 'RealOnly' if self.stage <= 1 else 'Real+Fake'
        logger.info(f"Dataset [S{self.stage} {self.split}]: {len(self.video_samples)} samples "
                    f"(real={n_r}, fake={n_f}, mode={mode})")

    def __len__(self):
        return len(self.video_samples)

    def __getitem__(self, idx):
        video_path, label = self.video_samples[idx]
        try:
            # --- Video ---
            vr = decord.VideoReader(
                video_path, ctx=decord.cpu(0),
                width=self.spatial_size, height=self.spatial_size
            )
            total_frames = len(vr)

            if self.split == 'train':
                start = random.randint(0, max(0, total_frames - self.clip_len))
            else:
                start = max(0, (total_frames - self.clip_len) // 2)

            indices = list(range(start, min(total_frames, start + self.clip_len)))
            while len(indices) < self.clip_len:
                indices.append(indices[-1])

            frames = vr.get_batch(indices).float() / 255.0
            frames = frames.permute(3, 0, 1, 2)  # [C, T, H, W]

            # --- Audio ---
            try:
                container = self._av.open(video_path)
                audio_stream = container.streams.audio[0]
                audio_frames = container.decode(audio_stream)
                samples = []
                for af in audio_frames:
                    samples.append(af.to_ndarray())
                audio_np = np.concatenate(samples, axis=1)  # [ch, samples]
                audio_np = audio_np.mean(axis=0)  # stereo -> mono
                sr = audio_stream.codec_context.sample_rate
                container.close()

                mel = compute_mel(audio_np, sr=sr, mel_fb=self.mel_fb)
            except Exception:
                # Audio decode failed -> return zeros
                mel = torch.zeros(80, 16)

            return frames, mel, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ============================================================
# 2. Stage 1 Losses
# ============================================================
def path_a_recon_loss(outputs):
    X_orig = outputs['X_main_orig']
    X_m = outputs['X_main_m']
    A_tilde = outputs['A_tilde']
    mask = outputs['mask']
    B, N, D = X_orig.shape

    eye = torch.eye(N, device=X_orig.device).unsqueeze(0)
    A_no_self = A_tilde * (1.0 - eye)
    A_no_self = A_no_self / (A_no_self.sum(dim=-1, keepdim=True) + 1e-8)

    X_recon = torch.bmm(A_no_self, X_m)
    mask_f = mask.unsqueeze(-1).float()
    loss = F.mse_loss(X_recon * mask_f, X_orig * mask_f, reduction='sum')
    return loss / (mask_f.sum() * D + 1e-8)


def path_b_recon_loss(outputs):
    X_low_orig = outputs['X_low_orig']
    X_high_orig = outputs['X_high_orig']
    X_low_m = outputs['X_low_m']
    X_high_m = outputs['X_high_m']
    A_joint = outputs['A_joint_lh']
    mask_low = outputs['mask_low_b']
    mask_high = outputs['mask_high_b']

    B, N, D = X_low_orig.shape

    eye = torch.eye(2 * N, device=X_low_orig.device).unsqueeze(0)
    A_no_self = A_joint * (1.0 - eye)
    A_no_self = A_no_self / (A_no_self.sum(dim=-1, keepdim=True) + 1e-8)

    X_joint_orig = torch.cat([X_low_orig, X_high_orig], dim=1)
    X_joint_m = torch.cat([X_low_m, X_high_m], dim=1)
    X_recon = torch.bmm(A_no_self, X_joint_m)

    mask_joint = torch.cat([mask_low, mask_high], dim=1).unsqueeze(-1).float()
    loss = F.mse_loss(X_recon * mask_joint, X_joint_orig * mask_joint, reduction='sum')
    return loss / (mask_joint.sum() * D + 1e-8)


# ============================================================
# 3. Training Engine
# ============================================================
def train_msc(model, root_dir, batch_size=64, epochs_s1=5, epochs_s2=20,
              lr=1e-4, resume_path=None, save_every=200, use_audio=False,
              fake_ratio=3.0, num_workers=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    save_dir = Path("msc_checkpoints")
    save_dir.mkdir(exist_ok=True)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    current_stage = 1
    start_epoch = 0

    # Pick dataset class based on data type
    DatasetClass = MSCAudioVideoDataset if use_audio else MSCVideoDataset
    dataset_kwargs = lambda split, stage: (
        {'root_dir': root_dir, 'split': split, 'stage': stage, 'fake_ratio': fake_ratio}
        if use_audio else
        {'root_dir': root_dir, 'split': split, 'stage': stage}
    )

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw = ckpt.get('model_state_dict', ckpt)
        clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}
        # Filter mismatched keys (e.g. classifier dimension change)
        model_dict = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in clean.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                filtered[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(filtered, strict=False)
        if skipped:
            logger.info(f"  Skipped {len(skipped)} mismatched keys: {skipped[:3]}...")
        current_stage = ckpt.get('stage', 1)
        start_epoch = ckpt.get('epoch', 0)
        logger.info(f"Resumed: Stage {current_stage}, Epoch {start_epoch+1}")

    try:
        # ============================================================
        # Stage 1: Masked Reconstruction (Real videos only)
        # ============================================================
        if current_stage == 1:
            logger.info("=" * 55)
            logger.info("Stage 1: Masked Reconstruction (Real videos only)")
            logger.info("=" * 55)

            ds_train = DatasetClass(**dataset_kwargs('train', 1))
            ds_val = DatasetClass(**dataset_kwargs('val', 1))

            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True,
                                  prefetch_factor=4, persistent_workers=True)
            dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                num_workers=2, drop_last=False, pin_memory=True,
                                prefetch_factor=4, persistent_workers=True)

            opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scaler = GradScaler('cuda')
            best_val = float('inf')

            for epoch in range(start_epoch, epochs_s1):
                model.train()
                loss_total = 0.0

                pbar = tqdm(dl_train, desc=f"[S1] E{epoch+1}/{epochs_s1}",
                            leave=False, dynamic_ncols=True)
                for bi, batch_data in enumerate(pbar):
                    if use_audio:
                        videos, audio_mel, _ = batch_data
                        audio_mel = audio_mel.to(device)
                    else:
                        videos, _ = batch_data
                        audio_mel = None

                    videos = videos.to(device)
                    opt.zero_grad()

                    with autocast(device_type=device.type):
                        outputs = model(videos, audio_mel=audio_mel,
                                        mask_ratio=0.5, return_all=True)
                        loss_a = path_a_recon_loss(outputs)
                        loss_b = path_b_recon_loss(outputs)
                        loss = loss_a + 0.5 * loss_b

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()

                    loss_total += loss.item()
                    pbar.set_postfix({
                        'Loss': f"{loss.item():.3f}",
                        'A': f"{loss_a.item():.3f}",
                        'B': f"{loss_b.item():.3f}",
                    })

                    if save_every > 0 and bi > 0 and bi % save_every == 0:
                        torch.save({
                            'stage': 1, 'epoch': epoch + 1, 'batch': bi,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                        }, save_dir / "latest.pth")

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_data in dl_val:
                        if use_audio:
                            videos, audio_mel, _ = batch_data
                            audio_mel = audio_mel.to(device)
                        else:
                            videos, _ = batch_data
                            audio_mel = None

                        videos = videos.to(device)
                        with autocast(device_type=device.type):
                            outputs = model(videos, audio_mel=audio_mel,
                                            mask_ratio=0.5, return_all=True)
                            loss = path_a_recon_loss(outputs) + 0.5 * path_b_recon_loss(outputs)
                        val_loss += loss.item()

                avg_train = loss_total / max(1, len(dl_train))
                avg_val = val_loss / max(1, len(dl_val))
                logger.info(f"[S1] Epoch {epoch+1}/{epochs_s1} | "
                            f"Train: {avg_train:.4f} | Val: {avg_val:.4f}")

                if avg_val < best_val:
                    best_val = avg_val
                    torch.save(model.state_dict(), save_dir / "best_stage1.pth")

                torch.save({
                    'stage': 1, 'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, save_dir / "latest.pth")

            current_stage = 2
            start_epoch = 0
            best_s1 = save_dir / "best_stage1.pth"
            if best_s1.exists():
                ckpt = torch.load(best_s1, map_location=device, weights_only=False)
                model.load_state_dict(ckpt, strict=False)
                logger.info("Loaded best Stage 1 weights for Stage 2.")

        # ============================================================
        # Stage 2: Discriminative Fine-tuning
        # ============================================================
        if current_stage == 2:
            logger.info("=" * 55)
            logger.info("Stage 2: BCE Fine-tuning (Real + Fake)")
            logger.info("=" * 55)

            ds_train = DatasetClass(**dataset_kwargs('train', 2))
            ds_val = DatasetClass(**dataset_kwargs('val', 2))

            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True,
                                  prefetch_factor=4, persistent_workers=True)
            dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                                num_workers=2, drop_last=False, pin_memory=True,
                                prefetch_factor=4, persistent_workers=True)

            opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scaler = GradScaler('cuda')
            best_acc = 0.0

            for epoch in range(start_epoch, epochs_s2):
                model.train()
                loss_t, acc_t, msc_r, msc_f, nb = 0.0, 0.0, 0.0, 0.0, 0

                pbar = tqdm(dl_train, desc=f"[S2] E{epoch+1}/{epochs_s2}",
                            leave=False, dynamic_ncols=True)
                for batch_data in pbar:
                    if use_audio:
                        videos, audio_mel, labels = batch_data
                        audio_mel = audio_mel.to(device)
                    else:
                        videos, labels = batch_data
                        audio_mel = None

                    videos = videos.to(device)
                    labels = labels.to(device).unsqueeze(1)

                    opt.zero_grad()
                    with autocast(device_type=device.type):
                        outputs = model(videos, audio_mel=audio_mel,
                                        mask_ratio=0.0, return_all=False)
                        loss = bce_loss(outputs['logits'], labels)

                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()

                    with torch.no_grad():
                        preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
                        acc = (preds == labels).float().mean().item()

                    loss_t += loss.item()
                    acc_t += acc
                    msc = outputs['msc_lh'].detach()
                    real_m = (labels.squeeze() == 0)
                    fake_m = (labels.squeeze() == 1)
                    if real_m.any():
                        msc_r += msc[real_m].mean().item()
                    if fake_m.any():
                        msc_f += msc[fake_m].mean().item()
                    nb += 1

                    pbar.set_postfix({
                        'Loss': f"{loss.item():.3f}",
                        'Acc': f"{acc*100:.0f}%",
                        'MSC_R': f"{msc_r/max(1,nb):.3f}",
                        'MSC_F': f"{msc_f/max(1,nb):.3f}",
                    })

                    if save_every > 0 and nb > 0 and nb % save_every == 0:
                        torch.save({
                            'stage': 2, 'epoch': epoch + 1, 'batch': nb,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                        }, save_dir / "latest.pth")

                # Validation
                model.eval()
                v_loss, v_acc, v_msc_r, v_msc_f, nv = 0.0, 0.0, 0.0, 0.0, 0
                with torch.no_grad():
                    for batch_data in dl_val:
                        if use_audio:
                            videos, audio_mel, labels = batch_data
                            audio_mel = audio_mel.to(device)
                        else:
                            videos, labels = batch_data
                            audio_mel = None

                        videos = videos.to(device)
                        labels = labels.to(device).unsqueeze(1)
                        with autocast(device_type=device.type):
                            outputs = model(videos, audio_mel=audio_mel,
                                            mask_ratio=0.0, return_all=False)
                            loss = bce_loss(outputs['logits'], labels)

                        preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
                        v_loss += loss.item()
                        v_acc += (preds == labels).float().mean().item()

                        msc = outputs['msc_lh']
                        real_m = (labels.squeeze() == 0)
                        fake_m = (labels.squeeze() == 1)
                        if real_m.any():
                            v_msc_r += msc[real_m].mean().item()
                        if fake_m.any():
                            v_msc_f += msc[fake_m].mean().item()
                        nv += 1

                avg_l = loss_t / max(1, nb)
                avg_a = (acc_t / max(1, nb)) * 100
                avg_vl = v_loss / max(1, nv)
                avg_va = (v_acc / max(1, nv)) * 100
                r_msc = v_msc_r / max(1, nv)
                f_msc = v_msc_f / max(1, nv)

                logger.info(
                    f"[S2] E{epoch+1}/{epochs_s2} | "
                    f"Train Loss:{avg_l:.4f} Acc:{avg_a:.1f}% | "
                    f"Val Loss:{avg_vl:.4f} Acc:{avg_va:.1f}% | "
                    f"MSC Real:{r_msc:.4f} Fake:{f_msc:.4f} Gap:{r_msc-f_msc:.4f}"
                )

                if avg_va > best_acc:
                    best_acc = avg_va
                    torch.save({
                        'stage': 2, 'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'val_acc': avg_va,
                    }, save_dir / "best_stage2.pth")
                    logger.info(f"  -> Best (Acc={best_acc:.1f}%)")

                torch.save({
                    'stage': 2, 'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, save_dir / "latest.pth")

    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Interrupted:\n{traceback.format_exc()}")
        torch.save({
            'stage': current_stage,
            'epoch': epoch + 1 if 'epoch' in dir() else 0,
            'model_state_dict': model.state_dict(),
        }, save_dir / "crash.pth")


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp')
    parser.add_argument('--data_type', type=str, default='purevideo',
                        choices=['purevideo', 'fakeavceleb'],
                        help='Dataset type: purevideo (Kinetics+Sora/etc) or '
                             'fakeavceleb (FakeAVCeleb_v1.2 with audio)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs_s1', type=int, default=5)
    parser.add_argument('--epochs_s2', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fake_ratio', type=float, default=3.0,
                        help='Fake:real ratio for FakeAVCeleb (default 3:1)')
    parser.add_argument('--num_workers', type=int, default=6,
                        help='DataLoader workers')
    parser.add_argument('--no_resume', action='store_true',
                        help='Force fresh training, ignore existing checkpoints')
    parser.add_argument('--save_every', type=int, default=200,
                        help='Save checkpoint every N batches')
    args = parser.parse_args()

    use_audio = (args.data_type == 'fakeavceleb')
    data_root = args.data_root
    if use_audio and 'FakeAVCeleb' not in data_root:
        data_root = str(Path(data_root) / 'FakeAVCeleb_v1.2')

    # Auto-detect latest checkpoint
    latest_ckpt = Path("msc_checkpoints/latest.pth")
    resume_path = None
    if not args.no_resume and latest_ckpt.exists():
        resume_path = str(latest_ckpt)
        logger.info(f"Auto-resuming from {resume_path}")

    mode = 'AudioVisual' if use_audio else 'PureVideo'
    logger.info(f"MSC Merged Training | Mode: {mode} | Data: {args.data_type}")
    logger.info(f"  S1 epochs: {args.epochs_s1}, S2 epochs: {args.epochs_s2}")
    logger.info(f"  Batch: {args.batch_size}, LR: {args.lr}")

    model = MSCDetector(
        embed_dim=384, hidden_dim=256, attn_dim=64,
        k_sparse=30, tau=2, use_audio=use_audio,
    )

    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params: {n_tot:,} total, {n_tr:,} trainable")

    train_msc(
        model=model, root_dir=data_root,
        batch_size=args.batch_size,
        epochs_s1=args.epochs_s1, epochs_s2=args.epochs_s2,
        lr=args.lr, resume_path=resume_path, save_every=args.save_every,
        use_audio=use_audio, fake_ratio=args.fake_ratio,
        num_workers=args.num_workers,
    )
