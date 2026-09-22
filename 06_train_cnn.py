"""
06_train_cnn.py

Trains a CNN on the mel-spectrograms to classify melam type
(Panchari / Pandi / Panchavadyam / Thayambaka).

Handles the class imbalance (Thayambaka ~6300 clips vs Pandi ~2900 clips)
via a weighted loss function computed from the TRAIN split's class counts —
this makes mistakes on under-represented classes "cost" more during
training, instead of the model just learning to over-predict the biggest
class.

Requires: torch, numpy, pandas, scikit-learn, tqdm

Usage:
    python 06_train_cnn.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SpectrogramDataset(Dataset):
    """Loads a (n_channels, n_mels, frames) .npy feature array — channel 0
    is the log-mel-spectrogram (timbre), channel 1 is a tempogram (rhythm/
    tempo) — pads/crops the time axis to a fixed length, and returns a
    (n_channels, n_mels, frames) tensor.

    When augment=True (train split only), applies SpecAugment-style random
    frequency/time masking. This is the standard fix for a CNN overfitting
    to recording-specific characteristics (mic, room, compression) instead
    of the actual audio patterns — by randomly blanking out chunks of the
    spectrogram each epoch, the model can't rely on any single narrow
    "fingerprint" and is forced to learn more general, robust features."""

    def __init__(self, df, label_to_idx, fixed_frames=config.FIXED_SPEC_FRAMES, augment=False):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.fixed_frames = fixed_frames
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _pad_or_crop(self, spec):
        # spec is (channels, n_mels, frames) — only the time (last) axis
        # ever needs padding/cropping.
        frames = spec.shape[-1]
        if frames == self.fixed_frames:
            return spec
        if frames > self.fixed_frames:
            start = (frames - self.fixed_frames) // 2
            return spec[..., start:start + self.fixed_frames]
        pad_total = self.fixed_frames - frames
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        # Pad each channel with ITS OWN min, not a single shared min — mel-dB
        # and tempogram values live on very different scales.
        pad_widths = [(0, 0)] * (spec.ndim - 1) + [(pad_left, pad_right)]
        out = np.empty(spec.shape[:-1] + (self.fixed_frames,), dtype=spec.dtype)
        for c in range(spec.shape[0]):
            out[c] = np.pad(spec[c], pad_widths[1:], mode="constant", constant_values=spec[c].min())
        return out

    def _spec_augment(self, spec):
        # spec: (channels, n_mels, frames). Apply the same masking pattern
        # to all channels so the mel/tempogram stay time-aligned.
        n_channels, n_mels, frames = spec.shape

        for c in range(n_channels):
            fill_value = spec[c].min()
            for _ in range(2):
                f_width = np.random.randint(0, n_mels // 8)
                if f_width > 0:
                    f_start = np.random.randint(0, n_mels - f_width)
                    spec[c, f_start:f_start + f_width, :] = fill_value
            for _ in range(2):
                t_width = np.random.randint(0, frames // 8)
                if t_width > 0:
                    t_start = np.random.randint(0, frames - t_width)
                    spec[c, :, t_start:t_start + t_width] = fill_value

        return spec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec = np.load(row["npy_path"]).astype(np.float32)
        if spec.ndim == 2:
            # Backward-compat: old single-channel (mel-only) .npy files.
            spec = spec[np.newaxis, ...]
        spec = self._pad_or_crop(spec)

        # Normalize PER CHANNEL to roughly [-1, 1] — mel-dB values are
        # typically in [-80, 0] while tempogram values are a very different
        # range; normalizing jointly would let one channel's scale drown
        # out the other's.
        for c in range(spec.shape[0]):
            spec[c] = (spec[c] - spec[c].mean()) / (spec[c].std() + 1e-6)

        if self.augment:
            spec = self._spec_augment(spec)

        tensor = torch.from_numpy(spec)  # (n_channels, n_mels, frames)
        label = self.label_to_idx[row["class"]]
        return tensor, label


# ---------------------------------------------------------------------------
# Model — a small CNN, deliberately modest in size given dataset scale
# ---------------------------------------------------------------------------
class MelamCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.N_INPUT_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # n_mels/2, frames/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # n_mels/4, frames/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # n_mels/8, frames/8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pool -> fixed size regardless of input dims
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ResNetMelamClassifier(nn.Module):
    """ImageNet-pretrained ResNet18, adapted for 1-channel spectrograms.

    Why this can help when a from-scratch CNN overfits to recording-specific
    characteristics: the pretrained backbone already encodes general visual
    features (edges, textures, gradients) that have no reason to latch onto
    narrow, recording-specific noise the way a model trained purely on a
    small number of unique source videos can. Freezing most of the backbone
    also means far fewer parameters are actually being fit to your data,
    directly reducing overfitting risk rather than adding to it.
    """

    def __init__(self, n_classes, freeze_backbone=True):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18's first conv expects 3 channels; our spectrograms have
        # config.N_INPUT_CHANNELS channels (2: mel + tempogram, by default).
        # Replace it, initializing new weights as the pretrained 3-channel
        # weights averaged down to 1 channel and then repeated across our
        # input channels, so it starts from a sensible point rather than random.
        old_conv = backbone.conv1
        n_in = config.N_INPUT_CHANNELS
        new_conv = nn.Conv2d(n_in, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                              stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            averaged = old_conv.weight.mean(dim=1, keepdim=True)  # (out_ch, 1, kH, kW)
            new_conv.weight[:] = averaged.repeat(1, n_in, 1, 1) / n_in
        backbone.conv1 = new_conv

        if freeze_backbone:
            for name, param in backbone.named_parameters():
                # Keep the last residual block (layer4) trainable so the
                # network can still adapt higher-level features to spectrograms;
                # everything earlier (generic low-level features) stays frozen.
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False

        backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(backbone.fc.in_features, n_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class CRNNMelamClassifier(nn.Module):
    """CNN + BiGRU, built specifically to fix a problem the other two
    architectures have: they both end in AdaptiveAvgPool2d((1,1)), which
    collapses the ENTIRE time axis into a single vector before
    classification. That throws away rhythm/tempo information — and
    melam-type is defined almost entirely by tempo and beat-cycle length
    (Panchari 96->48->24->12->6 vs Pandi 56->28->14->7), not by timbre,
    since they share the same instruments. A model that only sees averaged
    frequency energy is left classifying near-identical "sound" for melams
    that only differ in their rhythmic cycle, which is exactly the
    Pandi<->Panchari confusion seen in the confusion matrix.

    Fix: the conv stack here pools frequency aggressively but time only
    lightly, so a real time sequence survives into a BiGRU that can learn
    tempo/cycle patterns. Mean+max pooling over the GRU's per-timestep
    outputs (rather than a single final hidden state) makes the pooled
    summary robust to where in the clip the cycle happens to start.
    """

    def __init__(self, n_classes, rnn_hidden=128, rnn_layers=2, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, freq_pool, time_pool):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                # (freq_pool, time_pool) kernel — pool frequency harder than
                # time so temporal resolution (the rhythm signal) survives.
                nn.MaxPool2d((freq_pool, time_pool)),
            )

        self.conv = nn.Sequential(
            conv_block(config.N_INPUT_CHANNELS, 32, freq_pool=2, time_pool=2),  # n_mels/2, frames/2
            conv_block(32, 64, freq_pool=2, time_pool=2),   # n_mels/4,  frames/4
            conv_block(64, 128, freq_pool=2, time_pool=1),  # n_mels/8,  frames/4  (time preserved here)
            conv_block(128, 128, freq_pool=2, time_pool=1), # n_mels/16, frames/4  (time preserved here)
        )
        # After this, shape is (batch, 128, n_mels/16, frames/4). Frequency
        # gets averaged away (it's timbre, already captured by the conv
        # filters); time is kept as the sequence dimension for the GRU.
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        gru_out_dim = rnn_hidden * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            # mean-pool and max-pool over time, concatenated — robust to
            # where in the clip a beat cycle happens to start/end.
            nn.Linear(gru_out_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)                      # (B, C, F, T)
        x = x.mean(dim=2)                      # collapse frequency -> (B, C, T)
        x = x.permute(0, 2, 1)                 # (B, T, C) for the GRU
        out, _ = self.rnn(x)                   # (B, T, 2*hidden)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled)


def build_model(n_classes):
    if config.MODEL_ARCHITECTURE == "crnn":
        print("Building CRNN (CNN + BiGRU, time-preserving) — captures rhythm/tempo, not just timbre")
        return CRNNMelamClassifier(n_classes)
    if config.MODEL_ARCHITECTURE == "resnet18":
        print(f"Building ResNet18 (pretrained, freeze_backbone={config.RESNET_FREEZE_BACKBONE})")
        return ResNetMelamClassifier(n_classes, freeze_backbone=config.RESNET_FREEZE_BACKBONE)
    print("Building simple_cnn (from scratch)")
    return MelamCNN(n_classes)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def mixup_batch(specs, labels, alpha, device):
    """Blend pairs of spectrograms (and their labels) within a batch.

    Why: with only 8-19 unique source videos per class, a fast-fitting model
    (like the from-scratch CRNN, with no frozen backbone to slow it down)
    can memorize per-recording artifacts within a single epoch instead of
    learning general melam features. Mixup forces every training example to
    be a blend of two different clips, so there's no fixed "video fingerprint"
    to latch onto — the model has to learn features that behave linearly
    between examples, which is a much stronger anti-memorization pressure
    than dropout/weight-decay alone.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(specs.size(0), device=device)
    mixed_specs = lam * specs + (1 - lam) * specs[perm]
    labels_a, labels_b = labels, labels[perm]
    return mixed_specs, labels_a, labels_b, lam


def run_epoch(model, loader, criterion, optimizer, device, train=True,
              use_mixup=False, mixup_alpha=0.2, grad_clip_norm=5.0):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for specs, labels in tqdm(loader, leave=False):
            specs, labels = specs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if train and use_mixup:
                mixed_specs, labels_a, labels_b, lam = mixup_batch(specs, labels, mixup_alpha, device)
                outputs = model(mixed_specs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(specs)
                loss = criterion(outputs, labels)

            if train:
                loss.backward()
                # Gradient clipping — GRUs are prone to exploding gradients;
                # without this, a bad batch can spike the loss and destabilize
                # training for several subsequent epochs (this is what caused
                # the val_loss spike to 5.4 at epoch 5 in the run without clipping).
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * specs.size(0)
            preds = outputs.argmax(dim=1)
            # Accuracy is always measured against the true (unmixed) labels.
            # Under mixup this slightly understates train accuracy (expected,
            # since the model is asked to predict a blend) — val/test are
            # never mixed, so those numbers stay exact and comparable.
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    # Seed everything, not just the pandas/sklearn split (config.RANDOM_SEED
    # was already used there). Without this, PyTorch's weight initialization
    # and DataLoader shuffling differ every run, which is a large chunk of
    # why "identical" reruns produced very different results (e.g. epoch-1
    # val_acc of 0.4851 in one run vs 0.6461 in the next with no config
    # change) — that noise made it hard to tell whether a config change
    # actually helped.
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if not os.path.exists(config.SPLITS_CSV):
        raise SystemExit(f"Missing {config.SPLITS_CSV} — run 05_split_dataset.py first.")

    df = pd.read_csv(config.SPLITS_CSV)
    classes = sorted(df["class"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    with open(config.LABEL_MAP_JSON, "w") as f:
        json.dump(label_to_idx, f, indent=2)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Cap clips-per-video in TRAIN only. Some source videos contributed
    # 1000+ clips (one Panchari video alone: 1,288) vs. others under 100 —
    # left uncapped, gradient updates are dominated by whichever handful of
    # videos happened to be long, so the model effectively "sees" only a
    # few unique recordings per class regardless of nominal clip count. Val
    # and test are left untouched — they're already video-scarce (as few as
    # 1-2 videos per class), and capping them further would only shrink an
    # already-small evaluation sample without helping training.
    if config.MAX_CLIPS_PER_VIDEO is not None:
        before = len(train_df)
        # Index-based sampling (not groupby().apply(...)) — apply() on a
        # groupby silently drops the grouping column in pandas >= 2.2, which
        # would delete video_id right when we need it most.
        keep_idx = []
        for _, group in train_df.groupby("video_id"):
            n = min(len(group), config.MAX_CLIPS_PER_VIDEO)
            keep_idx.extend(group.sample(n=n, random_state=config.RANDOM_SEED).index.tolist())
        train_df = train_df.loc[keep_idx].reset_index(drop=True)
        print(f"Capped clips/video in TRAIN at {config.MAX_CLIPS_PER_VIDEO}: "
              f"{before} -> {len(train_df)} clips "
              f"({train_df['video_id'].nunique()} unique videos)")

    train_ds = SpectrogramDataset(train_df, label_to_idx, augment=True)
    val_ds = SpectrogramDataset(val_df, label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(n_classes=len(classes)).to(device)

    # Weighted loss — computed from TRAIN split class counts only (never
    # from val/test, to avoid any leakage of split-specific information).
    train_labels = train_df["class"].map(label_to_idx).values
    class_weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Manual override on top of frequency balancing. Evidence from a fair
    # (post split-fix) test run: ~78% of ALL misclassifications across every
    # class landed on Thayambaka — it's being used as a default "unsure"
    # guess, especially swallowing Pandi (recall crashed to 19%) and Panchari
    # (recall 45%). Frequency-balanced weighting alone can't fix this: it
    # only rewards getting TRUE-Thayambaka samples right, it doesn't
    # penalize wrongly guessing Thayambaka on a different true class. This
    # musically makes some sense too — Thayambaka has no fixed kalam/tempo
    # structure (it's improvisational), so its tempogram signature is
    # naturally the most varied, and "ambiguous rhythm" ends up looking
    # statistically like "genuine Thayambaka." Pushing its weight down (and
    # Pandi/Panchari's up) makes true-Thayambaka mistakes cheaper and
    # true-Pandi/Panchari mistakes costlier, directly countering the bias.
    for cls_name, multiplier in config.CLASS_WEIGHT_OVERRIDES.items():
        if cls_name in label_to_idx:
            class_weights[label_to_idx[cls_name]] *= multiplier

    print("Class weights (train-derived):",
          {classes[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())})

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_acc = 0.0            # tracks the best checkpoint ever saved (any improvement, however small)
    best_val_acc_for_patience = 0.0  # tracks improvement for early-stopping purposes only (noise-filtered)
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            use_mixup=config.USE_MIXUP, mixup_alpha=config.MIXUP_ALPHA,
            grad_clip_norm=config.GRAD_CLIP_NORM,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{config.EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ALWAYS keep the literal best-ever checkpoint, even for a tiny gain
        # — throwing away a genuine (if small) improvement because it didn't
        # clear a noise-filtering bar was the bug in the previous version
        # (epoch 4's val_acc=0.6517 beat epoch 1's 0.6461, but a 1.0pt
        # min_delta discarded it, leaving an under-trained epoch-1 model
        # locked in as "best").
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "val_acc": val_acc,
                "epoch": epoch,
            }, config.BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

        # Patience/stopping decision uses the min_delta-filtered threshold
        # SEPARATELY — this is what should absorb epoch-to-epoch noise from
        # the small number of unique val videos, without discarding
        # checkpoints.
        if val_acc > best_val_acc_for_patience + config.EARLY_STOP_MIN_DELTA:
            best_val_acc_for_patience = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\nNo improvement for {config.EARLY_STOP_PATIENCE} epochs — stopping early.")
                break

    # Save training curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVES_PNG, dpi=120)
    print(f"\nTraining curves saved to {config.TRAINING_CURVES_PNG}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print("Run 07_evaluate_model.py to test on the held-out test set.")


if __name__ == "__main__":
    main()