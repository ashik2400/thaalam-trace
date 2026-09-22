"""
08_predict.py

Run the trained model on a NEW audio file (a recording that was never part
of the dataset/training pipeline) — this is what you use to sanity-check the
model against real-world audio, not just the held-out test split.

Preprocessing here deliberately mirrors 03_generate_spectrograms.py and the
SpectrogramDataset in 06_train_cnn.py exactly (same sample rate, mel params,
frame count, per-clip normalization) — a mismatch there is the #1 reason a
model that scores well on the test set looks like it's guessing randomly on
new audio.

Two modes:
  1. Single clip (5-10s) -> one prediction with per-class probabilities.
  2. Long recording -> split into overlapping windows, run each through the
     model, and report both per-window predictions and an averaged verdict.
     Useful because a real festival recording is rarely a clean 7s clip.

Requires: torch, librosa, numpy (already in requirements.txt)

Usage:
    python 08_predict.py path/to/clip.wav
    python 08_predict.py path/to/long_recording.wav --windowed
    python 08_predict.py path/to/clip.mp3 --model models/best_model.pt
"""

import argparse
import os

import librosa
import numpy as np
import torch

import config
from importlib import import_module

train_module = import_module("06_train_cnn")
build_model = train_module.build_model


# ---------------------------------------------------------------------------
# Preprocessing — must match 03_generate_spectrograms.py + SpectrogramDataset
# EXACTLY, including the 2-channel (mel + tempogram) stack.
# ---------------------------------------------------------------------------
def make_features(y, sr):
    """Same 2-channel [mel, tempogram] stack as 03_generate_spectrograms.py.
    See that file's make_spectrogram() docstring for why: melam type is
    defined by tempo/beat-cycle length, not timbre, so the model needs a
    rhythm-explicit feature, not just a mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.HOP_LENGTH)
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=config.HOP_LENGTH, win_length=config.N_MELS
    )

    n_frames = min(log_mel.shape[1], tempogram.shape[1])
    log_mel = log_mel[:, :n_frames]
    tempogram = tempogram[:, :n_frames]
    return np.stack([log_mel, tempogram], axis=0)  # (2, n_mels, frames)


def pad_or_crop(spec, fixed_frames=config.FIXED_SPEC_FRAMES):
    frames = spec.shape[-1]
    if frames == fixed_frames:
        return spec
    if frames > fixed_frames:
        start = (frames - fixed_frames) // 2
        return spec[..., start:start + fixed_frames]
    pad_total = fixed_frames - frames
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    out = np.empty(spec.shape[:-1] + (fixed_frames,), dtype=spec.dtype)
    for c in range(spec.shape[0]):
        out[c] = np.pad(spec[c], (pad_left, pad_right), mode="constant", constant_values=spec[c].min())
    return out


def normalize(spec):
    # Per-channel — mel-dB and tempogram values live on very different
    # scales, must match SpectrogramDataset's per-channel normalization.
    out = spec.copy()
    for c in range(out.shape[0]):
        out[c] = (out[c] - out[c].mean()) / (out[c].std() + 1e-6)
    return out


def clip_to_tensor(y, sr):
    spec = make_features(y, sr)
    spec = pad_or_crop(spec)
    spec = normalize(spec).astype(np.float32)
    return torch.from_numpy(spec).unsqueeze(0)  # (1, n_channels, n_mels, frames)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]

    model = build_model(n_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.4f})")
    return model, classes


def predict_tensor(model, classes, tensor, device):
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    return [(classes[i], float(probs[i])) for i in order]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def predict_single(filepath, model, classes, device):
    y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    tensor = clip_to_tensor(y, sr)
    results = predict_tensor(model, classes, tensor, device)

    print(f"\n{os.path.basename(filepath)} — predicted: {results[0][0]} ({results[0][1]*100:.1f}%)")
    print("All classes:")
    for cls, p in results:
        bar = "#" * int(p * 40)
        print(f"  {cls:15s} {p*100:5.1f}%  {bar}")
    return results


def predict_windowed(filepath, model, classes, device, window_sec=None, hop_sec=3.5):
    window_sec = window_sec or config.CLIP_TARGET_SEC
    y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(y) < win_len:
        print("Recording shorter than one window — falling back to single-clip mode.")
        return predict_single(filepath, model, classes, device)

    all_probs = []
    pos = 0
    win_idx = 0
    print(f"\n{os.path.basename(filepath)} — sliding {window_sec:.1f}s windows, {hop_sec:.1f}s hop")
    while pos + win_len <= len(y):
        chunk = y[pos:pos + win_len]
        tensor = clip_to_tensor(chunk, sr)
        results = predict_tensor(model, classes, tensor, device)
        probs = dict(results)
        all_probs.append(probs)
        t_start = pos / sr
        print(f"  [{t_start:6.1f}s] {results[0][0]:15s} {results[0][1]*100:5.1f}%")
        pos += hop_len
        win_idx += 1

    avg = {cls: float(np.mean([p[cls] for p in all_probs])) for cls in classes}
    ranked = sorted(avg.items(), key=lambda x: -x[1])
    print(f"\nAveraged over {win_idx} windows — overall verdict: "
          f"{ranked[0][0]} ({ranked[0][1]*100:.1f}%)")
    for cls, p in ranked:
        bar = "#" * int(p * 40)
        print(f"  {cls:15s} {p*100:5.1f}%  {bar}")
    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to a new audio file (wav/mp3/m4a etc.)")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--windowed", action="store_true",
                         help="Treat input as a longer recording and slide a window across it")
    parser.add_argument("--window-sec", type=float, default=None,
                         help="Window length in seconds for --windowed mode (default: CLIP_TARGET_SEC)")
    parser.add_argument("--hop-sec", type=float, default=3.5,
                         help="Hop between windows in seconds for --windowed mode")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        raise SystemExit(f"File not found: {args.audio_path}")
    if not os.path.exists(args.model):
        raise SystemExit(f"Missing {args.model} — run 06_train_cnn.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)

    if args.windowed:
        predict_windowed(args.audio_path, model, classes, device,
                          window_sec=args.window_sec, hop_sec=args.hop_sec)
    else:
        predict_single(args.audio_path, model, classes, device)


if __name__ == "__main__":
    main()

# Note: clip_to_tensor / predict_tensor / load_model are reused directly by
# 09_predict_live.py for microphone input — preprocessing must stay identical
# across both scripts, so don't duplicate this logic elsewhere.