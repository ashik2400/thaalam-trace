"""
03_generate_spectrograms.py

Converts every segmented clip into a mel-spectrogram:
  - Saves a .npy array (log-mel-spectrogram) for CNN training
  - Saves a .png image (for visual inspection / your presentation slides)
  - Writes labels.csv: the final file your CNN training script will read

Requires: librosa, numpy, matplotlib, pandas, tqdm

Usage:
    python 03_generate_spectrograms.py
"""

import csv
import os

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config


def ensure_dirs():
    os.makedirs(config.SPECTROGRAM_DIR, exist_ok=True)
    for cls in config.CLASSES:
        os.makedirs(os.path.join(config.SPECTROGRAM_DIR, cls, "npy"), exist_ok=True)
        os.makedirs(os.path.join(config.SPECTROGRAM_DIR, cls, "png"), exist_ok=True)


def _resample_rows(arr, n_rows):
    """Resample a (rows, frames) array down to exactly n_rows rows.

    This is what lets the rhythm channel look far back in time without being
    forced to output N_MELS rows. The old code set win_length=N_MELS purely
    so the shapes would stack, which capped the lag window at 3.0s — it
    conflated "how far the feature looks" with "how many rows it returns".
    Those are separate concerns and this separates them.
    """
    src_rows = arr.shape[0]
    if src_rows == n_rows:
        return arr
    src_idx = np.arange(src_rows)
    tgt_idx = np.linspace(0, src_rows - 1, n_rows)
    out = np.empty((n_rows, arr.shape[1]), dtype=arr.dtype)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(tgt_idx, src_idx, arr[:, j])
    return out


def _first_prominent_peak(acf, lo, hi, rel=0.55):
    """First tall peak in an autocorrelation, not the global maximum.

    Autocorrelation of a periodic signal peaks at EVERY multiple of the
    period, so argmax lands on an arbitrary multiple and the beat estimate
    comes out 2x or 3x too slow.
    """
    from scipy.signal import find_peaks
    seg = acf[lo:hi]
    if seg.size < 3:
        return None
    peaks, _ = find_peaks(seg)
    if peaks.size == 0:
        return None
    tall = peaks[seg[peaks] >= rel * seg[peaks].max()]
    return int(tall[0] if tall.size else peaks[0]) + lo


def _beat_normalized_acf(onset_env, sr, hop, n_rows, n_frames,
                         frame_sec=2.0):
    """Autocorrelation with the lag axis measured in BEATS, not seconds.

    A tempogram reports the dominant periodicity, which is the beat — and
    Panchari and Pandi both have beats. What separates them is how beats
    GROUP: 6 versus 7. Rescaling the lag axis by the estimated beat period
    puts a 6-grouping at 6 and a 7-grouping at 7 regardless of tempo, so a
    kalam-1 clip and a kalam-5 clip of the same melam produce the same
    profile. That tempo-invariance is the point: it is what makes
    slow-kalam clips usable instead of impossible.

    Computed over short overlapping windows so the output keeps a time axis
    and can stack with the mel channel.
    """
    max_beats = getattr(config, "BEATNORM_MAX_BEATS", 9)
    lo = max(2, int(getattr(config, "BEATNORM_MIN_BEAT_SEC", 0.08) * sr / hop))
    hi_cap = int(getattr(config, "BEATNORM_MAX_BEAT_SEC", 4.0) * sr / hop)

    win = max(int(frame_sec * sr / hop), 32)
    out = np.zeros((n_rows, n_frames), dtype=np.float32)
    axis = np.linspace(0, max_beats, n_rows, endpoint=False)

    for t in range(n_frames):
        centre = t
        a, b = max(0, centre - win // 2), min(len(onset_env), centre + win // 2)
        seg = onset_env[a:b]
        if seg.size < 16:
            continue
        seg = seg - seg.mean()
        acf = librosa.autocorrelate(seg, max_size=seg.size)
        if acf[0] <= 0:
            continue
        acf = acf / acf[0]
        hi = min(acf.size - 1, hi_cap)
        if hi <= lo:
            continue
        beat_lag = _first_prominent_peak(acf, lo, hi)
        if not beat_lag:
            continue
        src = axis * beat_lag
        ok = src < acf.size - 1
        col = np.zeros(n_rows, dtype=np.float32)
        col[ok] = np.interp(src[ok], np.arange(acf.size), acf)
        out[:, t] = col
    return out


def _cymbal_onset_env(y, sr, hop, fmin, fmax):
    """Onset strength restricted to the ilathalam (cymbal) frequency band.

    The cymbals keep the thalam — they're the timekeeper both drummers and
    listeners lock onto — and they sit well above the chenda's fundamental,
    so band-limiting exposes the timekeeper instead of the drum wash on top
    of it. Validated on real segmented audio with
    11_probe_rhythm_features.py: cymbal-band beat-normalized ACF scored
    66-70% on Panchari-vs-Pandi across 7/15/30s clips, against ~52-55% for
    a full-band tempogram and ~54% for mel timbre alone. That's the reason
    this is the default rhythm feature rather than a tempogram.
    """
    S = np.abs(librosa.stft(y=y, n_fft=config.N_FFT, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return librosa.onset.onset_strength(
        S=librosa.power_to_db(S[band] ** 2), sr=sr, hop_length=hop)


def make_rhythm_channel(y, sr, n_rows, n_frames):
    """Channel 2 of the network input. See config.RHYTHM_FEATURE."""
    hop = config.HOP_LENGTH
    mode = getattr(config, "RHYTHM_FEATURE", "beat_norm_acf")

    if mode == "beat_norm_acf":
        fmin = getattr(config, "BEATNORM_CYMBAL_FMIN_HZ", 2000)
        fmax = getattr(config, "BEATNORM_CYMBAL_FMAX_HZ", 9000)
        onset_env = _cymbal_onset_env(y, sr, hop, fmin, fmax)
        return _beat_normalized_acf(onset_env, sr, hop, n_rows, n_frames)

    # Other modes use the full-band envelope — kept for comparison, not
    # recommended (see probe results above).
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    if mode == "tempogram_legacy":
        # The original setting. win_length=N_MELS caps the lag window at
        # N_MELS * hop / sr = 3.0s, so any cycle slower than 3 seconds is
        # outside the window entirely — which is most of kalams 1 and 2.
        # Kept only so probe runs have a baseline to beat.
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                       hop_length=hop, win_length=config.N_MELS)
        return _resample_rows(tg, n_rows)

    # "tempogram_wide": set the lag window by musical requirement,
    # then squeeze the lag axis down to n_rows afterwards.
    max_lag_sec = getattr(config, "TEMPOGRAM_MAX_LAG_SEC", 20.0)
    win = int(max_lag_sec * sr / hop)
    win = max(16, min(win, max(len(onset_env) - 1, 16)))
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr,
                                   hop_length=hop, win_length=win)
    return _resample_rows(tg, n_rows)


def make_spectrogram(filepath):
    """Returns (features, log_mel, sr) where features is (2, N_MELS, frames).

    Channel 1 is a log-mel spectrogram — timbre. That carries most of the
    task: Panchavadyam uses thimila/maddalam/edakka, and Thayambaka uses no
    kombu and no kuzhal, so three of the four classes are separable by
    instrumentation alone.

    Channel 2 is a rhythm feature, selected by config.RHYTHM_FEATURE. Its
    only real job is Panchari vs Pandi — same instruments, 6-beat thalam
    versus 7-beat. That is the one genuinely rhythmic decision in the task.

    What changed from the previous version: the rhythm channel no longer
    has its lag window dictated by N_MELS. See make_rhythm_channel.
    """
    y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    rhythm = make_rhythm_channel(y, sr, config.N_MELS, log_mel.shape[1])

    # Frame counts can differ by one between the two features depending on
    # librosa's internal padding — trim both to the shorter length.
    n_frames = min(log_mel.shape[1], rhythm.shape[1])
    log_mel = log_mel[:, :n_frames]
    rhythm = rhythm[:, :n_frames].astype(np.float32)

    # float32 explicitly: at 30s clips these arrays are 4.3x larger than
    # before, and letting a float64 tempogram through would double the disk
    # footprint of the whole dataset for no benefit.
    features = np.stack([log_mel, rhythm], axis=0).astype(np.float32)
    return features, log_mel, sr


def save_png(log_mel, sr, out_path, title):
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(log_mel, sr=sr, hop_length=config.HOP_LENGTH,
                              x_axis="time", y_axis="mel", cmap="magma")
    plt.title(title, fontsize=9)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def main():
    # Guard against the two config values silently drifting apart. Before
    # this, FIXED_SPEC_FRAMES was the literal 300 with a comment claiming it
    # was derived from CLIP_TARGET_SEC. Raising clip length without editing
    # it would have cropped every clip back to 7 seconds — no error, no
    # warning, and training curves that still looked plausible.
    expected = int(config.CLIP_TARGET_SEC * config.SAMPLE_RATE / config.HOP_LENGTH) + 1
    if abs(config.FIXED_SPEC_FRAMES - expected) > 2:
        raise SystemExit(
            f"config.FIXED_SPEC_FRAMES={config.FIXED_SPEC_FRAMES} does not match "
            f"CLIP_TARGET_SEC={config.CLIP_TARGET_SEC}s, which needs ~{expected} "
            f"frames. Clips would be silently cropped to "
            f"{config.FIXED_SPEC_FRAMES * config.HOP_LENGTH / config.SAMPLE_RATE:.1f}s.")

    print(f"Clip length {config.CLIP_TARGET_SEC}s -> {config.FIXED_SPEC_FRAMES} frames")
    print(f"Rhythm channel: {getattr(config, 'RHYTHM_FEATURE', 'tempogram_wide')}")

    ensure_dirs()

    if not os.path.exists(config.SEGMENTS_METADATA_CSV):
        raise SystemExit(f"Missing {config.SEGMENTS_METADATA_CSV} — run 02_segment_audio.py first.")

    with open(config.SEGMENTS_METADATA_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Overwrite, not append: this script has no "already processed" skip
    # logic (unlike 01_download_videos.py), so append mode would duplicate
    # every clip_id row on a rerun. That matters now more than before,
    # since regenerating with the new 2-channel [mel, tempogram] features
    # means you'll rerun this on data you've already run it on once.
    with open(config.LABELS_CSV, "w", newline="", encoding="utf-8") as out_f:
        fieldnames = ["clip_id", "class", "npy_path", "png_path", "source_wav"]
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Generating spectrograms for {len(rows)} clips...")
        for row in tqdm(rows):
            cls, clip_id, filepath = row["class"], row["clip_id"], row["filepath"]
            if not os.path.exists(filepath):
                continue

            npy_path = os.path.join(config.SPECTROGRAM_DIR, cls, "npy", f"{clip_id}.npy")
            png_path = os.path.join(config.SPECTROGRAM_DIR, cls, "png", f"{clip_id}.png")

            try:
                features, log_mel, sr = make_spectrogram(filepath)
            except Exception as e:
                print(f"  ! Failed on {filepath}: {e}")
                continue

            np.save(npy_path, features)  # (2, n_mels, frames): [mel, tempogram]
            save_png(log_mel, sr, png_path, title=f"{cls} — {clip_id}")  # mel only, for visual inspection

            writer.writerow({
                "clip_id": clip_id,
                "class": cls,
                "npy_path": npy_path,
                "png_path": png_path,
                "source_wav": filepath,
            })

    print(f"\nDone. Final training-ready labels file: {config.LABELS_CSV}")

    # Quick class balance summary — useful to paste straight into your presentation
    from collections import Counter
    with open(config.LABELS_CSV, newline="", encoding="utf-8") as f:
        counts = Counter(r["class"] for r in csv.DictReader(f))
    print("\nClass balance:")
    for cls in config.CLASSES:
        print(f"  {cls:15s} {counts.get(cls, 0)}")


if __name__ == "__main__":
    main()