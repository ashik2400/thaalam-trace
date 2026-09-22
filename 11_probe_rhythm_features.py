"""
11_probe_rhythm_features.py

Does any rhythm feature actually separate Panchari from Pandi on YOUR audio?

Run this BEFORE retraining. A training run takes hours and answers the
question "did accuracy move?", which confounds the feature, the labels, the
clip length and the split all at once. This answers one question directly:
given clips you believe are correctly labelled, is the 6-vs-7 distinction
present in the feature at all?

Scope is deliberately narrow — Panchari vs Pandi only. The other two classes
are separable by instrumentation (Panchavadyam uses thimila/maddalam/edakka;
Thayambaka uses no kombu or kuzhal), so a mel-spectrogram already handles
them. Panchari vs Pandi is the only genuinely rhythmic decision in the task,
and it is where every previous run has failed.

Grouping is by performance_id, so a feature cannot score well by recognising
a recording it has already seen.

Usage:
    python 11_probe_rhythm_features.py
    python 11_probe_rhythm_features.py --max-per-class 400 --clip-sec 30
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

TARGET_CLASSES = ("Panchari", "Pandi")


# --------------------------------------------------------------------------
# Candidate features
# --------------------------------------------------------------------------

def feat_tempogram_current(y, sr, hop):
    """What the pipeline uses today: tempogram, win_length=N_MELS (~3.0s lag).

    Included as the baseline to beat. If a new feature can't clear this,
    it isn't worth the pipeline change.
    """
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr,
                                   hop_length=hop, win_length=config.N_MELS)
    return tg.mean(axis=1)


def feat_tempogram_wide(y, sr, hop):
    """Same idea, but a lag window long enough to contain a slow cycle."""
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    win = min(int(20.0 * sr / hop), max(len(onset) - 1, 16))  # up to ~20 s lag
    tg = librosa.feature.tempogram(onset_envelope=onset, sr=sr,
                                   hop_length=hop, win_length=win)
    return tg.mean(axis=1)


def _cymbal_onset(y, sr, hop, fmin=2000, fmax=9000):
    """Onset strength restricted to the ilathalam band.

    The cymbals keep the thalam, and they sit well above the chenda's
    fundamental, so band-limiting should expose the timekeeper instead of
    the drum wash on top of it.
    """
    S = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return librosa.onset.onset_strength(
        S=librosa.power_to_db(S[band] ** 2), sr=sr, hop_length=hop)


def _first_prominent_peak(acf, lo, hi, rel=0.55):
    """First tall peak, not the global max.

    Autocorrelation of a periodic signal peaks at every multiple of the
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
    return int((tall[0] if tall.size else peaks[0])) + lo


def _beat_normalized_acf(onset_env, sr, hop, max_beats=9, bins_per_beat=12):
    """Autocorrelation with the lag axis measured in beats, not seconds.

    This is the point of the whole exercise. A tempogram reports the beat
    RATE, which both melams have. Rescaling the lag axis by the estimated
    beat period makes the feature tempo-invariant, so a 6-grouping lands at
    6 whether it came from kalam 1 or kalam 5 — which is what makes
    slow-kalam clips usable rather than impossible.
    """
    env = onset_env - onset_env.mean()
    if env.size < 8 or not np.any(env):
        return np.zeros(max_beats * bins_per_beat)
    acf = librosa.autocorrelate(env, max_size=env.size)
    if acf[0] <= 0:
        return np.zeros(max_beats * bins_per_beat)
    acf = acf / acf[0]

    lo = max(2, int(0.08 * sr / hop))              # fastest plausible beat
    hi = min(acf.size - 1, int(4.0 * sr / hop))    # slowest plausible beat
    if hi <= lo:
        return np.zeros(max_beats * bins_per_beat)
    beat_lag = _first_prominent_peak(acf, lo, hi)
    if not beat_lag:
        return np.zeros(max_beats * bins_per_beat)

    axis = np.linspace(0, max_beats, max_beats * bins_per_beat, endpoint=False)
    src = axis * beat_lag
    ok = src < acf.size - 1
    out = np.zeros_like(axis)
    out[ok] = np.interp(src[ok], np.arange(acf.size), acf)
    return out


def feat_beat_norm_full(y, sr, hop):
    """Beat-normalized ACF on the full-band onset envelope."""
    return _beat_normalized_acf(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop), sr, hop)


def feat_beat_norm_cymbal(y, sr, hop):
    """Beat-normalized ACF on the cymbal band only."""
    return _beat_normalized_acf(_cymbal_onset(y, sr, hop), sr, hop)


def feat_mel_baseline(y, sr, hop):
    """Mel-spectrogram summary — a pure timbre control.

    Panchari and Pandi use the same instruments, so this SHOULD score near
    chance. If it doesn't, the model is separating them by recording
    characteristics rather than by music, and the split still leaks.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=config.N_FFT,
                                         hop_length=hop, n_mels=config.N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])


FEATURES = {
    "mel summary (timbre control)": feat_mel_baseline,
    "tempogram, win=128 (current)": feat_tempogram_current,
    "tempogram, ~20s lag window": feat_tempogram_wide,
    "beat-normalized ACF, full band": feat_beat_norm_full,
    "beat-normalized ACF, cymbal band": feat_beat_norm_cymbal,
}


# --------------------------------------------------------------------------

def load_clips(max_per_class, clip_sec):
    seg = pd.read_csv(config.SEGMENTS_METADATA_CSV)
    seg = seg[seg["class"].isin(TARGET_CLASSES)].copy()

    corr_path = getattr(config, "LABEL_CORRECTIONS_CSV", None)
    if corr_path and os.path.exists(corr_path):
        corr = pd.read_csv(corr_path)
        drop = set(corr[corr.action == "drop"].video_id)
        relabel = dict(zip(corr[corr.action == "relabel"].video_id,
                           corr[corr.action == "relabel"].new_class))
        seg = seg[~seg.video_id.isin(drop)]
        seg["class"] = seg.apply(
            lambda r: relabel.get(r.video_id, r["class"]), axis=1)
        seg = seg[seg["class"].isin(TARGET_CLASSES)]
        print(f"Applied corrections: {len(drop)} dropped, {len(relabel)} relabelled.")
    else:
        print("WARNING: no label_corrections.csv — probing UNCORRECTED labels.")

    perf_path = getattr(config, "PERFORMANCE_GROUPS_CSV", None)
    if perf_path and os.path.exists(perf_path):
        pm = pd.read_csv(perf_path)
        lookup = dict(zip(pm.video_id, pm.performance_id))
        seg["performance_id"] = seg.video_id.map(lambda v: lookup.get(v, v))
    else:
        print("WARNING: no performance_groups.csv — grouping by video_id, "
              "which will not catch re-uploads.")
        seg["performance_id"] = seg.video_id

    # Sample evenly across performances rather than taking the first N clips,
    # so one long recording can't dominate the probe the way it dominates
    # training.
    #
    # NOTE: this deliberately does NOT use
    #   grp.groupby("performance_id").apply(lambda g: g.sample(...))
    # As of pandas 3.0, apply() on a groupby silently DROPS the column you
    # grouped on from the result (previously a DeprecationWarning, now the
    # default). That produced a DataFrame missing "performance_id" entirely
    # -> AttributeError further down. Looping over groups explicitly sidesteps
    # it and works the same way on older pandas too.
    out = []
    for cls, grp in seg.groupby("class"):
        n_perf = max(1, grp.performance_id.nunique())
        per_perf = max(1, max_per_class // n_perf)
        parts = []
        for _, g in grp.groupby("performance_id"):
            parts.append(g.sample(min(len(g), per_perf), random_state=config.RANDOM_SEED))
        picked = pd.concat(parts) if parts else grp.iloc[0:0]
        out.append(picked.head(max_per_class))
    result = pd.concat(out).reset_index(drop=True)
    assert "performance_id" in result.columns, (
        "performance_id missing after sampling — grouping/merge changed above")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--clip-sec", type=float, default=None,
                    help="Truncate each clip to this length. Defaults to "
                         "whatever the segments already are.")
    args = ap.parse_args()

    df = load_clips(args.max_per_class, args.clip_sec)
    print(f"\nProbing {len(df)} clips "
          f"({dict(df.groupby('class').size())}) across "
          f"{df.performance_id.nunique()} performances.\n")

    n_perf = df.performance_id.nunique()
    if n_perf < 4:
        raise SystemExit(f"Only {n_perf} performances — too few to cross-"
                         f"validate honestly. Fix the dataset first.")

    hop = config.HOP_LENGTH
    audio = []
    errors = {}          # exception message (first occurrence) -> count
    missing_files = 0
    print("Loading audio...")
    for _, r in df.iterrows():
        fp = r.filepath
        if not os.path.exists(fp):
            missing_files += 1
            audio.append((None, None))
            continue
        try:
            y, sr = librosa.load(fp, sr=config.SAMPLE_RATE, mono=True)
            if args.clip_sec:
                y = y[:int(args.clip_sec * sr)]
            audio.append((y, sr))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            errors[msg] = errors.get(msg, 0) + 1
            audio.append((None, None))

    keep = [i for i, (y, _) in enumerate(audio) if y is not None and y.size > hop * 8]
    df = df.iloc[keep].reset_index(drop=True)
    audio = [audio[i] for i in keep]
    print(f"{len(df)} clips loaded.")

    if missing_files or errors:
        print(f"\n{missing_files} file(s) not found on disk, "
              f"{sum(errors.values())} raised an error while loading:")
        for msg, count in sorted(errors.items(), key=lambda x: -x[1])[:6]:
            print(f"  [{count}x] {msg}")
        print()

    if len(df) == 0:
        raise SystemExit(
            "0 clips loaded — see the errors printed above for the actual "
            "cause. Common ones: filepath column in segments_metadata.csv "
            "points somewhere that no longer exists (re-run "
            "02_segment_audio.py if you moved the project folder), or a "
            "package version mismatch (try: pip show soundfile librosa).")

    labels = df["class"].values
    groups = df["performance_id"].values
    n_splits = min(5, len(np.unique(groups)))

    print(f"{'feature':<36} {'accuracy':>10}   {'vs chance':>10}")
    print("-" * 60)
    results = {}
    for name, fn in FEATURES.items():
        X = []
        for (y, sr) in audio:
            try:
                X.append(fn(y, sr, hop))
            except Exception:
                X.append(None)
        dim = max((len(v) for v in X if v is not None), default=0)
        X = np.array([v if v is not None and len(v) == dim else np.zeros(dim)
                      for v in X])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=0.1))
        scores = cross_val_score(clf, X, labels, groups=groups,
                                 cv=GroupKFold(n_splits=n_splits))
        acc = scores.mean()
        results[name] = acc
        print(f"{name:<36} {acc*100:9.1f}%   {(acc-0.5)*100:+9.1f}pp")

    print("\nHow to read this:")
    print("  Chance is 50%. Grouping is by performance, so a feature cannot")
    print("  score by recognising a recording it has already heard.")
    print()
    mel = results.get("mel summary (timbre control)", 0)
    best_rhythm = max((v for k, v in results.items() if "mel" not in k),
                      default=0)
    if mel > 0.70:
        print("  The timbre control scored high. Panchari and Pandi use the")
        print("  same instruments, so it should not be able to. Something is")
        print("  still leaking — most likely recording identity. Re-check the")
        print("  performance groups before trusting anything else here.")
    if best_rhythm < 0.60:
        print("  No rhythm feature cleared 60%. Do not retrain yet: either the")
        print("  clips are too short to contain a cycle, or the labels under")
        print("  test are still wrong. A training run would hide which.")
    elif best_rhythm > mel + 0.10:
        print("  A rhythm feature beat the timbre control by a clear margin.")
        print("  That is the signal worth building the input representation")
        print("  around.")


if __name__ == "__main__":
    main()