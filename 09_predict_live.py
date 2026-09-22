"""
09_predict_live.py

Records audio from the laptop mic and classifies it live — no file saved
to disk unless you pass --save. Reuses the exact same preprocessing as
08_predict.py (clip_to_tensor / predict_tensor / load_model), so results
are directly comparable to file-based predictions.

Requires: sounddevice (new dependency — pip install sounddevice)
  Linux also needs the system PortAudio library:
      sudo apt-get install libportaudio2
  Mac/Windows: sounddevice ships PortAudio, no extra install needed.

Two modes:
  1. Single-shot (default): records one window_sec clip, classifies it, exits.
       python 09_predict_live.py
       python 09_predict_live.py --seconds 7

  2. Continuous: records + classifies in a loop until Ctrl+C — useful for
     holding the laptop near a live/recorded performance and watching
     predictions update as the melam plays.
       python 09_predict_live.py --continuous
       python 09_predict_live.py --continuous --seconds 7 --interval 5
"""

import argparse
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

import config
from importlib import import_module

predict_module = import_module("08_predict")
clip_to_tensor = predict_module.clip_to_tensor
predict_tensor = predict_module.predict_tensor
load_model = predict_module.load_model


def record_clip(seconds, sr):
    print(f"Recording {seconds:.1f}s from mic...", end=" ", flush=True)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    print(f"done. (peak={peak:.4f}, rms={rms:.5f})")
    if peak < 0.01:
        print("  ! WARNING: signal is near-silent (peak < 0.01). The mic likely isn't "
              "picking up real audio — check input device, mic volume/permissions, or "
              "move the speaker closer. Prediction below is not meaningful until this is fixed.")
    return audio


def print_prediction(results, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}predicted: {results[0][0]} ({results[0][1]*100:.1f}%)")
    for cls, p in results:
        bar = "#" * int(p * 40)
        print(f"    {cls:15s} {p*100:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.BEST_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--seconds", type=float, default=config.CLIP_TARGET_SEC,
                         help="Recording length per window (default matches training clip length)")
    parser.add_argument("--continuous", action="store_true",
                         help="Keep recording + classifying in a loop until Ctrl+C")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Pause between recordings in --continuous mode (seconds)")
    parser.add_argument("--save", metavar="PATH_PREFIX", default=None,
                         help="If set, save each recorded clip as PATH_PREFIX_<n>.wav")
    parser.add_argument("--device", type=int, default=None,
                         help="Input device index (run 'python -m sounddevice' to list devices)")
    args = parser.parse_args()

    if args.device is not None:
        sd.default.device = (args.device, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)
    sr = config.SAMPLE_RATE

    print(f"\nListening via mic (sample rate {sr} Hz). Ctrl+C to stop.\n")

    n = 0
    try:
        while True:
            audio = record_clip(args.seconds, sr)

            if args.save:
                out_path = f"{args.save}_{n:03d}.wav"
                sf.write(out_path, audio, sr)
                print(f"  saved -> {out_path}")

            tensor = clip_to_tensor(audio, sr)
            results = predict_tensor(model, classes, tensor, device)
            print_prediction(results, label=f"take {n}")
            print()
            n += 1

            if not args.continuous:
                break
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()