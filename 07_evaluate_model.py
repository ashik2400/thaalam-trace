"""
07_evaluate_model.py

Loads the best checkpoint from 06_train_cnn.py and evaluates it on the
held-out TEST split (never seen during training or validation).

Prints per-class precision/recall/F1 and saves a confusion matrix image —
the confusion matrix is especially useful for your presentation, since it
shows exactly which melams get confused with each other (e.g. Pandi vs
Panchari, which are rhythmically related).

Requires: torch, numpy, pandas, scikit-learn, matplotlib, tqdm

Usage:
    python 07_evaluate_model.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from importlib import import_module

train_module = import_module("06_train_cnn")
SpectrogramDataset = train_module.SpectrogramDataset
build_model = train_module.build_model


def main():
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise SystemExit(f"Missing {config.BEST_MODEL_PATH} — run 06_train_cnn.py first.")
    if not os.path.exists(config.SPLITS_CSV):
        raise SystemExit(f"Missing {config.SPLITS_CSV} — run 05_split_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    label_to_idx = checkpoint["label_to_idx"]
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]

    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})")

    model = build_model(n_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(config.SPLITS_CSV)
    test_df = df[df["split"] == "test"]
    print(f"Test set size: {len(test_df)}")

    test_ds = SpectrogramDataset(test_df, label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for specs, labels in tqdm(test_loader, desc="Evaluating"):
            specs = specs.to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    cm = confusion_matrix(all_labels, all_preds)

    # Plot and save confusion matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PNG, dpi=120)
    print(f"\nConfusion matrix saved to {config.CONFUSION_MATRIX_PNG}")

    overall_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nOverall test accuracy: {overall_acc:.4f}")


if __name__ == "__main__":
    main()