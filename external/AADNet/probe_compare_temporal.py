import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def temporal_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "flat":
        return x.reshape(x.shape[0], -1)

    if mode == "mean":
        pooled = x.mean(axis=-1)
        return pooled.reshape(pooled.shape[0], -1)

    if mode == "max":
        pooled = x.max(axis=-1)
        return pooled.reshape(pooled.shape[0], -1)

    if mode == "mean_std":
        mean = x.mean(axis=-1).reshape(x.shape[0], -1)
        std = x.std(axis=-1).reshape(x.shape[0], -1)
        return np.concatenate([mean, std], axis=1)

    raise ValueError(f"Unsupported mode: {mode}")


def build_representations(data: np.lib.npyio.NpzFile):
    reps = {
        "pre_fc": (
            data["train_pre_fc"],
            data["valid_pre_fc"],
            data["test_pre_fc"],
        ),
        "logits": (
            data["train_logits"],
            data["valid_logits"],
            data["test_logits"],
        ),
        "eeg_mean": tuple(temporal_features(data[f"{split}_eeg_branch"], "mean") for split in ["train", "valid", "test"]),
        "eeg_max": tuple(temporal_features(data[f"{split}_eeg_branch"], "max") for split in ["train", "valid", "test"]),
        "eeg_mean_std": tuple(temporal_features(data[f"{split}_eeg_branch"], "mean_std") for split in ["train", "valid", "test"]),
        "aud_mean": tuple(temporal_features(data[f"{split}_aud_branch"], "mean") for split in ["train", "valid", "test"]),
        "aud_max": tuple(temporal_features(data[f"{split}_aud_branch"], "max") for split in ["train", "valid", "test"]),
        "aud_mean_std": tuple(temporal_features(data[f"{split}_aud_branch"], "mean_std") for split in ["train", "valid", "test"]),
    }
    return reps


def evaluate_representation(X_train, X_valid, X_test, y_train, y_valid, y_test):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000)
    )
    probe.fit(X_train, y_train)

    valid_pred = probe.predict(X_valid)
    test_pred = probe.predict(X_test)

    return accuracy_score(y_valid, valid_pred), accuracy_score(y_test, test_pred), X_train.shape


def run_one(npz_path: Path):
    print(f"\n=== {npz_path} ===")
    data = np.load(npz_path)
    reps = build_representations(data)

    y_train = data["train_labels"]
    y_valid = data["valid_labels"]
    y_test = data["test_labels"]

    results = []
    for name, (X_train, X_valid, X_test) in reps.items():
        valid_acc, test_acc, shape = evaluate_representation(X_train, X_valid, X_test, y_train, y_valid, y_test)
        results.append((name, valid_acc, test_acc, shape))

    results.sort(key=lambda x: x[2], reverse=True)
    for name, valid_acc, test_acc, shape in results:
        print(f"{name:12s} | valid={valid_acc:.4f} | test={test_acc:.4f} | shape={shape}")


def main():
    parser = argparse.ArgumentParser(description="Compare probe performance across pre_fc/logits and temporally pooled branch features.")
    parser.add_argument("npz_files", nargs="+", help="One or more .npz probe feature files")
    args = parser.parse_args()

    for path in args.npz_files:
        run_one(Path(path))


if __name__ == "__main__":
    main()
