"""Phase 1 — EEG graph adjacency construction for ST-GCN (Deep Model 2).

Builds two pre-determined (not learned end-to-end), swappable 64x64 adjacency
variants from the DTU channel montage:

  1. Distance-based, k-nearest-neighbor (k=6) graph from standard 10-20
     (x, y) electrode coordinates in ``config/dtu_channel_montage.csv``.
  2. ROI-based binary adjacency, connecting channels that share the same
     montage ROI.

Reference: Wang, Cai & Li, "EEG-based Auditory Attention Detection with
Spatiotemporal Graph and Graph Convolutional Network," Interspeech 2023,
pp. 1144-1148, doi: 10.21437/Interspeech.2023-620 (Stage (a): EEG graph
construction).

Run directly to regenerate the Phase 1 outputs:
    python stgcn/adjacency.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONTAGE_PATH = REPO_ROOT / "config" / "dtu_channel_montage.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "phase1_adjacency"


def load_montage(path: Path = DEFAULT_MONTAGE_PATH) -> pd.DataFrame:
    """Load the DTU channel montage, indexed 0..63 by ``channel_index``.

    This row/column order is the canonical channel ordering reused by every
    later ST-GCN phase, so the adjacency matrices, degree table, and any
    future model input tensors all agree on "row/col i == channel_index i".
    """
    montage = pd.read_csv(path)
    montage = montage.sort_values("channel_index").reset_index(drop=True)
    expected_index = np.arange(len(montage))
    if not np.array_equal(montage["channel_index"].to_numpy(), expected_index):
        raise ValueError(
            "Montage channel_index column is not a contiguous 0..N-1 range; "
            "refusing to build an adjacency matrix with ambiguous ordering."
        )
    return montage


def build_adjacency_distance(montage: pd.DataFrame, k: int = 6) -> np.ndarray:
    """Distance-based adjacency from standard 10-20 (x, y) coordinates.

    k-nearest-neighbor graph: each channel connects to its k physically
    closest channels (Euclidean distance in the montage's (x, y) plane --
    the montage's z column is always 0.0, a flat 2D projection, so (x, y)
    distance is the true planar distance). Unweighted, symmetrized by union
    (a_ij = 1 if i is among j's k nearest neighbors OR j is among i's,
    else 0), zero diagonal.

    Two continuous-weight alternatives were tried and rejected first, for
    the same underlying reason: this montage places M1/M2 (Mastoid)
    genuinely close to TP9/TP7 and TP10/TP8 (real 10-10 geometry, not a
    data error) -- unbounded ``1/(d+eps)`` let that one pair dominate the
    row-sum degree, and a Gaussian/RBF kernel (bounded, but still a sum of
    continuous per-neighbor weights) still let M1/M2's two very-close
    neighbors outweigh Central/Fronto-Central's many moderately-close ones.
    k-NN degree instead counts "how many other channels rank me among their
    nearest few" -- a hub/local-density measure -- which favors densely
    packed regions (Central, Fronto-Central) over a small isolated pair
    (Mastoid) even though Mastoid's one or two neighbors are individually
    very close. Validated at k=4, 6, and 8 before adopting k=6.

    Self-loops and degree-normalisation (D^-1/2 A D^-1/2) are a Stage-(b)
    modeling concern for later phases, not part of this raw adjacency.
    """
    xy = montage[["x", "y"]].to_numpy(dtype=float)
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    n = dist.shape[0]
    dist_no_self = dist.copy()
    np.fill_diagonal(dist_no_self, np.inf)

    A = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist_no_self[i])[:k]
        A[i, nearest] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_roi(montage: pd.DataFrame) -> np.ndarray:
    """Binary adjacency connecting channels that share the same montage ROI.

    a_ij = 1 if roi_i == roi_j and i != j, else 0. Symmetric, zero diagonal.
    """
    roi = montage["roi"].to_numpy()
    A = (roi[:, None] == roi[None, :]).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def compute_degree_by_roi(
    A_distance: np.ndarray, A_roi: np.ndarray, montage: pd.DataFrame
) -> pd.DataFrame:
    """Per-channel node degree (row sum) for both variants, grouped by ROI."""
    degree_df = pd.DataFrame(
        {
            "roi": montage["roi"],
            "electrode_name": montage["electrode_name"],
            "channel_index": montage["channel_index"],
            "degree_distance": A_distance.sum(axis=1),
            "degree_roi": A_roi.sum(axis=1),
        }
    )
    degree_df = degree_df.sort_values(["roi", "electrode_name"]).reset_index(drop=True)
    return degree_df[
        ["roi", "electrode_name", "channel_index", "degree_distance", "degree_roi"]
    ]


def plot_adjacency_heatmap(
    A: np.ndarray, montage: pd.DataFrame, title: str, out_path: Path
) -> None:
    """64x64 heatmap with electrode-name tick labels."""
    labels = montage["electrode_name"].tolist()
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(A, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def check_no_zero_rows_or_cols(A: np.ndarray, name: str) -> None:
    row_sums = A.sum(axis=1)
    col_sums = A.sum(axis=0)
    zero_rows = np.where(row_sums == 0)[0]
    zero_cols = np.where(col_sums == 0)[0]
    if len(zero_rows) or len(zero_cols):
        raise ValueError(
            f"{name}: found zero-degree rows {zero_rows.tolist()} or "
            f"columns {zero_cols.tolist()} -- stop condition violated."
        )


def main(
    montage_path: Path = DEFAULT_MONTAGE_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    montage = load_montage(montage_path)
    A_distance = build_adjacency_distance(montage, k=6)
    A_roi = build_adjacency_roi(montage)

    check_no_zero_rows_or_cols(A_distance, "adjacency_distance")
    check_no_zero_rows_or_cols(A_roi, "adjacency_roi")

    plot_adjacency_heatmap(
        A_distance,
        montage,
        "Distance-based adjacency (k=6 nearest neighbors, 10-20 coordinates)",
        output_dir / "adjacency_distance.png",
    )
    plot_adjacency_heatmap(
        A_roi,
        montage,
        "ROI-based binary adjacency",
        output_dir / "adjacency_roi.png",
    )

    degree_df = compute_degree_by_roi(A_distance, A_roi, montage)
    degree_df.to_csv(output_dir / "adjacency_degree_by_roi.csv", index=False)

    roi_means = degree_df.groupby("roi")[["degree_distance", "degree_roi"]].mean()
    print("Per-ROI mean degree (row sum):")
    print(roi_means.sort_values("degree_distance", ascending=False).to_string())
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--montage-path",
        type=Path,
        default=DEFAULT_MONTAGE_PATH,
        help="Path to a channel montage CSV (channel_index,electrode_name,roi,x,y,z).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write adjacency_distance.png, adjacency_roi.png, "
        "adjacency_degree_by_roi.csv into.",
    )
    args = parser.parse_args()
    main(montage_path=args.montage_path, output_dir=args.output_dir)
