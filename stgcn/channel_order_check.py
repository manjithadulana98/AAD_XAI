"""Mandatory pre-step (before ST-GCN Phase 2) -- channel ordering safety check.

Compares, index by index, the channel name ordering the external/AADNet DTU
pipeline actually uses to build its model-facing EEG tensor against the
channel name ordering used to build the Phase 1 adjacency matrices from
``config/dtu_channel_montage.csv``.

Where the AADNet ordering comes from (verified against
``external/AADNet/aadnet/dataset.py``, ``DTUDataset``): the raw DTU ``.mat``
files are read in their own native per-file channel order
(``__cls__.channels``), then the final model-facing ``eeg`` array is built by
indexing into that raw array with ``selected_idx = [channels.index(ch) for ch
in selected_chs]`` (dataset.py line ~428, applied at line ~436), where
``selected_chs`` is exactly the ``dataset.channels`` list in the DTU YAML
configs. So the YAML ``channels`` list -- not the raw .mat column order, and
not ``ordinary_channels`` (only used for average-referencing) -- is the
authoritative order of the tensor AADNet's model actually consumes. That list
is identical, character for character, across every DTU config variant in
``external/AADNet/config/`` (SI, SS, kaggle, pilot, NSR), so it is stable
regardless of which one Phase 2 ultimately runs with.

Run directly:
    python stgcn/channel_order_check.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
AADNET_YAML_PATH = REPO_ROOT / "external" / "AADNet" / "config" / "config_AADNet_SI_DTU_kaggle.yml"
MONTAGE_PATH = REPO_ROOT / "config" / "dtu_channel_montage.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "channel_order_check.csv"


def load_aadnet_channel_order(yaml_path: Path = AADNET_YAML_PATH) -> list[str]:
    """Extract the ``dataset.channels`` list from an AADNet DTU YAML config.

    Parsed with a targeted regex rather than a full YAML load, since the
    upstream config files use inline Python-style comments after the list
    (``# Fuglsang 64``) that trip up a strict YAML parser; the list itself is
    a simple flat sequence of quoted channel names.
    """
    text = yaml_path.read_text()
    match = re.search(r"channels:\s*\[([^\]]*)\]", text)
    if not match:
        raise ValueError(f"Could not find a 'channels: [...]' list in {yaml_path}")
    raw_items = match.group(1).split(",")
    return [item.strip().strip("'\"") for item in raw_items]


def load_montage_channel_order(montage_path: Path = MONTAGE_PATH) -> list[str]:
    montage = pd.read_csv(montage_path).sort_values("channel_index").reset_index(drop=True)
    return montage["electrode_name"].tolist()


def build_channel_order_check(
    aadnet_channels: list[str], montage_channels: list[str]
) -> pd.DataFrame:
    n = max(len(aadnet_channels), len(montage_channels))
    rows = []
    for i in range(n):
        a = aadnet_channels[i] if i < len(aadnet_channels) else None
        m = montage_channels[i] if i < len(montage_channels) else None
        rows.append({"index": i, "aadnet_channel_name": a, "montage_channel_name": m, "match": a == m})
    return pd.DataFrame(rows)


def main() -> None:
    aadnet_channels = load_aadnet_channel_order()
    montage_channels = load_montage_channel_order()

    print(f"AADNet channel count: {len(aadnet_channels)}")
    print(f"Montage channel count: {len(montage_channels)}")

    check_df = build_channel_order_check(aadnet_channels, montage_channels)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    check_df.to_csv(OUTPUT_PATH, index=False)

    n_match = int(check_df["match"].sum())
    n_total = len(check_df)
    print(f"\nExact index-for-index matches: {n_match} / {n_total}")

    aadnet_set = set(aadnet_channels)
    montage_set = set(montage_channels)
    only_in_aadnet = sorted(aadnet_set - montage_set)
    only_in_montage = sorted(montage_set - aadnet_set)
    print(f"\nChannels only in AADNet's list (absent from montage): {only_in_aadnet}")
    print(f"Channels only in montage (absent from AADNet's list): {only_in_montage}")
    print(f"\nOutput written to: {OUTPUT_PATH}")

    if n_match < n_total:
        print(
            "\nSTOP CONDITION FAILED: AADNet and montage channel orderings do "
            "not match index-for-index. Do not proceed to model code -- see "
            "channel_order_check.csv for the exact per-index mismatches."
        )


if __name__ == "__main__":
    main()
