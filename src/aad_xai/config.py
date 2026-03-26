from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Sequence, Optional, Literal
from pathlib import Path
import json


@dataclass(frozen=True)
class PreprocessConfig:
    sfreq_out: int = 64
    bandpass_hz: tuple[float, float] = (1.0, 8.0)  # envelope-tracking band
    reref: Optional[Literal["average"]] = "average"


@dataclass(frozen=True)
class WindowConfig:
    lengths_s: Sequence[float] = (1.0, 5.0, 10.0, 60.0)
    overlap_s: float = 0.0  # overlap allowed *within split only*; never across split boundaries


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    model: Literal["trf", "cnn", "stgcn", "aadnet_ext"] = "cnn"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 5        # early-stopping patience (epochs)
    num_seeds: int = 3
    device: str = "cuda"


@dataclass(frozen=True)
class RunConfig:
    """Top-level configuration that groups all sub-configs.

    Instantiate directly or load from a JSON file with ``RunConfig.from_json``.
    """
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: Literal["synthetic", "kul", "dtu"] = "synthetic"
    dataset_root: str = "data_raw"
    output_dir: str = "runs"

    # ------------------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        """Serialize full config to JSON for reproducibility logging."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        """Deserialize config from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            preprocess=PreprocessConfig(**raw.get("preprocess", {})),
            window=WindowConfig(**{k: tuple(v) if k == "lengths_s" else v for k, v in raw.get("window", {}).items()}),
            split=SplitConfig(**raw.get("split", {})),
            train=TrainConfig(**raw.get("train", {})),
            dataset=raw.get("dataset", "synthetic"),
            dataset_root=raw.get("dataset_root", "data_raw"),
            output_dir=raw.get("output_dir", "runs"),
        )

