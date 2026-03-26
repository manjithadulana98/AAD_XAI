from __future__ import annotations
import json
import datetime
from pathlib import Path
from typing import Any


def save_json(path: str | Path, obj: Any) -> None:
    """Write a JSON-serializable object to *path* (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def get_run_dir(
    base: str | Path,
    model_name: str,
    seed: int,
    window_s: float | None = None,
) -> Path:
    """Return a deterministic run directory like ``runs/cnn/seed42/win1.0/``.

    Creates the directory on disk so that callers can immediately write into it.
    """
    parts = [str(model_name), f"seed{seed}"]
    if window_s is not None:
        parts.append(f"win{window_s}")
    run_dir = Path(base).joinpath(*parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_run_metadata(run_dir: Path, **kwargs: Any) -> None:
    """Persist arbitrary run metadata (config, split info, …) to *run_dir/meta.json*."""
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        **kwargs,
    }
    save_json(run_dir / "meta.json", meta)

