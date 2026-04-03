from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np

@dataclass(frozen=True)
class Trial:
    eeg: np.ndarray          # (n_channels, n_times)
    sfreq: float
    label: int              # Task-1: 0/1 for attended candidate stream A vs B
    subject_id: str
    trial_id: str
    group_id: str           # story/session id for leakage control
    audio_a: Optional[np.ndarray] = None
    audio_b: Optional[np.ndarray] = None
    audio_sr: Optional[int] = None

class BaseDataset:
    """Return Trial objects. Implement this for each dataset."""
    def trials(self) -> Iterator[Trial]:
        raise NotImplementedError
