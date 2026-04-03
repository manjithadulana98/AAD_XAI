from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import scipy.io
from .base import BaseDataset, Trial


class DTUDataset(BaseDataset):
    """Loader for the DTU EEG+audio AAD dataset (Zenodo / COCOHA-style).

    Expected layout (adjust to your release)::

        root/
          EEG/
            subj01/
              trial01.mat   # keys vary; common: 'data', 'fs', 'attended'
              trial02.mat
            subj02/
              ...
          stimuli/
            ...

    The exact schema depends on the DTU release version.  The parsing
    below provides a best-effort template.  Adjust ``_parse_subject_dir``
    to match your files.

    Parameters
    ----------
    root : str | Path
        Path to the extracted DTU dataset folder.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def trials(self) -> Iterator[Trial]:
        eeg_dir = self.root / "EEG"
        if not eeg_dir.exists():
            raise FileNotFoundError(
                f"Expected EEG folder at {eeg_dir}. "
                "Update loader to match your extracted DTU dataset."
            )

        for subj_dir in sorted(eeg_dir.iterdir()):
            if subj_dir.is_dir():
                yield from self._parse_subject_dir(subj_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_subject_dir(subj_dir: Path) -> Iterator[Trial]:
        """Parse all trial MAT files for a single DTU subject.

        TODO: Adapt field names / file patterns to match your DTU release.
        """
        subject_id = subj_dir.name

        for idx, mat_path in enumerate(sorted(subj_dir.glob("*.mat"))):
            try:
                mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
            except NotImplementedError:
                raise NotImplementedError(
                    f"{mat_path.name} is a v7.3 MAT file — use h5py."
                )

            # TODO: replace keys below with those in your MAT files
            if "data" not in mat:
                raise KeyError(
                    f"Key 'data' not found in {mat_path.name}. "
                    f"Available: {list(mat.keys())}. "
                    "Update _parse_subject_dir for your schema."
                )

            eeg_data = np.asarray(mat["data"], dtype=np.float32)
            sfreq = float(mat.get("fs", mat.get("fsample", 512)))
            label = int(mat.get("attended", 0))

            if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
                eeg_data = eeg_data.T

            trial_id = f"{subject_id}_T{idx:03d}"
            group_id = f"{subject_id}_{mat_path.stem}"

            yield Trial(
                eeg=eeg_data,
                sfreq=sfreq,
                label=label,
                subject_id=subject_id,
                trial_id=trial_id,
                group_id=group_id,
            )

