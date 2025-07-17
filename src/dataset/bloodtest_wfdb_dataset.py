import os
import time
from typing import Any, Optional

import pandas as pd
import torch
import wfdb
from sklearn.model_selection import KFold  # type: ignore
from torch.utils.data import Dataset


class BloodECGDataset(Dataset[tuple[torch.Tensor, torch.LongTensor]]):
    """
    A PyTorch Dataset that pairs ECG waveforms with corresponding blood test percentiles.
    Supports train/validation splits on a patient (subject) level using k-fold cross-validation.
    """

    def __init__(
        self,
        base_dir: str,
        ecg_csv_path: str,
        blood_csv_path: str,
        split: str,
        transform: Any = None,
        ignore_index: int = -100,
        load_n_ecgs: Optional[int] = None,
        k_folds: int = 10,
        val_fold: int = 0,
        seed: int = 42,
        catalogue_path: Optional[str] = None,
    ) -> None:
        # Basic parameters
        self.base_dir = base_dir
        self.transform = transform
        self.ignore_index = ignore_index

        self.catalogue_path = catalogue_path
        self.test_catalogue: Optional[pd.DataFrame] = None
        if self.catalogue_path is not None:
            self.test_catalogue = pd.read_csv(self.catalogue_path)

        # Load ECG metadata
        ecgs = pd.read_csv(ecg_csv_path, parse_dates=["charttime"])
        if load_n_ecgs is not None:
            ecgs = ecgs.iloc[:load_n_ecgs].copy()

        # Split subjects into folds
        subjects = ecgs["subject_id"].unique()
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(subjects))
        if val_fold < 0 or val_fold >= k_folds:
            raise ValueError(f"val_fold must be in [0, {k_folds-1}], got {val_fold}.")
        train_idx, val_idx = folds[val_fold]
        train_subjects = subjects[train_idx]
        val_subjects = subjects[val_idx]

        # Filter ECGs by split
        if split == "train":
            self.ecgs = ecgs[ecgs["subject_id"].isin(train_subjects)].reset_index(drop=True)
        elif split == "val":
            self.ecgs = ecgs[ecgs["subject_id"].isin(val_subjects)].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'.")

        # Load blood data
        blood = pd.read_csv(blood_csv_path, parse_dates=["charttime"])
        print(f"Loaded {len(self.ecgs)} ECG records for split='{split}' and {len(blood)} blood test records.")

        # Prepare label itemids
        self.itemids: list[int] = sorted(blood["itemid"].unique())

        # Index blood tests by subject for quick lookup
        print("Indexing blood tests by subject...")
        self.blood_by_subject: dict[int, pd.DataFrame] = {
            subj: df.sort_values("charttime") for subj, df in blood.groupby("subject_id")
        }

    def __len__(self) -> int:
        return len(self.ecgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        time.sleep(1e-2)  # NOTE
        row = self.ecgs.iloc[idx]
        subject = int(row["subject_id"])
        ecg_time = row["charttime"]
        wf_path = os.path.join(self.base_dir, row["waveform_path"])
        record = wfdb.rdrecord(wf_path)
        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        if self.transform:
            signal = self.transform(signal)

        target: torch.Tensor = torch.full((len(self.itemids),), self.ignore_index, dtype=torch.long)

        subj_blood = self.blood_by_subject.get(subject)
        if subj_blood is not None:
            lower = ecg_time - pd.Timedelta(hours=24)
            upper = ecg_time + pd.Timedelta(hours=24)
            window = subj_blood[(subj_blood["charttime"] >= lower) & (subj_blood["charttime"] <= upper)]
            if not window.empty:
                # for each lab itemid, pick the draw closest in time
                for i, item in enumerate(self.itemids):
                    df_item = window[window["itemid"] == item]
                    if not df_item.empty:
                        deltas = (df_item["charttime"] - ecg_time).abs()
                        closest_idx = deltas.idxmin()
                        pct = int(df_item.loc[closest_idx, "percentile"])
                        target[i] = pct

        return signal, target

    @property
    def num_tests(self) -> int:
        """Return the number of unique blood test itemids."""
        return len(self.itemids)

    @property
    def itemids(self) -> list[int]:
        """Return the list of blood test itemids."""
        return self._itemids

    @itemids.setter
    def itemids(self, itemids: list[int]) -> None:
        """Set the list of blood test itemids."""
        self._itemids = sorted(itemids)

    @itemids.deleter
    def itemids(self) -> None:
        """Delete the list of blood test itemids."""
        self._itemids = []

    def get_test_names(self) -> list[str]:
        """Return the names of the blood tests corresponding to itemids."""  # either use test catalogue or itemid
        if self.test_catalogue is not None:
            return [
                self.test_catalogue.loc[self.test_catalogue["itemid"] == itemid, "label"].values[0]
                for itemid in self.itemids
            ]
        else:
            return [str(itemid) for itemid in self.itemids]


if __name__ == "__main__":
    # Quick smoke test for train and validation splits
    base_dir = "/dataset/physionet.org/files/mimic-iv-ecg-diagnostic/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
    ecg_csv = "src/scripts/ecg_with_blood_24h.csv"
    blood_csv = "src/scripts/blood_with_ecg_24h.csv"

    # Training split
    train_ds = BloodECGDataset(
        base_dir=base_dir,
        ecg_csv_path=ecg_csv,
        blood_csv_path=blood_csv,
        k_folds=10,
        val_fold=0,
        seed=42,
        split="train",
    )
    print(f"Train set: {len(train_ds)} samples, {train_ds.num_tests} tests.")

    # Validation split
    val_ds = BloodECGDataset(
        base_dir=base_dir, ecg_csv_path=ecg_csv, blood_csv_path=blood_csv, k_folds=10, val_fold=0, seed=42, split="val"
    )
    print(f"Validation set: {len(val_ds)} samples, {val_ds.num_tests} tests.")
