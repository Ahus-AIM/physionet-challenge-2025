import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from helper_code import age_string, label_string, load_text, sex_string
except ImportError:
    age_string = "# Age:"
    sex_string = "# Sex:"
    label_string = "# Chagas label:"

    def load_text(filename: str) -> str:
        with open(filename, "r") as f:
            string = f.read()
        return string


class WFDBDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
        max_num_samples: Optional[int] = None,
        k_folds: int = 1,
        val_fold: int = 0,
        use_val: bool = False,
        shuffle_seed: int = 42,
        return_demographics: bool = False,
        reweighting_csv: Optional[str] = None,
    ) -> None:
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.max_num_samples = max_num_samples
        self.reweighting_df: Optional[pd.DataFrame] = None
        if reweighting_csv is not None:
            self.reweighting_df = pd.read_csv(reweighting_csv, dtype=str)
            print(f"Reweighting DataFrame loaded with {len(self.reweighting_df)} entries.")
        else:
            print("No reweighting DataFrame provided, using default labels.")

        self.k_folds = k_folds
        self.val_fold = val_fold
        self.use_val = use_val
        self.shuffle_seed = shuffle_seed
        self.return_demographics = return_demographics

        self.records: List[str] = []

        self._gather_records(self.root_dir)

        # K-fold split logic
        indices = np.arange(len(self.records))
        if self.k_folds > 1:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(indices)
            fold_sizes = np.full(self.k_folds, len(indices) // self.k_folds, dtype=int)
            fold_sizes[: len(indices) % self.k_folds] += 1
            folds = []
            current = 0
            for size in fold_sizes:
                folds.append(indices[current : current + size])
                current += size
            if self.use_val:
                chosen = folds[self.val_fold]
            else:
                chosen = np.concatenate([folds[i] for i in range(self.k_folds) if i != self.val_fold])
            self.records = [self.records[i] for i in chosen]

        if self.max_num_samples:
            self.records = self.records[: self.max_num_samples]

    def _gather_records(self, directory: str) -> None:
        """Recursively searches for .hea files and stores their corresponding records."""
        files: List[str] = []
        for root, _, filenames in os.walk(directory):
            files.extend(os.path.join(root, file) for file in filenames if file.endswith(".hea"))
        for file in tqdm(files, desc="Processing files"):
            record_name = os.path.splitext(file)[0]
            self.records.append(record_name)

    def __len__(self) -> int:
        return len(self.records)

    def get_age(self, header: str) -> int:
        """Extracts the age from the header."""
        try:
            age_str = header.split(age_string)[1].split("\n")[0]
            return int(age_str)
        except (IndexError, ValueError):
            return -1

    def get_sex(self, header: str) -> int:
        """Extracts the sex from the header."""
        try:
            sex_str = header.split(sex_string)[1].split("\n")[0]
            sex_integer = 1 if sex_str.lower().startswith("m") else 0
            return sex_integer
        except IndexError:
            return -1

    def get_patient_id(self, header: str) -> str:
        """Extracts the patient ID from the header."""
        try:
            patient_id = header.split("# Patient ID:")[1].split("\n")[0].strip()
            return patient_id
        except IndexError:
            return "unknown"

    def __getitem__(  # type: ignore
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[Tuple[torch.Tensor, int, int], torch.Tensor]]:
        record_path = self.records[idx]
        label = torch.Tensor([self.get_chagas_label(idx)])
        record = wfdb.rdrecord(record_path)
        signal = torch.tensor(record.p_signal, dtype=torch.float32).T

        if self.reweighting_df is not None and self.use_val is False:
            patient_id = self.get_patient_id(load_text(record_path + ".hea"))
            if patient_id in self.reweighting_df["patient_id"].values:
                label = torch.tensor(
                    [
                        float(
                            self.reweighting_df[self.reweighting_df["patient_id"] == patient_id][
                                "chagas_fraction"
                            ].values[0]
                        )
                    ],
                    dtype=torch.float32,
                )

        if signal.shape[1] < 2700:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            signal = self.transform(signal)
        if self.return_demographics:
            header = load_text(record_path + ".hea")
            age = self.get_age(header)
            sex = self.get_sex(header)
            return (signal, age, sex), label
        return signal, label

    def get_chagas_label(self, idx: int) -> int:
        record_path = self.records[idx]
        header = load_text(record_path + ".hea")
        label_str = header.split(label_string)[1].split("\n")[0]
        return 1 if "true" in label_str.lower() else 0
