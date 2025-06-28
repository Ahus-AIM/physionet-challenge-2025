import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset
from tqdm import tqdm

from helper_code import label_string, load_text


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
    ) -> None:
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.max_num_samples = max_num_samples

        self.k_folds = k_folds
        self.val_fold = val_fold
        self.use_val = use_val
        self.shuffle_seed = shuffle_seed

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record_path = self.records[idx]
        label = torch.Tensor([self.get_chagas_label(idx)])
        record = wfdb.rdrecord(record_path)
        signal = torch.tensor(record.p_signal, dtype=torch.float32).T
        if signal.shape[1] < 2700:
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            signal = self.transform(signal)
        return signal, label

    def get_chagas_label(self, idx: int) -> int:
        record_path = self.records[idx]
        header = load_text(record_path + ".hea")
        label_str = header.split(label_string)[1].split("\n")[0]
        if "true" in label_str.lower():
            label = 1
        else:
            label = 0
        return label
