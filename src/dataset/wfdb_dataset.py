import os
import wfdb
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from helper_code import load_text, label_string
from typing import Optional, Any, Tuple, List


class WFDBDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root_dir: str, transform: Optional[Any] = None, max_num_samples: Optional[int] = None) -> None:
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.max_num_samples = max_num_samples
        self.records: List[str] = []
        self.labels: List[int] = []

        self._gather_records(self.root_dir)

    def _gather_records(self, directory: str) -> None:
        """Recursively searches for .hea files and stores their corresponding records and labels."""
        root = directory
        files = os.listdir(directory)
        files = [file for file in files if file.endswith(".hea")]
        if self.max_num_samples:
            files = files[: self.max_num_samples]
        for file in tqdm(files, desc="Processing files"):
            record_name = os.path.splitext(file)[0]
            record_path = os.path.join(root, record_name)

            self.records.append(record_path)

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
