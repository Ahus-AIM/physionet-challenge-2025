from typing import Any, List, Optional, Tuple

import h5py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class HDF5InMemoryDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, hdf5_files: str, labels_csv: str, transform: Optional[Any] = None):
        """
        Args:
            labels_csv (str): Path to the CSV file with labels.
        """
        labels_df = pd.read_csv(labels_csv)
        labels_df["exam_id"] = labels_df["exam_id"].astype(str)
        labels_df["chagas"] = labels_df["chagas"].astype(int)
        self.labels_map = dict(zip(labels_df["exam_id"], labels_df["chagas"]))

        self.transform = transform

        self.exam_ids: List[str] = []
        self.tracings = []
        self.labels = []

        h5_files = sorted(hdf5_files)
        h5_files = [fname for fname in h5_files if fname.endswith(".hdf5")]

        for fname in tqdm(h5_files, desc="Loading HDF5 files"):
            print(f"Loading {fname} into memory... Num samples currently in memory: {len(self.exam_ids)}")
            with h5py.File(fname, "r") as f:
                # pass
                file_exam_ids = f["exam_id"][:].astype(str)
                crop = (4096 - 2700) // 2
                file_tracings = f["tracings"][:, crop:-crop, 4:]  # Load entire array into memory
                print(file_tracings.shape)
                # file_tracings = torch.zeros(len(file_exam_ids))

            for exam_id, tracing in zip(file_exam_ids, file_tracings):
                if exam_id in self.labels_map:  # Ensure label exists
                    self.exam_ids.append(exam_id)
                    self.tracings.append(torch.tensor(tracing, dtype=torch.float32))
                    self.labels.append(torch.tensor(self.labels_map[exam_id], dtype=torch.float32).unsqueeze(0))

        self.tracings = torch.stack(self.tracings)  # type: ignore
        self.labels = torch.stack(self.labels)  # type: ignore

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tracings, labels = self.tracings[idx].T, self.labels[idx]

        if self.transform:
            tracings = self.transform(tracings)

        return tracings, labels


# Usage Example
if __name__ == "__main__":
    hdf5_dir = "/data/physionet-challenge-2025/code15/"
    labels_csv = "/data/physionet-challenge-2025/code15/code15_chagas_labels/code15_chagas_labels.csv"
    batch_size = 32

    dataset = HDF5InMemoryDataset(hdf5_dir, labels_csv)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)  # No extra batching needed

    for signals, labels in dataloader:
        print(f"Batch shape: {signals.shape}, {labels.shape}")  # Expected: (32, C, T), (32, 1)

    print(f"Total batches: {len(dataset)}")
