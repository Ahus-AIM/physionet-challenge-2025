from typing import Any, Dict, List

import numpy as np
import scipy.signal as signal
import torch

try:
    from src.utils import import_class_from_path
except ImportError:
    from utils import import_class_from_path  # type: ignore


class HighPassButterworth(torch.nn.Module):
    def __init__(self, cutoff: float = 0.5, fs: float = 400, order: int = 4) -> None:
        """
        High-pass Butterworth filter transform for PyTorch tensors.

        Args:
            cutoff (float): Cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            order (int): Filter order.
        """
        super().__init__()
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype="high", analog=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the high-pass filter to a batch of signals.

        Args:
            x (torch.Tensor): Input tensor of shape (channels, time) or (channels, time).

        Returns:
            torch.Tensor: Filtered signal with the same shape as input.
        """
        x_np = x.detach().cpu().numpy()
        x_reversed_np = x_np[..., ::-1].copy()
        x_np_pad = np.concatenate([x_reversed_np, x_np, x_reversed_np], axis=-1)
        filtered = signal.filtfilt(self.b, self.a, x_np_pad, axis=-1).copy()
        filtered = filtered[..., x_np.shape[-1] : 2 * x_np.shape[-1]]
        return torch.tensor(filtered, dtype=x.dtype, device=x.device)


class Clip(torch.nn.Module):
    def __init__(self, min_val: float = -5.0, max_val: float = 5.0) -> None:
        """
        Clip signal values between a minimum and maximum value.

        Args:
            min_val (float): Minimum value to clip.
            max_val (float): Maximum value to clip.
        """
        super(Clip, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.min_val, max=self.max_val)


class HannWindow(torch.nn.Module):
    def __init__(self, power: float = 0.25) -> None:
        """
        Apply a Hann window to the signal.

        Args:
            power (float): Power of the Hann window.
        """
        super(HannWindow, self).__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(x.size(-1), device=x.device).pow(self.power)
        return x * window


class Normalize(torch.nn.Module):
    def __init__(self) -> None:
        super(Normalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


class SoftNormalize(torch.nn.Module):
    def __init__(self) -> None:
        super(SoftNormalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std.sqrt()


class CenterCrop(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super(CenterCrop, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        middle = x.size(1) // 2
        return x[:, middle - self.size // 2 : middle + self.size // 2]


class RandomCrop(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = np.random.randint(0, x.size(1) - self.size + 1)
        return x[:, start : start + self.size]


class DropChannels(torch.nn.Module):
    def __init__(self) -> None:
        super(DropChannels, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[4:]


class ComposedTransform(torch.nn.Module):
    def __init__(self, transform_config: List[Dict[Any, Any]]) -> None:
        """
        Initializes a composed transform based on the given configuration.

        Args:
            transform_config (dict): Configuration containing class path and transforms.
        """
        super(ComposedTransform, self).__init__()
        transforms = []
        for transform_def in transform_config:
            transform_class = import_class_from_path(transform_def["class_path"])
            transform = transform_class(**transform_def.get("KWARGS", {}))
            transforms.append(transform)
        self.transforms = transforms

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            signal = transform.forward(signal)
        return signal


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean: float = 0.0, std: float = 1e-5) -> None:
        """
        Adds Gaussian noise to the input signal.

        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise


class NanToZero(torch.nn.Module):
    def __init__(self) -> None:
        """
        Converts NaN values in the input tensor to zero.
        """
        super(NanToZero, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Replace NaN values with zero.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with NaN values replaced by zero.
        """
        return torch.nan_to_num(x, nan=0.0)


class CropZeroPadded(torch.nn.Module):
    def __init__(self, min_num_samples: int = 1200) -> None:
        """
        Crops zero-padded signals to remove leading and trailing zeros.
        """
        super(CropZeroPadded, self).__init__()
        self.min_num_samples = min_num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stddev = torch.std(x, dim=0)
        nonzero_indices = torch.nonzero(stddev)[:, 0]
        if nonzero_indices.numel() < 100:
            print("The signal is all zeros...")
            return x
        first_nonzero = int(torch.min(nonzero_indices))
        last_nonzero = int(torch.max(nonzero_indices)) + 1
        if last_nonzero - first_nonzero < self.min_num_samples:
            print("The signal is (almost) all zeros...")
            return x
        return x[:, first_nonzero:last_nonzero]


class Resample(torch.nn.Module):
    def __init__(self, input_fs: float = 500.0, target_fs: float = 400.0) -> None:
        """
        Resample a 1D signal tensor from input_fs to target_fs using polyphase filtering.

        Args:
            input_fs (float): Original sampling frequency in Hz. Default: 500.
            target_fs (float): Desired sampling frequency in Hz. Default: 400.
        """
        super(Resample, self).__init__()
        # compute integer up/down factors
        from math import gcd

        up = int(target_fs)
        down = int(input_fs)
        g = gcd(up, down)
        self.up = up // g
        self.down = down // g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply resampling to a batch of signals.

        Args:
            x (torch.Tensor): Input tensor of shape (..., time).

        Returns:
            torch.Tensor: Resampled tensor with time dimension scaled by up/down.
        """
        # move to numpy for scipy
        orig_dev = x.device
        x_np = x.detach().cpu().numpy()
        # use polyphase filtering for high-quality resampling
        y_np = signal.resample_poly(x_np, self.up, self.down, axis=-1)
        # convert back to torch tensor
        y = torch.tensor(y_np, dtype=x.dtype, device=orig_dev)
        return y


class RandomMultiply(torch.nn.Module):
    def __init__(self, min_value: float = 0.7, max_value: float = 1.3) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multiplier = self.min_value + np.random.rand() * (self.max_value - self.min_value)
        return x * multiplier
