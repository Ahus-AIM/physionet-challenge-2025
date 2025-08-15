from typing import Any, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import resample_poly

from helper_code import (
    get_sampling_frequency,
    get_signal_names,
    load_header,
    load_signals,
    reorder_signal,
)


def resample_signal(signal: NDArray[Any], rate: float, target_rate: float) -> NDArray[Any]:
    """
    Resample the signal to the target rate.
    """
    if rate == target_rate:
        return signal
    else:
        up = int(target_rate)
        down = int(rate)
        resampled: NDArray[Any] = resample_poly(signal, up, down, axis=1)
        return resampled


def detect_zero_padding(signal: NDArray[Any]) -> Tuple[int, int, bool]:
    stddev = np.std(signal, axis=0)
    nonzero_indices = np.nonzero(stddev)[0]
    if len(nonzero_indices) < 100:
        print("The signal is all zeros...")
        return 0, signal.shape[1], False
    first_nonzero = int(np.min(nonzero_indices))
    last_nonzero = int(np.max(nonzero_indices)) + 1
    return first_nonzero, last_nonzero, True


def normalize_signal(signal: torch.Tensor) -> torch.Tensor:
    mean: torch.Tensor = torch.mean(signal, dim=2, keepdim=True)
    std: torch.Tensor = torch.std(signal, dim=2, keepdim=True).clamp(min=1e-5)
    signal = (signal - mean) / std
    return signal


def reorder_channels(signal: NDArray[Any], header: str) -> NDArray[Any]:
    desired_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal_names = get_signal_names(header)
    lower_signal_names = [name.lower() for name in signal_names]
    lower_desired_order = [name.lower() for name in desired_order]
    if set(lower_signal_names) == set(lower_desired_order):
        signal = reorder_signal(signal, lower_signal_names, lower_desired_order)
    return signal


def drop_channels(signal: NDArray[Any], num_target_channels: int) -> NDArray[Any]:
    return signal[-num_target_channels:]


def process_signal(signal: NDArray[Any], header: str) -> Tuple[NDArray[Any], bool]:
    assert (
        signal.shape[1] > signal.shape[0]
    ), f"The signal should have more columns than rows but has shape {signal.shape}"

    # Crop the signal if there is zero padding
    first_nonzero, last_nonzero, signal_has_values = detect_zero_padding(signal)
    signal = signal[:, first_nonzero:last_nonzero]
    signal = reorder_channels(signal, header)

    # Bandpass and resample filter the signal
    rate: float = get_sampling_frequency(header)  # type: ignore
    target_rate: float = 400.0
    signal = resample_signal(signal, rate, target_rate)

    return signal, signal_has_values


def classify_from_record(record: str, model: torch.nn.Module) -> Tuple[int, float]:
    header: str = load_header(record)
    signal, _ = load_signals(record)

    signal, signal_has_values = process_signal(signal.T, header)
    signal = drop_channels(signal, int(model.num_in_channels))  # type: ignore
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    num_chunks = 22
    chunk_size_samples = 800
    signal_size = signal_tensor.shape[2]
    jump_size = (signal_size - chunk_size_samples) // num_chunks

    signal_tensor_chunks = []
    for i in range(num_chunks):
        start = i * jump_size
        end = start + chunk_size_samples
        if end > signal_size:
            break
        signal_tensor_chunk = signal_tensor[:, :, start:end]
        signal_tensor_chunks.append(signal_tensor_chunk)

    probability_outputs = []
    for signal_tensor_chunk in signal_tensor_chunks[1:-1]:
        signal_tensor_chunk = normalize_signal(signal_tensor_chunk)

        probability = model(signal_tensor_chunk, sigmoid_first=True)
        probability_outputs.append(probability.mean().item())

    probability_output = np.mean(probability_outputs)

    probability_threshold = 0.5
    binary_output = int(probability_output > probability_threshold)

    if not signal_has_values:
        probability_output = 0.0
        binary_output = 0

    return binary_output, probability_output
