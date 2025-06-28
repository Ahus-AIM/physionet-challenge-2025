from typing import Any, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.signal import butter, lfilter, resample_poly

from helper_code import (
    get_sampling_frequency,
    get_signal_names,
    load_header,
    load_signals,
    reorder_signal,
)


def bandpass_and_resample(
    signal: NDArray[Any], rate: float, target_rate: float, lowcut: float, highcut: float
) -> NDArray[Any]:
    """
    Bandpass filter the data between lowcut and highcut, and then resample to target_rate.
    """
    padded_signal = np.concatenate([signal[:, ::-1], signal, signal[:, ::-1]], axis=1)
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist

    if low < 0.0:
        print(f"Lowcut {lowcut} is too low for the sampling frequency {rate}. Setting to 0.0.")
        low = 1e-5
    if high > 1.0:
        print(f"Highcut {highcut} is too high for the sampling frequency {rate}. Setting to 1.0.")
        high = 1 - 1e-5

    b, a = butter(1, [low, high], btype="band")
    padded_filtered = lfilter(b, a, padded_signal, axis=1)

    resampled_signal: NDArray[Any]
    if rate == target_rate:
        signal_length = padded_filtered.shape[1] // 3
        resampled_signal = padded_filtered[:, signal_length : 2 * signal_length]
        return resampled_signal
    else:  # Resample using polyphase filtering
        up = int(target_rate)
        down = int(rate)
        padded_resampled = resample_poly(padded_filtered, up, down, axis=1)

        signal_length = padded_resampled.shape[1] // 3
        resampled_signal = padded_resampled[:, signal_length : 2 * signal_length]
        return resampled_signal


def detect_zero_padding(signal: NDArray[Any]) -> Tuple[int, int, bool]:
    stddev = np.std(signal, axis=0)
    nonzero_indices = np.nonzero(stddev)[0]
    if len(nonzero_indices) < 100:
        print("The signal is all zeros...")
        return 0, signal.shape[1], False
    first_nonzero = int(np.min(nonzero_indices))
    last_nonzero = int(np.max(nonzero_indices)) + 1
    return first_nonzero, last_nonzero, True


def normalize_signal(signal: NDArray[Any]) -> NDArray[Any]:
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    std = np.clip(std, a_min=1e-5, a_max=None)
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

    # Bandpass and resample filter the signal
    rate: float = get_sampling_frequency(header)  # type: ignore
    target_rate: float = 400.0
    lowcut: float = 0.5
    highcut: float = 150.0
    signal = bandpass_and_resample(signal, rate, target_rate, lowcut, highcut)

    # Normalize the signal
    signal = normalize_signal(signal)

    # Ensure the channels are always in the same order
    signal = reorder_channels(signal, header)

    return signal, signal_has_values


def classify_from_record(record: str, model: torch.nn.Module) -> Tuple[int, float]:
    header: str = load_header(record)
    signal, _ = load_signals(record)

    signal, signal_has_values = process_signal(signal.T, header)
    signal = drop_channels(signal, int(model.num_in_channels))  # type: ignore
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    logit = model(signal_tensor)
    probability_output = torch.sigmoid(logit).item()

    probability_threshold = 0.5
    binary_output = int(probability_output > probability_threshold)

    if not signal_has_values:
        probability_output = 0.0
        binary_output = 0

    return binary_output, probability_output
