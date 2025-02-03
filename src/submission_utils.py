import torch
import numpy as np
from typing import Tuple, Any
from numpy.typing import NDArray
from scipy.signal import butter, lfilter

from helper_code import load_header, get_sampling_frequency, load_signals, reorder_signal, get_signal_names


def bandpass_filter(
    signal: NDArray[Any], rate: float, lowcut: float, highcut: float, pad_len: int = 2048
) -> NDArray[Any]:
    """
    Bandpass filter the data between lowcut and highcut.
    """
    padded_signal = np.pad(signal, pad_len, mode="reflect")
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    padded_filtered = lfilter(b, a, padded_signal, axis=0)
    filtered: NDArray[Any] = padded_filtered[pad_len:-pad_len]
    return filtered


def detect_zero_padding(signal: NDArray[Any]) -> Tuple[int, int]:
    stddev = np.std(signal, axis=0)
    first_nonzero = int(np.min(np.nonzero(stddev)))
    last_nonzero = int(np.max(np.nonzero(stddev)))
    return first_nonzero, last_nonzero


def normalize_signal(signal: NDArray[Any]) -> NDArray[Any]:
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
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


def resample_signal(signal: NDArray[Any], rate: float, target_rate: float) -> NDArray[Any]:
    raise NotImplementedError("Resampling is not implemented yet.")


def drop_channels(signal: NDArray[Any], num_target_channels: int) -> NDArray[Any]:
    return signal[-num_target_channels:]


def process_signal(signal: NDArray[Any], header: str) -> NDArray[Any]:
    assert (
        signal.shape[1] > signal.shape[0]
    ), f"The signal should have more columns than rows but has shape {signal.shape}"

    # Crop the signal if there is zero padding
    first_nonzero, last_nonzero = detect_zero_padding(signal)
    signal = signal[:, first_nonzero:last_nonzero]

    # Bandpass filter the signal
    lowcut: float = 0.5
    highcut: float = 150.0
    rate: float = get_sampling_frequency(header) or 400.0  # NOTE should we use 400.0 as default rate or throw an error?
    signal = bandpass_filter(signal, rate, lowcut, highcut)

    # Normalize the signal
    signal = normalize_signal(signal)

    # Ensure the channels are always in the same order
    signal = reorder_channels(signal, header)

    # Resample the signal if needed
    target_rate: float = 400.0
    if not np.all(np.isclose(rate, target_rate, atol=10.0)):
        signal = resample_signal(signal, rate, target_rate)

    return signal


def classify_from_record(record: str, model: torch.nn.Module) -> Tuple[int, float]:
    header: str = load_header(record)
    signal, _ = load_signals(record)

    signal = process_signal(signal.T, header)
    signal = drop_channels(signal, model.num_in_channels)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    logit = model(signal_tensor)
    probability_output = torch.sigmoid(logit).item()

    probability_threshold = 0.5
    binary_output = int(probability_output > probability_threshold)

    return binary_output, probability_output
