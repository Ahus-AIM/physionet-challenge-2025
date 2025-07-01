#!/usr/bin/env python

import argparse
import os
import os.path
import sys
from contextlib import contextmanager
from typing import Iterator, Optional, Dict, List, Any

import h5py
import numpy as np
import pandas as pd
import wfdb

from helper_code import is_boolean, is_integer, sanitize_boolean_value


def get_parser() -> argparse.ArgumentParser:
    description: str = "Prepare the SaMi-Trop dataset for the Challenge."
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--signal_file", type=str, required=True)  # exams.hdf5
    parser.add_argument("-d", "--demographics_file", type=str, required=True)  # exams.csv
    parser.add_argument("-f", "--signal_format", type=str, required=False, default="dat", choices=["dat", "mat"])
    parser.add_argument("-o", "--output_path", type=str, required=True)
    return parser


@contextmanager
def suppress_stdout() -> Iterator[None]:
    with open(os.devnull, "w") as devnull:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = stdout


def convert_dat_to_mat(record: str, write_dir: Optional[str] = None) -> None:
    import wfdb.io.convert  # type: ignore

    cwd: Optional[str] = None
    if write_dir:
        cwd = os.getcwd()
        os.chdir(write_dir)

    with suppress_stdout():
        wfdb.io.convert.matlab.wfdb_to_matlab(record)

    os.remove(record + ".hea")
    os.remove(record + ".dat")

    os.rename(record + "m" + ".hea", record + ".hea")
    os.rename(record + "m" + ".mat", record + ".mat")

    output_string: str = ""
    with open(record + ".hea", "r") as f:
        for line in f:
            if line.startswith("#Creator") or line.startswith("#Source"):
                continue
            line = line.replace(record + "m", record)
            output_string += line

    with open(record + ".hea", "w") as f:
        f.write(output_string)

    if write_dir and cwd is not None:
        os.chdir(cwd)


def fix_checksums(record: str, checksums: Optional[np.ndarray] = None) -> None:
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)
        signals: np.ndarray = np.asarray(x.d_signal)
        checksums_arr = np.atleast_1d(np.sum(signals, axis=0, dtype=np.int16))
    else:
        checksums_arr = np.atleast_1d(checksums)

    header_filename: str = os.path.join(record + ".hea")
    lines: List[str] = []
    num_leads: Optional[int] = None

    with open(header_filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                arrs = line.split(" ")
                num_leads = int(arrs[1])
            if num_leads is not None and 0 < i <= num_leads and not line.startswith("#"):
                arrs = line.split(" ")
                arrs[6] = str(checksums_arr[i - 1])
                line = " ".join(arrs)
            lines.append(line)

    with open(header_filename, "w") as f:
        f.writelines(lines)


def run(args: argparse.Namespace) -> None:
    exam_ids: List[int] = []
    exam_id_to_age: Dict[int, int] = {}
    exam_id_to_sex: Dict[int, str] = {}

    df: pd.DataFrame = pd.read_csv(args.demographics_file)
    for _, row in df.iterrows():
        row_exam_id: Any = row["exam_id"]
        assert is_integer(row_exam_id)
        int_exam_id: int = int(row_exam_id)
        exam_ids.append(int_exam_id)

        row_age: Any = row["age"]
        assert is_integer(row_age)
        int_age: int = int(row_age)
        exam_id_to_age[int_exam_id] = int_age

        row_is_male: Any = row["is_male"]
        assert is_boolean(row_is_male)
        sex_str: str = "Male" if sanitize_boolean_value(row_is_male) else "Female"
        exam_id_to_sex[int_exam_id] = sex_str

    lead_names: List[str] = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    sampling_frequency: int = 400
    units: str = "mV"
    gain: int = 1000
    baseline: int = 0
    num_bits: int = 16
    fmt: str = str(num_bits)

    os.makedirs(args.output_path, exist_ok=True)
    num_exam_ids: int = len(exam_ids)

    with h5py.File(args.signal_file, "r") as f:
        for i in range(num_exam_ids):
            cur_exam_id: int = exam_ids[i]
            physical_signals: np.ndarray = np.array(f["tracings"][i], dtype=np.float32)
            num_samples, num_leads = np.shape(physical_signals)
            assert num_leads == 12

            r: int = 0
            while r < num_samples and np.all(physical_signals[r, :] == 0):
                r += 1
            s: int = num_samples
            while s > r and np.all(physical_signals[s - 1, :] == 0):
                s -= 1

            if r >= s:
                continue
            physical_signals = physical_signals[r:s, :]

            digital_signals: np.ndarray = gain * physical_signals
            digital_signals = np.round(digital_signals)
            digital_signals = np.clip(digital_signals, -(2 ** (num_bits - 1)) + 1, 2 ** (num_bits - 1) - 1)
            digital_signals[~np.isfinite(digital_signals)] = -(2 ** (num_bits - 1))
            digital_signals = np.asarray(digital_signals, dtype=np.int32)

            patient_age: int = exam_id_to_age[cur_exam_id]
            patient_sex: str = exam_id_to_sex[cur_exam_id]
            source: str = "SaMi-Trop"
            label: bool = True

            comments: List[str] = [
                f"Age: {patient_age}",
                f"Sex: {patient_sex}",
                f"Chagas label: {label}",
                f"Source: {source}",
            ]

            record: str = str(cur_exam_id)
            wfdb.wrsamp(
                record,
                fs=sampling_frequency,
                units=[units] * num_leads,
                sig_name=lead_names,
                d_signal=digital_signals,
                fmt=[fmt] * num_leads,
                adc_gain=[gain] * num_leads,
                baseline=[baseline] * num_leads,
                write_dir=args.output_path,
                comments=comments,
            )

            if args.signal_format in ("mat", ".mat"):
                convert_dat_to_mat(record, write_dir=args.output_path)

            checksums_arr: np.ndarray = np.atleast_1d(np.sum(digital_signals, axis=0, dtype=np.int16))
            fix_checksums(os.path.join(args.output_path, record), checksums=checksums_arr)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
