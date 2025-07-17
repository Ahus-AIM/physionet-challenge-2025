# Ahus AIM Entry for the George B. Moody PhysioNet Challenge 2025

## What's in this repository?

This repository contains team Ahus AIM's entry for the [George B. Moody PhysioNet Challenge 2025](https://physionetchallenges.org/2025/).

## Linting & Static Analysis
– Black, isort, Flake8, Mypy  
Run all checks via:
./lint_src.sh

## How do I run these scripts?

First, you can download and create data for these scripts by following the instructions in the following section.

## How do I create data for these scripts?

You can use the scripts in this repository to convert the [CODE-15% dataset](https://zenodo.org/records/4916206) to [WFDB](https://wfdb.io/) format. These instructions use `code15_hdf5` as the path for the input data files and `code15_wfdb` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine.

1. Download and unzip one or more of the `exam_part` files and the `exams.csv` file in the [CODE-15% dataset](https://zenodo.org/records/4916206).

2. Download and unzip the Chagas labels, i.e., the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) file.

3. Convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_code15_data.py \
            -i code15_hdf5/exams_part0.hdf5 code15_hdf5/exams_part1.hdf5 \
            -d code15_hdf5/exams.csv \
            -l code15_hdf5/code15_chagas_labels.csv \
            -o code15_wfdb

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings. You can include more or fewer of these files to increase or decrease the number of ECG recordings, respectively. You may want to start with fewer ECG recordings to debug your code.

# Installation

1. Create virtual environment with Python 3.12:
   ```bash
   python3.12 -m venv venv
   ```
2. Activate the environment:
   ```bash
   source venv/bin/activate
   ```
3. Upgrade pip and install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

**NOTE:** You can also run the code as in the PhysioNet Challenge evaluation.

## Training & Fine‑tuning

1. Edit or create a config (YAML/JSON) to specify data paths, transforms, model/loss, hyperparameters (and optional search space).
2. Launch on CUDA 0:
   CUDA_VISIBLE_DEVICES=0 \
     python -m src.train --config /path/to/config.yaml
3. Outputs:
   - If search space is not defined: results go to ./sandbox/weights
   - If search space is defined: results go to ~/ray_results/. In this case, you can inspect the results and fetch the best weights with the scripts in src/scripts.



## Evaluation code

The evaluation code used to score submissions is not stored here, see [evaluation code repository](https://github.com/physionetchallenges/evaluation-2025) for code and instructions for evaluating your entry using the Challenge scoring metric.

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Frequently asked questions (FAQ) for this year's Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
