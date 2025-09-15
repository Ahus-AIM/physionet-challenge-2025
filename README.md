# Detection of Chagas Disease from the ECG: Ahus AIM's Entry for the George B. Moody PhysioNet Challenge 2025

![Tests](https://github.com/Ahus-AIM/physionet-challenge-2025/actions/workflows/submission.yml/badge.svg?branch=main) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31211/) [![Preprint](https://img.shields.io/badge/Preprint-PDF-blue?logo=arxiv)](docs/preprint.pdf) [![Poster](https://img.shields.io/badge/Poster-PDF-orange?logo=adobeacrobatreader)](docs/poster.pdf)


This repository contains team Ahus AIM's entry for the [Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025](https://physionetchallenges.org/2025/).


# Installation

Create virtual environment with Python 3.12 and install requirements:
```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training & Fine‑tuning

1. Edit or create a train config file, see examples in the [config folder](src/config/).

2. Launch training: 
   ```bash
   python -m src.train --config /src/config/your_modified_config_file.yaml
   ```

3. Outputs:
   - If search space is not defined, results go to: `./sandbox/weights/`
   - If search space is defined, results go to: `~/ray_results/`

You can use the [plotting script](src/scripts/plot_ray_results.py) and the [weight gathering script](src/scripts/get_model_weights_from_experiment.py) to summarize the result of a training run.

## Inference and Evaluation
See the [submission script](.github/workflows/submission.yml) for an example of how to build and test this submission, as specified in this year's Challenge procedure. See [evaluation code repository](https://github.com/physionetchallenges/evaluation-2025) for code and instructions for evaluation using Challenge scoring metric.

## Linting & Static Analysis
To apply linting and static typechecking:
```bash
./lint_src.sh
```

## Useful links

* [Challenge website](https://physionetchallenges.org/2025/)
* [Evaluation code](https://github.com/physionetchallenges/evaluation-2025)
* [Frequently asked questions (FAQ) for the 2025 Challenge](https://physionetchallenges.org/2025/faq/)
* [Frequently asked questions (FAQ) about the Challenges in general](https://physionetchallenges.org/faq/)
