#!/usr/bin/env python
import os
import sys
from typing import Tuple

import torch
from numpy.typing import NDArray

from src.config.default import get_cfg
from src.submission_utils import classify_from_record
from src.train import load_and_train
from src.utils import load_model as load_torch_model

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
def train_model(data_folder: str, model_folder: str, verbose: bool) -> None:
    this_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(this_path, "src/config/inception_wfdb.yml")
    config = get_cfg(config_path)
    config.DATASET.TRAIN.KWARGS["root_dir"] = data_folder
    config.DATASET.VAL.KWARGS["root_dir"] = data_folder
    model: torch.nn.Module = load_and_train(ray_config=None, config=config)

    if verbose:
        print(f"Saving model to {model_folder}")

    torch.save(model.state_dict(), os.path.join(model_folder, "inception.pth"))
    with open(os.path.join(model_folder, "inception.yml"), "w") as f:
        f.write(config.dump())

    return None


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder: str, verbose: bool) -> torch.nn.Module:
    files = os.listdir(model_folder)
    config_path = [file for file in files if file.endswith("inception.yml")][0]
    config_path = os.path.join(model_folder, config_path)
    weights_path = [file for file in files if file.endswith("inception.pth")][0]
    weights_path = os.path.join(model_folder, weights_path)

    if verbose:
        print(f"Loading model from {config_path} and {weights_path}")

    config = get_cfg(config_path)
    model = load_torch_model(config)
    model.load_weights(weights_path)  # type: ignore
    model.eval()

    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record: str, model: torch.nn.Module, verbose: str) -> Tuple[bool, float]:
    binary_output, probability_output = classify_from_record(record, model)

    if verbose:
        print(f"Binary output: {binary_output}, Probability output: {probability_output}")

    return bool(binary_output), probability_output
