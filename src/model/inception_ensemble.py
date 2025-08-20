import os
from collections import OrderedDict
from typing import Any

import torch

from src.model.inception import InceptionNetworkWithDownsampling


class InceptionEnsemble(torch.nn.Module):
    def __init__(
        self, weights_dir: str, weights_startswith: str = "load_and_train_", inception_kwargs: dict[str, Any] = {}
    ) -> None:
        super(InceptionEnsemble, self).__init__()
        models_list = []
        all_weights = [f for f in os.listdir(weights_dir) if f.startswith(weights_startswith)]
        for weight_file in all_weights:
            model = InceptionNetworkWithDownsampling(**inception_kwargs)
            model.load_weights(os.path.join(weights_dir, weight_file))
            models_list.append(model)
        self.models: torch.nn.ModuleList = torch.nn.ModuleList(models_list)
        self.num_in_channels = self.models[0].num_in_channels

    def forward(self, x: torch.Tensor, sigmoid_first: bool = False, pre_sigmoid_constant: float = 0.0) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        if sigmoid_first:
            stacked = torch.stack(outputs, dim=1) + pre_sigmoid_constant
            return torch.sigmoid(stacked).mean(dim=1)
        return torch.stack(outputs, dim=1).mean(dim=1)

    def load_weights(self, path: str) -> None:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, (OrderedDict, tuple)):
            state = loaded if isinstance(loaded, OrderedDict) else loaded[0]
            state = OrderedDict(
                (k.replace("_orig_mod.", ""), v) for k, v in state.items()
            )  # Removes prefix added by torch.compile
            self.load_state_dict(state)
        else:
            print("Could not load weights")
