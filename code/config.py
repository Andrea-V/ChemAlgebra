from types import SimpleNamespace
import gpustat
import os, sys
from copy import deepcopy

def best_gpus(n_gpus=1):
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)

    n_best_gpus = list(map(lambda x: x[0], sorted(zip(ids, ratios), key=lambda x: x[1])[:n_gpus]))

    print("- Best GPUs:", n_best_gpus)
    return n_best_gpus


def make_config(cfg_param):
    cfg = deepcopy(cfg_param)

    cfg["data"]["augm"] = SimpleNamespace(**cfg["data"]["augm"])
    cfg["data"] = SimpleNamespace(**cfg["data"])

    cfg["model"]["gru"] = SimpleNamespace(**cfg["model"]["gru"])
    cfg["model"]["transformer"] = SimpleNamespace(**cfg["model"]["transformer"])
    cfg["model"] = SimpleNamespace(**cfg["model"])

    cfg["loss"] = SimpleNamespace(**cfg["loss"])
    cfg["training"] = SimpleNamespace(**cfg["training"])
    cfg["log"] = SimpleNamespace(**cfg["log"])
    cfg["eval"] = SimpleNamespace(**cfg["eval"])
    cfg = SimpleNamespace(**cfg)
    return cfg


default_config = {
    "devices": best_gpus(4),
    "name": "FINAL_RUN_CROSS", #"FINAL_RUN",
    "data": {
        "cross": True, # load cross-distribution data for train and eval instead of in-, x-, out-
        "path_prefix": "./data",
        "batch_size": 128,
        "augm": {
            "type": 1, # 5, 10
            "n": 1, # 5, 10
            "repr": "smiles" # "formula",
        },
        "n_workers": 128,
    },
    "model": {
        "max_length": 200,
        "gru": {
            "depth": 6,
            "hidden_size": 512, # NB decoder will have hidden_size*2 when encoder=bidirectional
            "attn_heads": 8,
        },
        "transformer": {
            "encoder_depth": 6,     # 10
            "decoder_depth": 6,     # 10
            "embedding_size": 512,  #256,
            "attn_heads": 8,        # 8
            "d_model": 512, # 256
            "d_ff": 512,  # 256
            "beam_width": 10,
        }
    },
    "loss": {
        "label_smoothing": 0.1,
    },
    "training": {
        "min_delta": 0.0,
        "early_stopping_patience": 50,
        "scheduler_patience": 10,
        "max_epochs": 500,
        "lr": 1e-5,
        "fast_dev_run": False,
        "val_check_interval": 500
    },
    "log": {
    },
    "eval": {
        "final_preds_path": "./final_predictions"
    }
}


if __name__ == "__main__":
    print(best_gpus(3))