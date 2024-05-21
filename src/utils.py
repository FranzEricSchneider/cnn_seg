import argparse
import copy
import json
import numpy
from pathlib import Path
import wandb

from config import CONFIG


def key_string(key):
    key = str(key)
    # Shorten the key to produce shorter names
    if len(key) > 5:
        key = ("".join([split[0] for split in key.split("_")])).upper()
    return key


def tensor2np(tensor):
    if isinstance(tensor, numpy.ndarray):
        return tensor
    else:
        return tensor.detach().cpu().numpy()


def login_wandb(config):
    keyfile = Path(config["keyfile"])
    assert (
        keyfile.is_file()
    ), f"Need to populate {keyfile} with json containing wandb key"
    wandb.login(key=json.load(keyfile.open("r"))["key"])


def wandb_run(config):
    name = "-".join(
        [
            f"{key_string(key)}:{config[key]}"
            if (key in config and not isinstance(config[key], dict))
            else key
            for key in config["wandb_print"]
        ]
    )
    run = wandb.init(
        # Wandb creates random run names if you skip this field
        name=name,
        # Allows reinitalizing runs
        reinit=True,
        # Insert specific run id here if you want to resume a previous run
        # run_id=
        # You need this to resume previous runs, but comment out reinit=True when using this
        # resume="must"
        # Project should be created in your wandb account
        project="cnn-segmentation",
        config=config,
    )
    return run


def load_config():
    config = copy.copy(CONFIG)

    parser = argparse.ArgumentParser(description="Set config via command line")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to dir with all images and labels. Expected format is"
        " data_dir/(train | test | val)/(images | masks)",
    )
    parser.add_argument("--wandb-print", nargs="+", default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-l", "--lr", type=float, default=None)
    parser.add_argument("-w", "--wd", type=float, default=None)
    parser.add_argument("-g", "--train-augmentation-path", default=None)
    args = parser.parse_args()

    # Blindly fill arguments into the config
    for key in config.keys():
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                config[key] = value

    print("\n" + "=" * 36 + " CONFIG " + "=" * 36)
    for k, v in config.items():
        print(f"\t{k}: {v}")
    print("=" * 80)

    return config
