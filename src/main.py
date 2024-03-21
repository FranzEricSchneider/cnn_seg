import torch

from loader import get_loaders
from model import SegModel
from train import run_train
from utils import load_config, login_wandb, wandb_run


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config()
    run = None
    if config["wandb"]:
        login_wandb(config)
        run = wandb_run(config)
    loaders = get_loaders(config)
    model = SegModel(config, device, run)
    if config["train"]:
        run_train(loaders, model, config, device, run)
    else:
        save_inference(models, loaders, ("train", "val", "test"), config)


if __name__ == "__main__":
    main()
