import torch

from loader import get_loaders
from model import model_from_pth, SegModel
from train import run_train, save_inference
from utils import load_config, login_wandb, wandb_run


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config()

    run = None
    if config["wandb"] or config["model"] is not None:
        login_wandb(config)
    if config["wandb"]:
        run = wandb_run(config)

    loaders = get_loaders(config)

    if config["model"] is None:
        model = SegModel(config, device, run)
    else:
        model = model_from_pth(config["model"], device)

    if config["train"]:
        run_train(loaders, model, config, device, run, debug=False)
    else:
        save_inference(model, loaders, ("train", "val", "test"), config, device)


if __name__ == "__main__":
    main()
