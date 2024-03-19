from loader import get_loaders
from model import run_train, SegModel
from utils import load_config, login_wandb, wandb_run


def main():
    config = load_config()
    run = None
    if config["wandb"]:
        login_wandb(config)
        run = wandb_run(config)
    loaders = get_loaders(config)
    model = SegModel(config, run)
    if config["train"]:
        run_train(loaders, model, config)
    else:
        save_inference(models, loaders, ("train", "val", "test"), config)


if __name__ == "__main__":
    main()
