from os import getenv

CONFIG = {
    "data_dir": None,
    "extension": "jpg",

    "train": True,

    "wandb": False,
    "wandb_print": [
        "architecture",
        "encoder",
    ],
    "keyfile": getenv("HOME") + "/wandb.json",

    "architecture": "unet",
    "encoder": "resnet18",

    # "models": [],
    # # "models": ["checkpoint.pth"],
    # # "models": [{"name": "checkpoint.pth", "run_path": "image-regression/3q34k58v", "replace": True}],
    # # "models": [{"name": "checkpoint.pth", "run_path": "image-regression/fqsx3zdk", "replace": True},
    # #            {"name": "checkpoint.pth", "run_path": "image-regression/37z196qx", "replace": True},
    # #            {"name": "checkpoint.pth", "run_path": "image-regression/phan45yu", "replace": True}],
    # # "models": ["checkpoint.pth",
    # #            {"name": "checkpoint.pth", "run_path": "image-regression/sk5209ak", "replace": True}],
    # # If we define a model using a path to a .pth file, this needs to be a list
    # # to the corresponding wandb config.yaml file
    # "config_paths": None,

    "lr": 1e-4,
    "scheduler": "constant",
    "StepLR_kwargs": {"step_size": 5, "gamma": 0.2},
    "LRTest_kwargs": {"min_per_epoch": 0.05, "runtime_min": 6, "start": 1e-6, "end": 1.0},
    "OneCycleLR_kwargs": {"max_lr": 5e-3, "min_lr": 5e-5},
    "CosMulti_kwargs": {"epoch_per_cycle": 5, "eta_min": 6e-6},
    # mode: min (ideally value goes down) or max (opposite)
    # factor: scale factor for LR
    # patience: number of epochs with no improvement after which LR is reduced
    "ReduceLROnPlateau_kwargs": {"mode": "min", "patience": 2, "factor": 0.5, "min_lr": 6e-6},

    # When doing an LRTest, what frequency do you want to report from the
    # batches? A tri of 1 will report every batch
    "train_report_iter": 1,

    # Augmentations are stored in a json file as (name, kwargs). They are
    # applied in the loader phase. The way to experiment with augmentations is
    # to make a copy of the file, set those that you want, and then select the
    # files one-by-one as a command-line argument.
    "train_augmentation_path": "train_augmentations.json",
    "val_augmentation_path": "val_augmentations.json",
    "test_augmentation_path": "test_augmentations.json",

    # Randomly view one validation image per batch
    "vis_val_images": True,

    # Save all (augmented) training images to /tmp/ for debug (slow)
    "log_training_images": False,

    # Increase if you can handle it, generally
    "batch_size": 6,

    "epochs": 24,
    "wd": 0.01,
    "eval_report_iter": 1,

}
