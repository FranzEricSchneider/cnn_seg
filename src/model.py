import json
from matplotlib import pyplot
import numpy
from os import rename
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import wandb
import yaml

import segmentation_models_pytorch as smp
import torchseg

from utils import tensor2np
from vis import vis_image


class SegModel:
    def __init__(self, config, device, run=None, disable_areas=False, **kwargs):

        # All of the necessary arguments as a dictionary
        self.config = config
        # Either None or a wandb run that can be used to track stats
        self.run = run

        self.model = torchseg.create_model(
            self.config["architecture"],
            encoder_name=self.config["encoder"],
            in_channels=3,
            classes=1,
            **kwargs,
        )
        self.model.to(device)

        # Preprocessing parameters based on the encoder
        params = smp.encoders.get_preprocessing_params(self.config["encoder"])
        self.std = torch.tensor(params["std"]).view(1, 3, 1, 1).to(device)
        self.mean = torch.tensor(params["mean"]).view(1, 3, 1, 1).to(device)

        # TODO: Try other loss options:
        # https://smp.readthedocs.io/en/latest/losses.html
        self.loss_fn = torchseg.losses.DiceLoss(torchseg.losses.BINARY_MODE, from_logits=False)

        # Save outputs from various stages
        self.outputs = {"train": [], "val": [], "test": []}

        # Save whether or not to check the mask for invalid areas (slower)
        self.disable_areas = disable_areas

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        # Run image through the model
        mask = self.model(image)
        return mask

    def process_batch(self, image, mask, stage, keep_output=False):
        """
        Arguments:
            image: batch of images that have already been transformed
                (tensor, shape BCHW)
            mask: batch of masks that have already been transformed
                (tensor, shape BCHW) where the channel value is 1
            stage: string specifying train/val/test. So far is used for
                optional validation in the "val" stage, and also records output
                stats by stage. It's up to the code that calls this function to
                clear self.outputs between batches
            keep_output: If True, we will store the output mask as a floating
                point numpy array in the self.output[stage]. This is memory
                intensive and likely should not be done during training.

        returns: loss (defined by self.loss_fn()) as a tensor, such that
            loss.backward() works if desired
        """

        # Shape of the image should be BCHW
        assert image.ndim == 4
        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be BCHW, for binary C=1 (weird)
        assert mask.ndim == 4
        assert mask.shape[1] == 1
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Convert the logits into a 0-1 sigmoid valu
        # This code was copied from the DiceLoss source code
        #   "Using Log-Exp as this gives more numerically stable result and
        #   does not cause vanishing gradient on extreme values 0 and 1"
        predicted_mask = F.logsigmoid(logits_mask).exp()

        # Predicted mask contains logits, and loss_fn param `from_logits` is
        # set to True
        if self.disable_areas:
            # We have somewhat hackily defined a mask value of 0.5 as an area
            # where the value is unknown - remove those areas from the loss
            # computation
            good_areas = (mask < 0.25) | (mask > 0.75)
            loss = self.loss_fn(predicted_mask * good_areas, mask * good_areas)
        else:
            loss = self.loss_fn(predicted_mask, mask)

        # Convert mask values to probabilities, then apply thresholding
        threshold_mask = (predicted_mask > 0.5).float()
        # The first time through, save a debug image
        if (
            stage == "val"
            and self.config["vis_val_images"]
            and len(self.outputs[stage]) == 0
        ):
            impath = Path(f"/tmp/val_vis_{int(time.time() * 1e6)}.jpg")
            vis_image(
                tensor=image[0],
                gt_mask=mask[0],
                pred_mask=threshold_mask[0],
                save_path=impath,
            )
            if self.run is not None:
                self.run.log({"val_vis_first_image": wandb.Image(str(impath))})

        # Compute TN/FP/TP/FN pixels as reporting stats for each image
        tp, fp, fn, tn = smp.metrics.get_stats(
            threshold_mask.int(), mask.int(), mode="binary"
        )
        # It's important to cost loss to numpy to avoid a huge GPU load
        # building up over time, since loss is connected to all other GPU
        # computations
        self.outputs[stage].append(
            {"loss": tensor2np(loss), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        )
        if keep_output:
            self.outputs[stage][-1]["mask"] = tensor2np(predicted_mask)

        return loss

    def record_epoch_end(self, stage, lr=None):

        # For now if there is no wandb run active just end. In the future we
        # might add some non-wandb reporting above this
        if self.run is None:
            return

        # Aggregate batch step metics
        tp, fp, fn, tn = [
            torch.cat([x[key] for x in self.outputs[stage]])
            for key in ["tp", "fp", "fn", "tn"]
        ]
        # Calculate IoU score for each image and then compute the mean
        mean_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        # Aggregate mask intersection over whole dataset and then compute IoU.
        # For datasets with "empty" images (no target class) a large gap could
        # be observed. Empty images influence dataset_iou much less.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        log_values = {
            f"{stage}_loss": numpy.mean([x["loss"] for x in self.outputs[stage]]),
            f"{stage}_tp": tensor2np(tp).mean(),
            f"{stage}_tn": tensor2np(tn).mean(),
            f"{stage}_fp": tensor2np(fp).mean(),
            f"{stage}_fn": tensor2np(fn).mean(),
            f"{stage}_per_im_iou": mean_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        if lr is not None:
            log_values.update({"lr": lr})
        self.run.log(log_values)


def load_wandb_config(run_path=None, new_file="wandb_config.yaml", config_path=None):
    """
    The wandb config splits config values into desc (description) and value
    (the stuff we want). Undo that.
    """
    if config_path is None:
        path = wandb.restore(name="config.yaml", run_path=run_path, replace=True)
        path = Path(path.name)
        rename(path, new_file)
        config_path = new_file
    wandb_dict = yaml.safe_load(config_path.open("r"))
    loaded_config = {}
    for key, value in wandb_dict.items():
        if isinstance(value, dict) and "value" in value:
            loaded_config[key] = value["value"]
    return loaded_config


def model_from_pth(settings, device, run, disable_areas):

    if isinstance(settings, dict):
        path = wandb.restore(**settings)
        path = Path(path.name)
        load_file = path.parent.joinpath(
            f"{settings['run_path'].replace('/', '_')}.pth"
        )
        rename(path, load_file)
        config = load_wandb_config(
            run_path=settings["run_path"], new_file=load_file.with_suffix(".yaml")
        )

    else:
        raise NotImplementedError(f"Unknown setting type: {type(settings)}")

    model = SegModel(config, device, run=run, disable_areas=disable_areas)
    model.model.load_state_dict(
        torch.load(load_file, map_location=torch.device(device))["model_state_dict"]
    )

    return model


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run a really basic validation model through the first"
        " part of a dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "rootdir",
        help="Path to images/ and masks/ root dir for visualization",
        type=Path,
    )
    parser.add_argument(
        "--count",
        help="Number of batches to process",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    imdir = args.rootdir / "images"
    maskdir = args.rootdir / "masks"

    from loader import SegmentationDataset

    dataset = SegmentationDataset(
        sorted(imdir.glob("*jpg")),
        sorted(maskdir.glob("*npy")),
        transforms=None,
    )

    import os
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
    )

    model = SegModel(
        config={
            "architecture": "FPN",
            "encoder": "resnet18",
            "lr": 1e-3,
            "vis_val_images": True,
        },
        device="cpu",
    )
    model.model.eval()

    for i, (image, mask, _) in enumerate(dataloader):
        if i == args.count:
            break
        print(f"Batch {i + 1} / {args.count}")
        model.process_batch(image, mask, "val")
    print("Check for viz images in /tmp/, model output will be nonsense")
