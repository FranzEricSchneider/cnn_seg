import json
from matplotlib import pyplot
import numpy
import time
import torch
import wandb

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchseg

from utils import tensor2np


def vis_image(tensor, gt_mask, pred_mask, save_path):

    from matplotlib import pyplot

    figure = pyplot.figure(figsize=(6, 2.5))

    pyplot.subplot(1, 4, 1)
    pyplot.imshow(tensor2np(tensor).transpose(1, 2, 0))  # convert CHW -> HWC
    pyplot.title("Image")
    pyplot.axis("off")

    pyplot.subplot(1, 4, 2)
    pyplot.imshow(tensor2np(gt_mask).squeeze(), vmin=0, vmax=1)
    pyplot.title("Ground truth")
    pyplot.axis("off")

    pyplot.subplot(1, 4, 3)
    pyplot.imshow(tensor2np(pred_mask).squeeze(), vmin=0, vmax=1)
    pyplot.title("Prediction")
    pyplot.axis("off")

    pyplot.subplot(1, 4, 4)
    pyplot.imshow(1 - tensor2np(pred_mask).squeeze(), vmin=0, vmax=1)
    pyplot.title("1 - Prediction")
    pyplot.axis("off")

    pyplot.tight_layout()
    pyplot.savefig(save_path)
    pyplot.close(figure)


# Inspired by
# https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
class SegModel(pl.LightningModule):
    def __init__(self, config, run=None, **kwargs):

        # All of the necessary arguments as a dictionary
        self.config = config
        # Either None or a wandb run that can be used to track stats
        self.run = run

        super().__init__()
        self.model = torchseg.create_model(
            self.config["architecture"],
            encoder_name=self.config["encoder"],
            in_channels=3,
            # I don't think it counts the background as a class, which is
            # weird but whatever
            classes=1,
            **kwargs,
        )

        # preprocessing parameters based on the encoder
        params = smp.encoders.get_preprocessing_params(self.config["encoder"])
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # TODO: Try other loss options:
        # https://smp.readthedocs.io/en/latest/losses.html
        self.loss_fn = torchseg.losses.DiceLoss(
            torchseg.losses.BINARY_MODE, from_logits=True
        )

        # Save outputs from various stages
        self.outputs = {"train": [], "val": [], "test": []}

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch[0]

        # Shape of the image should be BCHW
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be BCHW, for binary C=1 (weird)
        assert mask.ndim == 4
        assert mask.shape[1] == 1

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is
        # set to True
        loss = self.loss_fn(logits_mask, mask)

        # Convert mask values to probabilities, then apply thresholding
        pred_mask = (logits_mask.sigmoid() > 0.5).float()

        # Compute TN/FP/TP/FN pixels for each image
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.int(), mask.int(), mode="binary"
        )

        # The first time through, save a debug image
        if (
            stage == "val"
            and self.config["vis_val_images"]
            and len(self.outputs[stage]) == 0
        ):
            vis_image(
                tensor=image[0],
                gt_mask=mask[0],
                pred_mask=pred_mask[0],
                save_path=f"/tmp/val_vis_{int(time.time() * 1e6)}.jpg",
            )

        self.outputs[stage].append(
            {
                "loss": tensor2np(loss),
                "tp": tensor2np(tp),
                "fp": tensor2np(fp),
                "fn": tensor2np(fn),
                "tn": tensor2np(tn),
            }
        )

        return loss

    def shared_epoch_end(self, outputs, stage):

        # aggregate step metics
        tp = torch.cat([torch.tensor(x["tp"]) for x in outputs])
        fp = torch.cat([torch.tensor(x["fp"]) for x in outputs])
        fn = torch.cat([torch.tensor(x["fn"]) for x in outputs])
        tn = torch.cat([torch.tensor(x["tn"]) for x in outputs])

        # Per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        if self.run is not None:
            wandb.log(
                {
                    f"{stage}_loss": numpy.mean([x["loss"] for x in outputs]),
                    f"{stage}_tp": tensor2np(tp).mean(),
                    f"{stage}_tn": tensor2np(tn).mean(),
                    f"{stage}_fp": tensor2np(fp).mean(),
                    f"{stage}_fn": tensor2np(fn).mean(),
                    f"{stage}_per_im_iou": per_image_iou,
                    f"{stage}_dataset_iou": dataset_iou,
                }
            )

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def on_training_epoch_start(self):
        super().on_training_epoch_start()
        self.outputs["train"] = []

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_training_epoch_end(self):
        return self.shared_epoch_end(self.outputs["train"], "train")

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.outputs["val"] = []

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end(self.outputs["val"], "val")

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.outputs["test"] = []

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_test_epoch_end(self):
        return self.shared_epoch_end(self.outputs["test"], "test")

    # TODO: Try other optimizers and settings
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])


def run_train(loaders, model, config):

    # TODO: Go through the vast number of options
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        detect_anomaly=False,
    )

    train_data, val_data, _ = loaders
    trainer.fit(
        model,
        train_dataloaders=train_data,
        val_dataloaders=val_data,
    )


if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "rootdir",
        help="Path to images/ and masks/ root dir for visualization",
        type=Path,
    )
    args = parser.parse_args()

    imdir = args.rootdir / "images"
    maskdir = args.rootdir / "masks"

    from loader import SegmentationDataset

    dataset = SegmentationDataset(
        sorted(imdir.glob("*jpg")),
        sorted(maskdir.glob("*png")),
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

    model = SegModel({"architecture": "FPN", "encoder": "resnet18", "lr": 1e-3})
    # TODO: Look at other arguments
    # TODO: Look into learning rate schedulers
    trainer = pl.Trainer(gpus=0, max_epochs=5)
    trainer.fit(model, train_dataloaders=dataloader)
