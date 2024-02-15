import json
from matplotlib import pyplot
import numpy
import torch
import wandb

import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from utils import tensor2np


# Inspired by
# https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb
class SegModel(pl.LightningModule):
    def __init__(self, config, run=None, **kwargs):

        # All of the necessary arguments as a dictionary
        self.config = config
        # Either None or a wandb run that can be used to track stats
        self.run = run

        super().__init__()
        self.model = smp.create_model(
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

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

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

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Let's compute metrics for some threshold first convert mask values to
        # probabilities, then apply thresholding
        pred_mask = (logits_mask.sigmoid() > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.int(), mask.int(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

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

    # TODO: Try removing the stage from shared_step
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    # TODO: Try other optimizers and settings
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])


def run_train(loaders, model, config):

    # TODO: Go through the vast number of options
    trainer = pl.Trainer(
        # TODO: Check if this is True
        gpus=0,
        max_epochs=config["epochs"],
        detect_anomaly=True,
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
