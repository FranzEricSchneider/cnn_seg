"""
Dataloader and related tools
"""

import cv2
import json
import numpy
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import wandb

from utils import tensor2np


TMPDIR = Path("/tmp/")


# TODO: Check out expanded transforms:
# https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
def build_transform(augpath):
    assert augpath.is_file()

    def catch_cases(kwargs):
        if "dtype" in kwargs and kwargs["dtype"] == "torch.float32":
            kwargs["dtype"] = torch.float32
        return kwargs

    return v2.Compose(
        [
            getattr(v2, name)(**catch_cases(kwargs))
            for name, kwargs in json.load(augpath.open("r"))
        ]
    )


# TODO: Add a debug step where we grab an image from each dataset or something
def get_loaders(config):

    loaders = []
    for subdir, shuffle in (("train", True), ("val", False), ("test", False)):

        augpath = Path(config[f"{subdir}_augmentation_path"])

        imdir = config["data_dir"] / subdir / "images"
        maskdir = config["data_dir"] / subdir / "masks"

        dataset = SegmentationDataset(
            sorted(imdir.glob("*.jpg")),
            sorted(maskdir.glob("*.npy")),
            transforms=build_transform(augpath),
        )

        loaders.append(
            DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=shuffle,
                num_workers=os.cpu_count() // 2,
            )
        )

        if config["wandb"]:
            # Save our augmentation file
            wandb.save(str(augpath))
            # Save a list of all of the files for this stage (e.g. train/test)
            path = Path(TMPDIR).joinpath(f"{subdir}_files.json")
            json.dump(
                sorted(
                    [
                        (impath.name, maskpath.name)
                        for impath, maskpath in zip(dataset.impaths, dataset.maskpaths)
                    ]
                ),
                path.open("w"),
                indent=4,
            )
            wandb.save(str(path))

    return loaders


# Inspired by
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
class SegmentationDataset(Dataset):
    def __init__(self, impaths, maskpaths, transforms):
        """
        Arguments:
            impaths: list of Path objects for images openable with cv2.imread
            maskpaths: list of Path objects for masks (.npy boolean arrays,
                openable with numpy.load)
            transforms: None (default transforms will be used) or a v2.Compose
                object with sequenced augmentations

        NOTE: when iterated through, impaths and maskpaths must correspond
        """
        self.impaths = impaths
        self.maskpaths = maskpaths
        assert len(impaths) == len(maskpaths)

        if transforms is None:
            tlist = [v2.ToPILImage()]
            try:
                tlist += [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            except AttributeError:
                # Handle older torchvision
                tlist += [v2.ToTensor()]
            self.transforms = v2.Compose(tlist)
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):

        # Load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask, then convert it to 0-255 image type
        image = cv2.cvtColor(cv2.imread(str(self.impaths[idx])), cv2.COLOR_BGR2RGB)
        mask = (numpy.load(self.maskpaths[idx]) * 255).astype(numpy.uint8)

        # TODO: We're going to need to split space-based transformations that
        # should apply to both, and color-based transformations which shouldn't
        image, mask = self.transforms(image, mask)

        # Return a tuple of the image and its mask
        return (image, mask)


# Test out the loader
if __name__ == "__main__":

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Visualize a random image/mask from the rootdir",
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

    dataset = SegmentationDataset(
        sorted(imdir.glob("*jpg")),
        sorted(maskdir.glob("*npy")),
        transforms=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=1,
    )

    from matplotlib import pyplot

    sample = dataset[0]
    pyplot.subplot(1, 2, 1)
    # Change from CHW to HWC
    pyplot.imshow(tensor2np(sample[0]).transpose(1, 2, 0))
    pyplot.subplot(1, 2, 2)
    # For visualization we have to remove 3rd dimension of mask
    pyplot.imshow(tensor2np(sample[1]).squeeze())
    pyplot.show()
