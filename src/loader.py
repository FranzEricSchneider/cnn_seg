import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToPILImage, ToTensor

# Inspired by
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
class SegmentationDataset(Dataset):
    def __init__(self, impaths, maskpaths, transforms):
        # NOTE: when iterated through, impaths and maskpaths must correspond
        self.impaths = impaths
        self.maskpaths = maskpaths
        if transforms is None:
            self.transforms = Compose(
                [
                    ToPILImage(),
                    ToTensor(),
                ]
            )
        else:
            self.transforms = transforms

        assert len(impaths) == len(maskpaths)

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):

        # Load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.cvtColor(cv2.imread(str(self.impaths[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.maskpaths[idx]), 0)

        # Check to see if we are applying anything to both image and its mask
        image = self.transforms(image)
        mask = self.transforms(mask)

        # Return a tuple of the image and its mask
        return (image, mask)


def tensor2np(tensor):
    return tensor.detach().cpu().numpy()


# Test out the loader
if __name__ == "__main__":

    from pathlib import Path

    root = Path("/home/fschneider/Downloads/SIMSEG/")
    imdir = root / "images"
    maskdir = root / "masks"

    dataset = SegmentationDataset(
        sorted(imdir.glob("*jpg")),
        sorted(maskdir.glob("*png")),
        transforms=None,
    )

    import os

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
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
