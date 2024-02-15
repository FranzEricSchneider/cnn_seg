"""
Helper functions to build a dataset
"""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize(root, size):

    for name, globtxt in (("images", "*jpg"), ("masks", "*png")):
        for impath in tqdm(sorted((root / name).glob(globtxt))):
            image = Image.open(impath)
            resized = image.resize(size, Image.ANTIALIAS)
            resized.save(impath)

    print(f"Done reformatting {root.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--rootdir",
        help="Path to images/ and masks/ root dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-H",
        "--height",
        help="New image height",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-W",
        "--width",
        help="New image width",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    assert args.width % 32 == 0
    assert args.height % 32 == 0

    resize(args.rootdir, (args.width, args.height))
