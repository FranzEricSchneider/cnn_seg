"""
Take a dump of image and black/white mask files, then convert them to a
train/test/val dataset of the right size and type. The final type should have
.jpg images of a size divisible by 32, and .npy boolean masks of the same size.
We will assume on the input side that the images/masks will have corresponding
names with an added flag in the mask name:
    image_0001.jpg
    image_mask_0001.jpg
That way you can provide the flag ('_mask' in this case) and the code will
first look for all files matching the flags (*_mask*.jpg) and then go one by
one and find the images with names where the flag is removed.
"""

import argparse
import cv2
from pathlib import Path
import numpy
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    ANTIALIAS = Image.Resampling.LANCZOS
except AttributeError:
    ANTIALIAS = Image.ANTIALIAS


def process(savedir, mpath, flag, size):
    """
    Take in a directory and a mask file, then save a resized version of the
    mask and image in the appropriate types (.npy boolean mask and .jpg image).

    NOTE: For right now, dark in the mask images will be treated as True, this
    could be parametrized later.

    Arguments:
        savedir: Path() to a directory we want to save into
        mpath: Path() to a mask file
        flag: Part of the mask path name that we can remove to get the
            corresponding image. E.g.
                render_0001.jpg         [image]
                render_bw_0001.jpg    [mask]
                flag: '_bw'
        size: (width, height) that we want to resize the images to
    """
    mask = resize(mpath, size).mean(axis=2)
    boolmask = numpy.zeros(mask.shape, dtype=bool)
    boolmask[mask < 128] = True
    numpy.save(savedir / "masks" / mpath.stem, boolmask)

    impath = mpath.parent / mpath.name.replace(flag, "")
    resize(impath, size, savepath=savedir / "images" / impath.name)


def resize(impath, size, savepath=None):
    resized = Image.open(impath).resize(size, ANTIALIAS)
    if savepath is not None:
        resized.save(savepath)
    else:
        return numpy.array(resized)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--datadir",
        help="Path to where all the images and mask files are found",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--savedir",
        help="Path to where we will save the reworked files in train/val/test."
        " It is assumed that this path will not yet exist.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--maskflag",
        help="String component of the file names that is unique to mask files"
        " and, when removed, gives you the name of the corresponding image. See"
        " the assumptions in the intro text. IF THIS IS NONE then we assume"
        " there are no masks and we must generate empty mask files so that"
        " unlabled images can be evaluated.",
    )
    parser.add_argument(
        "--valfrac",
        help="Fraction of files to use for validation.",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--testfrac",
        help="Fraction of files to use for test.",
        type=float,
        default=0.2,
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
    parser.add_argument(
        "--random-seed",
        help="Random seed to use for file splits.",
        type=int,
        default=12345,
    )
    args = parser.parse_args()

    # Gather all of the masks
    assert args.datadir.is_dir(), f"{args.datadir} not found"

    # If there are no masks, generate empty ones so that we can evaluate
    # unlabeled images
    maskflag = args.maskflag
    if maskflag is None:
        maskflag = "_empty-mask"
        for impath in args.datadir.glob(f"*jpg"):
            cv2.imwrite(
                str(impath.with_name(f"{impath.stem}{maskflag}.jpg")),
                numpy.zeros((args.height, args.width, 3), dtype=numpy.uint8),
            )

    all_masks = sorted(args.datadir.glob(f"*{maskflag}*jpg"))
    assert len(all_masks) > 0, f"No files found with f{args.datadir}/*{maskflag}*jpg"

    # Gather all of the corresponding image files
    all_images = [mask.parent / mask.name.replace(maskflag, "") for mask in all_masks]
    for impath in all_images:
        assert impath.is_file(), f"{impath} expected but not found"

    # Split the lists into train/val/test
    for fraction in [args.valfrac, args.testfrac]:
        assert 0.0 < fraction < 1.0, f"Fraction {fraction} should be 0-1"
    maskpaths = {}
    maskpaths["train"], other = train_test_split(
        all_masks,
        test_size=args.valfrac + args.testfrac,
        random_state=args.random_seed,
        shuffle=True,
    )
    maskpaths["val"], maskpaths["test"] = train_test_split(
        other,
        test_size=args.testfrac / (args.valfrac + args.testfrac),
        random_state=args.random_seed,
        shuffle=True,
    )

    # Build the directory we want to save files in
    assert not args.savedir.is_dir(), f"{args.savedir} should not exist yet"
    args.savedir.mkdir()
    for subdir in maskpaths.keys():
        (args.savedir / subdir).mkdir()
        for subsubdir in ("images", "masks"):
            (args.savedir / subdir / subsubdir).mkdir()

    assert args.width % 32 == 0, f"Width {args.width} needs to be divisible by 32"
    assert args.height % 32 == 0, f"Height {args.height} needs to be divisible by 32"
    # Load the masks and images, resize them, and save them
    for subdir, mpaths in maskpaths.items():
        print(f"{subdir}:")
        for mpath in tqdm(mpaths):
            process(
                savedir=args.savedir / subdir,
                mpath=mpath,
                flag=maskflag,
                size=(args.width, args.height),
            )


if __name__ == "__main__":
    main()
