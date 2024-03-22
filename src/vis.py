import cv2
import numpy
from pathlib import Path


def torch_img_to_array(torch_img, sigma=3):
    # Convert torch to numpy
    float_image = torch_img.movedim(0, -1).detach().cpu().numpy()
    # If torch images are normalized, then we can
    # 1) scale that down to get a certain number of sigmas into -0.5 - 0.5
    # 2) add 0.5 so we're from 0 - 1
    # 3) clip so the high-sigma values are maxed at 0 and 1 instead of clipping
    if float_image.min() < 0:
        float_image = numpy.clip((float_image / (2 * sigma)) + 0.5, 0, 1)
    uint8_image = (float_image * 255).astype(numpy.uint8)
    # Convert RGB and grayscale to BGR
    if uint8_image.shape[2] == 3:
        uint8_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
    else:
        uint8_image = numpy.dstack([uint8_image] * 3)
    return uint8_image


def save_debug_images(
    savedir,
    torch_imgs,
    torch_masks,
    impaths=None,
    prefix="debug_",
):
    # Enforce this
    if impaths is not None:
        impaths = [Path(impath) for impath in impaths]

    new_impaths = []

    for i, (img, mask) in enumerate(zip(torch_imgs, torch_masks)):
        uint8_image = numpy.hstack(
            [
                torch_img_to_array(img),
                torch_img_to_array(mask),
            ]
        )
        if impaths is not None:
            uint8_image = numpy.hstack((cv2.imread(str(impaths[i])), uint8_image))
            name = prefix + impaths[i].name
        else:
            name = prefix + f"img{i:03}.jpg"
        new_impaths.append(savedir / name)
        cv2.imwrite(str(new_impaths[-1]), uint8_image)
    return new_impaths
