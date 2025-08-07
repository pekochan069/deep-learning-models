import cv2
from cv2.typing import MatLike


def bucket_crop_image(image: MatLike, width: int, height: int) -> list[MatLike]:
    raw_h, raw_w = image.shape[:2]
    h = (raw_h // height) * height
    w = (raw_w // width) * width
    start_x = (raw_w - w) // 2
    start_y = (raw_h - h) // 2
    return [
        image[y : y + height, x : x + x + width]
        for y in range(start_y, h + start_y, height)
        for x in range(start_x, w + start_x, width)
    ]


def quad_crop_image(image: MatLike, width: int, height: int) -> list[MatLike]:
    h, w = image.shape[:2]

    if width * 2 < w and height * 2 < h:
        # crop center of the image into 4 quadrants
        # center - width / 2 : center + width / 2
        mid_y = h // 2
        mid_x = w // 2
        return [
            image[
                mid_y - height // 2 : mid_y + height // 2,
                mid_x - width // 2 : mid_x + width // 2,
            ]
        ]

    else:
        return [cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)]
