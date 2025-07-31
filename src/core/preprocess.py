from cv2.typing import MatLike


def bucket_crop_image(image: MatLike, width: int, height: int) -> list[MatLike]:
    h, w = image.shape[:2]
    h = (h // height) * height
    w = (w // width) * width
    return [
        image[y : y + height, x : x + width]
        for y in range(0, h, height)
        for x in range(0, w, width)
    ]
