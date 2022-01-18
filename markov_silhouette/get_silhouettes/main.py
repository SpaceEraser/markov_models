import logging
import os
from random import shuffle, seed
from typing import Iterator
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
from imagehash import dhash

from process_silhouette import image_from_silhouette_url
from image_search import silhouette_urls_from_noun
from noun_gen import noun_gen

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if os.getenv("PYTHONHASHSEED") is not None:
    SEED = int(os.getenv("PYTHONHASHSEED"))
    np.random.seed(SEED)
    seed(SEED)


def silhouette_gen() -> Iterator[np.ndarray]:
    from itertools import islice

    nouns = list(noun_gen())
    # nouns = ["power of attorney"]
    shuffle(nouns)

    logger.debug(f"Loaded {len(nouns)} nouns")

    for noun in nouns:
        logger.debug(f"Using noun {noun}")

        image_urls = silhouette_urls_from_noun(noun)
        images_maybe = (
            image_from_silhouette_url(image_url) for image_url in image_urls
        )
        yield from islice(filter(
            lambda x: x is not None, images_maybe
        ), 5)


def get_next_image_index(folder: str, prefix: str) -> int:
    from itertools import count
    for i in count(1):
        for child in Path(folder).iterdir():
            if child.name == f"{prefix}{i}{child.suffix}":
                break
        else:
            return i


def hash_image(im):
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im, mode='L')
    if not isinstance(im, Image.Image):
        raise Exception("Image must be either numpy.ndarray or PIL.image")
    return dhash(im, hash_size=10)


def is_image_duplicate(im: np.ndarray) -> bool:
    is_image_duplicate.hashes = getattr(is_image_duplicate, "hashes", [])
    imhash = hash_image(im)

    for h in is_image_duplicate.hashes:
        if h - imhash < 6:
            break
    else:
        is_image_duplicate.hashes.append(imhash)
        return False

    return True


def main():
    from itertools import islice
    folder = "../../silhouettes"
    prefix = "image"

    logger.debug("Priming hash DB")
    for child in Path(folder).iterdir():
        try:
            im = Image.open(child)
        except Exception:
            continue
        is_image_duplicate(im)
    logger.debug("Done priming hash DB")

    silhouettes = silhouette_gen()
    silhouettes = (im for im in silhouettes if not is_image_duplicate(im))

    for image in islice(silhouettes, 1000):
        i = get_next_image_index(folder, prefix)
        filename = f"{folder}/{prefix}{i}.png"
        cv.imwrite(filename, image, [cv.IMWRITE_PNG_COMPRESSION, 9])
        logger.debug(f"Wrote out {filename}")


if __name__ == "__main__":
    main()
