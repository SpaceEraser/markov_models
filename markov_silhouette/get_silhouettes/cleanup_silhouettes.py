import os
from pathlib import Path
from typing import *
from PIL import Image
import numpy as np
from main import hash_image
import logging

from markov_silhouette.get_silhouettes.process_silhouette import grayscale_img, rescale_grayscale_image, \
    check_preprocess_image

logging.getLogger("PIL.Image").setLevel(logging.ERROR)


def cleanup_image(path) -> Optional[np.ndarray]:
    try:
        im = Image.open(path)
    except Exception:
        return None
    try:
        im = grayscale_img(im)
    except Exception:
        return None
    im = rescale_grayscale_image(im)
    im = check_preprocess_image(im)

    if im is None:
        return None

    return im


def main():
    folder = "../../silhouettes"
    files = list(p for p in Path(folder).iterdir() if p.name.endswith(".png"))

    hashes = []
    failures = []
    duplicates = []
    changed = []
    for child in files:
        try:
            old_im = np.asarray(Image.open(child), dtype=np.uint8)
        except Exception:
            continue

        im = cleanup_image(child)
        if im is None:
            failures.append(child)
            print(f"{child} failed checks")
            continue

        try:
            imhash = hash_image(im)
        except Exception:
            continue

        for p, h in hashes:
            diff = h - imhash
            if diff < 6:
                break
        else:
            hashes.append((child, imhash))
            if old_im.shape != im.shape or np.any(old_im != im):
                changed.append((child, old_im, im))
            continue
        duplicates.append(child)
        print(f"{child} is a duplicate of {p} (d={diff})")

    print(f"failures ({len(failures)}/{len(files)}):")
    for p in failures:
        print(f"    {p}", end='')
        os.remove(p)
        print("... removed")

    print(f"duplicates  ({len(duplicates)}/{len(files)}):")
    for p in duplicates:
        print(f"    {p}", end='')
        os.remove(p)
        print("... removed")

    print(f"changed  ({len(changed)}/{len(files)}):")
    for p, _old, new in changed:
        print(f"    {p}", end='')
        Image.fromarray(new, mode='L').save(p)
        print("... saved")


if __name__ == "__main__":
    main()
