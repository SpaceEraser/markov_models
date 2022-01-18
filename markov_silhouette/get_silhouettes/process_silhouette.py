import logging
from typing import *

import cv2 as cv
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TARGET_SIZE = 1000
ASPECT_RATIO_CUTOFF = 0.1
SILHOUETTE_RANGE_CUTOFF = 10
CONTOUR_AREA_PERCENT_CUTOFF = 0.01
CONTOUR_REDUCTION_AREA_PERCENT_CUTOFF = 0.8


def has_narrow_parts(im: np.ndarray) -> bool:
    from shapely.geometry import Polygon, MultiPolygon

    def get_polygon(cont, hier, index) -> Optional[Polygon]:
        parent = cont[index].squeeze()
        if len(parent) < 3:
            return None
        children = []
        index = hier[index, 2]
        while index >= 0:
            c = cont[index].squeeze()
            if len(c) >= 3:
                children.append(c)
            index = hier[index, 0]
        return Polygon(parent, children)

    def remove_dups(nparr: np.ndarray) -> np.ndarray:
        i = 0
        j = 1
        out = []
        while i < len(nparr):
            if i + j != len(nparr) and np.all(nparr[i] == nparr[i + j]):
                j += 1
                continue
            out.append(nparr[i])
            i += j
            j = 1
        return np.array(out, dtype=nparr.dtype)

    contours, hierarchy = cv.findContours(255 - im, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if contours is None or hierarchy is None or len(hierarchy) == 0:
        logger.debug("Failed to find image contours")
        return True
    hierarchy = hierarchy[0]

    mp = []
    for i in range(len(contours)):
        if hierarchy[i, 3] != -1:
            continue
        poly = get_polygon(contours, hierarchy, i)
        if poly is None:
            continue
        mp.append(poly)
    if len(mp) == 0:
        logger.debug("No valid contours found for image")
        return True
    mp = MultiPolygon(mp)
    error = mp.buffer(0.0001).simplify(0.0001).difference(mp.buffer(-10).buffer(10)).area

    return error / mp.area > 0.005


def trim_image(im: np.ndarray) -> np.ndarray:
    startmask = im == im[0, 0]
    endmask = im == im[-1, -1]
    sr = np.argmax(~np.all(startmask, axis=1))
    er = im.shape[0] - np.argmax(~np.all(endmask, axis=1)[::-1])
    sc = np.argmax(~np.all(startmask, axis=0))
    ec = im.shape[1] - np.argmax(~np.all(endmask, axis=0)[::-1])
    return im[sr:er, sc:ec]


def trim_image_completely(im):
    while True:
        old_im_shape = im.shape
        im = trim_image(im)
        if old_im_shape == im.shape:
            break
    return im


def get_image_edges(im: np.ndarray) -> np.ndarray:
    return np.concatenate([im[0, :-1], im[:-1, -1], im[-1, 1:], im[1:, 0]])


def recolor_for_dominant(im: np.ndarray) -> np.ndarray:
    edges = get_image_edges(im)
    num_whites = np.count_nonzero(edges)
    num_blacks = len(edges) - num_whites

    if num_blacks > num_whites:
        logger.debug("Switching colors, image edges are black dominant")
        return 255 - im
    return im


def is_silhouette(im: np.ndarray) -> bool:
    # TODO: allow any image with mainly 2 colors (not just black and white)
    #       idea: do proper bimodal check, get location of 2 peaks, threshold on point exactly between them
    try:
        low, high = np.sort(
            KMeans(n_clusters=2, n_init=1).fit(im.reshape(-1, 1)).cluster_centers_.squeeze()
        )
        logger.debug(f"Image cluster center is {(low, high)}")
    except Exception as e:
        logger.error(f"Image clustering failed: {e}")
        return False

    return low <= SILHOUETTE_RANGE_CUTOFF and high >= 255 - SILHOUETTE_RANGE_CUTOFF


def remove_small_parts(im: np.ndarray) -> np.ndarray:
    # label and collect stats on CCs of inverted image
    num_cc, labelled_im, stats, _centroids = cv.connectedComponentsWithStats(255 - im, connectivity=4)

    # count number of small area CCs
    small_cc_labels = [i for i in range(num_cc) if stats[i, cv.CC_STAT_AREA] < 15 * 15]

    # remove small CCs
    for i in small_cc_labels:
        im[labelled_im == i] = 255

    return im


def is_silhouette_simple(im: np.ndarray) -> bool:
    def count_holes(hier, i) -> int:
        i = hier[i, 2]
        n = 0
        while i >= 0:
            n += 1
            i = hier[i, 0]
        return n

    def contour_area(cont, hier, i) -> Tuple[float, float]:
        outer_area = cv.contourArea(cont[i])
        i = hier[i, 2]
        children_area = 0
        while i >= 0:
            children_area += cv.contourArea(cont[i])
            i = hier[i, 0]
        return outer_area, children_area

    contours, hierarchy = cv.findContours(255 - im, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    if contours is None or hierarchy is None or len(hierarchy) == 0:
        return False

    hierarchy = hierarchy[0]
    num_contours = len(hierarchy)
    holes = [count_holes(hierarchy, i) for i in range(len(hierarchy))]
    parent_child_areas = [contour_area(contours, hierarchy, i) for i in range(len(contours))]

    # imc = np.stack([im] * 3, axis=-1)
    # for i in range(len(contours)):
    #     if hierarchy[i,3] == -1:
    #         c = (0,0,255)
    #     else:
    #         c = (0,255,0)
    #     imc = cv.drawContours(imc, [contours[i]], 0, color=c, thickness=2)
    #     imc = cv.putText(imc, str(parent_child_areas[i][0]), contours[i][0][0], cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    #
    # print(f"{holes = }")
    # print(f"{parent_child_areas = }")
    # cv.imshow("contours", imc)
    # cv.waitKey()

    children_not_too_prominent = all(c / p < 0.2 for p, c in parent_child_areas if p > 0.0)
    parents_dont_have_too_many_holes = all(h <= 1 for h in holes)
    return num_contours <= 2 and parents_dont_have_too_many_holes and children_not_too_prominent


def make_silhouette_well_formed(im: np.ndarray) -> np.ndarray:
    """
    Smooth and threshold image so that only 2 colors are present.
    """
    assert im.ndim == 2

    # smooth image while preserving edges
    proc_im = cv.bilateralFilter(im, 9, 75, 75)

    # threshold
    proc_im[proc_im > 127] = 255
    proc_im[proc_im <= 127] = 0

    return proc_im


def download_img(url: str) -> Optional[Image.Image]:
    import io
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=5) as req:
            byte_arr = bytearray(req.read())
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    try:
        return Image.open(io.BytesIO(byte_arr))
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")


def grayscale_img(im: Image.Image) -> np.ndarray:
    if im.mode != 'RGBA':
        return np.asarray(im.convert('L'), dtype=np.uint8)
    im = np.asarray(im, dtype=np.uint8).astype(float)
    a = im[:, :, 3:] / 255
    im = im[:, :, :3]
    im = a * im + (1 - a) * 255
    return np.asarray(Image.fromarray(im.astype(np.uint8), mode='RGB').convert('L'), dtype=np.uint8)


def rescale_grayscale_image(im: np.ndarray) -> np.ndarray:
    scaled = im.astype(float)
    mi, ma = np.min(scaled), np.max(scaled)
    if mi == ma:
        return im
    scaled = ((scaled - mi) / (ma - mi)) * 255
    return scaled.astype(im.dtype)


def resize_image(im: np.ndarray) -> np.ndarray:
    if im.shape[0] == TARGET_SIZE and im.shape[1] == TARGET_SIZE:
        return im

    old_size = im.shape
    ratio = TARGET_SIZE / np.max(im.shape)
    im = cv.resize(im, dsize=None, fx=ratio, fy=ratio, interpolation=cv.INTER_AREA)
    logger.debug(f"Resized image: {old_size} -> {im.shape}")
    return im


def is_image_too_small(im: np.ndarray) -> bool:
    return im.shape[0] < TARGET_SIZE / 3 or im.shape[1] < TARGET_SIZE / 3


def check_preprocess_image(im: np.ndarray) -> Optional[np.ndarray]:
    if not is_silhouette(im):
        logger.debug("Downloaded image is not a silhouette")
        return None

    if is_image_too_small(im):
        logger.debug(f"Image is too small: {im.shape[0]}x{im.shape[1]}")
        return None

    im = resize_image(im)

    im = make_silhouette_well_formed(im)

    im = remove_small_parts(im)
    im = 255 - remove_small_parts(255 - im)

    im = trim_image_completely(im)
    im = recolor_for_dominant(im)

    if is_image_too_small(im):
        logger.debug(f"Image is too small after cleaning: {im.shape[0]}x{im.shape[1]}")
        return None

    im = resize_image(im)

    if has_narrow_parts(im):
        logger.debug("Silhouette has too many narrow parts")
        return None

    if np.all(im == im[0, 0]):
        logger.debug("Image is empty")
        return None

    if not is_silhouette_simple(im):
        logger.debug("Silhouette has too many parts, or too many holes in parts, or too big big holes in parts")
        return None

    num_white = np.count_nonzero(im)
    if num_white / im.size < 0.2:
        logger.debug(f"White is not prominent enough: {num_white} / {im.size} < 0.2")
        return None

    aspect_ratio = im.shape[0] / im.shape[1]
    if aspect_ratio < ASPECT_RATIO_CUTOFF or aspect_ratio > 1.0 / ASPECT_RATIO_CUTOFF:
        logger.debug(f"Image aspect ratio is unacceptable {aspect_ratio}")
        return None

    return im


def image_from_silhouette_url(url: str) -> Optional[np.ndarray]:
    logger.debug(f"Downloading image from {url}")
    im = download_img(url)
    if im is None:
        logger.debug("Image failed to download")
        return None
    try:
        im = grayscale_img(im)
    except Exception as e:
        logger.debug(f"Failed to grayscale image: {e}")
        return None
    im = rescale_grayscale_image(im)
    im = check_preprocess_image(im)

    if im is None:
        logger.debug(f"Image failed preprocessing checks")
        return None

    return im


def main():
    for name in ["../../silhouettes/image1.png", "../../silhouettes/image5.png", "../../silhouettes/image4.png"]:
        im = Image.open(name)
        # cv.imshow(f"before {name}", np.asarray(im, dtype=np.uint8))
        im = grayscale_img(im)
        cv.imshow(f"removed {name}", im)
        cv.waitKey()
        cv.destroyAllWindows()
        # im = check_preprocess_image(im)
        # if im is None:
        #     print("Silhouette failed checking and processing")
        # else:
        #     cv.imshow(f"after {name}", im)
        # cv.waitKey()
        # cv.destroyAllWindows()


if __name__ == "__main__":
    main()
