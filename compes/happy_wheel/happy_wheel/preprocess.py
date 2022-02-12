"""preprocess.py
preprocess dataset (image).
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from . import DATA_ROOT, RAW_TEST_DATA_DIR, RAW_TRAIN_DATA_DIR, logger

MAX_IMAGE_WIDTH = 768
MAX_IMAGE_HEIGHT = 512


def run(dirname: str = "preprocessed"):
    """Run preprocessing image."""
    logger.info("Start preprocessing")
    output_root = DATA_ROOT / dirname
    preprocess(RAW_TRAIN_DATA_DIR, output_root / "train_images")
    preprocess(RAW_TEST_DATA_DIR, output_root / "test_images")
    logger.info("Finish preprocessing")


def preprocess(input_dir: Path, output_dir: Path):
    """Preprocess images in `input_dir` and save to `output_dir`.
    Resize image so that image width <= max_width (768) and image height <=  max_height (512).
    """
    images = sorted(input_dir.glob("**/*.jpg"))
    with ThreadPoolExecutor(16) as executor:
        executor.map(lambda x: _resize_and_save(x, input_dir, output_dir), images)


def _resize_and_save(img_path: Path, input_dir: Path, output_dir: Path):
    """ """
    if not img_path.is_file():
        logger.warning(f"File does not exist : {img_path}. Skip preprocessing.")
        return
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning(f"Failed to load image : {img_path}. Skip preprocesssing.")
        return

    h, w, _ = img.shape
    scale = min(MAX_IMAGE_HEIGHT / h, MAX_IMAGE_WIDTH / w)
    dst_w = min(int(w * scale), MAX_IMAGE_WIDTH)
    dst_h = min(int(h * scale), MAX_IMAGE_HEIGHT)

    dst_img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)

    output_path = output_dir / img_path.relative_to(input_dir)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output_path), dst_img)
    logger.debug(f"Save preprocessed image to {output_path}.")


if __name__ == "__main__":
    run()
