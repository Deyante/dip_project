import glob
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

STORE = 320      # stored crop size
MARGIN = 0.2     # margin around the person box, must be larger than the training box jitter
OUT = "/kaggle/working/crops.pkl"


def find_clip_dir():
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "clip_img" in dirs:
            return os.path.join(root, "clip_img")
        dirs[:] = [d for d in dirs if not d.startswith(("matting", "clip_"))]


def crop(img, x0, y0, x1, y1):
    # crop with zero padding where the box leaves the image
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def process(img_path):
    mask_path = (img_path.replace("/clip_img/", "/matting/")
                 .replace("/clip_", "/matting_").replace(".jpg", ".png"))
    try:
        img = cv2.imread(img_path)
        alpha = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        alpha = alpha[:, :, 3] if alpha.ndim == 3 else alpha
        ys, xs = np.where(alpha > 127)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    except Exception:
        return None   # missing or broken file, or empty mask

    # person box + margin, resized to STORE x STORE
    h_img, w_img = img.shape[:2]
    w, h = x1 - x0, y1 - y0
    bx0, by0 = int(x0 - MARGIN * w), int(y0 - MARGIN * h)
    bx1, by1 = int(x1 + MARGIN * w), int(y1 + MARGIN * h)
    sx, sy = STORE / (bx1 - bx0), STORE / (by1 - by0)
    img = cv2.resize(crop(img, bx0, by0, bx1, by1), (STORE, STORE), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(crop(alpha, bx0, by0, bx1, by1), (STORE, STORE), interpolation=cv2.INTER_AREA)

    # person box and original image borders, in the stored crop's coordinates
    to_crop = lambda x, y: ((x - bx0) * sx, (y - by0) * sy)
    box = to_crop(x0, y0) + to_crop(x1, y1)
    bounds = to_crop(0, 0) + to_crop(w_img, h_img)
    return {
        "session": img_path.split("/clip_img/")[1].split("/")[0],
        "img": cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(),
        "mask": cv2.imencode(".png", alpha)[1].tobytes(),
        "box": box,
        "bounds": bounds,
    }


def main():
    t0 = time.time()
    clip_dir = find_clip_dir()
    paths = sorted(glob.glob(f"{clip_dir}/*/*/*.jpg"))
    print(f"{len(paths)} images in {clip_dir}  ({time.time() - t0:.0f}s)")

    samples = []
    with ThreadPoolExecutor(64) as pool:   # reading from Kaggle storage is slow, so read many files at once
        for i, s in enumerate(pool.map(process, paths)):
            if s is not None:
                samples.append(s)
            if i % 2000 == 0:
                print(f"{i}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT, "wb") as f:
        pickle.dump(samples, f)
    print(f"saved {len(samples)} crops, {os.path.getsize(OUT) / 1e9:.2f} GB  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
