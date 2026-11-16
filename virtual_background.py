import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from template_mask import TEMPLATE
from train.train import UNet, SIZE, MEAN, STD

BACKGROUND = "banarange.png"
WEIGHTS = "runs/full1/best.pt"

detector = YOLO("yolov8n.pt")
unet = UNet(pretrained=False)
unet.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
unet.eval()


def detect_people(frame):
    """YOLO person boxes as (x0, y0, x1, y1), clipped to the frame."""
    result = detector(frame, device="cpu", classes=[0], imgsz=320, verbose=False)[0]
    h, w = frame.shape[:2]
    boxes = []
    for x0, y0, x1, y1 in result.boxes.xyxy.int().tolist():
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 - x0 > 10 and y1 - y0 > 10:
            boxes.append((x0, y0, x1, y1))
    return boxes


# ---------- the three methods: frame + box -> mask of the box (float, 0..1) ----------

def mask_template(frame, box):
    x0, y0, x1, y1 = box
    return cv2.resize(TEMPLATE, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def mask_grabcut(frame, box):
    x0, y0, x1, y1 = box
    # GrabCut is slow: run it on a small copy of the frame
    scale = 240 / frame.shape[1]
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    h, w = small.shape[:2]
    rx0, ry0 = max(int(x0 * scale), 1), max(int(y0 * scale), 1)
    rx1, ry1 = min(int(x1 * scale), w - 1), min(int(y1 * scale), h - 1)

    # outside the box = sure background, template core = probably person, rest of the box = probably background
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[ry0:ry1, rx0:rx1] = cv2.GC_PR_BGD
    core = cv2.resize(TEMPLATE, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_NEAREST)
    gc[ry0:ry1, rx0:rx1][core == 1] = cv2.GC_PR_FGD
    cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK)

    person = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    return cv2.resize(person, (frame.shape[1], frame.shape[0]))[y0:y1, x0:x1]


def mask_model(frame, box):
    x0, y0, x1, y1 = box
    crop = cv2.resize(frame[y0:y1, x0:x1], (SIZE, SIZE))
    crop = (crop[:, :, ::-1] / 255.0 - MEAN) / STD    # BGR -> RGB, normalize
    x = torch.from_numpy(crop.transpose(2, 0, 1).astype(np.float32))[None]
    with torch.no_grad():
        prob = torch.sigmoid(unet(x))[0, 0].numpy()
    return cv2.resize(prob, (x1 - x0, y1 - y0))


METHODS = {"1": ("template", mask_template), "2": ("grabcut", mask_grabcut), "3": ("model", mask_model)}


def replace_background(frame, background, method):
    boxes = detect_people(frame)
    out = background.copy()
    if not boxes:
        return out
    alpha = np.zeros(frame.shape[:2], np.float32)
    for box in boxes:
        x0, y0, x1, y1 = box
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], method(frame, box))
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)[:, :, None]    # soften the edge

    # blend only inside the area that contains people, the rest is pure background
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    a = alpha[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = (a * frame[y0:y1, x0:x1] + (1 - a) * background[y0:y1, x0:x1]).astype(np.uint8)
    return out


def run_webcam():
    cam = cv2.VideoCapture(0)
    background, key = None, "3"
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        if background is None:
            background = cv2.resize(cv2.imread(BACKGROUND), (frame.shape[1], frame.shape[0]))

        t0 = time.time()
        name, method = METHODS[key]
        out = replace_background(frame, background, method)
        fps = 1 / (time.time() - t0)
        cv2.putText(out, f"{name}  {fps:.0f} fps   [1] template  [2] grabcut  [3] model  [q] quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("virtual background", out)

        pressed = chr(cv2.waitKey(1) & 0xFF)
        if pressed == "q":
            break
        if pressed in METHODS:
            key = pressed
    cam.release()
    cv2.destroyAllWindows()


def run_image(path):
    # test on one photo: saves the result of each method
    frame = cv2.imread(path)
    background = cv2.resize(cv2.imread(BACKGROUND), (frame.shape[1], frame.shape[0]))
    for name, method in METHODS.values():
        t0 = time.time()
        out = replace_background(frame, background, method)
        print(f"{name}: {time.time() - t0:.2f}s")
        cv2.imwrite(f"result_{name}.png", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_image(sys.argv[1])
    else:
        run_webcam()
