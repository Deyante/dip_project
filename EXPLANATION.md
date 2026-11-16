# Virtual background: what has been done, step by step

## How to run it

```
python virtual_background.py                 # webcam: keys 1/2/3 switch method, q quits
python virtual_background.py some_photo.jpg  # one photo: saves result_template/grabcut/model.png
```

The background image is `banarange.png` (change `BACKGROUND` at the top of `virtual_background.py`).
Packages needed: `opencv-python`, `numpy`, `torch`, `torchvision`, `ultralytics`.

## The pipeline (one webcam frame)

1. **Read a frame** from the webcam (`cv2.VideoCapture(0)`).
2. **Find the people** with YOLOv8-nano (`detect_people`). This is the detection step from
   `main2.py` in the dip_project repo: `YOLO('yolov8n.pt')` run on CPU. We keep only the
   "person" class and get one box `(x0, y0, x1, y1)` per person. The scene-description parts of
   `main2.py` (prompts, Ollama, tracker, side panel) are not used.
3. **Turn each box into a mask** with one of three methods (below). Every method has the same
   form: `mask = method(frame, box)`, a mask the size of the box with values from 0 (background)
   to 1 (person).
4. **Combine the masks** of all people into one full-frame mask (per-pixel maximum) and blur it
   slightly so the edge is soft.
5. **Composite**: `output = mask * frame + (1 - mask) * background`. Everything outside the boxes
   is pure background.
6. **Show** the result with the method name and the frames per second.

## The three methods

| Key | Method | How it works | Code |
|---|---|---|---|
| 1 | **Template** (naive mask) | A fixed head-and-shoulders shape, drawn once, is stretched to the box. It ignores the image content. | `template_mask.py`, `mask_template` |
| 2 | **GrabCut** (Rother et al. 2004) | Classical, no training. Pixels outside the box are background for sure; the template's shape is the first guess for the person. GrabCut learns the colors of person and background (Gaussian mixtures) and cuts between them with a graph cut. Run on a 240-pixel-wide copy of the frame for speed. | `mask_grabcut` |
| 3 | **Our trained model** | The box is cropped, resized to 256x256 and passed through our U-Net, which gives the probability of "person" for each pixel. | `mask_model`, `train/train.py` |

## How the model was trained

1. **Data**: AISegment Matting Human dataset (34,425 portrait photos with human-made masks), on Kaggle.
2. **Preparation** (`prep/prep.py`, a CPU job on Kaggle): for each photo, cut out the person's box
   with a 20% margin, resize to 320x320, and save all crops in one file (1.15 GB). Needed because
   reading the 31 GB dataset photo by photo was far too slow (about 3 photos per second).
3. **Model** (`train/train.py`): a U-Net. The encoder is MobileNetV2 pretrained on ImageNet; the
   decoder (upsampling, skip connections, two convolutions per level) is written by us.
4. **Training**: 20 epochs on a Kaggle T4 GPU (43 minutes). Loss = binary cross-entropy + Dice.
   Each training crop uses a randomly shifted and resized box (up to 15%), so the model copes
   with imperfect detector boxes, plus a random flip and brightness/color changes.
5. **Split**: by photo session (21,761 training / 12,664 validation photos), so near-identical
   photos never sit on both sides.
6. **Result**: validation IoU **0.987**. For reference, answering "the whole box is the person"
   scores 0.708. Outputs are in `runs/full1/` (`best.pt`, `log.csv`, `samples.png`).

## Measured so far

Test on one 1280x720 webcam photo, on this laptop's CPU (`result_compare.png` shows the three outputs):

| Method | Speed | What the output looks like |
|---|---|---|
| Template | about 10 fps | Rough: shows patches of the real room around the head and shoulders |
| GrabCut | about 4 fps | Follows the body, with some jagged edges |
| Model | about 6 fps | Cleanest outline |

YOLO alone takes about 50 ms per frame; the rest is the mask and the compositing.

## Not done yet

- The live webcam loop has only been tested on a still photo, not on the camera itself.
- No accuracy numbers on webcam images yet (needs the hand-checked webcam test set).
- The model has only seen portrait photos, not webcam frames (fine-tuning planned later).
- Speed-ups still open: run YOLO every few frames, export the model to ONNX, smooth the mask over time.
