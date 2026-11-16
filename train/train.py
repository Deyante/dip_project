import glob
import pickle
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SIZE = 256        # crops are resized to SIZE x SIZE
EPOCHS = 20
SUBSET = None     # number of images to use, None = whole dataset
BATCH = 32
LR = 1e-3
JITTER = 0.15     # random box jitter, fraction of box width/height
OUT = "/kaggle/working"

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------- data ----------
# crops.pkl comes from prep.py: one person crop per image, with the person box
# and the original image borders given in the crop's coordinates

def load_samples():
    path = glob.glob("/kaggle/input/**/crops.pkl", recursive=True)[0]
    with open(path, "rb") as f:
        return pickle.load(f)


def jitter_box(x0, y0, x1, y1, rng):
    w, h = x1 - x0, y1 - y0
    dx0, dx1 = rng.uniform(-JITTER, JITTER, 2) * w
    dy0, dy1 = rng.uniform(-JITTER, JITTER, 2) * h
    return x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1


def crop(img, x0, y0, x1, y1):
    # crop with zero padding where the box leaves the image
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = cv2.imdecode(np.frombuffer(s["img"], np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(s["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)

        # training: random jitter, validation: fixed jitter per image
        # the jittered box stays inside the original photo, like a real detector box
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        x0, y0, x1, y1 = jitter_box(*s["box"], rng)
        bx0, by0, bx1, by1 = s["bounds"]
        box = [int(round(v)) for v in (max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1))]
        img = cv2.resize(crop(img, *box), (SIZE, SIZE))
        mask = (cv2.resize(crop(mask, *box), (SIZE, SIZE)) > 127).astype(np.float32)

        if self.train:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            gain = rng.uniform(0.7, 1.3) * rng.uniform(0.9, 1.1, 3)
            img = np.clip(img * gain + rng.uniform(-30, 30), 0, 255)

        img = (img[:, :, ::-1] / 255.0 - MEAN) / STD    # BGR -> RGB, normalize
        img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        return img, torch.from_numpy(mask.copy())[None]


# ---------- model ----------

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """MobileNetV2 encoder + our own U-Net decoder."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        f = torchvision.models.mobilenet_v2(weights=weights).features
        # output channels: 16 (1/2), 24 (1/4), 32 (1/8), 96 (1/16), 320 (1/32)
        self.stages = nn.ModuleList([f[0:2], f[2:4], f[4:7], f[7:14], f[14:18]])
        self.ups = nn.ModuleList([
            conv_block(320 + 96, 96), conv_block(96 + 32, 32),
            conv_block(32 + 24, 24), conv_block(24 + 16, 16),
        ])
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], 1))
        x = self.head(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


def loss_fn(logits, mask):
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * mask).sum((2, 3)) + 1) / (p.sum((2, 3)) + mask.sum((2, 3)) + 1)
    return bce + dice.mean()


# ---------- training ----------

def evaluate(model, loader, device):
    """IoU of the model, and IoU of the trivial answer 'the whole box is the person'."""
    model.eval()
    inter = union = person = pixels = 0.0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device) > 0.5
            with torch.autocast("cuda"):
                pred = model(img) > 0
            inter += (pred & mask).sum().item()
            union += (pred | mask).sum().item()
            person += mask.sum().item()
            pixels += mask.numel()
    return inter / union, person / pixels


def save_samples(model, dataset, device, path):
    model.eval()
    rows = []
    for i in range(0, len(dataset), len(dataset) // 8)[:8]:
        img, mask = dataset[i]
        with torch.no_grad():
            pred = torch.sigmoid(model(img[None].to(device)))[0, 0].cpu().numpy()
        rgb = ((img.numpy().transpose(1, 2, 0) * STD + MEAN) * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        gray = lambda m: cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rows.append(np.hstack([rgb, gray(mask[0].numpy()), gray(pred)]))
    cv2.imwrite(path, np.vstack(rows))


def main():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))

    samples = load_samples()
    print("crops loaded:", len(samples))
    random.seed(0)
    random.shuffle(samples)
    if SUBSET:
        samples = samples[:SUBSET]

    # split by shooting session so near-identical photos don't end up in both sets
    sessions = sorted({s["session"] for s in samples})
    val_sessions = set(sessions[::10])
    train_s = [s for s in samples if s["session"] not in val_sessions]
    val_s = [s for s in samples if s["session"] in val_sessions]
    print(f"train: {len(train_s)}  val: {len(val_s)}  sessions: {len(sessions)}")

    train_ds, val_ds = Crops(train_s, True), Crops(val_s, False)
    train_dl = torch.utils.data.DataLoader(train_ds, BATCH, shuffle=True, num_workers=4, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, BATCH, num_workers=4)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(train_dl))
    scaler = torch.amp.GradScaler()

    best = 0.0
    with open(f"{OUT}/log.csv", "w") as log:
        log.write("epoch,train_loss,val_iou,whole_box_iou,seconds\n")
        for epoch in range(EPOCHS):
            model.train()
            t0, total = time.time(), 0.0
            for img, mask in train_dl:
                img, mask = img.to(device), mask.to(device)
                with torch.autocast("cuda"):
                    loss = loss_fn(model(img).float(), mask)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item()

            iou, whole = evaluate(model, val_dl, device)
            secs = time.time() - t0
            print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_dl):.4f}  "
                  f"val IoU {iou:.4f}  (whole box: {whole:.4f})  {secs:.0f}s", flush=True)
            log.write(f"{epoch + 1},{total / len(train_dl):.4f},{iou:.4f},{whole:.4f},{secs:.0f}\n")
            log.flush()
            if iou > best:
                best = iou
                torch.save(model.state_dict(), f"{OUT}/best.pt")

    model.load_state_dict(torch.load(f"{OUT}/best.pt"))
    save_samples(model, val_ds, device, f"{OUT}/samples.png")
    print("best val IoU:", best)


if __name__ == "__main__":
    main()
