import sys
import numpy as np
import cv2

# Head + shoulders template (200 wide x 300 tall), 1 = person, 0 = background
TEMPLATE = np.zeros((300, 200), dtype=np.uint8)
cv2.ellipse(TEMPLATE, (100, 70), (45, 60), 0, 0, 360, 1, -1)          # head
cv2.rectangle(TEMPLATE, (80, 120), (120, 170), 1, -1)                 # neck
cv2.ellipse(TEMPLATE, (100, 230), (95, 70), 0, 180, 360, 1, -1)       # shoulders
cv2.rectangle(TEMPLATE, (5, 230), (195, 299), 1, -1)                  # torso


def apply_mask(image, background, mask, x, y, width, height):
    h_img, w_img = image.shape[:2]
    out = cv2.resize(background, (w_img, h_img))
    m = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # clip the box to the image borders
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + width, w_img), min(y + height, h_img)
    if x0 >= x1 or y0 >= y1:
        return out

    m = m[y0 - y:y1 - y, x0 - x:x1 - x]
    # copy the person (mask == 1) from the image onto the background
    out[y0:y1, x0:x1][m == 1] = image[y0:y1, x0:x1][m == 1]
    return out


if __name__ == "__main__":
    img = cv2.imread("C:/Users/anant/Pictures/Camera Roll/neutre_anant.jpg")
    background = cv2.imread("./banarange.png")
    if img is None or background is None:
        sys.exit("Could not read the image or the background")

    result = apply_mask(img, background, TEMPLATE, 200, 150, 1000, 680)
    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
