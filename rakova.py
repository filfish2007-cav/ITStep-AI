import numpy as np
import cv2
from ultralytics import YOLO

image = cv2.imread("data/lesson_seg/tumor1.jpg")
cv2.imshow("image", image)

model = YOLO("data/lesson_seg/brain-tumor-seg.pt")

results = model.predict(image, device='mps')
result = results[0]

res = result.plot()
cv2.imshow("result", res)

masks = result.masks
print(masks)

masks_data = masks.data.cpu().numpy()
height, width, channels = image.shape

PIXEL_AREA = 0.0025  # площа одного пікселя у заданих одиницях

for i in range(len(masks_data)):
    mask = masks_data[i]
    mask = cv2.resize(mask, (width, height)).astype(bool)  # (width, height), не (height, width)!

    area_px = int(mask.sum())
    area_real = area_px * PIXEL_AREA

    if area_real < 10:
        tumor_type = "small"
    elif area_real <= 25:
        tumor_type = "middle"
    else:
        tumor_type = "large"

    print(f"tumor {i}: area_px={area_px}, area_real={area_real:.4f}, type={tumor_type}")

    copy = image.copy()
    copy[~mask] = 0

    cv2.imshow(tumor_type, copy)

cv2.waitKey(0)
cv2.destroyAllWindows()