import numpy as np
import cv2
from ultralytics import YOLO

image_corp = cv2.imread("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_seg/crop3.jpg")
cv2.imshow("image",image_corp)

model = YOLO("data/lesson_seg/crop-seg.pt")

results = model.predict(image_corp,
                        device='mps',)

result = results[0]

res = result.plot()
cv2.imshow("result",res)

masks = result.masks
print(masks)

masks_data = masks.data
masks_data = masks_data.cpu().numpy()

height, width, channels = image_corp.shape

for i in range(len(masks_data)):
    mask = masks_data[i]
    mask = cv2.resize(mask, (height, width)).astype(bool)
    copy = image_corp.copy()
    copy[~mask] = 255
    cv2.imshow(f"plant{i}",copy)


## task 2

image_corp = cv2.imread("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_seg/crop3.jpg")
cv2.imshow("image",image_corp)

model = YOLO("data/lesson_seg/crop-seg.pt")

results = model.predict(image_corp,
                        device='mps',)

result = results[0]

res = result.plot()
cv2.imshow("result",res)

masks = result.masks
print(masks)

masks_data = masks.data
masks_data = masks_data.cpu().numpy()
mask_list = []

for mask in masks_data:
    mask_sum = mask.sum()
    mask_list.append(mask_sum)

print(mask_list)


biggest_mask = max(mask_list)
print(biggest_mask)

for i in range(len(mask_list)):
    if biggest_mask == mask_list[i]:
        break

print(i)

mask3 = masks_data[i]

mask3_uint = mask3.astype(np.uint8)
mask3_uint *= 255

cv2.imshow("mask3", mask3_uint)








cv2.waitKey(0)