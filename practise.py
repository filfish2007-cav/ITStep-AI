import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.

image = cv2.imread(
    "data/lesson1/Lenna.png",  # шлях до файлу
    cv2.IMREAD_GRAYSCALE,   # прапорець як читати зображення(чорнобіле)
)

# cv2.imshow("Lenna", image)
# cv2.waitKey(0)
# print(image.dtype)
# print(image.shape)
# print(image.max())
# print(image.min())

# Завдання 2
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# такі зображень:
#  Верхній лівий кут розміром 100х50
#  Центральний квадрат розміром 100х100
#  Верхню половину
#  Нижню половину
#  Ліву половину
#  Праву половину

# segment1 = image[0:100,0:50]
# cv2.imshow("segment1", segment1)
# cv2.waitKey(0)

segment2 = image[78:178,78:178]
cv2.imshow("segment2", segment2)
print(segment2.shape)
cv2.waitKey(0)


# lower part
# cv2.imshow("bottom", image[129:255, :])
# left part
# cv2. 1mshow("Leftside", image:,: 128])
# right part
# cv2. imshow("nightside", Imagel:, 129:255])


# --- Image 1: Top black border ---
img1 = image.copy()
img1[:20, :] = 0  # Black out top 20 rows

# --- Image 2: Left black border ---
img2 = image.copy()
img2[:, :30] = 0  # Black out left 30 columns

# --- Image 3: Centered smaller frame ---
img3 = np.zeros((256, 256), dtype=np.uint8)
# Resize and paste into the center directly via slicing
img3[48:208, 48:208] = cv2.resize(image, (160, 160))

# --- Save outputs ---
cv2.imwrite('output1.png', img1)
cv2.imwrite('output2.png', img2)
cv2.imwrite('output3.png', img3)


# Завантажуємо оригінальне зображення
image = cv2.imread('data/Lenna.png', cv2.IMREAD_GRAYSCALE)

# 1. Створення маски для пікселів > 128
mask = image > 128
new_mask = mask.astype(np.uint8) * 255
cv2.imshow("mask", new_mask)

# 2. Створення та виведення ЗАПЕРЕЧЕННЯ маски (інверсія)
# Використовуємо оператор ~ для інверсії логічного масиву
inv_mask = (~mask).astype(np.uint8) * 255
cv2.imshow("inverted mask", inv_mask)

# 3. Заміна пікселів, які не відповідають масці, на 0
image[~mask] = 0
cv2.imshow("new image", image)

# Збереження результату (прибрано зайву кому)
cv2.imwrite("new_image.png", image)

cv2.waitKey(0)
