

# дз
# import numpy as np
#
#
# nums = np.array(
#     [
#     [1,2,3,4],
#     [5,6,7,8],
#     [9,10,11,12],
#     [13,14,15,16]
#     ]
# )
#
# print(nums[[0, 0, 1, 0]])

# #  зображення у opencv
# import utils
#
# utils.lesson1_image()

# читання зображення
import cv2


image = cv2.imread(
    "data/lesson1/cameraman.png",  # шлях до файлу
    cv2.IMREAD_GRAYSCALE,   # прапорець як читати зображення(чорнобіле)
)

print(type(image))
print(image.shape)
print(image.dtype)   # uint8  (0 - 255)
print(image)


# показати зображення
# cv2.imshow(
#     "original",  # назва для зображення
#     image           # саме зображення
# )

# щоб щображення показувалось довго треба зациклити програму
# чекаємо поки не буде натиснута якась кнопка
# cv2.waitKey(0)




# зміна розміру зображення

# не правильно
# new_image = image.reshape(128, 512)
# cv2.imshow("reshaped", new_image)


# new_image = cv2.resize(
#     image,
#     (500, 500)   # новий розмір
# )


# в пропорціях
# new_image = cv2.resize(
#     image,
#     None,
#     fx=1.5,   # множник для висоти та ширини
#     fy=1.5,
# )

# print(new_image.shape)
#
# cv2.imshow("resized", new_image)




# segment = image[75:175, 20:240]
# print(segment.shape)
# print(segment.dtype)
# cv2.imshow("segment", segment)


# збільшення значення пікселів

# import numpy as np
# nums = np.array([100, 200], dtype=np.uint8)
# print(nums + 80)
#
# image = image.astype(np.int64)
# image[75:175, 20:240] += 80
#
# mask = image > 255
# image[mask] = 255
#
#
# image = image.astype(np.uint8)
#
# segment = image[75:175, 20:240]
#
# cv2.imshow("original", image)
# cv2.imshow("segment", segment)
# cv2.waitKey(0)

