# # task 1

import cv2

# img = cv2.imread("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson3/notes.png")
#
# img = cv2.resize(img, (600, 600))
#
# cv2. imshow("original", img)
#
# gray_image = cv2.ctColor(img, cv2.COLOR_BR2GRAY)
#
# cv2.imshow("gray", gray_image)
#
# # threshold = 128
# #
# # mask = gray_image < threshold
# # gray_image[mask] = 0
# # gray_image[~mask] = 0
# #
# # cv2.imshow("gray", gray_image)
#
# # gauss = cv2.GaussianBlur(gray_image, (3, 3), 1.8)
# #
# # cv2.imshow("gauss", gauss)
#
# # двосторонній фільтр
# bilat = cv2.bilateralFilter(
#     gray_image,
#     d=5,  # розмір ядра
#     sigmaColor=75,   # наскільки зберігати різкість кольору
#     sigmaSpace=50,   # те ж саме що й в GaussianBlur
# )
#
# cv2.imshow('bilat', bilat)
#
# # adaptive
#
#
# res = cv2.adaptiveThreshold(
#     bilat,
#     255,  #  інтенчивність для білого кольору
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # фурмула згортки(гаус)
#     cv2.THRESH_BINARY,    # це не чіпаємо
#     7,    # розмір ядра для згортки
#     3           # наскільки чутливою має бути бінарізація
# )
#
# cv2.imshow("adaptive", res)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# # task 2

img = cv2.imread("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson3/sudoku.jpg")
cv2.imshow('Original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

result = clahe.apply(gray)
cv2.imshow('Clahe', result)

gauss = cv2.GaussianBlur(gray, (3,3), 0)
cv2.imshow('Gauss', gauss)

res = cv2.adaptiveThreshold(
    result,
    255,  #  інтенчивність для білого кольору
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,   # фурмула згортки(гаус)
    cv2.THRESH_BINARY,    # це не чіпаємо
    7,    # розмір ядра для згортки
    3           # наскільки чутливою має бути бінарізація
)

cv2.imshow('adaptive+clahe', res)

resultgray = cv2.fastNlMeansDenoising(gray,None,h=10,templateWindowSize=7,searchWindowSize=21)
cv2.imshow('nlmean', resultgray)
cv2.waitKey(0)


