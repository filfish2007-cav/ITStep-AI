import cv2

# Налаштування CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# TASK 1
img1 = cv2.imread('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson3/sonet.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)        # Перетворення у відтінки сірого
equalized1 = clahe.apply(gray1)                      # Нормалізація освітлення
blurred1 = cv2.medianBlur(equalized1, 3)             # Очищення від шумів без розмиття тексту

# Адаптивна бінаризація
res1 = cv2.adaptiveThreshold(
    blurred1, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 35, 10
)

# TASK 2
img2 = cv2.imread('/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson3/sonet.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
equalized2 = clahe.apply(gray2)
blurred2 = cv2.medianBlur(equalized2, 3)

res2 = cv2.adaptiveThreshold(
    blurred2, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 35, 10
)

# Відображення результатів
cv2.imshow('Result 1', res1)
cv2.imshow('Result 2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()