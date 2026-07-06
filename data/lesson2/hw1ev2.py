import cv2
import numpy as np

# Завантаження зображення
img3 = cv2.imread("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson2/darken.png")
cv2.imshow("original darken", img3)
cv2.waitKey(0)

# Переведення в HSV формат
hsv = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

# Розбиття на канали H, S, V
h, s, v = cv2.split(hsv)

# --- Спосіб 1: Вирівнювання гістограми (HE) ---
new_v_he = cv2.equalizeHist(v)
new_hsv_he = cv2.merge((h, s, new_v_he))
res_he = cv2.cvtColor(new_hsv_he, cv2.COLOR_HSV2BGR)
cv2.imshow("equalizeHist result", res_he)
cv2.waitKey(0)

# --- Спосіб 2: Збільшення значення на 40% (float32 + clip) ---
# Переводимо в float32, щоб уникнути переповнення типу uint8 при множенні
v_float = v.astype(np.float32)

# Збільшуємо на 40% (коефіцієнт 1.4)
v_float = v_float * 1.4

# Обрізаємо значення в межах [0-255] та повертаємо в uint8
new_v_boost = np.clip(v_float, 0, 255).astype(np.uint8)

new_hsv_boost = cv2.merge((h, s, new_v_boost))
res_boost = cv2.cvtColor(new_hsv_boost, cv2.COLOR_HSV2BGR)
cv2.imshow("boosted value result", res_boost)

cv2.waitKey(0)