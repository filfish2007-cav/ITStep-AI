import cv2
import ultralytics


model = ultralytics.YOLO("yolo11s-pose.pt")

img = cv2.imread("data/lesson_pose/human.jpg")

results = model.predict(img)  # повертає список з одним результатом
result = results[0]  #  дістаємо один результат зі списку

# print(result)

# візуалізація
# res_im = result.plot()
# cv2.imshow("result", res_im)

# ключові точки
keypoints = result.keypoints
# print(keypoints)


# координати xy
xy = keypoints.xy

# позбутися tensor device
xy = xy.cpu().numpy()

# дістаємо точки для першого об'єкта
xy = xy[0]

# змінити тип даних на int
xy = xy.astype(int)  # 17 точок

# координати правої долоні
x_right_hand, y_right_hand = xy[10]

print(x_right_hand, y_right_hand)


# намалювати коло на зображення
cv2.circle(
    img,   # зображення де малювати коло
    center=(x_right_hand, y_right_hand),   # координати центру
    radius=15,   # радіус в пікселях
    color=(255, 0, 0),  # колір в BGR(синій)
    thickness=-1,   # товщина ліній, -1 означає повністю заповнити кольором
)


# накласти текст на зображення
cv2.putText(
    img,          # зображення де пишемо текст
    "Right Hand",   # текст
    (x_right_hand + 10, y_right_hand - 40),   # позиція, лівий нижній кут
    cv2.FONT_HERSHEY_SIMPLEX,       # шрифт
    1,                     # розмір шрифту
    (0, 0, 0),                # колір в BGR
    2                      # товщина ліній
)


cv2.imshow("result", img)


# ліва стопа
x_left_feet, y_left_feet = xy[15]

# праве плече
x_right_shoulder, y_right_shoulder = xy[6]





# чи справді праве плече знаходиться правіше за ліву стопу
if x_right_shoulder > x_left_feet:
    print("справді праве плече знаходиться правіше за ліву стопу")

else:
    print("неправда що праве плече знаходиться правіше за ліву стопу")


# чи справді праве плече знаходиться вище за ліву стопу
if y_right_shoulder < y_left_feet:
    print("справді праве плече знаходиться вище за ліву стопу")

else:
    print("неправда що праве плече знаходиться вище за ліву стопу")



cv2.waitKey(0)