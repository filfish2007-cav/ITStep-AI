import cv2
import numpy as np
from ultralytics import YOLO

video_path = "/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson_pose/sitting.mp4"
device = "cuda"  # або "cuda" / 0
model = YOLO("yolov8n-pose.pt")

# ==========================================
# Завдання 1: Відкрити відео, отримати перший кадр, показати
# ==========================================
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if ret:
    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow("Task 1 - First Frame", frame_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# Завдання 2: Застосувати модель YOLO Pose
# ==========================================
results = model(frame, device=device)
print("Task 2 - Results:", results)

# ==========================================
# Завдання 3: Метод plot() і відображення
# ==========================================
annotated_frame = results[0].plot()
cv2.imshow("Task 3 - Plot", cv2.resize(annotated_frame, (640, 360)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==========================================
# Завдання 4: Отримати keypoints, xy координати, тип та розмір
# ==========================================
keypoints = results[0].keypoints
print("Task 4 - Keypoints:", keypoints)

xy = keypoints.xy.cpu().numpy()
print("Task 4 - Coordinates (xy):", xy)
print("Task 4 - Type:", type(xy), xy.dtype)
print("Task 4 - Shape:", xy.shape)

# ==========================================
# Завдання 5: Точки для першого об'єкта (коліно=13, ліва рука=9, права рука=10)
# ==========================================
if len(xy) > 0:
    l_knee = tuple(xy[0][13].astype(int))
    l_hand = tuple(xy[0][9].astype(int))
    r_hand = tuple(xy[0][10].astype(int))

    img_task5 = frame.copy()
    cv2.circle(img_task5, l_knee, 8, (0, 255, 0), -1)      # Зелений
    cv2.circle(img_task5, l_hand, 8, (0, 0, 255), -1)      # Червоний
    cv2.circle(img_task5, r_hand, 8, (255, 255, 255), -1)  # Білий

    cv2.imshow("Task 5", cv2.resize(img_task5, (640, 360)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# Завдання 6, 7, 8: Малювання точок для відео, підрахунок присідань та визначення стану
# ==========================================
cap = cv2.VideoCapture(video_path)
squat_count = 0
state = "встає"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = model(frame, device=device, verbose=False)[0]
    pts = res.keypoints.xy.cpu().numpy()

    if len(pts) > 0 and len(pts[0]) > 0:
        l_knee_pt = tuple(pts[0][13].astype(int))
        l_hand_pt = tuple(pts[0][9].astype(int))
        r_hand_pt = tuple(pts[0][10].astype(int))

        # Завдання 6: Намалювати точки
        cv2.circle(frame, l_knee_pt, 5, (0, 255, 0), -1)
        cv2.circle(frame, l_hand_pt, 5, (0, 0, 255), -1)
        cv2.circle(frame, r_hand_pt, 5, (255, 255, 255), -1)

        l_knee_y = pts[0][13][1]
        l_hand_y = pts[0][9][1]

        # Завдання 8: Логіка підрахунку присідань та стану
        # Рука нижче коліна в координатах екрана означає Y_руки > Y_коліна
        if l_hand_y > l_knee_y:
            if state == "присідає":
                squat_count += 1
            state = "встає"
        else:
            state = "присідає"

    cv2.putText(frame, f"Squats: {squat_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"State: {state}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Task 6-8", cv2.resize(frame, (640, 360)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==========================================
# Завдання 9: 258-й кадр, рамка першого об'єкта та plot()
# ==========================================
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 257)  # Кадр 258 (індексація з 0)
ret, frame_258 = cap.read()
cap.release()

if ret:
    res_258 = model(frame_258, device=device)[0]
    boxes_258 = res_258.boxes
    print("Task 9 - Boxes:", boxes_258)
    if len(boxes_258) > 0:
        print("Task 9 - First Box:", boxes_258[0].xyxy.cpu().numpy())

    plotted_258 = res_258.plot()
    cv2.imshow("Task 9 - Frame 258", cv2.resize(plotted_258, (640, 360)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Висновки: Модель стабільно детектує об'єкт та обмежувальну рамку (box) на 258-му кадрі.

# ==========================================
# Завдання 10: Функція get_box_area та площі на 200 кадрі
# ==========================================
def get_box_area(box):
    xywh = box.xywh.cpu().numpy()[0]
    w = xywh[2]
    h = xywh[3]
    return w * h

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 199)  # Кадр 200
ret, frame_200 = cap.read()
cap.release()

if ret:
    res_200 = model(frame_200, device=device)[0]
    boxes_200 = res_200.boxes
    for i, box in enumerate(boxes_200):
        print(f"Task 10 - Box {i} Area:", get_box_area(box))

# ==========================================
# Завдання 11: Функція get_largets_box_id
# ==========================================
def get_largets_box_id(boxes):
    if len(boxes) == 0:
        return -1
    areas = [get_box_area(box) for box in boxes]
    return int(np.argmax(areas))

# ==========================================
# Завдання 12: Модифікація Завдання 8 для найбільшої рамки
# ==========================================
cap = cv2.VideoCapture(video_path)
squat_count = 0
state = "встає"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = model(frame, device=device, verbose=False)[0]
    boxes = res.boxes

    if len(boxes) > 0:
        largest_id = get_largets_box_id(boxes)
        pts = res.keypoints.xy.cpu().numpy()

        if len(pts) > largest_id and len(pts[largest_id]) > 0:
            l_knee_pt = tuple(pts[largest_id][13].astype(int))
            l_hand_pt = tuple(pts[largest_id][9].astype(int))
            r_hand_pt = tuple(pts[largest_id][10].astype(int))

            cv2.circle(frame, l_knee_pt, 5, (0, 255, 0), -1)
            cv2.circle(frame, l_hand_pt, 5, (0, 0, 255), -1)
            cv2.circle(frame, r_hand_pt, 5, (255, 255, 255), -1)

            l_knee_y = pts[largest_id][13][1]
            l_hand_y = pts[largest_id][9][1]

            if l_hand_y > l_knee_y:
                if state == "присідає":
                    squat_count += 1
                state = "встає"
            else:
                state = "присідає"

    cv2.putText(frame, f"Squats: {squat_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"State: {state}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Task 12", cv2.resize(frame, (640, 360)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()