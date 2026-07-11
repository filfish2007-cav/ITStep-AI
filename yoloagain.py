import cv2
import ultralytics

model = ultralytics.YOLO("yolo11s.pt")

# 1. Отримуємо перший кадр з відео
cap = cv2.VideoCapture("/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson8/animals.mp4")
success, frame = cap.read()
cap.release()

# виводимо кадр на екран
cv2.imshow("frame", frame)
cv2.waitKey(0)

# 2. Детекція об'єктів за допомогою YOLO
results = model.predict(
    frame,
    device="mps",
    conf=0.25,   # мінімальна впевненість, нижче якої об'єкт відкидається
    iou=0.5,     # поріг перекриття рамок для видалення дублікатів
)

result = results[0]

# виводимо результати в консоль
print(result)

# малюємо рамки на кадрі і показуємо
res = result.plot()
cv2.imshow("result", res)
cv2.waitKey(0)

# 3. Вирізаємо кожен знайдений об'єкт і показуємо окремо
boxes = result.boxes

for i, box in enumerate(boxes):
    # координати рамки (x1, y1) - лівий верхній кут, (x2, y2) - правий нижній
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # назва класу та впевненість моделі
    class_name = result.names[int(box.cls[0])]
    conf_value = float(box.conf[0])

    # вирізаємо об'єкт з кадру
    crop = frame[y1:y2, x1:x2]

    # показуємо вирізаний об'єкт в окремому вікні
    window_name = f"{i}_{class_name}_{conf_value:.2f}"
    cv2.imshow(window_name, crop)

# # task 2

model = ultralytics.YOLO("yolo11s.pt")

video_path = "/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson8/animals.mp4"
cap = cv2.VideoCapture(video_path)

# 1. Беремо перший кадр і показуємо результат детекції/трекінгу
success, frame = cap.read()

results = model.track(frame, device="mps", persist=True)
result = results[0]

res = result.plot()
cv2.imshow("detection", res)
cv2.waitKey(0)

# виводимо в консоль всі знайдені об'єкти з їх ID, щоб було зрозуміло, що вводити
for box in result.boxes:
    obj_id = int(box.id[0]) if box.id is not None else None
    class_name = result.names[int(box.cls[0])]
    print(f"ID: {obj_id}, клас: {class_name}")

# 2. Просимо користувача ввести ID об'єкта, який треба відслідковувати
target_id = int(input("Введіть ID об'єкта для відстеження: "))

# 3. Проходимо по всіх наступних кадрах відео
while True:
    success, frame = cap.read()
    if not success:
        break

    # показуємо оригінальне відео
    cv2.imshow("original", frame)

    # детекція + трекінг на поточному кадрі
    results = model.track(frame, device="mps", persist=True)
    result = results[0]

    # шукаємо серед знайдених об'єктів той, у якого ID співпадає з потрібним
    for box in result.boxes:
        if box.id is None:
            continue

        obj_id = int(box.id[0])

        if obj_id == target_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names[int(box.cls[0])]

            # малюємо рамку навколо потрібного об'єкта
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # показуємо кадр з рамкою відслідковуваного об'єкта
    cv2.imshow("tracked object", frame)

    # 25 мс затримка між кадрами, вихід по клавіші 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
