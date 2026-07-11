import cv2
from ultralytics import YOLO

# task 1

video_path = '/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson8/animals.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

cv2.imshow('First Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

model = YOLO('yolov8n.pt')
results = model(frame)

result_img = results[0].plot()
cv2.imshow('YOLO Detection', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

results_low = model(frame, conf=0.1, iou=0.3)
result_img_low = results_low[0].plot()
cv2.imshow('conf=0.1, iou=0.3', result_img_low)
cv2.waitKey(0)
cv2.destroyAllWindows()

results_high = model(frame, conf=0.7, iou=0.7)
result_img_high = results_high[0].plot()
cv2.imshow('conf=0.7, iou=0.7', result_img_high)
cv2.waitKey(0)
cv2.destroyAllWindows()

boxes = results[0].boxes

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped = frame[y1:y2, x1:x2]
    cv2.imshow(f'Object {i}', cropped)
    cv2.waitKey(0)

cv2.destroyAllWindows()

# task 2

video_path = '/Users/filipprybkin/PycharmProjects/AI/ITStep-AI/data/lesson8/animals.mp4'
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

# Читаємо перший кадр + detection
ret, frame = cap.read()
results = model.track(frame, persist=True)
result_img = results[0].plot()
cv2.imshow('Detection', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

track_id = int(input('Введіть ID об\'єкта для відстеження: '))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    cv2.imshow('Original video', frame)

    boxes = results[0].boxes
    if boxes.id is not None:
        for box, obj_id in zip(boxes, boxes.id):
            if int(obj_id) == track_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tracked_frame = frame.copy()
                cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(tracked_frame, f'ID {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Tracked object', tracked_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
