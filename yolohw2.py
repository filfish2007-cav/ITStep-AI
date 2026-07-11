import cv2
from ultralytics import YOLO

## task 1

video_path = 'data/lesson8/meetings.mp4'
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    results = model(frame)
    result_img = results[0].plot()

    cv2.imshow('Detection', result_img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

## task 2

video_path = 'data/lesson8/meetings.mp4'
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

started = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    results = model(frame, classes=[0])
    num_people = len(results[0].boxes)

    if num_people >= 5:
        started = True

    if started:
        result_img = results[0].plot()
        cv2.imshow('Video from 5 people', result_img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
