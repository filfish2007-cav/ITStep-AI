import cv2
import mediapipe as mp
from utils import get_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

cap = cv2.VideoCapture('squat.mp4')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imwrite('first_frame.jpg', frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, fps, (w, h))

LOW_ANGLE = 100   # присіла
HIGH_ANGLE = 160  # встала

counter = 0
stage = "up"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

        x1, y1 = hip.x * w, hip.y * h
        x2, y2 = knee.x * w, knee.y * h
        x3, y3 = ankle.x * w, ankle.y * h

        angle = get_angle(x1, y1, x2, y2, x3, y3)

        if angle < LOW_ANGLE:
            stage = "down"
        if angle > HIGH_ANGLE and stage == "down":
            stage = "up"
            counter += 1

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Angle: {int(angle)}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(frame, f"Squats: {counter}", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()

print("Squats counted:", counter)