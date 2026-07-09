import cv2

cap = cv2.VideoCapture('data/lesson7/meter.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Створюємо об'єкт для запису нового відео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/lesson7/meter_binary.mp4', fourcc, fps, (width, height), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # прибираємо шум
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # бінарізація кадру
    ret2, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # записуємо кадр в новий файл
    out.write(binary)

    cv2.imshow('binary', binary)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()