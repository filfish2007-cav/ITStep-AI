import cv2

# 1 open the video file
cap = cv2.VideoCapture('data/lesson7/text.mp4')

# 2 save the file
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define new size
new_width = width // 2
new_height = height // 2

# create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_resized.mp4', fourcc, fps, (new_width, new_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize image/frame
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Save the resized frame to the new file
    out.write(resized_frame)

    # Display the result
    cv2.imshow('Resized Video', resized_frame)

    # Press q to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# task 2

cap = cv2.VideoCapture('data/lesson7/text.mp4')

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# isColor=False because our output is a binary (grayscale) image
out = cv2.VideoWriter('binary_output.mp4', fourcc, fps, (width, height), isColor=False)

# frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Adaptive Thresholding
    # Adjust blockSize and C based on the lighting specific video
    binary_frame = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 5
    )

    # Write the processed binary frame to the new file
    out.write(binary_frame)

    cv2.imshow('Binary Video', binary_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# task 3

cap = cv2.VideoCapture('data/lesson7/shapes.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter('edges_output.mp4', fourcc, fps, (width, height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Перетворюємо на сірий для кращого виділення країв
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Виділення країв за допомогою алгоритму Canny
    # 100 та 200 пороги чутливості
    edges = cv2.Canny(gray, 100, 200)

    # Зберігаємо результат
    out.write(edges)

    # Показуємо процес
    cv2.imshow('Edges Detection', edges)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()