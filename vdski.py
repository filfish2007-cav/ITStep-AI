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