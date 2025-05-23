# test_camera.py
import cv2
import time

print("Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Camera opened successfully")

# Get initial camera properties
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Initial camera properties:")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")

# Try to read and display frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print info every 30 frames
        print(f"Frame {frame_count} shape: {frame.shape}")
    
    # Display the frame
    cv2.imshow('Webcam test', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

print(f"Total frames processed: {frame_count}")
cap.release()
cv2.destroyAllWindows()
