import cv2 as cv
import numpy as np
from PIL import Image

def get_limits(color): 
    # Convert the given BGR color to HSV
    c = np.uint8([[color]])
    # here, insert the bgr values which you want to convert to hsv
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

    # Handle hue wraparound for red
    lower_limit1 = np.array([hsvC[0][0][0] - 10, 100, 100], dtype=np.uint8)
    upper_limit1 = np.array([hsvC[0][0][0] + 10, 255, 255], dtype=np.uint8)

    # Wraparound for hue: second part of red range
    lower_limit2 = np.array([hsvC[0][0][0] - 10 + 180, 100, 100], dtype=np.uint8)
    upper_limit2 = np.array([180, 255, 255], dtype=np.uint8)

    return (lower_limit1, upper_limit1), (lower_limit2, upper_limit2)

# Use standard red in BGR format
red = [0, 0, 255]
cap = cv.VideoCapture(0)  # Open the default camera

# Check if the video capture has been successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True: 
    # ret is a boolean return value from cap.read(), 
    # where cap is the video capture object created with cv.VideoCapture(0).
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame from BGR to HSV color space for color detection
    hsvimage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    (lower_limit1, upper_limit1), (lower_limit2, upper_limit2) = get_limits(color=red)

    # Create two masks for the red range and combine them
    mask1 = cv.inRange(hsvimage, lower_limit1, upper_limit1)
    mask2 = cv.inRange(hsvimage, lower_limit2, upper_limit2)
    mask = cv.bitwise_or(mask1, mask2)  # Combine the masks

    # Convert mask to a PIL image to use getbbox for bounding box detection
    mask_ = Image.fromarray(mask)

    # Get the bounding box of the detected color region (if any)
    bbox = mask_.getbbox()

    # If a bounding box exists, draw a rectangle around it on the original frame
    if bbox is not None: 
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangle

    # Display the frame with detected color highlighted (if any)
    cv.imshow('Detected Color', frame)

    # Break on pressing 'd'
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv.destroyAllWindows()
