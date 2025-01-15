import cv2
import numpy as np

# Known parameters (adjust based on your setup)
REAL_DIAMETER = 5.0  # Real-world diameter of the object in cm (e.g., 5 cm for a circular object)
FOCAL_LENGTH = 700   # Focal length of the camera in pixels (calibrated for your camera)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for filtering (e.g., red)
    lower_color = np.array([100, 150, 0])  # Adjust based on color
    upper_color = np.array([140, 255, 255])  # Adjust based on color

    mask = cv2.inRange(hsv, lower_color, upper_color)

    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=100
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # Draw the outer circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            # Calculate the distance
            detected_diameter = 2 * r  # Diameter in pixels
            if detected_diameter > 0:
                distance = (FOCAL_LENGTH * REAL_DIAMETER) / detected_diameter
                # Display the distance on the frame
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (x - 50, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the original frame and the mask
    cv2.imshow('Circular Object Detection', frame)
    cv2.imshow('Color Mask', mask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()