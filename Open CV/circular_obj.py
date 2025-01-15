# import cv2
# import numpy as np

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     blurred = cv2.GaussianBlur(gray, (9, 9), 2)

#     circles = cv2.HoughCircles(
#         blurred, 
#         cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, 
#         param1=100, param2=30, minRadius=10, maxRadius=100
#     )

#     # If circles are detected, draw them
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles[0, :]:
#             x, y, r = circle
#             # Draw the outer circle
#             cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
#             # Draw the center of the circle
#             cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

#     cv2.imshow('Circular Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color = np.array([100, 150, 0])  # We can asjust the colors according to need such as its for blue
    upper_color = np.array([140, 255, 255])  # Adjust based on color

    mask = cv2.inRange(hsv, lower_color, upper_color)

    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

    # For the detection of circles in the filtered mask
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

    cv2.imshow('Circular Object Detection', frame)
    cv2.imshow('Color Mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()