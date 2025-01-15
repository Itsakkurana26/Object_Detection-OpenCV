import cv2
import numpy as np
import os

# Load YOLO model files
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
names_path = "coco.names"

# Check if files exist
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")
if not os.path.exists(names_path):
    raise FileNotFoundError(f"Names file not found: {names_path}")

# Load class names from coco.names
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(class_names)

# Initialize YOLO model
net = cv2.dnn.readNet(weights_path, config_path)

# Set backend and target (use GPU if available)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Use DNN_TARGET_CUDA for GPU

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start video capture (webcam or video file)
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Preprocess the input frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass
    detections = net.forward(output_layers)

    # Initialize lists for detected objects
    boxes, confidences, class_ids = [], [], []

    # Loop through each detection
    for output in detections:
        for detection in output:
            scores = detection[5:]  # Class probabilities
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter weak detections
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    if len(boxes) > 0:  # Check if there are any boxes detected
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Check if any valid indices were returned
        if len(indices) > 0:
            for i in indices.flatten():  # Ensure we get a flat list of indices
                x, y, w, h = boxes[i]
                label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green for boxes

                # Draw the bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(f"Class ID: {class_ids[i]}, Label: {class_names[class_ids[i]]}")

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam - Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam - Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Webcam - Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

#     cv2.imshow('Webcam - Face Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# Release resources
cap.release()
cv2.destroyAllWindows()