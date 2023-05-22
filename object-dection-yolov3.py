import cv2
import numpy as np

# Load the YOLO model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load the COCO class labels
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
image = cv2.imread('image.jpg')

# Get the image dimensions
height, width, _ = image.shape

# Create a blob from the image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input blob for the network
net.setInput(blob)

# Run forward pass and get the detections
outputs = net.forward()

# Loop over each detection
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Filter detections by confidence threshold
            # Get the coordinates of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate the top-left corner coordinates of the bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with bounding boxes and labels
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
