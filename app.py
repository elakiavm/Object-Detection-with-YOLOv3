import cv2
import numpy as np

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set random colors for each class
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize video stream
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video stream
    ret, frame = cap.read()

    # Create a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Perform forward pass to get the output layers
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to store detected objects' properties
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each output layer
    for output in layer_outputs:
        # Loop over each detected object
        for detection in output:
            # Extract the class ID and confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions
            if confidence > 0.5:
                # Get the center coordinates and dimensions of the detected object
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner of the bounding box
                x = int(center_x - width/2)
                y = int(center_y - height/2)

                # Add the detected object's properties to the lists
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes and labels on the input frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
