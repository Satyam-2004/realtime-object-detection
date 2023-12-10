import cv2
import numpy as np

# Load YOLO model and configuration
net = cv2.dnn.readNet('path to yolov3.weights', 'path to yolov3.cfg')

# Load COCO class labels
with open('path to coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Additional object classes
new_classes = [
    'lamp', 'mirror', 'pillow', 'picture frame', 'rug', 'curtain', 'shoe', 'backpack', 'camera', 'keyboard',
    'printer', 'monitor', 'mouse pad', 'wallet', 'headphones', 'sunglasses', 'umbrella', 'watch', 'bracelet',
    'ring', 'necklace', 'wine bottle', 'coffee cup', 'tea cup', 'fork', 'knife', 'spoon', 'plate', 'napkin',
    'tablecloth', 'candle', 'flower', 'plant', 'tree', 'grass', 'cloud', 'moon', 'sun', 'star', 'mountain',
    'ocean', 'river', 'lake', 'beach', 'desert', 'bridge', 'building', 'skyscraper', 'house', 'church', 'castle'
]

# Combine existing and new classes
classes += new_classes

# Load YOLO model configuration and weights
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Set confidence threshold and non-maximum suppression threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Open a connection to the camera (you can replace the argument with a video file path)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Perform forward pass and get predictions
    detections = net.forward(output_layer_names)

    # Process each detection
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Print information about the detected object
                class_name = classes[class_id]
                print(f"Detected: {class_name}, Confidence: {confidence:.2f}")

                # Scale the bounding box coordinates to the original image size
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw the bounding box and label on the frame
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('Object Detection (YOLO)', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
     