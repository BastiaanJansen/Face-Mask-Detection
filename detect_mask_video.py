from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

MODEL = "models/mask_detector.model"
MIN_CONFIDENCE = 0.50


def extract_face_roi(frame, detection):
    h, w = frame.shape[:2]

    # Compute the (x, y)-coordinates of the bounding box for the object
    box = detection * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = box.astype("int")
    # Ensure the bounding boxes fall within the dimensions of the frame
    start_x, start_y = (max(0, start_x), max(0, start_y))
    end_x, end_y = (min(w - 1, end_x), min(h - 1, end_y))

    # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
    face = frame[start_y: end_y, start_x: end_x]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)

    return face, start_x, start_y, end_x, end_y


def detect_faces(frame, face_net):
    # Grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > MIN_CONFIDENCE:
            face, start_x, start_y, end_x, end_y = extract_face_roi(frame, detections[0, 0, i, 3:7])
            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    return faces, locations


def predict_mask(faces, mask_net):
    predictions = []

    # Only make a predictions IF at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)

    # Return a 2-tuple of the face locations and their corresponding locations
    return predictions


if not os.path.exists(MODEL):
    print("[INFO] Model does not exist")
    exit(1)

# Load our serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the face mask detector model from disk
print("[INFO] Loading face mask detector model...")
mask_net = load_model(MODEL)

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


def draw_roi(frame, box, prediction):
    # Unpack the bounding box and predictions
    (start_x, start_y, end_x, end_y) = box
    (mask, withoutMask) = prediction
    # Determine the class label and color we'll use to draw the bounding box and text
    label = "Mask" if mask > withoutMask else "No Mask"
    color = (52, 173, 72) if label == "Mask" else (0, 0, 255)
    # Include the probability in the label
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    font_scale = 0.45
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_padding = 10

    # Display the label and bounding box rectangle on the output frame
    (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]

    cv2.rectangle(frame, (start_x - 1, start_y - text_height - (text_padding * 2)),
                  (start_x + text_width + (text_padding * 2), start_y), color, -1)
    cv2.putText(frame, label, (start_x + text_padding, start_y - text_padding), font, font_scale, (255, 255, 255, 255),
                1)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)


# Loop over the frames from the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    # Detect faces in the frame and determine if they are wearing a face mask or not
    faces, locations = detect_faces(frame, face_net)
    predictions = predict_mask(faces, mask_net)

    # Loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locations, predictions):
        draw_roi(frame, box, pred)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
