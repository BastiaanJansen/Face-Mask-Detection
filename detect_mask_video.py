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

    # compute the (x, y)-coordinates of the bounding box for the object
    box = detection * np.array([w, h, w, h])
    start_x, start_y, end_x, end_y = box.astype("int")
    # ensure the bounding boxes fall within the dimensions of the frame
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
    # grab the dimensions of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locations = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > MIN_CONFIDENCE:
            face, start_x, start_y, end_x, end_y = extract_face_roi(frame, detections[0, 0, i, 3:7])
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    return faces, locations


def predict_mask(faces, mask_net):
    predictions = []

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=32)
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return predictions


if not os.path.exists(MODEL):
    print("[INFO] Model does not exist")
    exit(1)

# load our serialized face detector model from disk
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] Loading face mask detector model...")
mask_net = load_model(MODEL)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    # detect faces in the frame and determine if they are wearing a face mask or not
    faces, locations = detect_faces(frame, face_net)
    predictions = predict_mask(faces, mask_net)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locations, predictions):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
