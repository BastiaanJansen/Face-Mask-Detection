from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2


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


def detect_faces(frame, face_net, min_confidence):
    try:
        # Construct a blob from the frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    except:
        return [], []

    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initialize our list of faces, their corresponding locations
    faces = []
    locations = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > min_confidence:
            face, start_x, start_y, end_x, end_y = extract_face_roi(frame, detections[0, 0, i, 3:7])
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    return faces, locations


def predict_mask(faces, mask_net, batch_size=32):
    predictions = []

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # For faster inference we'll make batch predictions on all faces at the same time
        faces = np.array(faces, dtype="float32")
        predictions = mask_net.predict(faces, batch_size=batch_size)

    return predictions
