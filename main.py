from imutils.video import VideoStream
from tensorflow.keras.models import load_model
from detect_mask_video import detect_faces, predict_mask
from helpers.draw import draw_face_mask_roi, draw_text
import time
import cv2
import os

MODEL = "models/mask_detector.model"
MIN_CONFIDENCE = 0.50
VIDEO_STREAM_SOURCE = 0

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

def video(model, min_confidence=0.50, video_stream_source=0):
    mask_net = load_trained_model(model)
    face_net = load_face_detector()

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=video_stream_source).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        frame = vs.read()
        # detect faces in the frame and determine if they are wearing a face mask or not
        faces, locations = detect_faces(frame, face_net, min_confidence)
        predictions = predict_mask(faces, mask_net)

        with_masks = 0
        without_masks = 0

        for mask, without_mask in predictions:
            if mask > without_mask:
                with_masks += 1
            else:
                without_masks += 1

        padding = 10

        text_width, text_height = draw_text(frame, f"Faces: {len(faces)}", (0, 0), padding, COLOR_BLACK, COLOR_WHITE)
        draw_text(frame, f"Without mask: {without_masks}", (0, text_height * 2), padding, COLOR_WHITE, (0, 0, 255))
        draw_text(frame, f"With mask: {with_masks}", (0, text_height), padding, COLOR_WHITE, (52, 173, 72))

        # loop over the detected face locations and their corresponding locations
        for box, prediction in zip(locations, predictions):
            draw_face_mask_roi(frame, box, prediction)

        # show the output frame
        if frame.any():
            cv2.imshow("Mask detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()


def load_trained_model(name):
    # load the face mask detector model from disk
    print("[INFO] Loading face mask detector model...")
    return load_model(name)


def load_face_detector():
    # load our serialized face detector model from disk
    print("[INFO] Loading face detector model...")
    proto_txt_path = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weights_path = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    return cv2.dnn.readNet(proto_txt_path, weights_path)


if __name__ == "__main__":
    if not os.path.exists(MODEL):
        print("[INFO] Model does not exist")
        exit(1)
    video(MODEL, MIN_CONFIDENCE, VIDEO_STREAM_SOURCE)
