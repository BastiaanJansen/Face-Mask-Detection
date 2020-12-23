import cv2

def draw_roi(frame, box, prediction):
    # unpack the bounding box and predictions
    (start_x, start_y, end_x, end_y) = box
    (mask, without_mask) = prediction
    # determine the class label and color we'll use to draw the bounding box and text
    label = "Mask" if mask > without_mask else "No Mask"
    color = (52, 173, 72) if label == "Mask" else (0, 0, 255)
    # include the probability in the label
    label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

    font_scale = 0.45
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_padding = 10

    # display the label and bounding box rectangle on the output frame
    (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]

    cv2.rectangle(frame, (start_x - 1, start_y - text_height - (text_padding * 2)),
                  (start_x + text_width + (text_padding * 2), start_y), color, -1)
    cv2.putText(frame, label, (start_x + text_padding, start_y - text_padding), font, font_scale, (255, 255, 255, 255),
                1)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)