import cv2


def draw_text(frame, text, location, padding, text_color, background_color):
    font_scale = 0.45
    font = cv2.FONT_HERSHEY_SIMPLEX

    start_x, start_y = location

    text_width, text_height = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    box_x = start_x - 1
    box_y = start_y + text_height + (padding * 2) - 1

    cv2.rectangle(frame, (box_x, box_y),
                  (start_x + text_width + (padding * 2), start_y), background_color, -1)
    cv2.putText(frame, text, (start_x + padding, start_y + (padding * 2)), font, font_scale, text_color,
                1)

    return text_width + padding * 2, text_height * 3


def draw_face_mask_roi(frame, box, prediction):
    # unpack the bounding box and predictions
    (start_x, start_y, end_x, end_y) = box
    (mask, without_mask) = prediction
    # determine the class label and color we'll use to draw the bounding box and text
    label = "Mask" if mask > without_mask else "No Mask"
    color = (52, 173, 72) if label == "Mask" else (0, 0, 255)
    # include the probability in the label
    label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

    padding = 10

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
    draw_text(frame, label, (start_x, start_y - padding * 3), padding, (255, 255, 255), color)
