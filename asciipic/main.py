from xml.etree.ElementPath import get_parent_map
import cv2
import numpy as np
from numba import jit


@jit(nopython=True)
def to_ascii_art(frame, images, box_height=12, box_width=16):
    height, width = frame.shape
    for i in range(0, height, box_height):
        for j in range(0, width, box_width):
            roi = frame[i:i + box_height, j:j + box_width]
            best_match = np.inf
            best_match_index = 0
            for k in range(1, images.shape[0]):
                total_sum = np.sum(np.absolute(np.subtract(roi, images[k])))
                if total_sum < best_match:
                    best_match = total_sum
                    best_match_index = k
            roi[:, :] = images[best_match_index]
    return frame


def generate_ascii_letters():
    images = []

    letters = """ \\" '(),-./:;[]_`{><|}~ <|> _* '"""
    for letter in letters:
        img = np.zeros((12, 16), np.uint8)
        img = cv2.putText(img, letter, (0, 11),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200)
        images.append(img)
    return np.stack(images)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Setup camera
cap = cv2.VideoCapture(0)
# Set a smaller resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1028)
images = generate_ascii_letters()
while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    can = cv2.Canny(gb, 80, 60)
    ascii_art = to_ascii_art(can, images)
    cv2.imshow('ASCII ART', ascii_art)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(20) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
