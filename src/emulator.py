import cv2
import time


def image_generator(filename):
    start = time.time()
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return
    while True:
        curr = time.time()
        cap.set(cv2.CAP_PROP_POS_MSEC, (curr - start) * 1000)
        res = cap.read()
        if res:
            yield res[1]
        else:
            return


