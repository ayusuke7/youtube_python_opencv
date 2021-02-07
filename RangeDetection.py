import cv2
import numpy as np


# color in space BGR
orange_color = [46, 64, 255]
blue_color = [235, 173, 138]

kernel = np.ones((5, 5), np.uint8)
camera = cv2.VideoCapture("sources/cubo.mkv")

janelas = []


def show(name, frame):

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    """ open new windows in position 50x50 """
    if name not in janelas:
        cv2.moveWindow(name, 50, 50)
        janelas.append(name)

    cv2.imshow(name, frame)


def findRangeHSVColor(bgr, thresh=60):

    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    min = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    max = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    return min, max


def createMaskByColor(img, min, max):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, min, max)

    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)

    return close


def init():

    min, max = findRangeHSVColor(orange_color, 40)

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        resize = cv2.resize(frame, (840, 480), fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)

        mask = createMaskByColor(resize, min, max)
        show("mask", mask)

        contours, hier = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(resize, (x, y), (x+w, y+h), orange_color, -1)

        show("frame", resize)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    init()
