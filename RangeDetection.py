import cv2
import numpy as np


# color in space BGR
blue_color = [231, 195, 142]
red_color = [89, 92, 214]

camera = cv2.VideoCapture("sources/futebol.mp4")

janelas = []


def show(name, frame):

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    """ open new windows in position 50x50 """
    if name not in janelas:
        cv2.moveWindow(name, 50, 50)
        janelas.append(name)

    cv2.imshow(name, frame)


def findRangeHSVColor(bgr, thresh=40):

    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    min = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    max = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    return min, max


def createMaskByColor(img, min, max):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, min, max)

    kernel = np.ones((15, 15), np.uint8)

    close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return close


def init():

    frameskip = 1

    min, max = findRangeHSVColor(red_color, 38)

    while camera.isOpened():

        ret, frame = camera.read()

        if not ret:
            break

        resize = cv2.resize(frame, (840, 480), fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)

        mask = createMaskByColor(resize, min, max)
        show("mask", mask)

        contours, hier = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if(h <= 20 and h >= (1.3)*w):

                a = 4 if h <= 18 else 2
                l = 4 if w <= 10 else 2

                posX = x-10
                posY = y-10

                width = posX + (w*l)
                height = posY + (h*a)

                cv2.rectangle(resize, (posX, posY),
                              (width, height), (255, 255, 255), 2)

        show("frame", resize)

        if cv2.waitKey(15) == 27:
            break

    # camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    init()
