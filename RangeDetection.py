import cv2
import numpy as np


# color in space BGR
orange_color = [46, 64, 255]
blue_color = [235, 173, 138]

kernel = np.ones((10, 10), np.uint8)
camera = cv2.VideoCapture("sources/futebol1.mp4")


def findRangeHSVColor(bgr, thresh=60):

    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    min = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    max = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    return min, max


def createMaskByColor(img, min, max):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, min, max)

    bitwise = cv2.bitwise_and(img, img, mask=mask)

    res_bgr = cv2.cvtColor(bitwise, cv2.COLOR_HSV2BGR)

    res_gray = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_OTSU)

    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return close


def init():

    min, max = findRangeHSVColor(orange_color, 40)

    idx = 0

    while camera.isOpened():

        ret, frame = camera.read()

        if not ret:
            break

        resize = cv2.resize(frame, (840, 480), fx=0, fy=0,
                            interpolation=cv2.INTER_CUBIC)

        mask = createMaskByColor(resize, min, max)
        cv2.imshow("mask", mask)

        contours, hier = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if(h >= (1.5) * w):
                cv2.rectangle(resize, (x, y), (x+w, y+h), orange_color, -1)

        cv2.imshow("frame", resize)

        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    init()
