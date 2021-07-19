import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fgbg = cv2.createBackgroundSubtractorMOG2()

color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_black = (0, 0, 0)

kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)
kernel10 = np.ones((10, 10), np.uint8)

def drawer_hand(frame):

    frame = frame.copy()
    fgmask = frame

    fgbg.setBackgroundRatio(0.005)
    fgmask = fgbg.apply(frame, fgmask)

    erode = cv2.erode(fgmask, kernel3, iterations=1)

    #blur = cv2.blur(erode, (3, 3))

    open = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel5)

    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel5)

    cv2.imshow("MASK", close)

    contours, _ = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        aprox = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        cv2.drawContours(frame, [aprox], -1, color_green, 2)

    cv2.imshow("HAND", frame)

def init():

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            return

        frame = cv2.flip(frame, 1)
        alt, lar = frame.shape[:2]

        roi = frame[int(alt/2):alt, int(lar/2)+50:lar]

        drawer_hand(roi)

        cv2.rectangle(roi, (2, 2), (lar - 2, alt - 2), color_red, 2)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    init()