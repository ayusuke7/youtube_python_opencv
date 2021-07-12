import cv2 as cv

cap = cv.VideoCapture("sources/video.mp4")

face_cascade = cv.CascadeClassifier()

eyes_cascade = cv.CascadeClassifier()


def frame_resize(frame):
    resize = cv.resize(frame, (960, 520), fx=0, fy=0,
                       interpolation=cv.INTER_CUBIC)
    return resize


def detect_and_show(frame):

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)

        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faceROI = frame_gray[y:y+h, x:x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (0, 255, 255), 2)

    return frame


def init():

    if not eyes_cascade.load("haars/haarcascade_eye.xml"):
        exit(0)

    if not face_cascade.load("haars/haarcascade_frontalface_alt.xml"):
        exit(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = frame_resize(frame)

        frame = detect_and_show(frame)

        cv.imshow("frame", frame)

        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    init()
