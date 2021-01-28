import cv2
import sys

source_video = "sources/zelda.mp4"
cap = cv2.VideoCapture(source_video)

if __name__ == "__main__":

    ret, first_frame = cap.read()

    if not ret:
        sys.exit()

    box = cv2.selectROI("SELECT ROI", first_frame, fromCenter=True, showCrosshair=False)
    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(first_frame, box)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        timer = cv2.getTickCount()

        #box (alt, lar, x, y)
        ok, box = tracker.update(frame)

        if ok:
            pt1 = (box[0], box[1])
            pt2 = ((box[0] + box[2]), (box[1]+box[3]))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2, 1)
        else:
            print("FALHOU")

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
