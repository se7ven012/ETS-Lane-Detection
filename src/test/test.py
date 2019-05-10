import time

import numpy as np
import cv2

import base
import oe


def main():
    isFollowing = False

    curBallRoi = None
    preBallRoi = 0
    preBalls = []
    curTarget = (0, 0)
    preTarget = (0, 0)

    drawing = 0
    counter = 0

    win = base.getWinFromTitle(base.getWins(), "1.mp4")
    if not win:
        print("cannot find the window")
        return False

    while True:
        image = base.getWinPic(win[0])
        if image is False:
            print("can't get image")
            break

        src = base.cvtPIL(image, cv2.COLOR_RGB2BGR)

        if src is None:
            print("img is empty")
            counter = 0
            continue

        drawing = src.copy()

        if isFollowing is True:
            hs, he, ws, we = oe.extendRect(preBallRoi, src.shape, 1, 20)
            curBallRoi = src[hs:he, ws:we]

            roiGray = cv2.cvtColor(curBallRoi, cv2.COLOR_BGR2GRAY)
            _, roiBinary = cv2.threshold(
                roiGray, 127, 255, cv2.THRESH_BINARY_INV)
            roiBinary = oe.Hole(roiBinary, 0)
            roiBinary = cv2.erode(roiBinary, smallKernel, iterations=1)
            roiBinary = cv2.dilate(roiBinary, smallKernel, iterations=1)

            cv2.imshow("roiGray", roiGray)
            cv2.imshow("roiBinary", roiBinary)

            contours, _ = cv2.findContours(
                roiBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(ws, hs))

            checkFind = False
            for contour in contours:
                rect = cv2.boundingRect(contour)

                if rect[2] < 5 and rect[3] < 5:
                    continue

                if oe.approxLenth(rect[2:4], 0.5):
                    cv2.rectangle(drawing, (rect[0], rect[1]), (
                        rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255))

                    preBalls.append((rect[0]+rect[2]//2, rect[1]+rect[3]//2))
                    checkFind = True
                    isFollowing = True
                    preBallRoi = rect
                    counter += 1
                    print(" state: 2 following ")

            if checkFind is False:
                print(" state: 3 missing ")
                isFollowing = False
                preBalls.clear()
        else:
            hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            G = cv2.inRange(hsv, minG, maxG)
            G = cv2.erode(G, bigKernel, iterations=1)
            G = cv2.dilate(G, bigKernel, iterations=1)

            contours, _ = cv2.findContours(
                G, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            checkFind = False
            for contour in contours:
                rect = cv2.boundingRect(contour)

                if oe.approxLenth(rect[2:4], 0.5):
                    cv2.rectangle(drawing, (rect[0], rect[1]), (
                        rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255))

                    preBalls.append((rect[0]+rect[2]//2, rect[1]+rect[3]//2))
                    checkFind = True
                    isFollowing = True
                    preBallRoi = rect
                    print(" state: 1 find ")

            if checkFind is False:
                print(" state: -1 can't find ")

        if len(preBalls) is 2:
            curTarget = oe.get2vsv(preBalls[1], preBalls[0])

            cv2.circle(drawing, (preBallRoi[0]+preBallRoi[2]//2,
                                 preBallRoi[1]+preBallRoi[3]//2), 2, (255, 0, 0), -1)
            cv2.circle(drawing, curTarget, 2, (0, 0, 255), -1)
            cv2.circle(drawing, preTarget, 2, (0, 255, 0), -1)

            preTarget = curTarget
            preBalls.pop(0)

        print(" target: x:", curTarget[0], " y:", curTarget[1])
        cv2.imshow("drawing", drawing)

        if curBallRoi is not None:
            cv2.imshow("curBallRoi", curBallRoi)

        cv2.imshow("src", src)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    minG = (35, 43, 46)
    maxG = (77, 255, 255)

    smallKernel = np.ones((3, 3), np.uint8)
    bigKernel = np.ones((7, 7), np.uint8)

    start = time.time()
    main()
    end = time.time()
    print(end - start)
