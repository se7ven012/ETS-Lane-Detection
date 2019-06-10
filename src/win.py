import cv2

import base


def funcname(parameter_list):
    


def main():
    win = base.getWinFromTitle(base.getWins(), "Simulator")
    if not win:
        print("can't find the window", end='')
        return False

    while True:
        image = base.getWinPic(win[0])
        if image is False:
            print("can't get image", end="")
            break

        src = base.cvtPIL(image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 127, 255)

        # cv2.imshow("src", src)
        cv2.imshow("gray", gray)
        cv2.imshow("canny", canny)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
