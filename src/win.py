#%%
import cv2
from PIL import Image
from scipy import stats
import numpy as np
import base
import windowsBase


def main():
    i = 0 
    win = base.getWinFromTitle(windowsBase.getWins(), "Simulator")
    if not win:
        print("can't find the window", end='')
        return False

    while True:
        image = windowsBase.getWinPic(win[0])
        if image is False:
            print("can't get image", end="")
            break
        #height=753
        #width=1290
        src = base.cvtPIL(image, cv2.COLOR_RGB2BGR)
        src_speed=src[505:522, 1000:1058]
        np_speed = np.array(src_speed)
        print(stats.mode(np_speed)[0][0])


        #cv2.imshow("src", src)
        #cv2.imshow("src_speed", src_speed)
        # cv2.imshow("gray", gray)
        # cv2.imshow("canny", canny)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# %%
