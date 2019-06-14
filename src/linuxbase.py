import os
import subprocess

from Xlib import X
from PIL import Image

from ewmh import EWMH


def getWins():
    '''
    getWins() -> retval.
    get all active window id and title.
    '''
    ewmh = EWMH()
    wins, winHDs = [], ewmh.getClientListStacking()

    for winHD in winHDs:
        try:
            title = ewmh.getWmName(winHD)
            pid = ewmh.getWmPid(winHD)
        except:
            continue
        if title is not None:
            wins.append((winHD, title, pid))

    return wins


def getWinPic(id, method=None):
    '''
    getWinPic(hWnd/id, method=None) -> image
    return False if the window absent suddently.
    @param id the window handle.
    @param method get returns by Xlib.get_image() if None, else by command line maim in terminal if other.
    '''
    try:
        _geo = id.get_geometry()

        if method is None:
            _raw_image = id.get_image(
                _geo.x, _geo.y, _geo.width, _geo.height, X.ZPixmap, 0xffffffff)

        else:
            filename = "temp"
            subprocess.call(['maim', filename, "-g", '%dx%d+%d+%d' %
                             (_geo.width, _geo.height, _geo.x, _geo.y+30)])
            im = Image.open(filename)
            os.unlink(filename)
            return im
    except:
        return False
    return Image.frombytes("RGB", (_geo.width, _geo.height), _raw_image.data, "raw", "BGRX")
