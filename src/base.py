import re

import PIL.ImageGrab
import cv2
import numpy
import win32gui


def getWinCallBack(id, titles):
    '''
    CallBack Function.
    @param titles get (id, title).
    '''
    # is a window? is enabled? is visible?
    if win32gui.IsWindow(id) and win32gui.IsWindowEnabled(id) and win32gui.IsWindowVisible(id):
        title = win32gui.GetWindowText(id)
        if title:
            titles.append((id, title))


def getWins():
    '''
    getWins() -> retval.
    get all active window id and title.
    '''
    wins = []
    win32gui.EnumWindows(getWinCallBack, wins)
    return wins


def getWinFromTitle(wins, word):
    '''
    getWinFromTitle(wins, name) -> id, title.
    return False if all titles are no match.
    @param wins what the getWins() return.
    @param word a key-word can find out the expected title.
    '''
    pattern = re.compile(word, re.S | re.I)
    win = [win for win in wins if pattern.findall(win[1])]
    return False if len(win) == 0 else win[0]


def getWinPic(id):
    '''
    getWinPic(hWnd/id) -> rect -> (left, top, right, bottom).
    return False if the window absent suddently.
    @param id the window handle.
    '''
    try:
        rect = win32gui.GetWindowRect(id)
    except:
        return False
    return PIL.ImageGrab.grab(rect)


def cvtPIL(src, code):
    '''
    cvtPIL(src, code) -> dst.
    The PIL is a 3-channel RGB image and the code should be cv2.RGB2...
    @param src input what the getWinPic() return.
    @param code color space conversion code (see #ColorConversionCodes)...
    '''
    return cv2.cvtColor(numpy.asarray(src, numpy.uint8), code)
