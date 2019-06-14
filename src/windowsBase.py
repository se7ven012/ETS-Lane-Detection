import PIL.ImageGrab
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
