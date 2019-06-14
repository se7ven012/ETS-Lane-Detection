import re

import cv2
import numpy


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


def cvtPIL(src, code):
    '''
    cvtPIL(src, code) -> dst.
    The PIL is a 3-channel RGB image and the code should be cv2.RGB2...
    @param src input what the getWinPic() return.
    @param code color space conversion code (see #ColorConversionCodes)...
    '''
    return cv2.cvtColor(numpy.asarray(src, numpy.uint8), code)
