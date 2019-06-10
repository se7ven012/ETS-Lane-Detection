import math
import numpy as np
import cv2


def boundingRect(rotateRect):
    center = rotateRect[0]
    size = rotateRect[1]
    angle = rotateRect[2]

    _angle = angle*math.pi/180.
    b = math.cos(_angle)*0.5
    a = math.sin(_angle)*0.5

    pt = [[0, 0], [0, 0], [0, 0], [0, 0]]
    pt[0][0] = center[0] - a*size[0] - b*size[1]
    pt[0][1] = center[1] + b*size[0] - a*size[1]
    pt[1][0] = center[0] + a*size[0] - b*size[1]
    pt[1][1] = center[1] - b*size[0] - a*size[1]
    pt[2][0] = 2*center[0] - pt[0][0]
    pt[2][1] = 2*center[1] - pt[0][1]
    pt[3][0] = 2*center[0] - pt[1][0]
    pt[3][1] = 2*center[1] - pt[1][1]

    r = [int(min(min(min(pt[0][0], pt[1][0]), pt[2][0]), pt[3][0])),
         int(min(min(min(pt[0][1], pt[1][1]), pt[2][1]), pt[3][1])),
         int(max(max(max(pt[0][0], pt[1][0]), pt[2][0]), pt[3][0])),
         int(max(max(max(pt[0][1], pt[1][1]), pt[2][1]), pt[3][1]))]
    r[2] -= r[0] - 1
    r[3] -= r[1] - 1

    return (r)


def Hole(src, mode=0):

    h, w = src.shape[:2]
    temp = np.zeros((h+2, w+2), np.uint8)
    temp[1:h+1, 1:w+1] = src
    mask = np.zeros((h+4, w+4), np.uint8)
    cv2.floodFill(temp, mask, (0, 0), 255)
    cut = temp[1:h+1, 1:w+1]

    if mode == 0:
        dst = cv2.bitwise_or(src, (255 - cut))
    elif mode == 1:
        dst = 255 - cut
    return dst


def convContoursIntersectiponRate(ps1, ps2):
    area, _ = cv2.intersectConvexConvex(np.array(ps1), np.array(ps2))
    return 2.0*area/(cv2.contourArea(ps1)+cv2.contourArea(ps2))


def maxLimiter(x, max_x):
    return max_x if x > max_x else x


def minLimiter(x, min_x):
    return min_x if x < min_x else x


def reduce(value, factor):
    return value * (1 - factor)


def enlarge(value, factor):
    return value * (1 + factor)


def approxLenth(shape, factor):
    return (shape[0] > reduce(shape[1], factor)) and (shape[0] < enlarge(shape[1], factor))


def get2vsv(pre, cur):
    return (2*pre[0]-cur[0], 2*pre[1]-cur[1])


def extendRect(rect, shape, mode, factor=0.1):
    '''
    extendRect(rect, shape, mode, factor) -> retval.
    @param shape : h, w, (e.g. src.shape)
    @param mode : if mode = 0, extend by multifly a factor, if mode = 1, extend by add a factor
    '''
    max_h, max_w = shape[:2]
    extend_w, increment_w, extend_h, increment_h = 0, 0, 0, 0

    if mode is 0:
        extend_w, extend_h = enlarge(rect[2], factor), enlarge(rect[3], factor)
        increment_w, increment_h = extend_w - rect[2], extend_h - rect[3]

    elif mode is 1:
        extend_w, extend_h = rect[2] + factor, rect[3] + factor
        increment_w, increment_h = factor, factor

    else:
        print(" extendRect MODE ERROR! ")

    w_start = int(minLimiter(rect[0] - increment_w, 0))
    w_end = int(maxLimiter(rect[0] + extend_w, max_w))
    h_start = int(minLimiter(rect[1] - increment_h, 0))
    h_end = int(maxLimiter(rect[1] + extend_h, max_w))

    return (h_start, h_end, w_start, w_end)
