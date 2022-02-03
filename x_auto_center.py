import cv2 
#import os
#import matplotlib.pyplot as plt
import numpy as np
#import pprint
#import json

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_img_thresh(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)
    #img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    #img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    #blur = cv2.GaussianBlur(img_gray, (16, 16), cv2.BORDER_DEFAULT)

    img_dst = np.zeros_like(img_blur)
    cv2.normalize(img_blur, img_dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
    img_gray_balanced = img_dst

    canny_low = 150
    canny_high = 200
    #blur_kernal = (3, 3)

    img_gray_balanced_int8 = np.uint8(img_gray_balanced)
    img_gray_balanced_canny = cv2.Canny(img_gray_balanced_int8, canny_low, canny_high)
    
    #img_gray_balanced_canny_blur = cv2.blur(img_gray_balanced_canny, blur_kernal)
    #img_gray_enhanced = img_gray_balanced_canny_blur + img_gray_balanced

    th, img_thresh = cv2.threshold(img_gray_balanced_canny, 200, 255, cv2.THRESH_BINARY_INV)
    return img_thresh

def find_cnts_img_thresh(img_thresh):
    cnts, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def get_center_of_contour(cnt):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def find_img_center(img):
    ix, iy = img.shape[1], img.shape[0]
    img_thresh = find_img_thresh(img)
    cnts = find_cnts_img_thresh(img_thresh)
    print("num of cnts", len(cnts))
    if len(cnts) == 1:
        cX, cY = get_center_of_contour(cnts[0])
        return (cX, cY)
    elif len(cnts) > 1:
        cts = []
        cix, ciy = ix // 2,  iy // 2
        cx, cy = 0, 0
        for cnt in cnts:
            cX, cY = get_center_of_contour(cnt)
            if (cX - cix) ** 2 + (cY - ciy) ** 2 < (cx - cix) ** 2 + (cy - ciy) ** 2:
                cx, cy = cX, cY
            cts.append((cX, cY))
        return cx, cy
    else:
        return (0, 0)

def find_center_box(img, cx, cy):
    ix, iy = img.shape[1], img.shape[0]

    dx = min(cx, ix - cx) 
    dy = min(cy, iy - cy)
    dd = min(dx, dy)

    pt0 = (cx - dd, cy - dd)
    pt1 = (cx + dd, cy + dd)

    return pt0, pt1

def mark_center(img, cx, cy):
    img_draw = img.copy()
    img_draw = find_img_thresh(img_draw)

    pt0, pt1 = find_center_box(img, cx, cy)

    cv2.circle(img_draw, (cx, cy), 5, (0, 0, 0), -1)
    cv2.rectangle(img_draw, pt0, pt1, (0, 0, 255), 1)

    cv2.imshow("center img", img_draw)
    cv2.waitKey(0)


def center_img(filename, args):
    img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
    cx, cy = find_img_center(img_org)

    if cx != 0 and cy != 0:
        print("HW_img", img_org.shape[1], img_org.shape[0])
        print("CT_img", cx, cy)
        mark_center(img_org, cx, cy)
    else:
        print("skip as the image has more than one contours")


if __name__ == "__main__":

    args = dotdict({"dir_dst": "few-shot-plants/planets-128/img",
                    "size": 128})

    #filename = "few-shot-plants/planets/img/S82_05.jpg"
    #filename = "few-shot-plants/planets/img/S82_03.jpg"
    #filename = "few-shot-plants/planets/img/S81_05.jpg"
    #filename = "few-shot-plants/planets/img/S75_11.jpg"
    filename = "few-shot-plants/planets/img/S71_26.jpg"
    filename = "few-shot-plants/planets/img/S01_09.jpg"
    filename = "few-shot-plants/planets/img/S01_29.jpg"
    filename = "few-shot-plants/planets/img/S03_13.jpg"
    filename = "few-shot-plants/planets/img/S05_02.jpg"
    filename = "few-shot-plants/planets/img/S05_12.jpg"
    center_img(filename, args)
    pass
