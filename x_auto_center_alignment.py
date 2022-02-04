import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np
#import pprint
#import json

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def find_img_balanced(img_gray):
    #img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    #img_blur = cv2.GaussianBlur(img_blur, (5, 5), cv2.BORDER_DEFAULT)
    #blur = cv2.GaussianBlur(img_gray, (16, 16), cv2.BORDER_DEFAULT)

    img_dst = np.zeros_like(img_gray)
    cv2.normalize(img_gray, img_dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
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

def find_img_thresh(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)

    thv = 250
    found_right_cnt = False
    for thv in range(254, 200, -2):
        th, img_thresh = cv2.threshold(img_blur, thv, 255, cv2.THRESH_BINARY_INV)
        cnts, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if len(cnt) > 4:
                found_right_cnt = True
                break
        if found_right_cnt:
            break

    return img_thresh, cnts

def get_center_of_contour(cnt):
    try:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    except:
        return 0, 0

def find_img_center(img):
    ix, iy = img.shape[1], img.shape[0]
    img_thresh, cnts = find_img_thresh(img)
    print("num of cnts", len(cnts))
    if len(cnts) == 1:
        cX, cY = get_center_of_contour(cnts[0])
        return (cX, cY), 1
    elif len(cnts) > 1:
        cts = []
        cix, ciy = ix // 2,  iy // 2
        cx, cy = 0, 0
        for cnt in cnts:
            cX, cY = get_center_of_contour(cnt)
            if (cX - cix) ** 2 + (cY - ciy) ** 2 < (cx - cix) ** 2 + (cy - ciy) ** 2:
                cx, cy = cX, cY
            cts.append((cX, cY))
        return (cx, cy), len(cnts)
    else:
        return (0, 0), 0

def find_center_box(img, cx, cy):
    ix, iy = img.shape[1], img.shape[0]

    dx = min(cx, ix - cx) 
    dy = min(cy, iy - cy)
    dd = min(dx, dy)

    pt0 = (cx - dd, cy - dd)
    pt1 = (cx + dd, cy + dd)

    return pt0, pt1

def mark_center_debug(img_org, img_new, cx, cy, pt0, pt1):
    img_draw = img_org.copy()
    cv2.circle(img_draw, (cx, cy), 5, (0, 0, 0), -1)
    cv2.rectangle(img_draw, pt0, pt1, (0, 0, 255), 1)

    img_thread, _ = find_img_thresh(img_org.copy())

    plt.figure()
    plt.subplot(141)
    plt.imshow(img_org)
    plt.subplot(142)
    plt.imshow(img_thread)
    plt.subplot(143)
    plt.imshow(img_draw)
    plt.subplot(144)
    plt.imshow(img_new)
    #plt.show()
    print("Pause to showup plt.imshow")
    plt.close()


def gen_aligned_img(img_org, pt0, pt1,  cx, cy, filename, args):

    nh = pt1[1] - pt0[1]
    nw = pt1[0] - pt0[0]
    img_new = np.zeros((nh, nw, img_org.shape[2]), dtype=img_org.dtype)
    #img_new = img_org[pt0[0]:pt1[0], pt0[1]:pt1[1]]
    img_new = img_org[pt0[1]:pt1[1], pt0[0]:pt1[0]]
    
    bname = os.path.basename(filename)
    pname = "{}/{}_{}x{}.jpg".format(args.dir_dst, bname.split(".")[0], nh, nw)

    if args.debug:
        mark_center_debug(img_org, img_new, cx, cy, pt0, pt1)

    cv2.imwrite(pname, img_new) 
    return pname
    

def center_img_and_align(filename, args):
    img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
    (cx, cy), counts = find_img_center(img_org)

    if cx != 0 and cy != 0 and counts > 0:
        if args.debug:
            print("HW_img", img_org.shape[1], img_org.shape[0])
            print("CT_img", cx, cy)
            print("cn_img", counts)

        pt0, pt1 = find_center_box(img_org, cx, cy)

        return gen_aligned_img(img_org, pt0, pt1, cx, cy, filename, args)

    else:
        print("skip as the image has more than one contours")
        return None

def do_exam_file():
    args = dotdict({"dir_dst": "few-shot-plants/planets-aligned",
                   "debug": True})
    os.makedirs(args.dir_dst, exist_ok=True)

    #filename = "few-shot-plants/planets/img/S82_05.jpg"
    #filename = "few-shot-plants/planets/img/S82_03.jpg"
    
    #filename = "few-shot-plants/planets/img/S81_05.jpg"
    
    #filename = "few-shot-plants/planets/img/S75_11.jpg"
    #filename = "few-shot-plants/planets/img/S71_26.jpg"
    #filename = "few-shot-plants/planets/img/S01_09.jpg"
    #filename = "few-shot-plants/planets/img/S01_29.jpg"
    
    #filename = "few-shot-plants/planets/img/S03_13.jpg"
    
    #filename = "few-shot-plants/planets/img/S05_02.jpg"
    #filename = "few-shot-plants/planets/img/S05_12.jpg"

    filename = "few-shot-plants/planets/img/S01_04.jpg"
    filename = "few-shot-plants/planets/img/S01_08.jpg"
    filename = "few-shot-plants/planets/img/S01_14.jpg"
    filename = "few-shot-plants/planets/img/S01_18.jpg"
    filename = "few-shot-plants/planets/img/S01_22.jpg"

    filename = "few-shot-plants/planets/img/S03_00.jpg"
    
    args.debug = True
    dst_path = center_img_and_align(filename, args)
    print(dst_path)

def do_exam_folder():
    args = dotdict({"dir_dst": "few-shot-plants/planets-aligned",
                    "debug": False})
    os.makedirs(args.dir_dst, exist_ok=True)

    dir_src = "few-shot-plants/planets/img"
    filenames = sorted(os.listdir(dir_src))
    filed_filenames = []
    for name in filenames:
        if not name.endswith(".jpg"):
            continue
        filename = "{}/{}".format(dir_src, name)
        dst_path = center_img_and_align(filename, args)
        if dst_path is not None:
            print(dst_path)
        else:
            filed_filenames.append(filename)


if __name__ == "__main__":
    #do_exam_file()
    do_exam_folder()


    
