import cv2 
import os

def resize_imgs(dir_src, dir_dst, to_size):
    names = os.listdir(dir_src)
    for name in names:
        if not name.endswith(".jpg"):
            continue
        filename = "{}/{}".format(dir_src, name)
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_sm = cv2.resize(img, to_size)
        dname = "{}/{}".format(dir_dst, name)
        cv2.imwrite(dname, img_sm)

def resize_to_small():
    dir_src = "few-shot_exam/resize_src"
    dir_dst = "few-shot_exam/resize_dst"
    resize_imgs(dir_src, dir_dst, (64,64))


def resize_to_256():
    dir_src = "few-shot-plants/planets_256_enhanced"
    dir_dst = "few-shot-plants/planets_256_resized"
    resize_imgs(dir_src, dir_dst, (256,256))


if __name__ == "__main__":
    #resize_to_small()
    resize_to_256()
