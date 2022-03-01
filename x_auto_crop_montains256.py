import cv2 
import os
#import matplotlib.pyplot as plt
import numpy as np
import pprint
#import json

PAD_EXP = 128
s_points = []
s_click_count = 0
s_click_quit = False

IMG_SIZE = 256
IMG_STRECH = 64
IMG_SLIDE = 64

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def crop_imgs_resized_img(args, params):
    #img_org = params.img_org
    img_rsz = params.img_rsz

    #rsz_height = img_rsz.shape[0]
    rsz_width = img_rsz.shape[1]

    no_slide = (rsz_width - IMG_SIZE) // IMG_SLIDE

    for n in range(no_slide):
        for m in [0, 1]:
            pt1 = (n * IMG_SLIDE + m * IMG_STRECH, m * IMG_STRECH)
            pt2 = (n * IMG_SLIDE + IMG_SIZE + m * IMG_STRECH, IMG_SIZE + m * IMG_STRECH)

            height = pt2[1] - pt1[1]
            width = pt2[0] - pt1[0]
            print(n, m, pt1, pt2, height, width) 

            img_crop = np.zeros((height, width, img_rsz.shape[2]))
            img_crop = img_rsz[pt1[1]:pt2[1], pt1[0]:pt2[0]]

            img_path = "{}/{}_{:02d}-{:02d}.jpg".format(args.dir_dst, args.img_name, n, m)
            cv2.imwrite(img_path, img_crop)
            print(colorstr("green", "generate {} out of {}".format(img_path, args.img_fname)))

def resize_img(img_org):
    img_height = img_org.shape[0]
    img_width = img_org.shape[1]

    ratio = img_height / (IMG_SIZE + IMG_STRECH)
    rsz_height = IMG_SIZE + IMG_STRECH
    rsz_width = int(img_width / ratio + 0.5)

    img_rsz = cv2.resize(img_org, (rsz_width, rsz_height))
    print("org {} -> {}".format(img_org.shape, img_rsz.shape))
    return img_rsz 

def crop_file(filename, args):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} not exist")

    dir_src = os.path.dirname(filename)
    fnm_src = os.path.basename(filename).split(".")[0]
    img_name = fnm_src

    img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
    img_rsz = resize_img(img_org)

    args.dir_src = dir_src
    args.img_fname = fnm_src
    args.img_name = img_name

    pprint.pprint(args)

    params = dotdict()
    params.img_org = img_org
    params.img_rsz = img_rsz

    crop_imgs_resized_img(args, params)


if __name__ == "__main__":
    src_dir = "few-shot-montains/montains_org"
    args = dotdict({"dir_dst": "few-shot-montains/montains/img"})

    names = os.listdir(src_dir)
    for name in names:
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            filename = "{}/{}".format(src_dir, name)
            crop_file(filename, args)

