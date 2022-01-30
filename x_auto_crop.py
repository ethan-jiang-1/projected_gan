import cv2 
import os
import matplotlib.pyplot as plt
import numpy as np
import pprint
import json

PAD_EXP = 128


s_points = []
s_click_count = 0
s_click_quit = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def reset_statics():
    global s_points, s_click_count, s_click_quit
    s_points = []
    s_click_count = 0
    s_click_quit = False


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


def create_expanded(img_org):
    shape = img_org.shape
    shape_exp = (shape[0] + 2 * PAD_EXP, shape[1] + 2 * PAD_EXP, shape[2])
    val_bg = img_org[0][0]
    img_exp = np.zeros(shape_exp, dtype=img_org.dtype)
    img_exp.fill(val_bg[0])
    img_exp[PAD_EXP:img_org.shape[0] + PAD_EXP, PAD_EXP:img_org.shape[1] + PAD_EXP] = img_org
    return img_exp


def find_crop_pts_from_org(args, params):
    global s_points, s_click_count, s_click_quit
    reset_statics()

    cfg_fname = args.dir_src + "/" + args.img_fname + ".json"
    if os.path.isfile(cfg_fname):
        print(colorstr("green", f"#Found crop data {cfg_fname}, skip GUI to locate."))
        with open(cfg_fname, "rt") as f:
            json_str = f.read()
            crop_data = json.loads(json_str)
            params.crop = dotdict(crop_data)
            return True

    img_org = params.img_org
    img_draw = img_org.copy()

    print(colorstr("blue", "#Let's try to locate crop range by four conners..."))
    window_name = "Wait for center points of 4 conners {}".format(args.img_nums)
    cv2.namedWindow(window_name)

    def update_grid(img_draw, points):
        xn = [x[0] for x in points]
        yn = [y[1] for y in points]
        xns = sorted(np.array(xn))
        yns = sorted(np.array(yn))
        xmin = (xns[0] + xns[1]) // 2
        xmax = (xns[2] + xns[3]) // 2
        ymin = (yns[0] + yns[1]) // 2
        ymax = (yns[2] + yns[3]) // 2
        xdelta = (xmax - xmin) // (args.img_num_x - 1)
        ydelta = (ymax - ymin) // (args.img_num_y - 1)

        xnum = args.img_num_x
        ynum = args.img_num_y

        params.crop = dotdict()
        params.crop.xdelta = int(xdelta)
        params.crop.ydelta = int(ydelta)
        params.crop.xmin = int(xmin) 
        params.crop.xmax = int(xmax)
        params.crop.ymin = int(ymin)
        params.crop.ymax = int(ymax) 
        params.crop.xnum = int(xnum)
        params.crop.ynum = int(ynum)
        params.crop.pts = []
        for ix in range(0, xnum):
            for iy in range(0, ynum):
                x = xmin + xdelta * ix 
                y = ymin + ydelta * iy
                cv2.circle(img_draw, (x, y), 5, (0, 0, 0), -1)
                pt0, pt1 = (int(x - xdelta//2), int(y - ydelta // 2)), (int(x + xdelta // 2), int(y + ydelta // 2))
                cv2.rectangle(img_draw, pt0, pt1, (0, 0, 255), 1)
                params.crop.pts.append((pt0, pt1))
        return params.crop

    def capture_event(event, x, y, flags, params):
        global s_click_count, s_points
        if event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_draw, (x, y), 3, (0, 0, 0), -1)
            if len(s_points) < 4:
                s_points.append((x, y))
            if len(s_points) == 4:
                update_grid(img_draw, s_points)
            s_click_count += 1
        else:
            #print(event)
            pass

    cv2.setMouseCallback(window_name, capture_event)

    #cv2.rectangle(img_draw, (0, 0), (img_draw.shape[1], img_draw.shape[0]), (0, 0,255), 1)
    while True:
        cv2.imshow(window_name, img_draw)
        if cv2.waitKey(1) == 13:
            break
        if s_click_count > 4:
            break
        elif s_click_count == 4:
            if not s_click_quit:
                print(colorstr("blue", "click one more time to quit..."))
                s_click_quit = True

        title = window_name + ": {} clicked".format(s_click_count)
        cv2.setWindowTitle(window_name, title)

    cv2.setWindowTitle(window_name, "Is Crop ok? wait for key press...")
    cv2.destroyAllWindows()
    print(s_points)
    print(params.crop)
    print("\n")
    keys = input(colorstr("yellow", "#Is crop ok (yes/YES/y/Y) ? "))
    if keys.upper()[0] == "Y":
        json_str = json.dumps(params.crop)
        with open(cfg_fname, "wt+") as f:
            f.write(json_str)
        return True
    print(colorstr("blue", "#Let's try one more time..."))
    return False

    
def crop_from_exp(args, params):
    pass

def show_imgs(params):
    img_org = params.img_org
    img_exp = params.img_exp

    plt.figure()
    plt.subplot(121)
    plt.axis('off')
    plt.imshow(img_org)
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(img_exp)
    print("Pause to showup plt.imshow")
    plt.close()


def crop_file(filename, args):
    if not os.path.isfile(filename):
        raise ValueError(f"{filename} not exist")

    dir_src = os.path.dirname(filename)
    fnm_src = os.path.basename(filename).split(".")[0]
    img_name = fnm_src.split("-")[0]
    img_nums = fnm_src.split("-")[1]

    img_org = cv2.imread(filename, cv2.IMREAD_COLOR)

    args.dir_src = dir_src
    args.img_fname = fnm_src
    args.img_name = img_name
    args.img_nums = img_nums
    args.img_num_x = int(img_nums.split("x")[0])
    args.img_num_y = int(img_nums.split("x")[1])
    pprint.pprint(args)

    params = dotdict()
    params.img_org = img_org
    params.img_exp = create_expanded(img_org)

    while True:
        if find_crop_pts_from_org(args, params):
            break
    crop_from_exp(args, params)
    #show_imgs(params)


if __name__ == "__main__":
    filename = "few-shot-plants/planets_org/S01-6x5.jpg"
    filename = "few-shot-plants/planets_org/S34-3x2.jpg"
    #filename = "few-shot-plants/planets_org/S32-12x4.jpg"

    args = dotdict({"dir_dst": "few-shot-plants/planets"})
    crop_file(filename, args)

