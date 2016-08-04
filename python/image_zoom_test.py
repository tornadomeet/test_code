"""
test whether resize direct or in direct will bring in difference
"""
import numpy as np
import cv2
# np.set_printoptions(threshold=np.nan)

def zoom_up(img, scale=2.0):
    h, w = img.shape[:-1]
    double_img = cv2.resize(img.copy(), (int(w*scale), int(h*scale)))
    quadruple_img = cv2.resize(img.copy(), (int(w*scale*2), int(h*scale*2)))
    quadruple_indir_img = cv2.resize(double_img.copy(), (int(w*scale*2), int(h*scale*2)))
    mean_diff_zoom = (quadruple_img - quadruple_indir_img).mean()
    print "zoom up, scale = {} ---> mean of diff_zoom is:{}".format(scale, mean_diff_zoom)

def zoom_down(img, scale=0.5):
    h, w = img.shape[:-1]
    half_img = cv2.resize(img.copy(), (int(w*scale), int(h*scale)))
    quarder_img = cv2.resize(img.copy(), (int(w*scale/2), int(h*scale/2)))
    quarder_indir_img = cv2.resize(half_img.copy(), (int(w*scale/2), int(h*scale/2)))
    mean_diff_zoom = (quarder_img - quarder_indir_img).mean()
    print "zoom down, scale = {} ---> mean of diff_zoom is:{}".format(scale, mean_diff_zoom)

def zoom_back(img, scale=0.5):
    h, w = img.shape[:-1]
    half_img = cv2.resize(img.copy(), (int(w*scale), int(h*scale)))
    double_img = cv2.resize(img.copy(), (int(w*scale*2), int(h*scale*2)))
    back_img = cv2.resize(double_img.copy(), (int(w*scale), int(h*scale)))
    back_diff_zoom = (half_img - back_img).mean()
    print "zoom back, scale = {} ---> mean of diff_zoom is:{}".format(scale, back_diff_zoom)

img = cv2.imread("test.jpg")
zoom_up(img, 2.0)
zoom_up(img, 2.1)
zoom_up(img, 3.1)
zoom_down(img, 0.5)
zoom_down(img, 0.6)
zoom_down(img, 0.3)
zoom_back(img, 0.5)
zoom_back(img, 0.25)
zoom_back(img, 1.3)

