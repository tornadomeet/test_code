"""
test whether resize direct or in direct will bring in difference
"""
import numpy as np
import cv2
# np.set_printoptions(threshold=np.nan)

def zoom(scale=2.0):
    img = cv2.imread("test.jpg")
    h, w = img.shape[:-1]
    double_img = cv2.resize(img.copy(), (int(w*scale), int(h*scale)))
    quadruple_img = cv2.resize(img.copy(), (int(w*scale*2), int(h*scale*2)))
    quadruple_indir_img = cv2.resize(double_img.copy(), (int(w*scale*2), int(h*scale*2)))
    mean_diff_zoom = (quadruple_img - quadruple_indir_img).mean()
    print "scale = {} ---> mean of diff_zoom is:{}".format(scale, mean_diff_zoom)

zoom(0.5)
zoom(0.6)
zoom(2.0)
zoom(2.1)


