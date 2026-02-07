import numpy as np
import cv2 
from pathlib import Path
import sys

# 1. Get the directory where this specific script lives
script_dir = Path(__file__).parent.resolve()
img_path=script_dir/"lena_translate.png"
lena=cv2.imread(img_path)


###############---------------UTILIIES---------------------#########################

# common functuions to be used for all question 1-3 for traslation rotation and scaling

#   THIS FUNCTION CREATES A GRID FOR OUR SOURCE COORDINATES ACCORDING
#    TO IMAGE HIEGHT AND WIDTH IN AFFINE FORM SO ITS LIKE [X,Y,1](HOMOGENOUS COORDINATES ) STACKED TO FORM 3D GRID
# WHERE EACH X AND Y DENOTE SRC COORDIANTE INDEX
def create_grid(h, w):                                  
    y, x = np.indices((h, w))                           
    ones = np.ones_like(x)
    coords = np.stack([x, y, ones], axis=0)  # (3, H, W)
    return coords.reshape(3, -1)             # (3, H*W)

def bilinear_interpolate(img, x, y):
    H, W ,c= img.shape

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W-1)
    x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    y1 = np.clip(y1, 0, H-1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1-x)*(y1-y)
    wb = (x1-x)*(y-y0)
    wc = (x-x0)*(y1-y)
    wd = (x-x0)*(y-y0)

    return (Ia.T*wa + Ib.T*wb + Ic.T*wc + Id.T*wd).T


###############--------------------------------############################
def main():

    tx,ty= 3.75 , 4.3

    T=np.array([[1,0,tx],
                [0,1,ty],
                [0,0,1]])

    h,w,c=lena.shape
    

    grid=create_grid(h,w)

    scr_coor= np.linalg.inv(T) @ grid

    x_src=scr_coor[0, :]
    y_src=scr_coor[1, :]

    valid= (x_src >= 0) & (x_src < w-1) & (y_src >= 0) & (y_src < h-1)

    output = np.zeros((h*w , c))
    output[valid] = bilinear_interpolate(
            lena,
            x_src[valid],
            y_src[valid]
        )


    lena_translated=output.reshape(h,w,c).astype(lena.dtype)

    cv2.imshow("lena",lena)
    cv2.imshow("lena_traslated",lena_translated)
    cv2.waitKey(0)


if __name__=='__main__':
    main()












