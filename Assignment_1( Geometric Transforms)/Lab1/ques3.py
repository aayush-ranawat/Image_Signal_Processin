import numpy as np
import cv2
from ques1 import create_grid , bilinear_interpolate
from pathlib import Path


script_path=Path(__file__).parent.resolve()
img_path=script_path/"cells_scale.png"

cells=cv2.imread(img_path)
h,w,c=cells.shape
cx,cy=w/2,h/2

#bringing homogenous coordiantes to centre so that scaling is about centre and not top left

T1=np.array([[1,0,cy],
            [0,1,cx],
            [0,0,1]])


# sacling by 0.8

S1=np.array([[0.8,0,0],
             [0,0.8,0],
             [0,0,1]])


#scaling by 1.3

S2=np.array([[1.3,0,0],
            [0,1.3,0],
            [0,0,1]])

T2=np.array([[1,0,-cy],
            [0,1,-cx],
            [0,0,1]])


coords = create_grid(h,w)


src_coords = np.linalg.inv(T1 @ S2 @ T2) @ coords  #scaling by 1.3

src_coords_2=np.linalg.inv(T1 @ S1 @ T2) @ coords   #sclain by 0.8


x_src = src_coords[0, :]
y_src = src_coords[1, :]


x_src_2=src_coords_2[0,:]
y_src_2=src_coords_2[1,:]




valid= (x_src >= 0) & (x_src < w-1) & (y_src >= 0) & (y_src < h-1)


output = np.zeros((h*w,c))
output_2=np.zeros((h*w,c))


output[valid] = bilinear_interpolate(cells, x_src[valid],  y_src[valid])

output_2[valid]=bilinear_interpolate(cells, x_src_2[valid], y_src_2[valid])

cells_scaled_up=output.reshape(h,w,c).astype(cells.dtype)
cells_scaled_down=output_2.reshape(h,w,c).astype(cells.dtype)


cv2.imshow("cells",cells)
cv2.imshow("cells_scaled_1.3",cells_scaled_up)
cv2.imshow("cells_scaled_0.8",cells_scaled_down)
cv2.waitKey(0)