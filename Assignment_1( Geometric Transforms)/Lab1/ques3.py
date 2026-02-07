import numpy as np
import cv2
from ques1 import create_grid , bilinear_interpolate
from pathlib import Path


script_path=Path(__file__).parent.resolve()
img_path=script_path/"cells_scale.png"

cells=cv2.imread(img_path)
h,w,c=cells.shape
cx,cy=w/2,h/2

T1=np.array([[1,0,cy],
            [0,1,cx],
            [0,0,1]])

S1=np.array([[0.8,0,0],
             [0,0.8,0],
             [0,0,1]])

S2=np.array([[1.3,0,0],
            [0,1.3,0],
            [0,0,1]])

T2=np.array([[1,0,-cy],
            [0,1,-cx],
            [0,0,1]])


coords = create_grid(h,w)
src_coords = np.linalg.inv(T1 @ S2 @ T2) @ coords

x_src = src_coords[0, :]
y_src = src_coords[1, :]

valid= (x_src >= 0) & (x_src < w-1) & (y_src >= 0) & (y_src < h-1)

output = np.zeros((h*w,c))
output[valid] = bilinear_interpolate(
     cells,
    x_src[valid],
    y_src[valid]
)

cells_scaled=output.reshape(h,w,c).astype(cells.dtype)

cv2.imshow("cells",cells)
cv2.imshow("cells_scaled",cells_scaled)

cv2.waitKey(0)