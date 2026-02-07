import numpy as np
import cv2
from ques1 import create_grid , bilinear_interpolate
from pathlib import Path

script_path=Path(__file__).parent.resolve()
img_path=script_path/"pisa_rotate.png"


pisa=cv2.imread(img_path)

h,w,c=pisa.shape
print(pisa.shape)
cx,cy= w/2 ,h/2

T1=np.array([[1,0,cy],
            [0,1,cx],
            [0,0,1]])

theta=np.deg2rad(-4.5)

R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
T2=np.array([[1,0,-cy],
            [0,1,-cx],
            [0,0,1]])

T_rot=  T1 @ R @ T2

coords = create_grid(h,w)
src_coords = np.linalg.inv(T_rot) @ coords

x_src = src_coords[0, :]
y_src = src_coords[1, :]

valid= (x_src >= 0) & (x_src < w-1) & (y_src >= 0) & (y_src < h-1)

output = np.zeros((h*w,c))
output[valid] = bilinear_interpolate(
    pisa,
    x_src[valid],
    y_src[valid]
)

pisa_rotated=output.reshape(h,w,c).astype(pisa.dtype)





cv2.imshow("pisa_rotated",pisa_rotated)
cv2.imshow("pisa",pisa)

cv2.waitKey(0)