import numpy as np
import cv2
from ques1 import create_grid , bilinear_interpolate
from pathlib import Path


#using create_grid , bilinear_interpolate from ques1


script_path=Path(__file__).parent.resolve()
img_path=script_path/"pisa_rotate.png"


pisa=cv2.imread(img_path)

h,w,c=pisa.shape
# print(pisa.shape)
cx,cy= w/2 ,h/2


#bringing homogenous coordiantes to centre so that t=roation is about centre and not top left
T1=np.array([[1,0,cy],
            [0,1,cx],
            [0,0,1]])


theta=np.deg2rad(-4.5)

# applying rotation

R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])


#translating the  homogenous coordiantes back to original config.
T2=np.array([[1,0,-cy],
            [0,1,-cx],
            [0,0,1]])

#net total tranformations
T_rot=  T1 @ R @ T2


 # creating a homogenous grid with shape (h*w,3) denoting all coordiantes homogenous in assumed translated image
    #1st row :x
    # 2nd row :y
    # 3rd row :1s
coords = create_grid(h,w)

   # applying int T to obtain coordiantes in src image
src_coords = np.linalg.inv(T_rot) @ coords


   # x and y in src coordinates(transformed from grid)
x_src = src_coords[0, :]
y_src = src_coords[1, :]

valid= (x_src >= 0) & (x_src < w-1) & (y_src >= 0) & (y_src < h-1)


#array for storing interpolated coordinates
output = np.zeros((h*w,c))
output[valid] = bilinear_interpolate(
    pisa,
    x_src[valid],
    y_src[valid]
)



#reshaping the array to original image size from homogenous shape
pisa_rotated=output.reshape(h,w,c).astype(pisa.dtype)   





cv2.imshow("pisa_rotated",pisa_rotated)
cv2.imshow("pisa",pisa)

cv2.waitKey(0)