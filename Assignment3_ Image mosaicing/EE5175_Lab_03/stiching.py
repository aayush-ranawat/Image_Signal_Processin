import numpy as np
import cv2 

from sift import *

img_1 , img_2 , img_3 = cv2.imread(r"img1.png") , cv2.imread(r"img2.png") ,cv2.imread(r"img3.png")

# img_1 , img_2 , img_3 = cv2.imread(r"img170.jpg",cv2.IMREAD_GRAYSCALE) , cv2.imread(r"img210.jpg",cv2.IMREAD_GRAYSCALE) ,cv2.imread(r"img240.jpg",cv2.IMREAD_GRAYSCALE)

img_1,img_2,img_3=cv2.resize(img_1,(640,360)) , cv2.resize(img_2,(640,360)) ,cv2.resize(img_3,(640,360))






correspondance_21= sift(img_2,img_1)

correspondance_23= sift(img_2,img_3)


print("correspondance found")



H21= find_homography_ransac(correspondance_21)   # i1= H21 i2

H23= find_homography_ransac(correspondance_23)   # i3 = h23 i2

print(f"homogrpahies are {H21} and {H23}")


final_image=create_mosaic(img_1,img_2,img_3,H21,H23)




# corr_img=show_correspondences(img_1,img_2,correspondance)










cv2.imshow("canvas",final_image)



cv2.waitKey(0)


