import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():

    img1 = cv2.imread(r'IMG1.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r'IMG2.png', cv2.IMREAD_GRAYSCALE)

    

    # given coorespondance checked from image and question


    p1_img_1=np.array([124,29,1])
    p1_img_2=np.array([248,93,1])

    p2_img_1=np.array([372,157,1])
    p2_img_2=np.array([399,328,1])


   

    # rotaion computation

    angle_1=np.arctan2((p2_img_1[1]-p1_img_1[1]),(p2_img_1[0]-p1_img_1[0]))
    angle_2=np.arctan2((p2_img_2[1]-p1_img_2[1]),(p2_img_2[0]-p1_img_2[0]))

    #net rotation in radians

    theta=angle_1-angle_2
   
   

    R=np.array([[np.cos(theta), (-np.sin(theta)),0], 
                [ np.sin(theta) , np.cos(theta),0],
                [0,0,1]] )
    
  

    
    translation= (p2_img_1 - R@p2_img_2)
    # translation=[0,0]


    # homogenious tranformation matrix

    T=np.array([[np.cos(theta), -np.sin(theta) , translation[0]],
                [ np.sin(theta) , np.cos(theta) , translation[1]],
                [0 , 0 , 1]])
    
    print(T)
    

    #using open cv wrap affine bilinear transform to interpolation and transform

     # it takes 2x3 matrix as input are are sliciing the above matrix

    img2_aligned=cv2.warpAffine(img2, T[:2,:] , (img1.shape[1],img1.shape[0]), flags=cv2.INTER_LINEAR) 


    # to find occlusiopn we calcul;ate abs difference btw pixel intensities of images
    diff_image = cv2.absdiff(img1, img2_aligned)


  #  applying binary thresholding
    _, change_mask = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    #applying the change mask on the original image

    masked_image=cv2.bitwise_and(img1, img1, mask=change_mask)








    cv2.imshow("img_2_alinged",img2_aligned)
    cv2.imshow("binary_mask",change_mask)
    cv2.imshow("masked_image",masked_image)
    cv2.imshow("img_2",img2)
    cv2.imshow("img_1",img1)

    cv2.waitKey(0)


    # Run the function
if __name__ == "__main__":
    main()
    


# Estimated Affine Transformation Matrix:
# [[   0.8654551     0.49921821 -137.06015943]
#  [  -0.49921821    0.8654551    72.31879117]]


   
    
  
    