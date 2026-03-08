import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def create_manual_gaussian_kernel(sigma):
    """Generates a manual Gaussian kernel based on the specified sigma."""
    # Handle the sigma = 0.0 case (no blurring / identity)
    if sigma == 0.0:
        return np.array([[1.0]])
        
    # Calculate kernel size based on the formula: ceil(6*sigma + 1)
    k_size = math.ceil(6 * sigma + 1)
    
    # Ensure center coordinate is correctly offset
    center = k_size // 2
    
    # Create a 2D coordinate grid
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    # Apply the 2D Gaussian formula
    # We omit the 1/(2*pi*sigma^2) constant here because we normalize at the end
    g_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel so the sum of all elements is 1
    g_kernel = g_kernel / g_kernel.sum()
    
    return g_kernel

def main():
    # 1. Load the image (ensure 'Mandrill.png' is in your working directory)
    # Read in grayscale or color; assuming color (BGR) here.
    image_path = r'lab_4/Mandrill.png'
    image = cv2.imread(image_path)
    
    
    # 2. Define the sigma values to test
    sigmas = [1.6, 1.2, 1.0, 0.6, 0.3, 0.0]
    
    # Prepare a plot to observe the outputs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 3. Iterate through each sigma, create the kernel, and apply it
    for i, sigma in enumerate(sigmas):
        # Generate the custom kernel
        kernel = create_manual_gaussian_kernel(sigma)
        
        # Apply the manual kernel using cv2.filter2D
        # -1 indicates the output image will have the same depth as the source
        blurred_image = cv2.filter2D(image, -1, kernel)
        
        # Display the results
        axes[i].imshow(blurred_image)
        axes[i].set_title(f"$\sigma$ = {sigma} | Size: {kernel.shape[0]}x{kernel.shape[1]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()