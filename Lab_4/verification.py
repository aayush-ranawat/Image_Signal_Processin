import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# Functions from Part 1: Space-Invariant
# ==========================================
def create_manual_gaussian_kernel(sigma):
    """Generates a standard 2D Gaussian kernel."""
    if sigma == 0.0:
        return np.array([[1.0]])
        
    k_size = math.ceil(6 * sigma + 1)
    if k_size % 2 == 0: 
        k_size += 1 # Ensure odd kernel size
        
    center = k_size // 2
    x, y = np.mgrid[-center:center+1, -center:center+1]
    
    g_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g_kernel / g_kernel.sum()

# ==========================================
# Functions from Part 2: Space-Variant
# ==========================================
def apply_space_variant_blur(image, sigma_matrix):
    """Applies a pixel-by-pixel Gaussian blur based on a sigma matrix."""
    rows, cols = image.shape
    output_image = np.zeros_like(image, dtype=np.float64)
    
    # Calculate padding based on the maximum sigma in the matrix
    max_sigma = np.max(sigma_matrix)
    max_k_size = math.ceil(6 * max_sigma + 1)
    if max_k_size % 2 == 0: 
        max_k_size += 1
    pad_size = max_k_size // 2
    
    # Pad using BORDER_REFLECT_101 to perfectly match cv2.filter2D's default border handling
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT_101)
    
    for m in range(rows):
        for n in range(cols):
            sigma = sigma_matrix[m, n]
            
            if sigma < 0.01:
                output_image[m, n] = image[m, n]
                continue
                
            k_size = math.ceil(6 * sigma + 1)
            if k_size % 2 == 0:
                k_size += 1
                
            center = k_size // 2
            
            x, y = np.mgrid[-center:center+1, -center:center+1]
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum() 
            
            m_pad, n_pad = m + pad_size, n + pad_size
            region = padded_image[m_pad - center : m_pad + center + 1, 
                                  n_pad - center : n_pad + center + 1]
                                  
            output_image[m, n] = np.sum(region * kernel)
            
    return np.clip(output_image, 0, 255).astype(np.uint8)

# ==========================================
# Part 3: Verification Main Logic
# ==========================================
def main():
    # Load the image
    image_path = 'lab_4/Nautilus.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load '{image_path}'. Ensure it is in the same directory.")
        return

    print("Running Verification Step 3...")

    # --- Step 3(a): Space-invariant blur with sigma = 1.0 ---
    print("-> Calculating space-invariant blur (Part 1)...")
    sigma_invariant = 1.0
    kernel_invariant = create_manual_gaussian_kernel(sigma_invariant)
    # cv2.filter2D applies the kernel globally
    blurred_invariant = cv2.filter2D(image, -1, kernel_invariant)

    # --- Step 3(b): Space-variant blur with sigma(m,n) = 1.0 ---
    print("-> Calculating space-variant blur (Part 2)...")
    # Create a matrix of exactly the same shape as the image, entirely filled with 1.0
    sigma_matrix = np.full(image.shape, 1.0, dtype=np.float64)
    blurred_variant = apply_space_variant_blur(image, sigma_matrix)

    # --- Verification ---
    # Calculate the absolute difference between the two resulting images
    difference = cv2.absdiff(blurred_invariant, blurred_variant)
    max_diff = np.max(difference)
    
    print("\n--- Verification Results ---")
    print(f"Maximum pixel difference between methods: {max_diff}")
    
    if max_diff == 0:
        print("Success: The blurred images obtained from the two steps are EXACTLY the same.")
    elif max_diff <= 2:
        print("Success: The blurred images are functionally the same (minor rounding differences due to float/uint8 conversions).")
    else:
        print("Warning: There is a noticeable difference between the methods.")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Nautilus.pgm')
    axes[0].axis('off')
    
    axes[1].imshow(blurred_invariant, cmap='gray')
    axes[1].set_title('3(a) Space-Invariant ($\sigma = 1.0$)')
    axes[1].axis('off')
    
    axes[2].imshow(blurred_variant, cmap='gray')
    axes[2].set_title('3(b) Space-Variant ($\sigma(m,n) = 1.0$)')
    axes[2].axis('off')

    # Display the difference image (multiplied to make tiny differences visible)
    axes[3].imshow(difference * 50, cmap='gray') 
    axes[3].set_title(f'Difference (Max diff: {max_diff})')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()