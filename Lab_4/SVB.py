import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def create_space_variant_sigma_matrix():
    """Generates the 195x195 matrix of sigma values using the derived A and B."""
    N = 195
    A = 2.0
    
    # Using the exact derived formula for B to avoid rounding errors
    # Equivalent to B ≈ 3588.40
    B = -(19012.5) / np.log(0.005)
    
    # Create a grid of (m, n) coordinates
    m, n = np.mgrid[0:N, 0:N]
    
    # Calculate the sigma matrix 
    # Center is N/2 = 97.5
    sigma_matrix = A * np.exp(-((m - 97.5)**2 + (n - 97.5)**2) / B)
    
    return sigma_matrix, N

def apply_space_variant_blur(image, sigma_matrix):
    """Applies a pixel-by-pixel Gaussian blur based on the sigma matrix."""
    N = image.shape[0]
    output_image = np.zeros_like(image, dtype=np.float64)
    
    # Maximum possible padding needed based on max sigma (A = 2.0)
    max_sigma = 2.0
    max_k_size = math.ceil(6 * max_sigma + 1)
    pad_size = max_k_size // 2
    
    # Pad the original image to handle borders
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    # Iterate through every single pixel
    for m in range(N):
        for n in range(N):
            sigma = sigma_matrix[m, n]
            
            # If sigma is virtually zero, skip blurring to save computation
            if sigma < 0.01:
                output_image[m, n] = image[m, n]
                continue
                
            # Determine kernel size for this specific pixel
            k_size = math.ceil(6 * sigma + 1)
            if k_size % 2 == 0:
                k_size += 1
                
            center = k_size // 2
            
            # Generate the specific 2D Gaussian kernel
            x, y = np.mgrid[-center:center+1, -center:center+1]
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum() # Normalize
            
            # Extract local neighborhood and apply kernel
            m_pad, n_pad = m + pad_size, n + pad_size
            region = padded_image[m_pad - center : m_pad + center + 1, 
                                  n_pad - center : n_pad + center + 1]
                                  
            output_image[m, n] = np.sum(region * kernel)
            
    return np.clip(output_image, 0, 255).astype(np.uint8)

def main():
    # Load the image
    image_path = r'lab_4/Globe.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image at '{image_path}'.")
        return

    # Create the matrix of sigma(m, n) and get the target size N=195
    sigma_matrix, N = create_space_variant_sigma_matrix()

    # Force the image to match the 195x195 constraint 
    if image.shape != (N, N):
        print(f"Resizing image from {image.shape} to ({N}, {N})...")
        image = cv2.resize(image, (N, N), interpolation=cv2.INTER_AREA)

    print("Applying space-variant blur... (This will take a moment)")
    blurred_image = apply_space_variant_blur(image, sigma_matrix)
    
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original Image ({N}x{N})')
    axes[0].axis('off')
    
    im = axes[1].imshow(sigma_matrix, cmap='hot')
    axes[1].set_title('$\sigma(m, n)$ Heatmap')
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(blurred_image, cmap='gray')
    axes[2].set_title('Space-Variant Blurred Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()