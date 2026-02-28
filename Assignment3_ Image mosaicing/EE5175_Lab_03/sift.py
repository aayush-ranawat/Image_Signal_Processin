import cv2
import numpy as np


def sift(img1, img2):
    """
    Computes point correspondences between two images using sift

    Args:
        img1 (np.array): Query image
        img2 (np.array): Target image

    Returns:
        points (np.array): A 2 X num_matches X 2 array.
                           `points[0]` are keypoints in img1 and the corresponding
                            keypoints in img2 are `points[1]`
    """
    # sift = cv2.xfeatures2d.SIFT_create()

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    correspondences = np.zeros((2, len(good_matches), 2))

    for i, match in enumerate(good_matches):
        correspondences[0, i, :] = np.flip(kp1[match.queryIdx].pt)
        correspondences[1, i, :] = np.flip(kp2[match.trainIdx].pt)

    return correspondences




def show_correspondences(img1, img2, correspondences):
    """
    Visualizes point correspondences by drawing lines between matched points.

    Args:
        img1 (np.array): Query image
        img2 (np.array): Target image
        correspondences (np.array): 2 x N x 2 array from your sift function
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. Create a blank canvas large enough to hold both images side-by-side
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 2. Convert grayscale to BGR if needed, so we can draw colorful lines
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()

    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()

    # 3. Place both images onto the canvas
    canvas[:h1, :w1] = img1_color
    canvas[:h2, w1:w1+w2] = img2_color

    num_matches = correspondences.shape[1]

    # 4. Iterate through the matches and draw them
    for i in range(5):
        # Extract points. Remember: your sift function returns (y, x).
        # We assign index 1 to X, and index 0 to Y.
        x1 = int(correspondences[0, i, 1])
        y1 = int(correspondences[0, i, 0])

        # For the second image, we must shift the X coordinate right by w1
        x2 = int(correspondences[1, i, 1]) + w1
        y2 = int(correspondences[1, i, 0])

        # Generate a random color for each connecting line
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Draw the keypoints (circles) and the match (line)
        cv2.circle(canvas, (x1, y1), 4, color, -1)
        cv2.circle(canvas, (x2, y2), 4, color, -1)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

    # 5. Display the result
    # Resize for display if the combined images are too large for your screen
    max_display_width = 1400
    if canvas_w > max_display_width:
        scale = max_display_width / canvas_w
        display_canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
    else:
        display_canvas = canvas


    return display_canvas





def calculate_homography(pts1, pts2):
    """
    Helper to calculate Homography H using 4 points via DLT.
    pts1, pts2 are (4, 2) arrays.
    """
    A = []
    for i in range(4):
        x, y = pts1[i][1], pts1[i][0]  # Assuming (row, col) -> (x, y)
        u, v = pts2[i][1], pts2[i][0]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def find_homography_ransac(correspondences, epsilon=10):
    """
    Implements the specific RANSAC algorithm provided.
    """
    # points[0] are img1 (source), points[1] are img2 (target)
    pts1 = correspondences[0]
    pts2 = correspondences[1]
    num_pts = pts1.shape[0]
    
    # Target size for consensus: 80% of remaining points
    # Step (e): d = 0.8 * |P| where |P| = total - 4
    d = 0.8 * (num_pts - 4)

    while True:
        # (b) Randomly pick four correspondences
        indices = np.random.choice(num_pts, 4, replace=False)
        R_pts1 = pts1[indices]
        R_pts2 = pts2[indices]
        
        # (c) Calculate the homography H
        try:
            H = calculate_homography(R_pts1, R_pts2)
        except np.linalg.LinAlgError:
            continue

        # (d) Check remaining correspondences (P = M\R)
        consensus_set = []
        remaining_indices = [i for i in range(num_pts) if i not in indices]
        
        for i in remaining_indices:
            # Transform point from img1 using H
            # Format: [y, x] -> [x, y, 1] for matrix math
            p = np.array([pts1[i][1], pts1[i][0], 1]) 
            p_prime_proj = np.dot(H, p)
            
            # Normalize so that z'' = 1
            if p_prime_proj[2] == 0: continue
            x_double_prime = p_prime_proj[0] / p_prime_proj[2]
            y_double_prime = p_prime_proj[1] / p_prime_proj[2]
            
            # Ground truth from img2
            x_prime = pts2[i][1]
            y_prime = pts2[i][0]
            
            # Calculate Euclidean error
            dist = np.sqrt((x_prime - x_double_prime)**2 + (y_prime - y_double_prime)**2)
            
            if dist < epsilon:
                consensus_set.append(i)
        
        # (e) Check if consensus set is large enough
        if len(consensus_set) > d:
            return H
        



import numpy as np

def create_mosaic(img1, img2, img3, H21, H23):
    # 1. Determine the size of the full mosaic
    # We transform the corners of all images into the I2 coordinate space
    # to find the minimum and maximum bounds of the canvas.
    h2, w2 = img2.shape[:2]
    h1, w1 = img1.shape[:2]
    h3, w3 = img3.shape[:2]

    # Corners of I2 (Reference)
    corners_i2 = np.array([[0, 0, 1], [w2, 0, 1], [w2, h2, 1], [0, h2, 1]]).T
    
    # Corners of I1 and I3 mapped back to I2 space
    # Since I1 = H21 * I2, then I2 = inv(H21) * I1
    inv_H21 = np.linalg.inv(H21)
    inv_H23 = np.linalg.inv(H23)
    
    def get_transformed_corners(h, w, inv_H):
        c = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        proj = np.dot(inv_H, c)
        return proj[:2] / proj[2]

    corners_i1_in_i2 = get_transformed_corners(h1, w1, inv_H21)
    corners_i3_in_i2 = get_transformed_corners(h3, w3, inv_H23)

    # All corners in I2 space to find canvas bounds
    all_x = np.concatenate([corners_i2[0], corners_i1_in_i2[0], corners_i3_in_i2[0]])
    all_y = np.concatenate([corners_i2[1], corners_i1_in_i2[1], corners_i3_in_i2[1]])
    
    x_min, x_max = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    y_min, y_max = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

    # 2. Create the empty canvas
    width = x_max - x_min
    height = y_max - y_min
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    count_mask = np.zeros((height, width, 1), dtype=np.float32)

    # 3. Target-to-Source Mapping (Inverse Mapping)
    # We iterate through the canvas and find which pixels belong to I1, I2, or I3
    for y_cvs in range(height):
        for x_cvs in range(width):
            # Coordinate in the I2 (reference) coordinate system
            x_i2 = x_cvs + x_min
            y_i2 = y_cvs + y_min
            p_i2 = np.array([x_i2, y_i2, 1.0])

            # Check Image 2 (Identity)
            if 0 <= x_i2 < w2 and 0 <= y_i2 < h2:
                canvas[y_cvs, x_cvs] += img2[int(y_i2), int(x_i2)]
                count_mask[y_cvs, x_cvs] += 1

            # Check Image 1 (I1 = H21 * I2)
            p_i1 = np.dot(H21, p_i2)
            x_i1, y_i1 = p_i1[0]/p_i1[2], p_i1[1]/p_i1[2]
            if 0 <= x_i1 < w1 and 0 <= y_i1 < h1:
                canvas[y_cvs, x_cvs] += img1[int(y_i1), int(x_i1)]
                count_mask[y_cvs, x_cvs] += 1

            # Check Image 3 (I3 = H23 * I2)
            p_i3 = np.dot(H23, p_i2)
            x_i3, y_i3 = p_i3[0]/p_i3[2], p_i3[1]/p_i3[2]
            if 0 <= x_i3 < w3 and 0 <= y_i3 < h3:
                canvas[y_cvs, x_cvs] += img3[int(y_i3), int(x_i3)]
                count_mask[y_cvs, x_cvs] += 1

    # 4. Average the values (Blending)
    # Avoid division by zero
    final_mosaic = np.divide(canvas, count_mask, out=np.zeros_like(canvas), where=count_mask!=0)
    return final_mosaic.astype(np.uint8)