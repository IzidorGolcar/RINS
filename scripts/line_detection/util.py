

import numpy as np
import cv2

# Recalibrated HSV ranges
DEFAULT_HSV = {
    #         (H_low, S_low, V_low)   (H_high, S_high, V_high)
    "yellow": ((20,  100, 100),       (35,  255, 255)),
    "green":  ((40,   80,  80),       (80,  255, 255)),
    
    # Blue: Lowered S_low to 50 to catch the light blue, 
    # but kept it above 30 to avoid gray floor noise.
    "blue":   ((95,   50,  100),      (130, 255, 255)), 
    
    # Red: Kept saturation high to avoid picking up brown/gray tones
    "red":    ((0,    120, 100),      (10,  255, 255)),
}

def line_mask(depth_image, rgb_image, ground_mask):
    if ground_mask is None:
        return None
    
    h, w = rgb_image.shape[:2]
    line_labels = np.zeros((h, w), dtype=np.uint8)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    ground_binary = (ground_mask > 0).astype(np.uint8)
    
    color_class_map = {'yellow': 1, 'red': 2, 'blue': 3, 'green': 4}
    
    for color_name, class_id in color_class_map.items():
        hsv_lo, hsv_hi = DEFAULT_HSV[color_name]
        
        if color_name == 'red':
            # Standard red wrap-around for HSV
            lower1, upper1 = np.array([0, 120, 100]), np.array([10, 255, 255])
            lower2, upper2 = np.array([170, 120, 100]), np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower1, upper1)
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:
            color_mask = cv2.inRange(hsv_image, np.array(hsv_lo), np.array(hsv_hi))
        
        # Filter by ground mask and clean up
        color_mask = cv2.bitwise_and(color_mask, ground_binary)
        
        # Small kernel to remove single-pixel noise without destroying thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        for label_id in range(1, num_labels):
            # Reduced area threshold slightly to catch distant line segments
            if stats[label_id, cv2.CC_STAT_AREA] > 15:
                line_labels[cc_labels == label_id] = class_id
                
    return line_labels



def ground_mask(
        depth_image,
        fx, fy, cx, cy,
        ground_distance_threshold=0.05,
        ground_min_normal_y=0.7,
        ground_min_y_over_z=0.5
    ):


    depth = np.array(depth_image, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Mask invalid depths
    valid_mask = np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
    if valid_mask.sum() < 50:
        return np.zeros_like(depth, dtype=np.uint8)

    h, w = depth.shape

    # Camera intrinsics
    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Use lower part of image to sample ground points (focus on plausible ground region)
    sample_mask = valid_mask.copy()
    sample_mask[: int(h * 0.4), :] = False

    pts_indices = np.where(sample_mask)
    pts_count = pts_indices[0].shape[0]
    if pts_count < 50:
        return np.zeros_like(depth, dtype=np.uint8)

    # Build Nx3 array of points (X,Y,Z)
    pts = np.stack((X[pts_indices], Y[pts_indices], Z[pts_indices]), axis=1)

    # Subsample for RANSAC if too many points
    max_samples = 4000
    if pts.shape[0] > max_samples:
        idx = np.random.choice(pts.shape[0], max_samples, replace=False)
        pts_sample = pts[idx]
    else:
        pts_sample = pts

    # RANSAC plane fitting
    best_inliers = None
    best_plane = None
    iterations = 300
    distance_threshold = ground_distance_threshold
    rng = np.random.default_rng()
    N = pts_sample.shape[0]
    if N < 3:
        return np.zeros_like(depth, dtype=np.uint8)

    for _ in range(iterations):
        # pick 3 random distinct indices
        i1, i2, i3 = rng.choice(N, size=3, replace=False)
        p1 = pts_sample[i1]
        p2 = pts_sample[i2]
        p3 = pts_sample[i3]
        # compute normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Reject planes that do not look like a floor in the camera optical frame.
        normal_y = abs(float(normal[1]))
        normal_z = abs(float(normal[2]))
        if normal_y < ground_min_normal_y:
            continue
        if normal_z > 1e-6 and (normal_y / normal_z) < ground_min_y_over_z:
            continue

        d = -np.dot(normal, p1)

        # distances of all sampled pts to plane
        distances = np.abs(np.dot(pts_sample, normal) + d)
        inliers = distances <= distance_threshold
        inlier_count = int(inliers.sum())
        if best_inliers is None or inlier_count > best_inliers:
            best_inliers = inlier_count
            best_plane = (normal.copy(), float(d))

    if best_plane is None:
        return np.zeros_like(depth, dtype=np.uint8)

    normal, d = best_plane

    # compute distance for all valid pixels
    pts_all = np.stack((X[valid_mask], Y[valid_mask], Z[valid_mask]), axis=1)
    distances_all = np.abs(np.dot(pts_all, normal) + d)
    ground_mask_vals = np.zeros_like(depth, dtype=bool)
    ground_mask_vals[valid_mask] = distances_all <= distance_threshold

    # Optional: remove small islands and smooth
    mask = (ground_mask_vals.astype(np.uint8) * 255)
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, small_kernel, iterations=2)
    except Exception:
        pass

    return mask