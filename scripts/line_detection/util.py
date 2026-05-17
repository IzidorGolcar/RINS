

import numpy as np
import cv2


DEFAULT_HSV = {
    #         (H_low, S_low, V_low)   (H_high, S_high, V_high)
    "yellow": ((20,  100, 100),       (35,  255, 255)),
    "green":  ((40,   80,  80),       (80,  255, 255)),
    "blue":   ((90,   20,  20),       (135, 255, 255)),
    "red":    (( 0,  100, 100),       (179, 255, 255)),
}

def _line_mask(depth_image, rgb_image, ground_mask):
    """Segment colored lines (yellow, red, green, blue) on the ground plane.
    Returns multiclass labels: 0=background, 1=yellow, 2=red, 3=blue, 4=green.
    """
    if ground_mask is None:
        return None
    
    h, w = rgb_image.shape[:2]
    
    # Initialize multiclass label map (0 = background)
    line_labels = np.zeros((h, w), dtype=np.uint8)
    
    # Convert RGB to HSV for color-based segmentation
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    
    # Restrict to ground region
    ground_binary = (ground_mask > 0).astype(np.uint8)
    
    # Create multiclass labels for each color
    color_class_map = {'yellow': 1, 'red': 2, 'blue': 3, 'green': 4}
    class_colors = {
        1: (255, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0),
        4: (0, 255, 0),
    }
    
    for color_name, class_id in color_class_map.items():
        if color_name not in DEFAULT_HSV:
            continue
        
        color_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Special handling for red which wraps around hue
        if color_name == 'red':
            # Red in HSV wraps: 0-10 and 170-179
            red_lo = cv2.inRange(hsv_image, np.array((0, 100, 100)), np.array((10, 255, 255)))
            red_hi = cv2.inRange(hsv_image, np.array((170, 100, 100)), np.array((179, 255, 255)))
            color_mask = cv2.bitwise_or(red_lo, red_hi)
        else:
            hsv_lo, hsv_hi = DEFAULT_HSV[color_name]
            color_mask = cv2.inRange(hsv_image, np.array(hsv_lo), np.array(hsv_hi))
        
        # Apply to ground region only
        color_mask = cv2.bitwise_and(color_mask, ground_binary)
        # Remove speckle noise before labeling
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        except Exception:
            pass

        num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        filtered = np.zeros((h, w), dtype=np.uint8)

        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            aspect_ratio = max(width, height) / max(min(width, height), 1)

            # Keep thin, elongated components that are large enough to be real floor lines.
            if area < 20:
                continue

            component_mask = (cc_labels == label_id)
            filtered[component_mask] = 255

        # Assign class label where mask is positive
        line_labels[filtered > 0] = class_id
    
    return line_labels


def _ground_mask(
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
    except Exception:
        pass

    return mask