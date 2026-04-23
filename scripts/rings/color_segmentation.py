import cv2
import numpy as np

class ObjectDetector:

    def _cluster(self, img, n_clusters=9, sample_size=10_000):
        h, w = img.shape[:2]
        mask = np.any(img != 0, axis=-1)
        foreground_indices = np.where(mask.flatten())[0]
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3)
        weights = np.array([0.2, 1.0, 1.0], dtype=np.float32)
        pixels_f = pixels.astype(np.float32) * weights

        fg_pixels = pixels_f[foreground_indices]
        
        if len(fg_pixels) == 0:
            return np.zeros_like(img)

        np.random.seed(42)
        n_fg = len(fg_pixels)
        if n_fg > sample_size:
            indices = np.random.choice(n_fg, sample_size, replace=False)
            sample = fg_pixels[indices]
        else:
            sample = fg_pixels
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS
        _, _, centers = cv2.kmeans(sample, n_clusters, None, criteria, 1, flags)
        full_labels = np.full(len(pixels_f), -1, dtype=np.int32)
        distances = np.sum((fg_pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        fg_labels = np.argmin(distances, axis=1)
        full_labels[foreground_indices] = fg_labels
        centers_lab = (centers / weights).astype(np.uint8).reshape(1, -1, 3)
        centers_rgb = cv2.cvtColor(
            cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3)
        output_img = np.zeros((h * w, 3), dtype=np.uint8)
        mask_labels = full_labels != -1
        output_img[mask_labels] = centers_rgb[full_labels[mask_labels]]

        return output_img.reshape(h, w, 3)

    def _segment_objects(
            self,
            clustered_img: np.ndarray,
            min_area=800,
            max_area=5000,
            morph_kernel_size=5,
            morph_iterations=2,
    ):
        h, w = clustered_img.shape[:2]
        r, g, b = clustered_img[:, :, 0].astype(np.int32), \
            clustered_img[:, :, 1].astype(np.int32), \
            clustered_img[:, :, 2].astype(np.int32)
        
        packed = (r << 16) | (g << 8) | b
        is_bg = np.all(clustered_img == 0, axis=-1)
        
        unique_packed = np.unique(packed[~is_bg])
        
        id_map = np.full((h, w), -1, dtype=np.int32)
        for i, val in enumerate(unique_packed):
            id_map[packed == val] = i
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        id_map = self._mode_filter(id_map, kernel, morph_iterations)
        
        label_map = np.zeros((h, w), dtype=np.int32)
        next_label = 1
        
        for cid in range(len(unique_packed)):
            mask = (id_map == cid).astype(np.uint8) * 255
            n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for comp in range(1, n_cc):
                if min_area <= stats[comp, cv2.CC_STAT_AREA] <= max_area:
                    label_map[cc_labels == comp] = next_label
                    next_label += 1

        return label_map

    def _mode_filter(self, id_map: np.ndarray, kernel: np.ndarray, iterations: int) -> np.ndarray:
        id_map_shifted = (id_map + 1).astype(np.uint8)
        for _ in range(iterations):
            id_map_shifted = cv2.medianBlur(id_map_shifted, ksize=kernel.shape[0])
        return id_map_shifted.astype(np.int32) - 1

    def get_labels(
            self,
            image,
            downscale_factor=3,
            n_clusters=10,
            sample_size=10_000,
            min_area=3200,
            max_area=6000,
            morph_kernel_size=5,
            morph_iterations=2,

    ):
        min_area = min_area / (downscale_factor ** 2)
        max_area = max_area / (downscale_factor ** 2)
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // downscale_factor, h // downscale_factor),
                           interpolation=cv2.INTER_AREA)
        clustered = self._cluster(small, n_clusters, sample_size)
        segmented = self._segment_objects(clustered, min_area, max_area, morph_kernel_size, morph_iterations)
        return cv2.resize(segmented, (w, h), interpolation=cv2.INTER_NEAREST)