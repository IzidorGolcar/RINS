import cv2
import numpy as np


class ObjectDetector:

    def _cluster(self, img, n_clusters=9, sample_size=10_000):
        h, w = img.shape[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        pixels = lab.reshape(-1, 3)
        weights = np.array([0.2, 1.0, 1.0], dtype=np.float32)
        pixels_f = pixels.astype(np.float32) * weights

        np.random.seed(42)
        n_pixels = len(pixels_f)
        if n_pixels > sample_size:
            indices = np.random.choice(n_pixels, sample_size, replace=False)
            sample = pixels_f[indices]
        else:
            sample = pixels_f

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS

        _, _, centers = cv2.kmeans(sample, n_clusters, None, criteria, 1, flags)

        distances = np.sum((pixels_f[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        centers_lab = (centers / weights).astype(np.uint8).reshape(1, -1, 3)
        centers_rgb = cv2.cvtColor(
            cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB
        ).reshape(-1, 3)

        return centers_rgb[labels].reshape(h, w, 3)

    def _segment_objects(
            self,
            clustered_img: np.ndarray,
            min_area=800,
            morph_kernel_size=5,
            morph_iterations=2,
    ):
        h, w = clustered_img.shape[:2]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        r, g, b = clustered_img[:, :, 0].astype(np.int32), \
            clustered_img[:, :, 1].astype(np.int32), \
            clustered_img[:, :, 2].astype(np.int32)
        packed = (r << 16) | (g << 8) | b
        unique_packed, cluster_ids = np.unique(packed, return_inverse=True)
        n_clusters = len(unique_packed)
        id_map = cluster_ids.reshape(h, w).astype(np.int32)
        id_map = self._mode_filter(id_map, kernel, morph_iterations)
        label_map = np.zeros((h, w), dtype=np.int32)
        next_label = 1
        for cid in range(n_clusters):
            mask = (id_map == cid).astype(np.uint8) * 255
            n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            for comp in range(1, n_cc):
                if stats[comp, cv2.CC_STAT_AREA] >= min_area:
                    label_map[cc_labels == comp] = next_label
                    next_label += 1

        return label_map

    def _mode_filter(self, id_map: np.ndarray, kernel: np.ndarray,
                     iterations: int) -> np.ndarray:
        for _ in range(iterations):
            n_clusters = int(id_map.max()) + 1
            vote_maps = np.stack([
                cv2.filter2D(
                    (id_map == cid).astype(np.float32), -1,
                    kernel.astype(np.float32)
                )
                for cid in range(n_clusters)
            ], axis=-1)
            id_map = vote_maps.argmax(axis=-1).astype(np.int32)
        return id_map

    def get_labels(
            self,
            image,
            downscale_factor=2,
            n_clusters=10,
            sample_size=10_000,
            min_area=3200,
            morph_kernel_size=5,
            morph_iterations=2,

    ):
        min_area = min_area / (downscale_factor ** 2)
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // downscale_factor, h // downscale_factor),
                           interpolation=cv2.INTER_AREA)
        clustered = self._cluster(small, n_clusters, sample_size)
        segmented = self._segment_objects(clustered, min_area, morph_kernel_size, morph_iterations)
        return cv2.resize(segmented, (w, h), interpolation=cv2.INTER_NEAREST)