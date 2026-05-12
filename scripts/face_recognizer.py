#!/usr/bin/env python3
"""Lightweight personnel face classifier.

Trained at startup from the portraits in ``personnel/`` (filenames encode
``<name>_<pronouns>_<role>.png``).  No deep models — three classical
classifiers are voted together so a wrong answer from one of them can be
overridden by the other two:

  - LBPH  (cv2.face.LBPHFaceRecognizer_create) — primary, robust to light
  - PCA / Eigenfaces                            — per-class reconstruction
  - HOG + cosine kNN                            — texture-based tiebreak

LBPH lives in ``opencv-contrib-python``; if it isn't installed we
silently fall back to PCA + HOG voting.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


_FILENAME_RE = re.compile(
    r'^(?P<name>[a-z]+)_(?P<pron>he_him|she_her)_(?P<role>[a-z_]+)\.(?:png|jpg|jpeg)$',
    re.IGNORECASE,
)


@dataclass
class Identity:
    label_id: int
    name: str
    role: str
    gender: str  # 'male' | 'female'
    score: float  # 0..1, higher = better


def _has_lbph() -> bool:
    return hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')


def _augment(crop: np.ndarray) -> list[np.ndarray]:
    """Tiny augmentation: rotations, flip, brightness shifts."""
    h, w = crop.shape[:2]
    out = [crop]
    centre = (w / 2.0, h / 2.0)
    for angle in (-10.0, 10.0):
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        out.append(cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.append(cv2.flip(crop, 1))
    for scale in (0.85, 1.15):
        bright = np.clip(crop.astype(np.int16) * scale, 0, 255).astype(np.uint8)
        out.append(bright)
    return out


def _preprocess(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def _hog_descriptor(size: tuple[int, int]) -> cv2.HOGDescriptor:
    # win = image size, block = 32, stride = 16, cell = 16, 9 bins.
    return cv2.HOGDescriptor(
        _winSize=size,
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9,
    )


class PersonnelRecognizer:
    """Triple-classifier face recogniser trained on personnel portraits."""

    def __init__(
        self,
        personnel_dir: str,
        img_size: tuple[int, int] = (128, 128),
        lbph_threshold: float = 80.0,
        pca_components: int = 25,
        hog_k: int = 3,
        min_score: float = 0.34,
    ):
        self.img_size = img_size
        self.lbph_threshold = lbph_threshold
        self.pca_components = pca_components
        self.hog_k = hog_k
        self.min_score = min_score

        self.people: dict[int, dict] = {}
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )

        crops, labels = self._load(personnel_dir)
        if not crops:
            raise RuntimeError(f'No usable portraits found in {personnel_dir!r}')

        self._train_lbph(crops, labels)
        self._train_pca(crops, labels)
        self._train_hog(crops, labels)

    # ------------------------------------------------------------------ load

    def _load(self, personnel_dir: str) -> tuple[list[np.ndarray], list[int]]:
        crops: list[np.ndarray] = []
        labels: list[int] = []
        next_label = 0

        for fname in sorted(os.listdir(personnel_dir)):
            m = _FILENAME_RE.match(fname)
            if not m:
                continue

            path = os.path.join(personnel_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            face = self._extract_face(img)
            if face is None:
                continue

            label_id = next_label
            next_label += 1
            self.people[label_id] = {
                'name': m['name'].lower(),
                'role': m['role'].lower(),
                'gender': 'female' if m['pron'].lower() == 'she_her' else 'male',
            }

            for variant in _augment(face):
                crops.append(_preprocess(variant, self.img_size))
                labels.append(label_id)

        return crops, labels

    def _extract_face(self, bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
        )
        if len(faces) == 0:
            h, w = gray.shape
            side = min(h, w)
            cy, cx = h // 2, w // 2
            half = side // 2
            return gray[cy - half:cy + half, cx - half:cx + half]
        # Pick the biggest face if multiple.
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        return gray[y:y + h, x:x + w]

    # -------------------------------------------------------------- training

    def _train_lbph(self, crops: list[np.ndarray], labels: list[int]) -> None:
        self._lbph = None
        if not _has_lbph():
            return
        self._lbph = cv2.face.LBPHFaceRecognizer_create()
        self._lbph.train(crops, np.array(labels, dtype=np.int32))

    def _train_pca(self, crops: list[np.ndarray], labels: list[int]) -> None:
        flat = np.array([c.flatten().astype(np.float32) for c in crops])
        self._pca_mean = flat.mean(axis=0)
        centred = flat - self._pca_mean

        # SVD-based PCA — k < n_samples so SVD is cheap.
        k = min(self.pca_components, centred.shape[0] - 1, centred.shape[1])
        u, s, vt = np.linalg.svd(centred, full_matrices=False)
        self._pca_basis = vt[:k]  # (k, D)

        # Per-class mean in PCA space, used as the class prototype.
        proj = centred @ self._pca_basis.T  # (N, k)
        self._pca_class_mean: dict[int, np.ndarray] = {}
        for label in set(labels):
            mask = np.array(labels) == label
            self._pca_class_mean[label] = proj[mask].mean(axis=0)

    def _train_hog(self, crops: list[np.ndarray], labels: list[int]) -> None:
        self._hog = _hog_descriptor(self.img_size)
        self._hog_db = np.array([self._hog.compute(c).flatten() for c in crops])
        # L2-normalise for cosine similarity.
        norms = np.linalg.norm(self._hog_db, axis=1, keepdims=True) + 1e-9
        self._hog_db = self._hog_db / norms
        self._hog_labels = np.array(labels)

    # --------------------------------------------------------------- predict

    def recognize(self, gray_face_crop: np.ndarray) -> Identity | None:
        if gray_face_crop is None or gray_face_crop.size == 0:
            return None
        if min(gray_face_crop.shape[:2]) < 16:
            return None

        prepped = _preprocess(gray_face_crop, self.img_size)

        votes: dict[int, float] = {}
        lbph_label, lbph_dist = self._predict_lbph(prepped)
        pca_label, pca_score = self._predict_pca(prepped)
        hog_label, hog_score = self._predict_hog(prepped)

        if lbph_label is not None:
            votes[lbph_label] = votes.get(lbph_label, 0.0) + 1.0
        if pca_label is not None:
            votes[pca_label] = votes.get(pca_label, 0.0) + pca_score
        if hog_label is not None:
            votes[hog_label] = votes.get(hog_label, 0.0) + hog_score

        if not votes:
            return None

        # Highest weighted vote; LBPH counts as 1.0 by default to dominate ties.
        winner = max(votes, key=lambda k: votes[k])
        score = votes[winner] / max(1, sum(1 for v in (lbph_label, pca_label, hog_label) if v is not None))
        if score < self.min_score:
            return None

        meta = self.people[winner]
        return Identity(
            label_id=winner,
            name=meta['name'],
            role=meta['role'],
            gender=meta['gender'],
            score=float(score),
        )

    def _predict_lbph(self, prepped: np.ndarray) -> tuple[int | None, float]:
        if self._lbph is None:
            return None, 0.0
        label, dist = self._lbph.predict(prepped)
        if dist > self.lbph_threshold:
            return None, 0.0
        # Map distance → score in [0, 1]. Closer is better.
        score = float(max(0.0, 1.0 - dist / self.lbph_threshold))
        return int(label), score

    def _predict_pca(self, prepped: np.ndarray) -> tuple[int | None, float]:
        v = prepped.flatten().astype(np.float32) - self._pca_mean
        proj = v @ self._pca_basis.T
        best_label, best_dist = None, float('inf')
        for label, mean_proj in self._pca_class_mean.items():
            d = float(np.linalg.norm(proj - mean_proj))
            if d < best_dist:
                best_dist, best_label = d, label
        if best_label is None:
            return None, 0.0
        # Normalise by the median pairwise distance so the score is roughly in [0, 1].
        ref = np.median([np.linalg.norm(a - b)
                         for a in self._pca_class_mean.values()
                         for b in self._pca_class_mean.values() if not np.array_equal(a, b)])
        score = float(max(0.0, 1.0 - best_dist / max(ref, 1e-6)))
        return best_label, score

    def _predict_hog(self, prepped: np.ndarray) -> tuple[int | None, float]:
        v = self._hog.compute(prepped).flatten()
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = self._hog_db @ v  # cosine, since both sides are L2-normalised
        top_idx = np.argpartition(-sims, min(self.hog_k, len(sims) - 1))[: self.hog_k]
        top_labels = self._hog_labels[top_idx]
        top_sims = sims[top_idx]

        best_label, best_score = None, -1.0
        for label in set(top_labels.tolist()):
            mask = top_labels == label
            s = float(top_sims[mask].mean())
            if s > best_score:
                best_label, best_score = label, s
        if best_label is None or best_score <= 0:
            return None, 0.0
        return int(best_label), max(0.0, best_score)

    # ------------------------------------------------------------------ misc

    def gender_for(self, label_id: int) -> str | None:
        meta = self.people.get(label_id)
        return meta['gender'] if meta else None

    def all_people(self) -> Iterable[dict]:
        return self.people.values()


def _self_test(personnel_dir: str) -> None:  # pragma: no cover
    """Quick check that the recognizer round-trips its own training images."""
    rec = PersonnelRecognizer(personnel_dir)
    correct = total = 0
    for fname in sorted(os.listdir(personnel_dir)):
        if not _FILENAME_RE.match(fname):
            continue
        path = os.path.join(personnel_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        face = rec._extract_face(img)
        ident = rec.recognize(face)
        expected = fname.split('_', 1)[0].lower()
        ok = ident is not None and ident.name == expected
        correct += int(ok)
        total += 1
        print(f'{fname:50s} -> {ident}  ({"OK" if ok else "FAIL"})')
    print(f'\n{correct}/{total} portraits correctly classified.')


if __name__ == '__main__':  # pragma: no cover
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else 'personnel'
    _self_test(arg)
