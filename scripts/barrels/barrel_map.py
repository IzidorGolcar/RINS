"""Barrel landmark tracker.

Close port of ``rings/ring_map.py`` with barrel-specific state:
colour history, orientation history, leak (sticky) and spill anchor.
Confirmation thresholds are retuned for the barrel use case (see plan).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Constants used by the tracker. See plan for rationale.
MERGE_DIST = 0.45     # m (prevent side-by-side barrels from merging)
CONFIRM_OBS = 12
CONFIRM_TRACE = 0.20

# Outlier gate: chi-squared 3 DoF, p=0.05
MAHAL_GATE = 7.815

# Spill: need this many independent confirmations before locking it in.
SPILL_CONFIRM_COUNT = 2


@dataclass
class BarrelLandmark:
    id: int
    position: np.ndarray
    covariance: np.ndarray
    observations: int = 0
    colour_history: List[str] = field(default_factory=list)
    orientation_history: List[str] = field(default_factory=list)

    # Spill state
    leaking: bool = False
    spill_position: Optional[np.ndarray] = None
    pending_spill_count: int = 0   # candidate observations from any camera
    confirmed_spill_cameras: set = field(default_factory=set)
    confirmed_cameras: set = field(default_factory=set)

    # Bottom-of-barrel anchor (for spill search seed and RViz spill marker)
    ground_anchor: Optional[np.ndarray] = None

    @property
    def colour(self) -> str:
        if not self.colour_history:
            return 'unknown'
        return Counter(self.colour_history).most_common(1)[0][0]

    @property
    def orientation(self) -> str:
        """Mode orientation, requires >=70% majority else 'ambiguous'."""
        if not self.orientation_history:
            return 'ambiguous'
        counts = Counter(self.orientation_history)
        top, n = counts.most_common(1)[0]
        if n / len(self.orientation_history) >= 0.7:
            return top
        return 'ambiguous'


class BarrelMap:
    def __init__(self):
        self.landmarks: List[BarrelLandmark] = []

        # Observation / process noise (same scale ring_map.py uses).
        self.R = np.eye(3) * 0.3
        self.Q = np.eye(3) * 0.01

    def _nearest(self, pos: np.ndarray, colour: str) -> Optional[tuple[int, float]]:
        best_i, best_d = None, float('inf')
        for i, lm in enumerate(self.landmarks):
            # Only match landmarks of the same color!
            if lm.colour != colour:
                continue
            d = float(np.linalg.norm(lm.position - pos))
            if d < best_d:
                best_i, best_d = i, d
        return (best_i, best_d) if best_i is not None else None

    def update(self,
               pos: np.ndarray,
               colour: str,
               orientation: str,
               ground_anchor: Optional[np.ndarray] = None,
               camera_id: str = 'unknown') -> Optional[BarrelLandmark]:
        """Insert or update a landmark; returns the affected landmark (or None if rejected)."""
        nearest = self._nearest(pos, colour)

        if nearest is None or nearest[1] > MERGE_DIST:
            lm = BarrelLandmark(
                id=len(self.landmarks),
                position=pos.copy(),
                covariance=np.eye(3) * 2.0,
                observations=1,
                colour_history=[colour],
                orientation_history=[orientation],
                ground_anchor=None if ground_anchor is None else ground_anchor.copy(),
            )
            lm.confirmed_cameras.add(camera_id)
            self.landmarks.append(lm)
            return lm

        i, _ = nearest
        lm = self.landmarks[i]
        lm.confirmed_cameras.add(camera_id)

        # Mahalanobis outlier rejection (verbatim from ring_map.py).
        innovation = pos - lm.position
        S = lm.covariance + self.R
        try:
            mahal = float(innovation @ np.linalg.inv(S) @ innovation)
        except np.linalg.LinAlgError:
            return None
        if mahal > MAHAL_GATE:
            return None

        # Kalman-style update.
        P_pred = lm.covariance + self.Q
        y = pos - lm.position
        S = P_pred + self.R
        K = P_pred @ np.linalg.inv(S)
        lm.position = lm.position + K @ y
        lm.covariance = (np.eye(3) - K) @ P_pred
        lm.observations += 1
        lm.colour_history.append(colour)
        lm.orientation_history.append(orientation)
        if ground_anchor is not None:
            # Running mean of ground anchor (cheap; we don't need a separate filter).
            if lm.ground_anchor is None:
                lm.ground_anchor = ground_anchor.copy()
            else:
                lm.ground_anchor = 0.7 * lm.ground_anchor + 0.3 * ground_anchor
        return lm

    def nearest_landmark(self, pos: np.ndarray, colour: str,
                         max_dist: float) -> Optional[BarrelLandmark]:
        nearest = self._nearest(pos, colour)
        if nearest is None:
            return None
        i, d = nearest
        if d > max_dist:
            return None
        return self.landmarks[i]

    def register_spill(self,
                       landmark: BarrelLandmark,
                       spill_pos: np.ndarray,
                       camera_id: str) -> None:
        """Record a spill candidate from a specific camera.

        A spill is locked in (``leaking=True``) only once it has been seen by
        at least two distinct cameras (OAK-D + top), which is the multi-camera
        agreement the plan calls out.
        """
        if landmark.leaking:
            # Sticky: update position with running mean, don't downgrade.
            if landmark.spill_position is None:
                landmark.spill_position = spill_pos.copy()
            else:
                landmark.spill_position = (
                    0.8 * landmark.spill_position + 0.2 * spill_pos
                )
            return

        landmark.pending_spill_count += 1
        landmark.confirmed_spill_cameras.add(camera_id)
        if landmark.spill_position is None:
            landmark.spill_position = spill_pos.copy()
        else:
            landmark.spill_position = 0.5 * landmark.spill_position + 0.5 * spill_pos

        if (
            len(landmark.confirmed_spill_cameras) >= 2
            or landmark.pending_spill_count >= SPILL_CONFIRM_COUNT + 1
        ):
            # Either two cameras agree, or one camera has confirmed enough times
            # that we accept it unilaterally (some scenes will only be seen from
            # one viewpoint).
            landmark.leaking = True

    def confirmed_landmarks(self) -> List[BarrelLandmark]:
        confirmed = []
        for lm in self.landmarks:
            if lm.orientation == 'ambiguous':
                continue
            
            # Highly robust filter: if it was seen by OAK-D, it confirms in 6 frames.
            # If it was only seen by the more jittery top camera, it requires 18 frames
            # to prevent transient false detections (like the conveyor opening) from confirming.
            has_oakd = 'oakd' in lm.confirmed_cameras
            required_obs = CONFIRM_OBS if has_oakd else (CONFIRM_OBS * 3)

            if lm.observations >= required_obs and float(np.trace(lm.covariance)) < CONFIRM_TRACE:
                confirmed.append(lm)
        return confirmed
