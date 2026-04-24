import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RingLandmark:
    id: int
    position: np.ndarray
    covariance: np.ndarray
    observations: int = 0
    color_history: list = field(default_factory=list)

    @property
    def color(self) -> tuple:
        if not self.color_history:
            return (0, 0, 0)
        arr = np.array(self.color_history, dtype=np.uint8)
        return tuple(np.median(arr, axis=0).astype(np.uint8).tolist())

    CONFIRM_OBS   = 6
    CONFIRM_TRACE = 0.22
    MERGE_DIST    = 0.5 # m 


class RingMap:
    def __init__(self):
        self.landmarks: List[RingLandmark] = []

        self.R = np.eye(3) * 0.3
        self.Q = np.eye(3) * 0.01

    def _nearest(self, pos: np.ndarray) -> Optional[tuple[int, float]]:
        best_i, best_d = None, float('inf')
        for i, lm in enumerate(self.landmarks):
            d = np.linalg.norm(lm.position - pos)
            if d < best_d:
                best_i, best_d = i, d
        return (best_i, best_d) if best_i is not None else None

    def update(self, pos: np.ndarray, color: tuple):
        nearest = self._nearest(pos)

        if nearest is None or nearest[1] > RingLandmark.MERGE_DIST:
            lm = RingLandmark(
                id=len(self.landmarks),
                position=pos.copy(),
                covariance=np.eye(3) * 2.0,
                observations=1,
                color_history=[color]
            )
            self.landmarks.append(lm)
        else:
            i, _ = nearest
            lm = self.landmarks[i]

            innovation = pos - lm.position
            S = lm.covariance + self.R
            mahal = float(innovation @ np.linalg.inv(S) @ innovation)

            if mahal > 7.815:
                self.get_logger().warn(
                    f'Rejected outlier observation, mahalanobis={mahal:.2f}'
                ) if hasattr(self, 'get_logger') else None
                return

            P_pred = lm.covariance + self.Q
            y = pos - lm.position
            S = P_pred + self.R
            K = P_pred @ np.linalg.inv(S)
            lm.position   = lm.position + K @ y
            lm.covariance = (np.eye(3) - K) @ P_pred
            lm.observations += 1
            lm.color_history.append(color)

    def confirmed_landmarks(self) -> List[RingLandmark]:
        return [
            lm for lm in self.landmarks
            if lm.observations >= RingLandmark.CONFIRM_OBS
            and np.trace(lm.covariance) < RingLandmark.CONFIRM_TRACE
        ]