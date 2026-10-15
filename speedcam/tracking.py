"""ByteTrack association (supervision), run on reference-frame coordinates."""
from __future__ import annotations

import warnings

import numpy as np
import supervision as sv

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from supervision.tracker.byte_tracker.core import ByteTrack

from .config import TrackingSettings


class Tracker:
    def __init__(self, s: TrackingSettings, fps: float):
        # supervision scales the buffer by frame_rate / 30
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.bt = ByteTrack(
                track_activation_threshold=s.activation_threshold,
                lost_track_buffer=max(1, int(round(30 * s.lost_buffer_s))),
                minimum_matching_threshold=s.match_threshold,
                frame_rate=fps,
                minimum_consecutive_frames=s.min_consecutive_frames,
            )

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """Return a track id per input box (-1 when the box is not (yet) part of a track)."""
        n = len(conf)
        ids = np.full(n, -1, int)
        dets = sv.Detections(
            xyxy=np.asarray(xyxy, np.float32).reshape(-1, 4), confidence=np.asarray(conf, np.float32),
            class_id=np.asarray(cls, int), data={"idx": np.arange(n)},
        )
        out = self.bt.update_with_detections(dets)
        if len(out) and out.tracker_id is not None:
            ids[out.data["idx"]] = out.tracker_id
        return ids
