import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.signal as signal


class TemporalPostProcessor:
    """
    Phase 7: Temporal Post-Processing
    Applies temporal filtering and dual-threshold hysteresis to convert raw frame-level
    probability matrices into discrete, stabilized sound event annotations.
    - 1.5s median filter smoothing
    - Dual-threshold hysteresis (P_high = 0.65, P_low = 0.35)
    - Pruning short transient events (< 1.0s)
    - Structured JSON and CSV export
    """
    def __init__(
        self,
        class_labels: Optional[List[str]] = None,
        p_high: float = 0.65,
        p_low: float = 0.35,
        median_filter_sec: float = 1.5,
        min_duration_sec: float = 1.0,
        hop_sec: float = 0.5
    ):
        self.class_labels = class_labels or [
            "erhu", "pipa", "dizi", "percussion", "strings_western", "sheng",
            "suona", "guzheng", "yangqin", "woodwind", "brass", "nonmusic"
        ]
        self.p_high = p_high
        self.p_low = p_low
        self.median_filter_sec = median_filter_sec
        self.min_duration_sec = min_duration_sec
        self.hop_sec = hop_sec

    def apply_median_filter(self, P: np.ndarray) -> np.ndarray:
        """
        Applies a 1D median filter across each class probability stream over time.
        """
        # Calculate odd kernel size
        raw_k = int(round(self.median_filter_sec / self.hop_sec))
        kernel_size = raw_k if raw_k % 2 == 1 else raw_k + 1
        kernel_size = max(3, kernel_size)

        P_smooth = np.zeros_like(P)
        for c in range(P.shape[1]):
            if len(P[:, c]) >= kernel_size:
                P_smooth[:, c] = signal.medfilt(P[:, c], kernel_size=kernel_size)
            else:
                P_smooth[:, c] = P[:, c]
        return P_smooth

    def apply_hysteresis(
        self,
        P: np.ndarray,
        timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Applies dual-threshold hysteresis state machine to extract discrete events.
        """
        events: List[Dict[str, Any]] = []
        num_frames, num_classes = P.shape

        for c_idx, class_name in enumerate(self.class_labels):
            probs = P[:, c_idx]
            in_event = False
            start_frame = 0
            event_probs: List[float] = []

            for t in range(num_frames):
                p = probs[t]

                if not in_event:
                    # Trigger onset when probability exceeds high threshold
                    if p >= self.p_high:
                        in_event = True
                        start_frame = t
                        event_probs = [float(p)]
                else:
                    # Sustain event while probability remains above low threshold
                    if p >= self.p_low:
                        event_probs.append(float(p))
                    else:
                        # Event offset
                        in_event = False
                        end_frame = t
                        start_time = float(timestamps[start_frame])
                        end_time = float(timestamps[end_frame])
                        duration = end_time - start_time

                        if duration >= self.min_duration_sec:
                            events.append({
                                "instrument": class_name,
                                "start_time": round(start_time, 2),
                                "end_time": round(end_time, 2),
                                "duration": round(duration, 2),
                                "confidence": round(float(np.mean(event_probs)), 4)
                            })

            # Handle event active at end of timeline
            if in_event:
                end_frame = num_frames - 1
                start_time = float(timestamps[start_frame])
                end_time = float(timestamps[end_frame]) + self.hop_sec
                duration = end_time - start_time
                if duration >= self.min_duration_sec:
                    events.append({
                        "instrument": class_name,
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "duration": round(duration, 2),
                        "confidence": round(float(np.mean(event_probs)), 4)
                    })

        # Sort chronologically by start time
        events.sort(key=lambda x: (x["start_time"], x["end_time"], x["instrument"]))
        return events

    def process(
        self,
        P: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs full Phase 7 pipeline on probability matrix P:
        1. Median filtering
        2. Hysteresis thresholding
        3. Duration pruning
        """
        if timestamps is None:
            timestamps = np.arange(len(P)) * self.hop_sec

        # 1. Median filtering
        P_smoothed = self.apply_median_filter(P)

        # 2. Dual hysteresis & duration pruning
        events = self.apply_hysteresis(P_smoothed, timestamps)
        return events

    def export_annotations(
        self,
        events: List[Dict[str, Any]],
        json_path: Optional[str] = "reports/ast_inference_eval_come_drink_with_me.json",
        csv_path: Optional[str] = "reports/ast_inference_eval_come_drink_with_me.csv",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Exports detected events to structured JSON and CSV formats.
        """
        # Save JSON
        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "metadata": metadata or {
                    "p_high": self.p_high,
                    "p_low": self.p_low,
                    "median_filter_sec": self.median_filter_sec,
                    "min_duration_sec": self.min_duration_sec,
                    "total_events": len(events)
                },
                "events": events
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            print(f"Saved JSON annotations ({len(events)} events) to: {json_path}")

        # Save CSV
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["start_time", "end_time", "duration", "instrument", "confidence"])
                for ev in events:
                    writer.writerow([
                        ev["start_time"],
                        ev["end_time"],
                        ev["duration"],
                        ev["instrument"],
                        ev["confidence"]
                    ])
            print(f"Saved CSV annotations to: {csv_path}")
