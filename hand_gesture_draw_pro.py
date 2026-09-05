"""
================================================================================
 Hand Gesture Recognition & Air-Drawing Studio  —  PRO EDITION
================================================================================

Author  : Aryan
Stack   : Python 3.10+, OpenCV, MediaPipe, NumPy
Purpose : Capstone / main-project grade real-time computer vision application.

WHAT THIS DOES
--------------
  1. Tracks up to two hands via webcam using MediaPipe Hands.
  2. Renders every finger (Thumb / Index / Middle / Ring / Pinky) in a unique
     color, with a persistent legend pinned to the top-left corner.
  3. Reports the live Up/Down state of every individual finger, and classifies
     the overall hand pose into a named gesture (Fist, Open Palm, Peace,
     Thumbs Up, OK Sign, Rock On, Call Me, Pointing, etc.).
  4. Provides a low-latency "air-drawing" canvas driven by the index finger,
     using a One-Euro filter for jitter-free *and* lag-free strokes — the
     single biggest quality upgrade over a naive moving-average approach.
  5. Ships with the tooling a real product would have: threaded capture for
     higher throughput, on-screen trackbars for live tuning, stroke-level
     undo, screenshotting, video recording, and structured logging.

WHY THE DRAWING FEELS FASTER NOW
---------------------------------
  The previous version smoothed the pen position with a fixed-size moving
  average. Averaging N frames always drags the cursor N/2 frames behind your
  actual fingertip — the faster you move, the further behind it falls.

  This version replaces that with a **One-Euro Filter** (Casiez et al.), an
  adaptive low-pass filter purpose-built for human-interface pointing:
    - When your finger moves SLOWLY   -> filter smooths heavily (kills jitter)
    - When your finger moves QUICKLY  -> filter relaxes almost completely
                                          (kills lag, tracks your finger tightly)
  You can also tune this live with the "Smoothness" trackbar, and control
  brush size with the "Brush Size" trackbar — no code edits required.

CONTROLS
--------
  Q / Esc      Quit
  D            Toggle draw mode ON/OFF
  C            Clear canvas (and stroke history)
  Z            Undo the last stroke
  1-5          Select draw color (Thumb / Index / Middle / Ring / Pinky)
  R            Start / stop recording output_recording.avi
  S            Save a screenshot to screenshots/
  H            Show / hide the help panel
  Trackbars    "Brush Size" and "Smoothness" (live-adjustable, top of window)

GESTURE -> DRAWING BEHAVIOUR (while draw mode is ON)
-----------------------------------------------------
  Only Index finger up   -> Pen down, draws in the active color
  Open Palm (5 up)       -> Clears canvas instantly
  Fist (0 up)            -> Pen lifted (pauses the current stroke)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

# ==============================================================================
# LOGGING
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HandGestureStudio")

Color = Tuple[int, int, int]  # BGR tuple, since OpenCV works in BGR


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class AppConfig:
    """All tunable parameters in one place, with sane, professional defaults."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    max_hands: int = 2
    model_complexity: int = 0          # 0 = fast/lite, 1 = accurate/heavier
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6

    window_name: str = "Hand Gesture Recognition & Air Drawing — PRO"

    default_brush_thickness: int = 10
    min_brush_thickness: int = 2
    max_brush_thickness: int = 40

    # One-Euro filter defaults. Lower min_cutoff = snappier / less lag.
    # Higher beta = more aggressive de-lag during fast movement.
    smoothing_min_cutoff: float = 1.2
    smoothing_beta: float = 0.7

    screenshot_dir: str = "screenshots"
    recording_filename: str = "output_recording.avi"
    recording_fps: float = 20.0


# ==============================================================================
# FINGER / LANDMARK CONSTANTS
# ==============================================================================

FINGER_COLORS: Dict[str, Color] = {
    "Thumb":  (0,   0,   255),   # Red
    "Index":  (0,   255, 0),     # Green
    "Middle": (255, 0,   0),     # Blue
    "Ring":   (0,   255, 255),   # Yellow
    "Pinky":  (255, 0,   255),   # Magenta
}
PALM_COLOR: Color = (180, 180, 180)
WRIST_COLOR: Color = (255, 255, 255)

FINGER_ORDER: List[str] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

FINGER_LANDMARKS: Dict[str, List[int]] = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20],
}
FINGER_TIPS: Dict[str, int] = {"Thumb": 4, "Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}
FINGER_PIPS: Dict[str, int] = {"Index": 6, "Middle": 10, "Ring": 14, "Pinky": 18}

_LANDMARK_TO_FINGER: Dict[int, str] = {
    idx: finger for finger, ids in FINGER_LANDMARKS.items() for idx in ids
}


def finger_of_landmark(idx: int) -> Optional[str]:
    """Return which finger a given MediaPipe landmark index belongs to (None = wrist)."""
    return _LANDMARK_TO_FINGER.get(idx)


# ==============================================================================
# GESTURE CLASSIFICATION
# ==============================================================================

class Gesture(str, Enum):
    FIST = "Fist"
    OPEN_PALM = "Open Palm"
    PEACE = "Peace / Victory"
    THUMBS_UP = "Thumbs Up"
    POINTING = "Pointing"
    THREE = "Three"
    FOUR = "Four"
    CALL_ME = "Call Me (Shaka)"
    ROCK_ON = "Rock On"
    GUN = "Gun"
    PINKY_UP = "Pinky Up"
    OK_SIGN = "OK Sign"
    STOP = "Stop (No Thumb)"
    UNKNOWN = "Unknown"


# Maps (Thumb, Index, Middle, Ring, Pinky) up/down booleans -> Gesture
_GESTURE_MAP: Dict[Tuple[bool, bool, bool, bool, bool], Gesture] = {
    (False, False, False, False, False): Gesture.FIST,
    (True,  True,  True,  True,  True):  Gesture.OPEN_PALM,
    (False, True,  True,  False, False): Gesture.PEACE,
    (True,  False, False, False, False): Gesture.THUMBS_UP,
    (False, True,  False, False, False): Gesture.POINTING,
    (False, True,  True,  True,  False): Gesture.THREE,
    (True,  True,  True,  True,  False): Gesture.FOUR,
    (True,  False, False, False, True):  Gesture.CALL_ME,
    (False, True,  False, False, True):  Gesture.ROCK_ON,
    (True,  True,  False, False, False): Gesture.GUN,
    (False, False, False, False, True):  Gesture.PINKY_UP,
    (False, True,  True,  True,  True):  Gesture.STOP,
}

_OK_SIGN_PINCH_THRESHOLD = 0.05  # normalized distance between thumb & index tips


def classify_gesture(finger_states: Dict[str, bool], landmarks) -> Gesture:
    """Classify a hand pose into a Gesture enum from per-finger up/down states."""
    thumb_tip = landmarks[FINGER_TIPS["Thumb"]]
    index_tip = landmarks[FINGER_TIPS["Index"]]
    pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

    if (
        pinch_dist < _OK_SIGN_PINCH_THRESHOLD
        and finger_states["Middle"]
        and finger_states["Ring"]
        and finger_states["Pinky"]
    ):
        return Gesture.OK_SIGN

    key = tuple(finger_states[f] for f in FINGER_ORDER)
    return _GESTURE_MAP.get(key, Gesture.UNKNOWN)


def get_finger_states(landmarks, hand_label: str) -> Dict[str, bool]:
    """
    Determine which fingers are extended.

    Non-thumb fingers: "up" when the tip sits above (smaller y) its PIP joint.
    Thumb: compared on the x-axis against its IP joint, mirrored by handedness,
    since the thumb extends sideways rather than vertically. This lightweight
    geometric heuristic avoids needing a trained classifier and runs in
    constant time per frame.
    """
    states: Dict[str, bool] = {}

    thumb_tip = landmarks[FINGER_TIPS["Thumb"]]
    thumb_ip = landmarks[3]
    states["Thumb"] = (
        thumb_tip.x > thumb_ip.x if hand_label == "Right" else thumb_tip.x < thumb_ip.x
    )

    for finger in ("Index", "Middle", "Ring", "Pinky"):
        tip = landmarks[FINGER_TIPS[finger]]
        pip = landmarks[FINGER_PIPS[finger]]
        states[finger] = tip.y < pip.y

    return states


# ==============================================================================
# ONE-EURO FILTER — low-lag, jitter-resistant pointer smoothing
# ==============================================================================

def _alpha(cutoff_freq: float, time_delta: float) -> float:
    """Smoothing coefficient for a first-order low-pass filter at a given cutoff."""
    r = 2.0 * math.pi * cutoff_freq * time_delta
    return r / (r + 1.0)


class _LowPassFilter:
    """Single-value exponential low-pass filter used internally by OneEuroFilter."""

    def __init__(self, initial_value: float) -> None:
        self._value = initial_value

    def apply(self, value: float, alpha: float) -> float:
        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value


class OneEuroFilter:
    """
    Adaptive low-pass filter for real-time signals (Casiez, Roussel & Vogel, 2012).

    Behaves like a heavy smoother when the input is nearly still (kills jitter)
    and like a pass-through when the input is moving quickly (kills lag) —
    exactly the trade-off a drawing cursor needs.
    """

    def __init__(self, timestamp: float, initial_value: float,
                 min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = _LowPassFilter(initial_value)
        self._dx_filter = _LowPassFilter(0.0)
        self._t_prev = timestamp

    def __call__(self, timestamp: float, value: float) -> float:
        t_e = max(timestamp - self._t_prev, 1e-6)
        self._t_prev = timestamp

        derivative = (value - self._x_filter.value) / t_e
        d_alpha = _alpha(self.d_cutoff, t_e)
        smoothed_derivative = self._dx_filter.apply(derivative, d_alpha)

        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        x_alpha = _alpha(cutoff, t_e)
        return self._x_filter.apply(value, x_alpha)


class PointSmoother:
    """Applies a OneEuroFilter independently to the x and y pixel coordinates."""

    def __init__(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._fx: Optional[OneEuroFilter] = None
        self._fy: Optional[OneEuroFilter] = None

    def set_params(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        if self._fx is not None:
            self._fx.min_cutoff = min_cutoff
            self._fx.beta = beta
            self._fy.min_cutoff = min_cutoff
            self._fy.beta = beta

    def reset(self) -> None:
        self._fx = None
        self._fy = None

    def update(self, x: float, y: float, timestamp: float) -> Tuple[int, int]:
        if self._fx is None:
            self._fx = OneEuroFilter(timestamp, x, self.min_cutoff, self.beta)
            self._fy = OneEuroFilter(timestamp, y, self.min_cutoff, self.beta)
            return int(x), int(y)
        return int(self._fx(timestamp, x)), int(self._fy(timestamp, y))


# ==============================================================================
# THREADED CAMERA CAPTURE
# ==============================================================================

class ThreadedCamera:
    """
    Reads frames from the webcam on a background thread.

    cv2.VideoCapture.read() blocks until the driver hands back a frame; running
    it on its own thread decouples capture from processing so the main loop
    always works with the most recent frame instead of stalling on I/O —
    a standard technique for improving perceived responsiveness.
    """

    def __init__(self, source: int, width: int, height: int) -> None:
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {source}. "
                "Verify a webcam is connected and not in use by another application."
            )

        self._lock = threading.Lock()
        self._ok, self._frame = self._cap.read()
        self._stopped = False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok, self._frame = ok, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def stop(self) -> None:
        self._stopped = True
        self._thread.join(timeout=1.0)
        self._cap.release()


# ==============================================================================
# DRAWING CANVAS WITH STROKE HISTORY (UNDO SUPPORT)
# ==============================================================================

@dataclass
class Stroke:
    color: Color
    thickness: int
    points: List[Tuple[int, int]] = field(default_factory=list)


class DrawingCanvas:
    """Off-screen RGB canvas that records strokes as discrete, undoable objects."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        self.strokes: List[Stroke] = []
        self._active: Optional[Stroke] = None

    def start_stroke(self, color: Color, thickness: int) -> None:
        self._active = Stroke(color=color, thickness=thickness)
        self.strokes.append(self._active)

    def extend_stroke(self, point: Tuple[int, int]) -> None:
        if self._active is None:
            return
        self._active.points.append(point)
        if len(self._active.points) >= 2:
            cv2.line(
                self.image,
                self._active.points[-2],
                self._active.points[-1],
                self._active.color,
                self._active.thickness,
                cv2.LINE_AA,
            )

    def end_stroke(self) -> None:
        self._active = None

    def undo(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self._redraw()

    def clear(self) -> None:
        self.strokes.clear()
        self._active = None
        self.image[:] = 0

    def _redraw(self) -> None:
        self.image[:] = 0
        for stroke in self.strokes:
            for p1, p2 in zip(stroke.points, stroke.points[1:]):
                cv2.line(self.image, p1, p2, stroke.color, stroke.thickness, cv2.LINE_AA)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class HandGestureStudio:
    """Owns the full pipeline: capture -> detect -> classify -> draw -> render."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        self.camera = ThreadedCamera(config.camera_index, config.frame_width, config.frame_height)

        mp_hands = mp.solutions.hands
        self._mp_hands_module = mp_hands
        self.hands = mp_hands.Hands(
            max_num_hands=config.max_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )

        self.canvas: Optional[DrawingCanvas] = None
        self.draw_mode = False
        self.show_help = True
        self.current_color_name = "Index"

        self.smoother = PointSmoother(config.smoothing_min_cutoff, config.smoothing_beta)
        self._was_drawing = False

        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None

        self._fps_samples: Deque[float] = deque(maxlen=30)
        self._prev_time = time.time()

        os.makedirs(config.screenshot_dir, exist_ok=True)
        self._init_window_and_trackbars()

        logger.info("Hand Gesture Studio initialized (model_complexity=%d).", config.model_complexity)

    # ------------------------------------------------------------------ #
    # Window / trackbars
    # ------------------------------------------------------------------ #
    def _init_window_and_trackbars(self) -> None:
        cv2.namedWindow(self.config.window_name)
        cv2.createTrackbar(
            "Brush Size", self.config.window_name,
            self.config.default_brush_thickness, self.config.max_brush_thickness,
            lambda _v: None,
        )
        # Smoothness trackbar: 1-30 -> min_cutoff 0.1-3.0 (lower = snappier/faster).
        cv2.createTrackbar(
            "Smoothness", self.config.window_name,
            int(self.config.smoothing_min_cutoff * 10), 30,
            lambda _v: None,
        )

    def _current_brush_thickness(self) -> int:
        raw = cv2.getTrackbarPos("Brush Size", self.config.window_name)
        return max(self.config.min_brush_thickness, raw)

    def _apply_smoothing_trackbar(self) -> None:
        raw = cv2.getTrackbarPos("Smoothness", self.config.window_name)
        min_cutoff = max(raw, 1) / 10.0
        self.smoother.set_params(min_cutoff, self.config.smoothing_beta)

    # ------------------------------------------------------------------ #
    # Rendering: skeleton, legend, panels
    # ------------------------------------------------------------------ #
    def _draw_colored_skeleton(self, frame: np.ndarray, hand_landmarks) -> None:
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        for start_idx, end_idx in self._mp_hands_module.HAND_CONNECTIONS:
            f_start, f_end = finger_of_landmark(start_idx), finger_of_landmark(end_idx)
            color = FINGER_COLORS[f_start] if f_start and f_start == f_end else PALM_COLOR
            cv2.line(frame, points[start_idx], points[end_idx], color, 3, cv2.LINE_AA)

        for idx, (x, y) in enumerate(points):
            finger = finger_of_landmark(idx)
            color = FINGER_COLORS[finger] if finger else WRIST_COLOR
            radius = 8 if idx in FINGER_TIPS.values() else 5
            cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_legend(self, frame: np.ndarray) -> None:
        x0, y0 = 10, 10
        line_h = 30
        panel_w = 230
        panel_h = line_h * (len(FINGER_ORDER) + 1) + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "FINGER COLOR LEGEND", (x0 + 10, y0 + 22),
                    self.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for i, finger in enumerate(FINGER_ORDER):
            cy = y0 + 22 + line_h * (i + 1)
            cv2.circle(frame, (x0 + 20, cy - 6), 8, FINGER_COLORS[finger], -1)
            marker = " <- draw color" if finger == self.current_color_name and self.draw_mode else ""
            cv2.putText(frame, f"{finger}{marker}", (x0 + 40, cy),
                        self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_finger_state_panel(self, frame: np.ndarray, finger_states: Dict[str, bool],
                                  gesture: Gesture, hand_label: str, origin_x: int) -> None:
        y0 = 10
        line_h = 26
        panel_w, panel_h = 220, line_h * (len(FINGER_ORDER) + 2) + 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (origin_x, y0), (origin_x + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"{hand_label} Hand", (origin_x + 10, y0 + 22),
                    self.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        for i, finger in enumerate(FINGER_ORDER):
            cy = y0 + 22 + line_h * (i + 1)
            cv2.circle(frame, (origin_x + 15, cy - 6), 6, FINGER_COLORS[finger], -1)
            state_txt = "Up" if finger_states[finger] else "Down"
            cv2.putText(frame, f"{finger}: {state_txt}", (origin_x + 30, cy),
                        self.FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Gesture: {gesture.value}", (origin_x + 10, y0 + panel_h - 8),
                    self.FONT, 0.5, (0, 255, 180), 1, cv2.LINE_AA)

    def _draw_help_panel(self, frame: np.ndarray) -> None:
        if not self.show_help:
            return
        h, w = frame.shape[:2]
        lines = [
            "H: help   D: draw mode   C: clear   Z: undo",
            "1-5: pick color (Thumb/Index/Middle/Ring/Pinky)",
            "R: record   S: screenshot   Q/Esc: quit",
        ]
        y0 = h - 20 * len(lines) - 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0 - 10), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (15, y0 + 20 * i + 12),
                        self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_fps(self, frame: np.ndarray) -> None:
        now = time.time()
        instant_fps = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now
        self._fps_samples.append(instant_fps)
        avg_fps = sum(self._fps_samples) / len(self._fps_samples)

        w = frame.shape[1]
        cv2.putText(frame, f"FPS: {avg_fps:.0f}", (w - 130, 30),
                    self.FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Drawing logic
    # ------------------------------------------------------------------ #
    def _handle_drawing(self, frame_shape: Tuple[int, int, int],
                         finger_states: Dict[str, bool], landmarks) -> None:
        assert self.canvas is not None
        h, w = frame_shape[:2]
        index_tip = landmarks[FINGER_TIPS["Index"]]
        raw_point = (index_tip.x * w, index_tip.y * h)

        all_up = all(finger_states.values())
        all_down = not any(finger_states.values())
        only_index = finger_states["Index"] and not any(
            finger_states[f] for f in ("Thumb", "Middle", "Ring", "Pinky")
        )

        if all_up:
            self.canvas.clear()
            self.smoother.reset()
            self._was_drawing = False
            return

        if all_down or not only_index:
            if self._was_drawing:
                self.canvas.end_stroke()
                self.smoother.reset()
            self._was_drawing = False
            return

        self._apply_smoothing_trackbar()
        smooth_point = self.smoother.update(raw_point[0], raw_point[1], time.time())

        if not self._was_drawing:
            thickness = self._current_brush_thickness()
            color = FINGER_COLORS[self.current_color_name]
            self.canvas.start_stroke(color, thickness)
            self._was_drawing = True

        self.canvas.extend_stroke(smooth_point)

    def _composite_canvas_onto_frame(self, frame: np.ndarray) -> np.ndarray:
        assert self.canvas is not None
        mask_gray = cv2.cvtColor(self.canvas.image, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY_INV)
        mask_inv_3c = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(frame, mask_inv_3c)
        return cv2.bitwise_or(background, self.canvas.image)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        logger.info("Application started. Press 'Q' or 'Esc' to quit.")
        try:
            while True:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    logger.warning("Frame not received from camera; retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if self.canvas is None:
                    self.canvas = DrawingCanvas(w, h)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        hand_label = "Right"
                        if results.multi_handedness and hand_idx < len(results.multi_handedness):
                            hand_label = results.multi_handedness[hand_idx].classification[0].label

                        self._draw_colored_skeleton(frame, hand_landmarks)

                        finger_states = get_finger_states(hand_landmarks.landmark, hand_label)
                        gesture = classify_gesture(finger_states, hand_landmarks.landmark)

                        panel_x = w - 230 - hand_idx * 230
                        self._draw_finger_state_panel(frame, finger_states, gesture, hand_label, panel_x)

                        if self.draw_mode and hand_idx == 0:
                            self._handle_drawing(frame.shape, finger_states, hand_landmarks.landmark)
                else:
                    if self._was_drawing and self.canvas is not None:
                        self.canvas.end_stroke()
                    self._was_drawing = False
                    self.smoother.reset()

                frame = self._composite_canvas_onto_frame(frame)

                self._draw_legend(frame)
                self._draw_help_panel(frame)
                self._draw_fps(frame)

                mode_color = (0, 255, 255) if self.draw_mode else (0, 0, 255)
                cv2.putText(frame, f"DRAW MODE: {'ON' if self.draw_mode else 'OFF'}",
                            (w // 2 - 100, 30), self.FONT, 0.6, mode_color, 2, cv2.LINE_AA)

                if self.recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                    cv2.circle(frame, (w - 30, 60), 8, (0, 0, 255), -1)

                cv2.imshow(self.config.window_name, frame)

                if not self._handle_keys(frame):
                    break
        finally:
            self.cleanup()

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _handle_keys(self, frame: np.ndarray) -> bool:
        """Return False to signal the main loop should stop."""
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            return False

        if key == ord('d'):
            self.draw_mode = not self.draw_mode
            self.smoother.reset()
            self._was_drawing = False
            logger.info("Draw mode %s.", "ENABLED" if self.draw_mode else "DISABLED")

        elif key == ord('c') and self.canvas is not None:
            self.canvas.clear()
            logger.info("Canvas cleared.")

        elif key == ord('z') and self.canvas is not None:
            self.canvas.undo()
            logger.info("Last stroke undone.")

        elif key == ord('h'):
            self.show_help = not self.show_help

        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            self.current_color_name = FINGER_ORDER[int(chr(key)) - 1]

        elif key == ord('s'):
            self._save_screenshot(frame)

        elif key == ord('r'):
            self._toggle_recording(frame)

        return True

    def _save_screenshot(self, frame: np.ndarray) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.screenshot_dir, f"shot_{timestamp}.png")
        cv2.imwrite(path, frame)
        logger.info("Screenshot saved -> %s", path)

    def _toggle_recording(self, frame: np.ndarray) -> None:
        self.recording = not self.recording
        if self.recording:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            h, w = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                self.config.recording_filename, fourcc, self.config.recording_fps, (w, h)
            )
            logger.info("Recording started -> %s", self.config.recording_filename)
        else:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            logger.info("Recording stopped.")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def cleanup(self) -> None:
        logger.info("Shutting down and releasing resources...")
        self.camera.stop()
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        self.hands.close()


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition & Air Drawing Studio (Pro Edition)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument(
        "--model-complexity", type=int, choices=(0, 1), default=0,
        help="MediaPipe model complexity: 0=fast, 1=accurate (default: 0)",
    )
    parser.add_argument(
        "--min-cutoff", type=float, default=1.2,
        help="One-Euro filter base cutoff; lower = snappier/faster strokes (default: 1.2)",
    )
    args = parser.parse_args()

    return AppConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        model_complexity=args.model_complexity,
        smoothing_min_cutoff=args.min_cutoff,
    )


def main() -> None:
    config = parse_args()
    app: Optional[HandGestureStudio] = None
    try:
        app = HandGestureStudio(config)
        app.run()
    except RuntimeError as err:
        logger.error(str(err))
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C). Shutting down gracefully.")
    finally:
        if app is not None:
            app.cleanup()


if __name__ == "__main__":
    main()
