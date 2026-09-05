# Color-Coded Hand Gesture Recognition & Air-Drawing App

> **Two versions are included:**
> - `hand_gesture_draw.py` — the original, simpler version.
> - `hand_gesture_draw_pro.py` — **recommended.** A professional-grade rewrite
>   with a One-Euro filter for lag-free drawing, threaded camera capture,
>   live brush/smoothness trackbars, stroke-level undo, logging, type hints,
>   dataclasses, and enums. Use this one for the main project submission.

A real-time computer vision project built with **OpenCV** and **MediaPipe** that:

- Tracks your hand(s) via webcam
- Draws **each finger in its own distinct color** (Thumb, Index, Middle, Ring, Pinky)
- Shows a **color legend fixed in the top-left corner**
- Detects the **individual up/down state of every finger**
- Recognizes full **hand gestures** (Fist, Open Palm, Peace, Thumbs Up, OK Sign, Rock On, Call Me, Pointing, etc.)
- Lets you **air-draw** on screen using your index finger, with color switching, clearing, screenshots, and video recording

This is designed as a complete, presentable main/capstone project — not just a bare demo.

## Features

| Feature | Description |
|---|---|
| Per-finger coloring | Each of the 5 fingers is rendered with a unique color across all its joints and bones |
| Color legend | Always-visible panel in the top-left corner mapping color → finger name |
| Finger state panel | Per-hand panel showing Up/Down for every finger in real time |
| Gesture recognition | Rule-based classifier covering 12+ common gestures + OK-sign pinch detection |
| Air drawing | Point with only your index finger to draw; open palm clears the canvas; fist pauses the pen |
| Color switching | Press `1`-`5` to manually pick the draw color (matches finger colors) |
| Screenshot capture | Press `S` to save the current frame to `screenshots/` |
| Video recording | Press `R` to start/stop saving `output_recording.avi` |
| FPS counter | Live performance readout in the top-right |
| Two-hand support | Tracks and labels up to two hands simultaneously (Left/Right) |
| On-screen help | Toggleable control cheat-sheet at the bottom of the window |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Recommended: the Pro edition
python hand_gesture_draw_pro.py
python hand_gesture_draw_pro.py --camera 1 --model-complexity 0 --min-cutoff 0.8

# Original simpler edition
python hand_gesture_draw.py
```

### Pro-edition CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--camera` | `0` | Webcam index |
| `--width` / `--height` | `1280` / `720` | Capture resolution |
| `--model-complexity` | `0` | MediaPipe model: `0`=fast, `1`=more accurate but slower |
| `--min-cutoff` | `1.2` | One-Euro filter base cutoff — **lower = faster, snappier strokes**; raise it if lines look too jittery |

## Controls

| Key | Action |
|---|---|
| `D` | Toggle draw mode on/off |
| `C` | Clear the drawing canvas |
| `Z` | Undo the last stroke *(Pro edition only)* |
| `1`–`5` | Select draw color (Thumb / Index / Middle / Ring / Pinky) |
| `R` | Start/stop recording the output video |
| `S` | Save a screenshot |
| `H` | Show/hide the help panel |
| `Q` / `Esc` | Quit |

The Pro edition also shows two live trackbars at the top of the window:
**"Brush Size"** and **"Smoothness"** — drag "Smoothness" down if drawing
still feels behind your finger, or up if strokes look too shaky.

## Why the Pro Edition Draws Faster

The original version smoothed the pen position with a fixed 5-frame moving
average, which always trails the real fingertip position — the faster you
move, the further behind the line falls. The Pro edition replaces this with
a **One-Euro Filter**, an adaptive filter built specifically for interactive
pointing: it smooths heavily when your hand is nearly still (killing jitter)
and relaxes almost completely during fast motion (killing lag). It also uses
a background thread for camera capture and MediaPipe's lite model
(`model_complexity=0`) so the whole pipeline runs at a higher, steadier FPS.

## How Gesture Detection Works

1. **MediaPipe Hands** returns 21 normalized (x, y, z) landmarks per hand.
2. For the four non-thumb fingers, a finger is considered **"up"** when its
   fingertip landmark sits above (smaller `y`) its PIP joint.
3. The **thumb** is evaluated on the `x`-axis relative to its IP joint (flipped
   depending on detected handedness), since the thumb moves sideways rather
   than vertically.
4. The five booleans `(Thumb, Index, Middle, Ring, Pinky)` are looked up in a
   gesture table (`GESTURE_MAP`) to name the pose. A separate Euclidean-distance
   check between the thumb tip and index tip detects the **OK sign** (a pinch).
5. If no exact pattern matches, the app falls back to reporting `"N Finger(s) Up"`.

## Possible Extensions

- Swap the rule-based classifier for a trained ML model (e.g., a small
  classifier on landmark coordinates) for more robust, angle-invariant recognition.
- Add gesture-triggered actions (e.g., control system volume, switch slides).
- Export drawings as PNG/SVG with a dedicated "Save Drawing" gesture.
- Add multi-color simultaneous drawing (one stroke per hand).

## Project Structure

```
hand_gesture_project/
├── hand_gesture_draw.py   # Main application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── screenshots/           # Auto-created; saved screenshots land here
```
