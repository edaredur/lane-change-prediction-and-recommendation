# Lane Change Prediction and Recommendation

A real-time Advanced Driver Assistance System (ADAS) prototype that detects surrounding vehicles, predicts their lane change intentions using deep learning, and provides lane change recommendations to the driver — all from a single forward-facing camera.

The system combines classical computer vision (Canny edge detection, Hough transform, bird's-eye view projection) with modern deep learning (YOLOv5 for vehicle detection, BiLSTM for lane change prediction) in a multi-threaded pipeline that runs in real time on consumer hardware.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Computer Vision Pipeline](#computer-vision-pipeline)
  - [Lane Detection](#lane-detection)
  - [Vehicle Detection and Tracking](#vehicle-detection-and-tracking)
  - [Distance Estimation](#distance-estimation)
  - [Lateral Offset Calculation](#lateral-offset-calculation)
  - [Ego Lane Estimation](#ego-lane-estimation)
- [Lane Change Prediction (BiLSTM)](#lane-change-prediction-bilstm)
  - [Network Architecture](#network-architecture)
  - [Feature Engineering](#feature-engineering)
  - [Sliding Window Inference](#sliding-window-inference)
  - [Model Performance](#model-performance)
- [Lane Change Recommendation Engine](#lane-change-recommendation-engine)
  - [Lane Availability Assessment](#lane-availability-assessment)
  - [Decision Logic](#decision-logic)
  - [Recommendation Stabilization](#recommendation-stabilization)
- [Bird's-Eye View Visualization](#birds-eye-view-visualization)
- [Threading Architecture](#threading-architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Output Format](#output-format)
- [Training Data](#training-data)

---

## Overview

The system processes video from a forward-facing camera (dashcam, webcam, or simulation screen capture) and performs the following tasks in real time:

1. **Detects lane markings** using edge detection and Hough line transforms
2. **Detects and tracks vehicles** using YOLOv5 with ByteTrack persistent tracking
3. **Estimates distances** via a dual-method calibration system (pinhole model + lane-width ruler)
4. **Calculates lateral offsets** of each vehicle relative to the detected lane boundaries
5. **Predicts lane change intentions** for each tracked vehicle using a BiLSTM neural network over a 30-frame sliding window
6. **Determines ego lane position** using a fused estimator with EMA smoothing and hysteresis
7. **Recommends lane changes** to the driver based on traffic density, time-to-collision, gap analysis, and safety checks
8. **Visualizes everything** on a combined dashboard with camera overlay, bird's-eye view, and real-time statistics

---

## System Architecture

The system is organized as a three-thread producer–consumer pipeline:

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│   Frame Reader   │────▶│  Processor Thread    │────▶│   Main Thread        │
│   (Thread 1)     │     │  (Thread 2)          │     │   (Display/Write)    │
│                  │     │                      │     │                      │
│ • VideoCapture   │  FQ │ • Lane detection     │  RQ │ • AsyncVideoWriter   │
│ • Frame grab     │────▶│ • YOLO detection     │────▶│ • cv2.imshow         │
│ • Raw recording  │     │ • BiLSTM prediction  │     │ • Keyboard handling  │
│ • Drop-oldest    │     │ • Decision engine    │     │ • Auto-scale display │
│                  │     │ • BEV rendering      │     │                      │
└──────────────────┘     │ • CSV logging        │     └──────────────────────┘
                         └─────────────────────┘
```

- **`FRAME_QUEUE`** (capacity 1): Single-frame buffer with drop-oldest policy to prevent latency buildup
- **`RESULT_QUEUE`** (capacity 5): Processed results buffer for display/recording
- **`SHUTDOWN_FLAG`**: Thread-safe event for graceful shutdown across all threads

---

## Computer Vision Pipeline

### Lane Detection

Lane markings are detected using a classical computer vision pipeline:

1. **Downscale** the frame by 50% (`LANE_DOWNSCALE = 0.5`) for faster processing
2. **Grayscale conversion** followed by **Gaussian blur** (5×5 kernel) to reduce noise
3. **Canny edge detection** with thresholds (low=70, high=180)
4. **Trapezoidal ROI mask** to focus on the road ahead
5. **Probabilistic Hough Line Transform** (`HoughLinesP`) with parameters:
   - ρ = 2 pixels, θ = 1°, threshold = 80 votes
   - Minimum line length = 60 px, maximum gap = 80 px
6. **Segment classification**: Lines are split into left/right groups by slope sign
7. **Averaging** with center-bias weighting and slope filtering (valid range: 0.4–2.5)
8. **Temporal smoothing**: Rolling average over the last 5 frames per side
9. **Vanishing point clipping**: Lines are clipped at their intersection to prevent overlap

**Line style classification** is performed by measuring edge support along each lane boundary:
- **Solid**: coverage ≥ 62% and max gap ≤ 22%
- **Dashed**: coverage ≤ 55% and max gap ≥ 28%

**Yellow line detection** uses HSV color space sampling along the right lane boundary and frame edges, with both strict and relaxed HSV thresholds.

### Vehicle Detection and Tracking

- **YOLOv5 Nano** (`yolov5nu.pt`) runs at 480px inference resolution with FP16 on CUDA
- Very low confidence threshold (0.10) to catch partially occluded vehicles
- Detects COCO classes: car (2), motorcycle (3), bus (5), truck (7)
- **ByteTrack** provides persistent multi-object tracking with unique IDs
- Fallback nearest-neighbor matching (80px radius) when ByteTrack IDs are unavailable
- **Frame skipping** support: cached detections reused for up to 6 missed frames
- **Truncated bbox detection**: Vehicles at frame edges are flagged with a 10-frame cooldown after leaving the edge, preventing false predictions from partial visibility
- **Oncoming vehicle filter**: Removes vehicles closing at > 14 m/s with lateral offset > 0.7 (normalized)

### Distance Estimation

A dual-method calibration system provides robust distance estimates:

1. **Raw pinhole model** (`calculate_longitudinal_distance`):
   - Uses camera height (1.2m), pitch (10°), and vertical FOV
   - Formula: `z = camera_height / tan(pitch + pixel_angle)`
   - Clipped to [0.5m, 200m]

2. **Lane-width ruler** (`distance_from_lane_width`):
   - Uses detected lane width as a known-size reference (3.6m real-world)
   - Formula: `Z = focal_length × lane_width_real / lane_width_pixels`
   - More accurate when both lane lines are visible

3. **EMA calibration factor**: A running exponential moving average of the ratio between the two methods, used as a correction factor when lane lines are partially lost.

### Lateral Offset Calculation

For each detected vehicle:

1. Get x-coordinates of left and right lane lines at the vehicle's y-position
2. Validate lane width (minimum 40px, maximum 90% of frame width)
3. Compute lane center = (left_x + right_x) / 2
4. **Offset in pixels** = vehicle_center_x − lane_center_x
5. **Offset in meters** = (offset_px / lane_width_px) × 3.6m
6. **Normalized offset** = offset_px / (lane_width_px / 2), giving a value in [-1, +1]
7. **Position label**: "center" if |normalized| < 0.3, otherwise "left_side" or "right_side"

### Ego Lane Estimation

The `EgoLaneEstimator` class determines whether the driver is in the center, leftmost, or rightmost lane using a fused evidence system:

**Evidence sources** (per side):
- Yellow line detection (strong indicator of road edge)
- Solid/dashed line style and edge-support score
- Lane boundary presence/absence
- Adjacent vehicle detection on opposite side (counter-evidence)

**Temporal fusion**:
- Exponential moving average (EMA) accumulation with τ = 1.5 seconds
- Hysteresis thresholds: enter = 0.60, exit = 0.35
- State machine with only adjacent-state transitions (CENTER ↔ RIGHT_LANE ↔ LEFT_LANE)

**Additional stabilization**:
- Streak-based locking: requires consistent evidence over multiple frames
- Edge-lock latches to prevent oscillation
- Side label correction: reclassifies close vehicles when ego is at an edge lane

---

## Lane Change Prediction (BiLSTM)

### Network Architecture

A compact Bidirectional LSTM network with 37,507 trainable parameters:

```
Input (batch, 30, 6)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  nn.LSTM                                             │
│  • input_size = 6      • hidden_size = 32            │
│  • num_layers = 2      • bidirectional = True        │
│  • dropout = 0.4       • batch_first = True          │
└──────────────────────────────────────────────────────┘
       │
       ▼  (take last timestep → 64-dim)
┌──────────────────────────────────────────────────────┐
│  Classifier Head                                     │
│  Linear(64 → 32) → ReLU → Dropout(0.4) → Linear(32 → 3) │
└──────────────────────────────────────────────────────┘
       │
       ▼
Output: 3-class logits → Softmax → Probabilities
```

**Classes:**
| ID | Label | Bounding Box Color |
|----|-------|--------------------|
| 0 | Keep Lane | Green (100, 200, 100) |
| 1 | Lane Change Left | Magenta (255, 0, 255) |
| 2 | Lane Change Right | Orange (0, 165, 255) |

### Feature Engineering

Six features are extracted per vehicle per frame:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `offset_normalized` | Lateral position relative to lane center [-1, +1] |
| 2 | `lat_vel` | Lateral velocity (Δ offset between frames) |
| 3 | `lat_acc` | Lateral acceleration (Δ lateral velocity between frames) |
| 4 | `longitudinal_distance_m` | Calibrated distance to ego vehicle |
| 5 | `long_vel` | Longitudinal velocity (Δ distance between frames) |
| 6 | `bbox_w_norm` | Bounding box width / lane width (size proxy, clipped [0, 2]) |

All features are standardized using a `StandardScaler` fitted during training (`training_data/scaler.pkl`).

### Sliding Window Inference

- Each tracked vehicle maintains a **deque of 30 frames** (the sequence length)
- **Minimum 15 frames** required before prediction begins — below this, returns "Keep Lane" with zero confidence (uncertain)
- **15–29 frames**: Front-padded with the first frame's features to fill the 30-frame window
- **30+ frames**: Full sliding window, oldest frame dropped as new one enters
- Predictions are cached per track and reused when data gaps occur (e.g., vehicle temporarily lost)
- **Confidence threshold**: Predictions with max probability < 0.70 are marked as uncertain

### Model Performance

Trained for 21 epochs (early stopping at epoch 6 out of 50 max) with batch size 64, learning rate 0.0003, and weight decay 0.0001.

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 72.10% |
| **Macro F1** | 0.7218 |
| **Macro Precision** | 0.7205 |
| **Macro Recall** | 0.7244 |

**Per-class performance:**

| Class | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Keep Lane | 0.632 | 0.659 | 0.607 |
| Lane Change Left | 0.768 | 0.747 | 0.790 |
| Lane Change Right | 0.765 | 0.755 | 0.776 |

---

## Lane Change Recommendation Engine

### Lane Availability Assessment

Before any lane change is recommended, the system checks whether the target lane **exists** and is **crossable**:

- **Not crossable**: Yellow line or solid white line detected on that side
- **Lane exists**: Dashed line detected, or vehicle evidence on that side
- **Edge position**: If the ego is at the leftmost/rightmost lane (from `EgoLaneEstimator`), blocks recommendations in that direction
- **Edge-lock latch**: Prevents oscillation when ego position is borderline
- **Adjacent vehicle evidence**: Tracked with hold timers for temporal smoothing

### Decision Logic

The `lane_change_decision()` function implements a multi-stage rule-based evaluation:

1. **Cut-in risk detection**: Distance-aware confidence thresholds detect vehicles cutting into the ego lane from either side. Closer vehicles trigger at lower confidence.

2. **Overtake need assessment**: Triggered when the closest center vehicle satisfies:
   - Time-to-collision (TTC) ≤ 3 seconds, or
   - Distance below safety threshold

3. **Safety evaluation** (per side):
   - No vehicles in the danger zone (< 12m)
   - Sufficient front gap in the target lane (> 45m)
   - Adequate improvement over current lane (> 10m minimum gap improvement)

4. **Forward gap improvement comparison**: Computes the closest vehicle distance in each adjacent lane and compares against the current lane to select the best option.

5. **Symmetric evaluation**: Both left and right are evaluated equally; the side with the best gap improvement wins.

6. **Speed bracket adaptation**: Decision parameters adjust based on estimated driving speed:
   - **Slow** (avg distance < 12m): Tighter thresholds
   - **Medium**: Default parameters
   - **Fast** (avg distance > 25m): More conservative thresholds
   - Requires 10 consistent frames before switching brackets

### Recommendation Stabilization

Raw decisions are smoothed through a multi-layer stabilization system:

- **Confirmation counter**: A new recommendation must persist for several consecutive frames before being adopted
- **Hold timer**: Once active, a recommendation is held for a minimum duration to prevent flickering
- **Immediate STAY override**: Safety-critical situations force an immediate return to STAY
- **Hard gate**: Recommendations toward unavailable lanes are blocked regardless of the decision logic
- **Edge voting** (`is_likely_at_edge`): A 4-signal voting system (2-of-4 required) combining state machine output, edge evidence, traffic asymmetry, and lateral offset history

---

## Bird's-Eye View Visualization

The system generates a real-time top-down bird's-eye view (BEV) alongside the camera feed:

- **Canvas**: 500×1000 pixels with drawn lane markings (solid outer, dashed center)
- **Distance mapping**: Sqrt-scaled (`bev_y = sqrt(distance/max_range)`) so nearby vehicles get more visual space
- **Ego vehicle icon**: Rendered at the bottom center
- **Danger zone**: Red-tinted region at the bottom (closest range)
- **Distance grid**: Horizontal lines at 10m, 20m, 40m, 60m, 80m with labels
- **Vehicle rendering**: 7-layer animated boxes:
  1. Glow effect (for lane-changing vehicles)
  2. Halo outline
  3. Filled rectangle (color-coded by prediction)
  4. Border
  5. Directional arrows (for predicted lane changes)
  6. Exclamation mark (for uncertain predictions)
  7. Text label with distance
- **Lane change animation**: 4-phase blinking cycle for vehicles predicted to change lanes
- **Recommendation highlight**: Yellow/magenta tint on the recommended target lane
- **Blocked lane indicators**: Red overlay with X marker on unavailable lanes

---

## Threading Architecture

| Component | Thread | Mechanism |
|-----------|--------|-----------|
| `frame_reader()` | Thread 1 | Reads frames, applies flip, writes raw recording, drop-oldest queue policy |
| `processor_thread()` | Thread 2 | Runs full CV + ML pipeline, pushes results |
| Main thread | Main | Displays output, writes combined video, handles keyboard |
| `AsyncVideoWriter` | Daemon | Background video encoding with frame-drop counting |
| `BatchCSVWriter` | Inline | Batched CSV writes (flush every 500 rows) |
| `PerformanceMonitor` | Inline | Rolling 30-sample FPS average |

All threads coordinate through a shared `threading.Event` (`SHUTDOWN_FLAG`) for graceful shutdown on 'q' key press or end-of-video.

---

## Project Structure

```
├── Program_Main.py                     # Main application (3247 lines)
│   ├── Configuration constants         #   Camera, YOLO, lane detection, BEV params
│   ├── WindowCapture                   #   Screen capture for sim racing (Assetto Corsa)
│   ├── Lane detection functions        #   Canny, Hough, smoothing, style classification
│   ├── Distance estimation             #   Dual pinhole + lane-width calibration
│   ├── Lateral offset calculation      #   Vehicle position relative to lane
│   ├── EgoLaneEstimator                #   Fused ego position estimator
│   ├── Decision functions              #   Lane change recommendation engine
│   ├── BEV rendering functions         #   Bird's-eye view visualization
│   ├── LaneOffsetProcessor             #   Central pipeline orchestrator
│   ├── Threading infrastructure        #   3-thread producer-consumer pipeline
│   └── CLI argument parser             #   12 command-line options
│
├── lane_change_predictor_BILSTM.py     # BiLSTM prediction module (334 lines)
│   ├── BiLSTM                          #   Neural network definition
│   ├── LaneChangePredictor             #   Real-time inference with sliding windows
│   ├── Feature extraction              #   6 kinematic features per frame
│   └── Label/color definitions         #   Shared constants
│
├── yolov5nu.pt                         # YOLOv5 nano model weights (5.3 MB)
├── models/
│   └── bilstm/
│       ├── bilstm_best.pt              # Best BiLSTM weights (157 KB)
│       ├── bilstm_final.pt             # Final epoch weights (157 KB)
│       ├── confusion_matrix.png        # Test set confusion matrix
│       ├── per_class_analysis.png      # Per-class metrics visualization
│       ├── training_history.png        # Loss/F1 curves over epochs
│       ├── training_history.pkl        # Raw training history data
│       └── training_report.txt         # Full training results
│
├── training_data/                      # (not included — ~113 MB)
│   ├── scaler.pkl                      # StandardScaler for feature normalization
│   ├── train.csv / val.csv / test.csv  # Raw split datasets
│   ├── sequences_*.npz                 # Preprocessed 30-frame sequences
│   ├── master_sequences.npz            # Combined sequence data
│   ├── metadata.pkl                    # Dataset metadata
│   └── prep_*.png                      # Preprocessing visualizations
│
├── run_tests.py                        # FPS benchmark across 5/10/15/30 FPS variants
└── README.md
```

---

## Requirements

- **Python** 3.8+
- **PyTorch** (CPU or CUDA)
- **OpenCV** (`cv2`)
- **Ultralytics** (YOLOv5)
- **NumPy**
- **pandas**
- **scikit-learn** (for StandardScaler deserialization)

Optional:
- **mss** (for screen capture / sim racing mode)
- **CUDA-capable GPU** (for real-time performance with FP16 inference)

```bash
pip install opencv-python torch ultralytics numpy pandas scikit-learn mss
```

---

## Usage

```bash
# Run with default video file
python Program_Main.py

# Run with webcam
python Program_Main.py --source webcam

# Run with specific camera index
python Program_Main.py --source cam:1

# Run with a video file
python Program_Main.py --source "path/to/video.mp4"

# Assetto Corsa screen capture mode
python Program_Main.py --source ac --title "Assetto Corsa" --ac-fps 60
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source`, `-s` | Config path | Input: `webcam`, `cam:N`, video path, or `ac` for screen capture |
| `--title`, `-t` | `"Assetto Corsa"` | Window title for screen capture mode |
| `--ac-fps` | 30 | Target FPS for screen capture |
| `--virtual-index` | 0 | OBS / virtual camera index |
| `--resize` | None | Resize video height (e.g., 720, 480) |
| `--yolo-skip` | 0 | Frames to skip between YOLO inference (0 = every frame) |
| `--no-display` | False | Disable preview window (headless mode) |
| `--bev` | False | Show bird's-eye view only (no camera overlay) |
| `--no-record` | False | Disable output video recording |
| `--flip` | False | Flip camera 180° (for inverted mounts) |
| `--debug` | False | Enable debug-level logging |
| `--debug-decision` | False | Show decision debug overlay on video |

---

## Output Format

All outputs are saved to the `output_lane_offset/` directory with auto-incrementing filenames.

### Detections CSV (per vehicle per frame)

| Field | Description |
|-------|-------------|
| `frame` | Frame index |
| `timestamp` | Elapsed time (seconds) |
| `fps` | Current processing FPS |
| `track_id` | Persistent vehicle tracking ID |
| `class` | YOLO detection class |
| `conf` | Detection confidence |
| `x1, y1, x2, y2` | Bounding box coordinates |
| `cx, cy` | Bounding box center |
| `longitudinal_distance_m` | Calibrated distance to vehicle |
| `lc_predicted_class` | BiLSTM prediction (0/1/2) |
| `lc_predicted_label` | Human-readable prediction |
| `lc_confidence` | Prediction confidence (0–1) |
| `lc_is_uncertain` | Whether prediction is below threshold |
| `lc_frames_buffered` | Frames accumulated in sliding window |
| `bbox_truncated` | Whether bbox is at frame edge |
| `lane_left_x, lane_right_x` | Lane boundary x-coordinates at vehicle y |
| `lane_center_x` | Lane center x-coordinate |
| `lane_width_px` | Lane width in pixels |
| `offset_px` | Lateral offset in pixels |
| `offset_m` | Lateral offset in meters |
| `offset_normalized` | Normalized offset [-1, +1] |
| `position` | Position label (center/left_side/right_side) |

### Decisions CSV (per frame)

Logs the full recommendation engine state including: desired vs. stable decisions, ego position, lane availability flags, edge lock states, lane styles, vehicle counts per zone, gap distances, improvement scores, and stabilization counters.

---

## Training Data

Training data is excluded from this repository due to its size (~113 MB). To use the BiLSTM predictor, place the following file in the `training_data/` directory:

- **`scaler.pkl`** — Required at runtime for feature normalization

The remaining training data files (`*.csv`, `*.npz`, `metadata.pkl`) are only needed if you want to retrain the model.

---

## License

This project is for educational and research purposes.
