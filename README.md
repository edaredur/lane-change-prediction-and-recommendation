# Lane Change Prediction and Recommendation

Real-time lane change prediction system using computer vision and deep learning. The system detects vehicles via YOLOv5, tracks their lane offset behavior, and predicts lane change intentions (Keep Lane, Lane Change Left, Lane Change Right) using a BiLSTM model.

## Features

- **Real-time vehicle detection** using YOLOv5
- **Lane offset tracking** with bird's-eye view (BEV) transformation
- **Lane change prediction** using a BiLSTM deep learning model
- Supports webcam and video file input
- CSV logging of detections and predictions
- Optional preview window with BEV visualization

## Project Structure

```
├── Program_Main.py                     # Main program — lane offset processing & prediction
├── lane_change_predictor_BILSTM.py     # BiLSTM predictor module
├── yolov5nu.pt                         # YOLOv5 nano model weights
├── models/
│   └── bilstm/
│       ├── bilstm_best.pt              # Best BiLSTM model weights
│       ├── bilstm_final.pt             # Final BiLSTM model weights
│       ├── confusion_matrix.png        # Training evaluation
│       ├── per_class_analysis.png
│       ├── training_history.png
│       ├── training_history.pkl
│       └── training_report.txt
├── training_data/                      # (not included — see below)
├── run_tests.py                        # Test runner
└── README.md
```

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- PyTorch
- Ultralytics (YOLOv5)
- NumPy
- pandas

Install dependencies:
```bash
pip install opencv-python torch ultralytics numpy pandas
```

## Usage

```bash
# Run with default video file
python Program_Main.py

# Run with webcam
python Program_Main.py --source webcam

# Run with specific camera
python Program_Main.py --source cam:1

# Run with a video file
python Program_Main.py --source "path/to/video.mp4"
```

### Options

| Flag | Description |
|------|-------------|
| `--source` | Input source: `webcam`, `cam:N`, or path to video |
| `--yolo-skip N` | Skip N frames between YOLO runs (0 = every frame) |
| `--no-display` | Disable preview window |
| `--bev` | Show BEV only in the preview window |
| `--no-record` | Disable output video recording (CSV only) |
| `--resize N` | Resize video height to N pixels |
| `--flip` | Flip camera 180° |
| `--debug` | Enable debug logging |

## Training Data

Training data files are not included in this repository due to their large size (~113 MB). Place the following files in the `training_data/` directory:

- `scaler.pkl` — StandardScaler for feature normalization
- `master_sequences.npz`, `sequences_train.npz`, `sequences_val.npz`, `sequences_test.npz`
- `train.csv`, `val.csv`, `test.csv`

## License

This project is for educational and research purposes.
