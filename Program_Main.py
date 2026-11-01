"""Lane offset processor using webcam or video file input.

CLI usage:
    python "Program_Update.py"
    python "Program_Update.py" --source webcam
    python "Program_Update.py" --source cam:1
    python "Program_Update.py" --source "path/to/video.mp4"

Useful options:
    --yolo-skip N   Skip N frames between YOLO runs (0 = every frame)
    --no-display    Disable preview window
    --bev           Show BEV only in the preview window
    --no-record     Disable output video recording (CSV only)
    --resize N      Resize video height to N (e.g., 720). If not set, keeps original resolution.
    --flip          Flip camera 180 degrees (upside down)
    --debug         Enable debug logging

Output:
    Writes CSV detections and optional preview video into output_lane_offset/.
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import math
from collections import deque
import time
import threading
import queue
import logging
import argparse
from functools import lru_cache


try:
    import torch
except Exception:
    torch = None

from lane_change_predictor_BILSTM import LaneChangePredictor, LABEL_COLORS, LABEL_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
VIDEO_PATH: str = "C:\Main Storage\Downloads\Program dengan Edge Detection - Copy\raw_webcam_006.mp4"
OUTPUT_DIR = "output_lane_offset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_AC_FPS      = 30
VIRTUAL_CAM_INDEX   = 0
TARGET_RESIZE_HEIGHT = None

YOLO_MODEL_PATH = "yolov5n.pt"
YOLO_CONF       = 0.10
YOLO_IOU        = 0.5
DETECT_CLASSES  = ["car", "truck", "bus"]
YOLO_SKIP_FRAMES = 0
YOLO_DEVICE_PREF = "cuda"
YOLO_IMGSZ      = 480
YOLO_HALF       = True

# ── Deteksi bounding box terpotong tepi frame ──────────────────────────────
# Saat kendaraan (mis. bus/truk) mendekat lalu keluar dari sisi kiri/kanan
# frame, x1/x2 dari YOLO sering "terjepit" ke batas frame sehingga titik
# tengah (cx) bergeser cepat secara tidak natural -- bukan karena kendaraan
# pindah lajur, tapi karena sebagian bounding box-nya sudah di luar pandangan
# kamera. TRUNCATED_EDGE_MARGIN_PX/FRAC menentukan seberapa dekat sisi bbox
# ke tepi frame sebelum dianggap "terpotong". TRUNCATED_COOLDOWN_FRAMES
# menahan status tsb selama beberapa frame setelah bbox kembali normal,
# karena posisi bbox butuh waktu untuk stabil kembali.
TRUNCATED_EDGE_MARGIN_PX   = 6
TRUNCATED_EDGE_MARGIN_FRAC = 0.01
TRUNCATED_COOLDOWN_FRAMES  = 10

CANNY_LOW           = 70
CANNY_HIGH          = 180
HOUGH_RHO           = 2
HOUGH_THETA         = np.pi / 180
HOUGH_THRESH        = 80
HOUGH_MIN_LINE_LEN  = 60
HOUGH_MAX_LINE_GAP  = 80

ENABLE_RIGHT_YELLOW_RIGHTMOST = True
YELLOW_HSV_LOWER    = (15, 80, 80)
YELLOW_HSV_UPPER    = (45, 255, 255)
YELLOW_SAMPLE_BAND_PX = 6
YELLOW_MIN_RATIO    = 0.22
YELLOW_SAMPLES      = 10

LANE_WIDTH_METERS   = 3.6
AVG_HISTORY         = 5
MIN_SLOPE           = 0.4
MAX_SLOPE           = 2.5
LANE_DOWNSCALE      = 0.5

CAMERA_HEIGHT_M     = 1.2
CAMERA_PITCH_DEG    = 10
CAMERA_PITCH_RAD    = np.deg2rad(CAMERA_PITCH_DEG)
CAMERA_HFOV_DEG     = 120          # FOV horizontal dashcam (wide-angle) utk kalibrasi jarak

BEV_WIDTH           = 500
BEV_HEIGHT          = 1000
BEV_LANE_WIDTH      = 100
BEV_VEHICLE_WIDTH   = 35
BEV_VEHICLE_HEIGHT  = 60
BEV_DANGER_ZONE     = 150
BEV_BG_COLOR        = (40, 40, 40)
BEV_LANE_LINE_COLOR = (200, 200, 200)
BEV_OBSERVER_COLOR  = (0, 255, 255)

# Optional: replace BEV "DANGER ZONE" text + observer triangle with an icon.
# Put `car.png` next to this file (PredictionRecommendation/car.png) or change the path below.
BEV_CAR_ICON_PATH   = os.path.join(os.path.dirname(__file__), "car.png")
BEV_CAR_ICON_SIZE   = (70, 70)  # (width, height) in pixels

CSV_BATCH_SIZE      = 500
WRITE_QUEUE_SIZE    = 30
DISPLAY_ENABLED     = True
DISPLAY_BEV_ONLY    = False
RECORDING_ENABLED   = True
FLIP_CAMERA         = False

# Decision debug overlay (shows why LEFT/RIGHT/STAY was chosen)
DEBUG_DECISION      = False

FILTER_ONCOMING             = True
ONCOMING_CLOSING_SPEED_MPS  = 14.0
ONCOMING_LATERAL_NORM_MIN   = 0.7
TRACK_MATCH_MAX_PX          = 80
TRACK_STALE_SEC             = 0.8

LANE_WIDTH_MIN_PX   = 40
LANE_WIDTH_MAX_FRAC = 0.9

FRAME_QUEUE   = queue.Queue(maxsize=1)
RESULT_QUEUE  = queue.Queue(maxsize=5)
SHUTDOWN_FLAG = threading.Event()


def normalize_source(source_value):
    """Normalize CLI input into canonical webcam/video source format.

    Mirrors the behavior in `Lane_Change.py`:
    - "webcam"/"cam"/"camera" -> "cam:0"
    - "cam:1" (or webcam:/camera:) -> "cam:1"
    - "0" -> "cam:0"
    - otherwise treat as a file path / URL
    """
    source = str(source_value).strip()
    source_lower = source.lower()

    if source_lower in ("webcam", "cam", "camera"):
        return "cam:0"

    if source_lower.startswith(("cam:", "webcam:", "camera:")):
        _, idx_str = source.split(":", 1)
        cam_index = int(idx_str) if idx_str.strip().isdigit() else 0
        return f"cam:{cam_index}"

    if source_lower.isdigit():
        return f"cam:{int(source_lower)}"

    return source


# ============ DEVICE ============

def _resolve_yolo_device():
    if torch is None:
        return None
    if YOLO_DEVICE_PREF.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _maybe_enable_cudnn_benchmark():
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# ============ WINDOW CAPTURE ============

try:
    import mss
except Exception:
    mss = None

_MSS_THREAD_LOCAL = threading.local()

try:
    import win32gui
    import win32con
except Exception:
    win32gui = None
    win32con = None

def find_window(title_substring):
    if win32gui is None:
        return None
    target = title_substring.lower()
    found  = None
    def _callback(hwnd, _):
        nonlocal found
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            text = win32gui.GetWindowText(hwnd)
        except Exception:
            text = ""
        if text and target in text.lower():
            found = hwnd
            return False
        return True
    win32gui.EnumWindows(_callback, None)
    return found

def get_client_rect(hwnd):
    if win32gui is None:
        return 0, 0, 0, 0
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    tl = win32gui.ClientToScreen(hwnd, (left, top))
    br = win32gui.ClientToScreen(hwnd, (right, bottom))
    return tl[0], tl[1], br[0], br[1]

class WindowCapture:
    def __init__(self, title_substring='Assetto Corsa', fps=30):
        self.title  = title_substring
        self.fps    = float(fps)
        self.hwnd   = find_window(self.title) if win32gui is not None else None
        self.opened = self.hwnd is not None
        self.sct    = None
        self._last_bbox = None
        self._delay = 1.0 / max(1.0, self.fps)
        self._width = None
        self._height = None
        self._last_capture_time = None
        if self.opened:
            l, t, r, b = get_client_rect(self.hwnd)
            self._width  = max(1, r - l)
            self._height = max(1, b - t)
            self._last_bbox = {'left': l, 'top': t, 'width': self._width, 'height': self._height}

    def isOpened(self):
        return bool(self.opened)

    def read(self):
        if not self.isOpened():
            return False, None
        if self._last_capture_time is not None:
            elapsed = time.time() - self._last_capture_time
            wait    = self._delay - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_capture_time = time.time()
        try:
            l, t, r, b = get_client_rect(self.hwnd)
            self._last_bbox = {'left': l, 'top': t, 'width': max(1,r-l), 'height': max(1,b-t)}
            try:
                _sct = getattr(_MSS_THREAD_LOCAL, 'sct')
            except Exception:
                _sct = None
            if _sct is None:
                if mss is None:
                    return False, None
                _sct = mss.mss()
                setattr(_MSS_THREAD_LOCAL, 'sct', _sct)
            sct_img = _sct.grab(self._last_bbox)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return True, img
        except Exception:
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:           return self.fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:   return float(self._width or 0)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:  return float(self._height or 0)
        if prop == cv2.CAP_PROP_FRAME_COUNT:   return 0.0
        return 0.0

    def set(self, prop, value):
        return True

    def release(self):
        try:
            if self.sct:
                self.sct.close()
        except Exception:
            pass


# ============ UTILITY ============

def get_next_filename(base_path, extension):
    directory = os.path.dirname(base_path)
    basename  = os.path.splitext(os.path.basename(base_path))[0]
    counter   = 1
    while True:
        path = os.path.join(directory, f"{basename}{extension}" if counter == 1 else f"{basename}_{counter:03d}{extension}")
        if not os.path.exists(path):
            return path
        counter += 1

def test_video_codec(codec_fourcc, fps, frame_size, test_path):
    try:
        writer = cv2.VideoWriter(test_path, codec_fourcc, fps, frame_size)
        if not writer.isOpened():
            writer.release()
            return False
        writer.write(np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8))
        writer.release()
        if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
            os.remove(test_path)
            return True
        return False
    except Exception:
        return False


# ============ BEV FUNCTIONS ============

@lru_cache(maxsize=8)
def _load_icon_bgra(path: str, width: int, height: int):
    if not path or not os.path.exists(path):
        return None
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim != 3 or icon.shape[2] not in (3, 4):
        return None
    if icon.shape[2] == 3:
        icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
        icon[:, :, 3] = 255
    if width > 0 and height > 0:
        icon = cv2.resize(icon, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return icon


def _overlay_bgra(dst_bgr: np.ndarray, icon_bgra: np.ndarray, x: int, y: int):
    if dst_bgr is None or icon_bgra is None:
        return
    h, w = icon_bgra.shape[:2]
    if h <= 0 or w <= 0:
        return

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(dst_bgr.shape[1], int(x) + w)
    y2 = min(dst_bgr.shape[0], int(y) + h)
    if x1 >= x2 or y1 >= y2:
        return

    ix1 = x1 - int(x)
    iy1 = y1 - int(y)
    ix2 = ix1 + (x2 - x1)
    iy2 = iy1 + (y2 - y1)

    roi = dst_bgr[y1:y2, x1:x2]
    icon = icon_bgra[iy1:iy2, ix1:ix2]

    alpha = (icon[:, :, 3:4].astype(np.float32) / 255.0)
    roi[:] = (alpha * icon[:, :, :3].astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)

BEV_MAX_RANGE_M = 80.0   # jangkauan jarak yang ditampilkan BEV (penuh tinggi panel)

def bev_y_from_distance(z, max_range=BEV_MAX_RANGE_M):
    """Petakan jarak (m) -> baris-y BEV dengan skala AKAR (near diberi ruang lebih
    banyak) supaya kendaraan dekat tidak menumpuk di dasar panel.
    z=0 -> dasar (y=H), z=max_range -> atas (y=0)."""
    z = max(0.0, min(float(z), float(max_range)))
    frac = (z / float(max_range)) ** 0.5
    return int(round(BEV_HEIGHT * (1.0 - frac)))


def create_birdseye_view():
    bev      = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    bev[:]   = BEV_BG_COLOR
    cx       = BEV_WIDTH // 2
    lx       = cx - BEV_LANE_WIDTH
    rx       = cx + BEV_LANE_WIDTH
    cv2.line(bev, (lx, 0), (lx, BEV_HEIGHT), BEV_LANE_LINE_COLOR, 1)
    cv2.line(bev, (rx, 0), (rx, BEV_HEIGHT), BEV_LANE_LINE_COLOR, 1)
    y = 0
    while y < BEV_HEIGHT:
        cv2.line(bev, (cx, y), (cx, min(y+30, BEV_HEIGHT)), BEV_LANE_LINE_COLOR, 1)
        y += 50
    danger_y = BEV_HEIGHT - BEV_DANGER_ZONE
    ov = bev.copy()
    cv2.rectangle(ov, (lx, danger_y), (rx, BEV_HEIGHT), (0, 0, 100), -1)
    cv2.addWeighted(ov, 0.3, bev, 0.7, 0, bev)

    car_icon = _load_icon_bgra(BEV_CAR_ICON_PATH, BEV_CAR_ICON_SIZE[0], BEV_CAR_ICON_SIZE[1])
    if car_icon is None:
        cv2.putText(bev, "DANGER ZONE", (cx-70, danger_y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    cv2.putText(bev, "LEFT",     (cx-BEV_LANE_WIDTH-70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    cv2.putText(bev, "OUR LANE", (cx-45, 30),                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(bev, "RIGHT",    (cx+BEV_LANE_WIDTH+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    if car_icon is None:
        oy = BEV_HEIGHT - 30
        cv2.fillPoly(bev, [np.array([[cx,oy],[cx-25,oy+30],[cx+25,oy+30]], np.int32)], BEV_OBSERVER_COLOR)
        cv2.putText(bev, "CAMERA", (cx-35, oy+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BEV_OBSERVER_COLOR, 1)
    else:
        ih, iw = car_icon.shape[:2]
        icon_x = int(cx - (iw / 2))
        icon_y = int(BEV_HEIGHT - ih - 8)
        _overlay_bgra(bev, car_icon, icon_x, icon_y)
    for dist in [5, 10, 20, 30, 45, 60, 80]:
        my = bev_y_from_distance(dist)
        if 0 <= my < BEV_HEIGHT:
            cv2.line(bev, (0, my), (BEV_WIDTH, my), (100,100,100), 1)
            cv2.putText(bev, f"{dist}m", (5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    return bev


def apply_bev_recommendation_highlight(bev, decision):
    if bev is None or bev.size == 0 or decision not in ("LEFT", "RIGHT"):
        return
    cx = BEV_WIDTH // 2
    lx = cx - BEV_LANE_WIDTH
    rx = cx + BEV_LANE_WIDTH
    ov = bev.copy()
    if decision == "RIGHT":
        cv2.rectangle(ov, (rx, 0), (BEV_WIDTH-1, BEV_HEIGHT-1), (0,220,220), -1)
        cv2.addWeighted(ov, 0.35, bev, 0.65, 0, bev)
        cv2.line(bev, (rx, 0), (rx, BEV_HEIGHT), BEV_LANE_LINE_COLOR, 1)
    else:
        cv2.rectangle(ov, (0, 0), (lx, BEV_HEIGHT-1), (220,0,220), -1)
        cv2.addWeighted(ov, 0.25, bev, 0.75, 0, bev)
        cv2.line(bev, (lx, 0), (lx, BEV_HEIGHT), BEV_LANE_LINE_COLOR, 1)


def draw_bev_lane_state(bev, ego_lane_index, left_available, right_available, decision):
    """Gambar posisi EGO di antara 3 lajur BEV + tandai sisi terblok jadi MERAH.

    - Lajur ego: tint hijau + marker EGO (segitiga kuning) digeser ke lajurnya.
    - Sisi tidak tersedia (median/garis solid/tepi): tint MERAH + 'X'.
    - Sisi yang direkomendasikan: panah hijau terang.
    Visual ini konsisten dengan logika rekomendasi (sisi merah = RIGHT/LEFT diblok).
    """
    if bev is None or bev.size == 0:
        return
    cx = BEV_WIDTH // 2
    lx = cx - BEV_LANE_WIDTH
    rx = cx + BEV_LANE_WIDTH
    H  = BEV_HEIGHT
    idx = ego_lane_index if ego_lane_index in (0, 1, 2) else 1
    RED = (0, 0, 200)

    # Pewarnaan MURNI dari posisi lajur ego:
    #   ego di paling KANAN (idx 2) -> hanya sisi KANAN yang tak ada -> merah.
    #   ego di paling KIRI  (idx 0) -> hanya sisi KIRI  yang tak ada -> merah.
    #   ego di TENGAH (idx 1)       -> tidak ada yang diblok.
    block_left  = (idx == 0)
    block_right = (idx == 2)

    if block_left:
        cv2.rectangle(bev, (0, 0), (lx, H - 1), RED, -1)
    if block_right:
        cv2.rectangle(bev, (rx, 0), (BEV_WIDTH - 1, H - 1), RED, -1)

    if block_left:
        cv2.putText(bev, "X", (lx // 2 - 16, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(bev, "NO LANE", (10, H // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if block_right:
        cv2.putText(bev, "X", ((rx + BEV_WIDTH) // 2 - 16, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(bev, "NO LANE", (rx + 12, H // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # (Marker ego TIDAK digambar di sini -- ikon mobil ego sudah ada di template
    #  BEV bawah-tengah. Hindari marker dobel/tumpuk di area danger zone.)



def map_to_birdseye(cx, cy, offset_normalized, frame_height, position=None):
    center_x     = BEV_WIDTH // 2
    OFFSET_LEFT  = -170
    OFFSET_RIGHT =  170
    if position == 'left_side':
        bev_x = center_x + OFFSET_LEFT
    elif position == 'right_side':
        bev_x = center_x + OFFSET_RIGHT
    else:
        bev_x = center_x
    cy_norm      = cy / frame_height
    depth_factor = cy_norm ** 1.5
    bev_y        = int(depth_factor * (BEV_HEIGHT - 100))
    return bev_x, bev_y


def draw_vehicle_on_birdseye(bev, bev_x, bev_y, position, in_danger_zone=False,
                              lc_class=None, lc_confidence=0.0, lc_is_uncertain=True,
                              blink_visible=True, blink_phase=0, track_id=None):
    """
    Gambar kendaraan di BEV dengan indikator lane change berkedip SUPER HEBOH.

    lc_class       : int|None -- kelas prediksi LSTM (0/1/2)
    lc_confidence  : float    -- confidence prediksi
    lc_is_uncertain: bool     -- True jika confidence di bawah threshold predictor
    blink_visible  : bool     -- True = fase nyala kedip, False = fase gelap
    blink_phase    : int      -- phase 0-3 untuk animasi multi-layer (warna & glow bergantian)
    track_id       : int|None -- untuk label di BEV
    """
    if position == 'center':
        base_color = (0, 255, 0)
    elif position == 'right_side':
        base_color = (0, 165, 255)
    else:
        base_color = (255, 0, 255)

    if in_danger_zone:
        base_color = tuple(int(c * 0.7) for c in base_color)
        base_color = (base_color[0], base_color[1], min(base_color[2]+100, 255))

    # Kendaraan diprediksi lane change jika kelas bukan Lurus, tidak uncertain,
    # dan confidence cukup. Cek is_uncertain agar konsisten dengan label teks.
    is_lane_changing = (lc_class is not None and lc_class != 0
                        and not lc_is_uncertain and lc_confidence >= 0.70)

    body_color = (255,0,255) if (is_lane_changing and lc_class==1) else \
                 (0,165,255) if (is_lane_changing and lc_class==2) else base_color

    hw = BEV_VEHICLE_WIDTH  // 2
    hh = BEV_VEHICLE_HEIGHT // 2
    x1 = max(0, min(bev_x-hw, BEV_WIDTH-1))
    x2 = max(0, min(bev_x+hw, BEV_WIDTH-1))
    y1 = max(0, min(bev_y-hh, BEV_HEIGHT-1))
    y2 = max(0, min(bev_y+hh, BEV_HEIGHT-1))
    m  = 2

    if is_lane_changing:
        # ── BLINK ENGINE: pakai time.time() agar kedip PASTI real-time ──
        # Frekuensi 4 Hz: nyala 0.13 s, gelap 0.12 s per siklus 0.25 s
        t_now   = time.time()
        nyala   = (t_now % 0.25) < 0.13
        # Fase warna berganti tiap 0.125 s (8 Hz visual variety)
        t_phase = int(t_now / 0.125) % 4

        acx, acy = (x1+x2)//2, (y1+y2)//2

        phase_colors = [
            body_color,
            (255, 255, 255),
            tuple(min(int(c * 1.5), 255) for c in body_color),
            (0, 255, 255) if lc_class == 1 else (255, 255, 0),
        ]
        flash_color  = phase_colors[t_phase]
        border_thick = [4, 7, 4, 7][t_phase]
        glow_pad     = [10, 16, 8, 18][t_phase]

        if nyala:
            # ── LAYER 1: outer glow alpha-blend ──
            gx1 = max(0, x1 - glow_pad); gy1 = max(0, y1 - glow_pad)
            gx2 = min(BEV_WIDTH-1, x2 + glow_pad); gy2 = min(BEV_HEIGHT-1, y2 + glow_pad)
            glow_layer = bev.copy()
            cv2.rectangle(glow_layer, (gx1, gy1), (gx2, gy2),
                          tuple(int(c*0.55) for c in flash_color), -1)
            cv2.addWeighted(glow_layer, 0.50, bev, 0.50, 0, bev)
            cv2.rectangle(bev, (gx1, gy1), (gx2, gy2), flash_color, 2)

            # ── LAYER 2: halo tengah ──
            mid = glow_pad // 2
            cv2.rectangle(bev,
                          (max(0,x1-mid), max(0,y1-mid)),
                          (min(BEV_WIDTH-1,x2+mid), min(BEV_HEIGHT-1,y2+mid)),
                          flash_color, 1)

            # ── LAYER 3: fill body ──
            cv2.rectangle(bev, (x1+m, y1+m), (x2-m, y2-m), flash_color, -1)

            # ── LAYER 4: border dalam tebal ──
            cv2.rectangle(bev, (x1, y1), (x2, y2), (255, 255, 255), border_thick)

            # ── LAYER 5: dua panah arah besar ──
            aLen = 20
            if lc_class == 1:
                cv2.arrowedLine(bev, (acx+aLen, acy),    (acx-aLen, acy),     (255,255,255), 3, tipLength=0.45)
                cv2.arrowedLine(bev, (acx+aLen+4,acy-7), (acx-aLen+4,acy-7),  flash_color,  2, tipLength=0.40)
            else:
                cv2.arrowedLine(bev, (acx-aLen, acy),    (acx+aLen, acy),     (255,255,255), 3, tipLength=0.45)
                cv2.arrowedLine(bev, (acx-aLen-4,acy-7), (acx+aLen-4,acy-7),  flash_color,  2, tipLength=0.40)

            # ── LAYER 6: tanda seru merah ──
            cv2.putText(bev, "!", (max(2, x1-3), max(14, y1-2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # ── LAYER 7: label dengan latar flash ──
            lbl = f"{'<< KIRI' if lc_class==1 else 'KANAN >>'} {lc_confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            lx_lbl = max(0, x1)
            ly_lbl = max(th+4, y1-5)
            cv2.rectangle(bev, (lx_lbl, ly_lbl-th-2), (lx_lbl+tw+6, ly_lbl+3),
                          tuple(int(c*0.75) for c in flash_color), -1)
            cv2.rectangle(bev, (lx_lbl, ly_lbl-th-2), (lx_lbl+tw+6, ly_lbl+3), (255,255,255), 1)
            cv2.putText(bev, lbl, (lx_lbl+3, ly_lbl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        else:
            # ── FASE GELAP: outline + panah redup, tidak hilang total ──
            cv2.rectangle(bev, (x1+m, y1+m), (x2-m, y2-m),
                          tuple(max(0, int(c*0.18)) for c in body_color), -1)
            cv2.rectangle(bev, (x1, y1), (x2, y2), body_color, 2)
            if lc_class == 1:
                cv2.arrowedLine(bev, (acx+10, acy), (acx-10, acy),
                                tuple(int(c*0.45) for c in body_color), 1, tipLength=0.5)
            else:
                cv2.arrowedLine(bev, (acx-10, acy), (acx+10, acy),
                                tuple(int(c*0.45) for c in body_color), 1, tipLength=0.5)

    else:
        # Kendaraan biasa -- tidak berubah dari aslinya
        cv2.rectangle(bev, (x1+m, y1+m), (x2-m, y2-m), body_color, -1)
        wsc = (255,255,255) if in_danger_zone else (80,80,80)
        cv2.rectangle(bev, (x1,y1), (x2,y2), wsc, 1)
        # Kaca depan
        wh = int((y2-y1)*0.25)
        cv2.rectangle(bev, (x1+m+1, y1+m+1), (x2-m-1, y1+wh), tuple(int(c*0.5) for c in body_color), -1)
        # Spion
        cv2.rectangle(bev, (x1-1, bev_y-2), (x1+1, bev_y+2), (200,200,200), -1)
        cv2.rectangle(bev, (x2-1, bev_y-2), (x2+1, bev_y+2), (200,200,200), -1)

    # Track ID
    if track_id is not None:
        idt = f"ID:{track_id}"
        (tw2,th2),_ = cv2.getTextSize(idt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(bev, idt, (max(0,x1), min(BEV_HEIGHT-2, y2+th2+2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)


# ============ LANE DETECTION ============

def is_bbox_truncated(x1, x2, frame_width,
                       margin_px=TRUNCATED_EDGE_MARGIN_PX,
                       margin_frac=TRUNCATED_EDGE_MARGIN_FRAC):
    """True bila salah satu sisi bounding box menempel/terjepit ke tepi frame.

    Dipakai untuk menandai kendaraan yang sedang keluar dari sisi kiri/kanan
    pandangan kamera (misal saat disusul dari dekat). Dalam kondisi ini,
    x1 (atau x2) YOLO biasanya "terjepit" ke 0 atau (frame_width-1), sehingga
    cx = (x1+x2)/2 tidak lagi mewakili posisi tengah kendaraan sebenarnya --
    melainkan bergeser mengikuti bagian bbox yang masih terlihat. Ini SERING
    disalahartikan sebagai gerakan lateral (lane change) oleh predictor,
    padahal murni efek geometris kendaraan meninggalkan frame.
    """
    if frame_width is None or frame_width <= 0:
        return False
    margin = max(float(margin_px), float(frame_width) * float(margin_frac))
    return bool(float(x1) <= margin or float(x2) >= (float(frame_width) - margin))


def is_valid_slope(slope):
    return MIN_SLOPE < abs(slope) < MAX_SLOPE

def average_lane_lines(lines, frame_h, frame_w=None, side=None):
    if len(lines) == 0:
        return None
    candidates = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2-y1)/(x2-x1)
        if not is_valid_slope(slope):
            continue
        intercept = y1 - slope*x1
        try:
            xb = (frame_h - intercept) / slope
        except Exception:
            continue
        if not np.isfinite(xb):
            continue
        candidates.append((float(slope), float(intercept), float(xb)))
    if not candidates:
        return None
    slope_sel     = float(np.mean([s for s,_,_ in candidates]))
    intercept_sel = float(np.mean([b for _,b,_ in candidates]))
    if frame_w is not None and side in ("left","right"):
        cx = float(frame_w)/2.0
        sc = [c for c in candidates if (c[2]<cx if side=="left" else c[2]>cx)]
        if sc:
            slope_sel, intercept_sel, _ = min(sc, key=lambda c: abs(c[2]-cx))
    y1o = frame_h
    y2o = int(frame_h*0.55)
    if abs(slope_sel) < 1e-6:
        return None
    return (int((y1o-intercept_sel)/slope_sel), y1o,
            int((y2o-intercept_sel)/slope_sel), y2o)

def get_x_at_y(line, y):
    if line is None:
        return None
    x1, y1, x2, y2 = line
    if abs(y2-y1) < 0.1:
        return int((x1+x2)/2)
    t = (y-y1)/(y2-y1)
    x = x1 + t*(x2-x1)
    if x < -1000 or x > 3000:
        return None
    return int(x)

def calculate_longitudinal_distance(cy, frame_height,
                                    camera_height=CAMERA_HEIGHT_M,
                                    pitch_rad=CAMERA_PITCH_RAD):
    y_norm = cy / frame_height
    fov_rad = np.deg2rad(60)
    angle   = pitch_rad + (y_norm-0.5)*fov_rad
    if angle <= 0:
        return 200.0
    return float(np.clip(camera_height/np.tan(angle), 0.5, 200.0))


def focal_px_from_hfov(img_width, hfov_deg=CAMERA_HFOV_DEG):
    """Focal length (pixel) dari FOV horizontal: f = (W/2)/tan(HFOV/2)."""
    return (float(img_width) / 2.0) / math.tan(math.radians(float(hfov_deg) / 2.0))


def distance_from_lane_width(cy, left_line, right_line, focal_px, lane_width_m=LANE_WIDTH_METERS):
    """Jarak longitudinal via 'penggaris lajur'.

    Untuk jalan datar + pinhole, lebar nyata lajur (lane_width_m) yang terlihat
    selebar w_px pixel di baris-y kendaraan -> jaraknya  Z = f * W_real / w_px.
    Memakai garis ego-lane yang diekstrapolasi ke baris roda kendaraan (cy).
    Mengembalikan None bila lajur tak tersedia / lebar tak masuk akal (dekat
    titik-lenyap). Ini menjadikan lajur sbg acuan skala -> tak perlu kalibrasi
    lapangan; otomatis menyesuaikan geometri kamera/jalan nyata per frame.
    """
    if left_line is None or right_line is None or not focal_px:
        return None
    xl = get_x_at_y(left_line, cy)
    xr = get_x_at_y(right_line, cy)
    if xl is None or xr is None:
        return None
    w_px = abs(float(xr) - float(xl))
    if w_px < 3.0:
        return None
    z = float(focal_px) * float(lane_width_m) / w_px
    return float(np.clip(z, 0.5, 200.0))

def calculate_lateral_offset(cx, cy, left_line, right_line):
    if left_line is None or right_line is None:
        return None
    lx = get_x_at_y(left_line, cy)
    rx = get_x_at_y(right_line, cy)
    if lx is None or rx is None:
        return None
    if rx < lx:
        lx, rx = rx, lx
    lw = rx - lx
    if lw < 40 or lw > 1920*0.8:
        return None
    lcx = (lx+rx)/2
    opx = cx - lcx
    om  = (opx/lw)*LANE_WIDTH_METERS
    onm = opx/(lw/2)
    pos = 'center' if abs(onm) < 0.3 else ('right_side' if onm > 0 else 'left_side')
    return {
        'lane_left_x'            : int(lx),
        'lane_right_x'           : int(rx),
        'lane_center_x'          : int(lcx),
        'lane_width_px'          : int(lw),
        'offset_px'              : int(opx),
        'offset_m'               : round(om, 3),
        'offset_normalized'      : round(onm, 3),
        'position'               : pos,
        'longitudinal_distance_m': round(calculate_longitudinal_distance(cy, 720), 2),
    }

def smooth_line(history):
    if not history:
        return None
    return tuple(np.mean(history, axis=0).astype(int))


def fit_line_from_segments(segments, frame_h, y_top_frac=0.55):
    """Fit a single line (x1,y1,x2,y2) from multiple Hough segments.

    Unlike average_lane_lines(..., side=...), this does not bias toward the image center.
    """
    if not segments:
        return None
    candidates = []
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not is_valid_slope(slope):
            continue
        intercept = y1 - slope * x1
        candidates.append((float(slope), float(intercept)))
    if not candidates:
        return None
    slope_sel = float(np.mean([s for s, _ in candidates]))
    intercept_sel = float(np.mean([b for _, b in candidates]))
    if abs(slope_sel) < 1e-6:
        return None
    y1o = int(frame_h)
    y2o = int(frame_h * float(y_top_frac))
    x1o = int((y1o - intercept_sel) / slope_sel)
    x2o = int((y2o - intercept_sel) / slope_sel)
    return (x1o, y1o, x2o, y2o)


def line_edge_support_stats(line, edge_img, y_start_frac=0.60, y_end_frac=0.95, band_px=3, step_px=6):
    """Measure how well a fitted line is supported by Canny edges.

    Returns coverage ratio and maximum consecutive gap ratio.
    Solid lines tend to have higher coverage and shorter max gaps.
    """
    if line is None or edge_img is None:
        return None
    h, w = edge_img.shape[:2]
    ys = int(max(0, min(h - 1, h * float(y_start_frac))))
    ye = int(max(0, min(h - 1, h * float(y_end_frac))))
    if ye <= ys:
        return None

    hit = []
    for y in range(ys, ye + 1, max(1, int(step_px))):
        x = get_x_at_y(line, y)
        if x is None:
            hit.append(0)
            continue
        x = int(x)
        b = int(max(1, band_px))
        x1 = max(0, x - b)
        x2 = min(w, x + b + 1)
        y1 = max(0, y - b)
        y2 = min(h, y + b + 1)
        patch = edge_img[y1:y2, x1:x2]
        hit.append(1 if patch.size > 0 and np.any(patch > 0) else 0)

    n = len(hit)
    if n == 0:
        return None
    coverage = float(sum(hit)) / float(n)

    # max consecutive zeros
    max_gap = 0
    cur = 0
    for v in hit:
        if v == 0:
            cur += 1
            if cur > max_gap:
                max_gap = cur
        else:
            cur = 0
    max_gap_ratio = float(max_gap) / float(n)

    return {
        "coverage": coverage,
        "max_gap_ratio": max_gap_ratio,
        "samples": n,
    }


def classify_line_style_from_edges(line, edge_img):
    """Classify a line as solid/dashed/unknown using edge support stats."""
    st = line_edge_support_stats(line, edge_img)
    if not st:
        return {"style": "unknown", "score": None, **(st or {})}
    cov = float(st["coverage"])
    gap = float(st["max_gap_ratio"])
    # Score favors continuous lines.
    score = cov - 0.7 * gap

    # Heuristics: tuneable, but robust enough for outer-boundary inference.
    if cov >= 0.62 and gap <= 0.22:
        style = "solid"
    elif cov <= 0.55 and gap >= 0.28:
        style = "dashed"
    else:
        style = "unknown"
    return {"style": style, "score": float(score), **st}


def select_outer_boundary_from_segments(segments, side, yref, edge_img, bin_px=50):
    """Select the most likely outer boundary line on a given side.

    We bin Hough segments by x position at yref, fit a line per bin, then pick among the
    outermost bins the line with the best 'solid' support score.
    """
    if not segments:
        return None

    # Bucket segments by x_at_yref
    buckets = {}
    for s in segments:
        x = get_x_at_y(s, int(yref))
        if x is None:
            continue
        b = int(x // max(1, int(bin_px)))
        buckets.setdefault(b, []).append(s)
    if not buckets:
        return None

    # Sort bins by their mean x (outermost first)
    bin_items = []
    for b, segs in buckets.items():
        xs = [get_x_at_y(seg, int(yref)) for seg in segs]
        xs = [x for x in xs if x is not None]
        if not xs:
            continue
        bin_items.append((float(np.mean(xs)), b, segs))
    if not bin_items:
        return None
    reverse = True if str(side) == "right" else False
    bin_items.sort(key=lambda t: t[0], reverse=reverse)

    # Evaluate top few outer bins
    best = None
    for mean_x, b, segs in bin_items[:4]:
        line = fit_line_from_segments(segs, frame_h=edge_img.shape[0])
        if line is None:
            continue
        st = classify_line_style_from_edges(line, edge_img)
        x_line = get_x_at_y(line, int(yref))
        cand = {
            "line": line,
            "x": x_line,
            "mean_x": mean_x,
            **st,
            "n_segs": int(len(segs)),
        }
        if best is None:
            best = cand
            continue
        # Prefer solid style, then higher score, then outermost
        best_solid = (best.get("style") == "solid")
        cand_solid = (cand.get("style") == "solid")
        if cand_solid and not best_solid:
            best = cand
        elif cand_solid == best_solid:
            if (cand.get("score") or -1e9) > (best.get("score") or -1e9):
                best = cand
    return best

def detect_yellow_on_right_lane(frame_bgr, right_line, frame_height):
    if not ENABLE_RIGHT_YELLOW_RIGHTMOST or frame_bgr is None or right_line is None:
        return False
    x1, y1, x2, y2 = right_line
    if y2 == y1:
        return False
    h, w = frame_bgr.shape[:2]
    ys = int(max(0, min(h-1, frame_height*0.60)))
    ye = int(max(0, min(h-1, frame_height*0.95)))
    if ye <= ys:
        return False
    total, yellow = 0, 0
    for y in np.linspace(ys, ye, num=YELLOW_SAMPLES):
        y = int(y)
        t = (y-y1)/(y2-y1)
        x = int(x1 + t*(x2-x1))
        b = int(YELLOW_SAMPLE_BAND_PX)
        patch = frame_bgr[max(0,y-b):min(h,y+b+1), max(0,x-b):min(w,x+b+1)]
        if patch.size == 0:
            continue
        hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Strict + relaxed mask to handle faded yellow
        mask_strict = cv2.inRange(hsv, np.array(YELLOW_HSV_LOWER, np.uint8), np.array(YELLOW_HSV_UPPER, np.uint8))
        low_relaxed = (int(YELLOW_HSV_LOWER[0]), 40, 60)
        high_relaxed = (int(YELLOW_HSV_UPPER[0]), 255, 255)
        mask_relaxed = cv2.inRange(hsv, np.array(low_relaxed, np.uint8), np.array(high_relaxed, np.uint8))
        mask = cv2.bitwise_or(mask_strict, mask_relaxed)
        total  += mask.size
        yellow += cv2.countNonZero(mask)
    return (yellow/total >= YELLOW_MIN_RATIO) if total > 0 else False


def detect_yellow_on_edge(frame_bgr, side="right", frame_height=None):
    """Detect yellow markings near the image edge.

    Helps infer rightmost/leftmost lane when the chosen lane boundary line is an inner dashed line,
    but an outer solid (often yellow) shoulder/median marking is visible.
    """
    if frame_bgr is None:
        return False
    h, w = frame_bgr.shape[:2]
    if frame_height is None:
        frame_height = h

    ys = int(max(0, min(h-1, frame_height*0.60)))
    ye = int(max(0, min(h-1, frame_height*0.95)))
    if ye <= ys:
        return False

    strip_w = max(6, int(0.06*w))
    if str(side).lower().startswith("r"):
        x1, x2 = max(0, w-strip_w), w
    else:
        x1, x2 = 0, min(w, strip_w)

    roi = frame_bgr[ys:ye, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_count = int(cv2.countNonZero(edges))
    if edge_count < 120:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_strict = cv2.inRange(hsv, np.array(YELLOW_HSV_LOWER, np.uint8), np.array(YELLOW_HSV_UPPER, np.uint8))
    low_relaxed = (int(YELLOW_HSV_LOWER[0]), 40, 60)
    high_relaxed = (int(YELLOW_HSV_UPPER[0]), 255, 255)
    mask_relaxed = cv2.inRange(hsv, np.array(low_relaxed, np.uint8), np.array(high_relaxed, np.uint8))
    mask = cv2.bitwise_or(mask_strict, mask_relaxed)

    edge_mask = edges > 0
    yellow_on_edges = int(np.count_nonzero((mask > 0) & edge_mask))
    score = float(yellow_on_edges) / float(edge_count) if edge_count > 0 else 0.0
    return score >= 0.20


def detect_white_on_edge(frame_bgr, side="right", frame_height=None):
    """Detect white lane/shoulder markings near the image edge.

    Uses HSV whiteness (low S, high V) gated by Canny edges to avoid bright non-line regions.
    """
    if frame_bgr is None:
        return False
    h, w = frame_bgr.shape[:2]
    if frame_height is None:
        frame_height = h

    ys = int(max(0, min(h-1, frame_height*0.60)))
    ye = int(max(0, min(h-1, frame_height*0.95)))
    if ye <= ys:
        return False

    strip_w = max(6, int(0.06*w))
    if str(side).lower().startswith("r"):
        x1, x2 = max(0, w-strip_w), w
    else:
        x1, x2 = 0, min(w, strip_w)

    roi = frame_bgr[ys:ye, x1:x2]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_count = int(cv2.countNonZero(edges))
    if edge_count < 120:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    white = (s <= 60) & (v >= 180)
    edge_mask = edges > 0
    white_on_edges = int(np.count_nonzero(white & edge_mask))
    score = float(white_on_edges) / float(edge_count) if edge_count > 0 else 0.0
    return score >= 0.22


# ============ WRITERS & MONITOR ============

class AsyncVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create VideoWriter: {path}")
        self.queue          = queue.Queue(maxsize=WRITE_QUEUE_SIZE)
        self.thread         = threading.Thread(target=self._run, daemon=True)
        self.running        = True
        self.frames_written = 0
        self.frames_dropped = 0
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                f = self.queue.get(timeout=1)
                if f is None:
                    break
                self.writer.write(f)
                self.frames_written += 1
                self.queue.task_done()
            except queue.Empty:
                continue

    def write(self, frame):
        try:
            self.queue.put_nowait(frame)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def release(self):
        self.queue.put(None)
        self.thread.join(timeout=5)
        self.running = False
        self.writer.release()

class BatchCSVWriter:
    def __init__(self, path, batch_size=CSV_BATCH_SIZE):
        self.path    = path
        self.bs      = batch_size
        self.buf     = []
        self.total   = 0
        self.first   = True

    def add_row(self, row):
        self.buf.append(row)
        if len(self.buf) >= self.bs:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        pd.DataFrame(self.buf).to_csv(self.path, mode='w' if self.first else 'a',
                                       header=self.first, index=False)
        self.total += len(self.buf)
        self.buf.clear()
        self.first = False

    def close(self):
        self.flush()

class PerformanceMonitor:
    def __init__(self, window=30):
        self.hist = deque(maxlen=window)
    def update(self, fps):
        self.hist.append(fps)
    def get_average_fps(self):
        return float(np.mean(self.hist)) if self.hist else 0.0


# ============ PIPELINE THREADS ============

def frame_reader(cap, fq, raw_writer=None, raw_size=None):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    frame_delay = 1.0 / fps

    while cap.isOpened() and not SHUTDOWN_FLAG.is_set():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_CAMERA:
            try:
                frame = cv2.flip(frame, -1)
            except Exception:
                pass
        # Simpan frame mentah (resize ke raw_size jika perlu) sebelum diproses
        if raw_writer is not None:
            try:
                if raw_size is not None and (frame.shape[1], frame.shape[0]) != raw_size:
                    raw_frame = cv2.resize(frame, raw_size, interpolation=cv2.INTER_AREA)
                else:
                    raw_frame = frame
                raw_writer.write(raw_frame)
            except Exception:
                pass

        try:
            fq.put_nowait(frame)
        except queue.Full:
            try:
                fq.get_nowait()
            except queue.Empty:
                pass
            try:
                fq.put_nowait(frame)
            except queue.Full:
                pass

        elapsed = time.time() - start_time
        sleep_time = frame_delay - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    try:
        fq.put(None, timeout=1)
    except Exception:
        pass

def processor_thread(processor, fq, rq):
    idx = 0
    while not SHUTDOWN_FLAG.is_set():
        try:
            frame = fq.get(timeout=1)
        except queue.Empty:
            continue
        if frame is None:
            break
        if hasattr(processor,'resize_scale') and processor.resize_scale != 1.0:
            frame = cv2.resize(frame, (processor.width, processor.height))
        fps = processor.perf_monitor.get_average_fps() if idx > 0 else 0.0
        t0  = time.time()
        combined, overlay, bev = processor.process_frame(frame, idx, fps)
        processor.perf_monitor.update(1.0/(time.time()-t0+1e-6))
        try:
            rq.put((combined, overlay, bev, idx), timeout=1)
        except queue.Full:
            if SHUTDOWN_FLAG.is_set():
                break
        idx += 1
    try:
        rq.put(None, timeout=1)
    except Exception:
        pass

def consume_results_main_thread(rq, async_writer):
    if DISPLAY_ENABLED:
        try:
            window_title = "Lane Detection BEV" if DISPLAY_BEV_ONLY else "Lane Detection Dashboard"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        except Exception as e:
            logger.error(f"Cannot create window: {e}")
    display_failed = False
    last_log = time.time()
    while not SHUTDOWN_FLAG.is_set():
        try:
            item = rq.get(timeout=1)
        except queue.Empty:
            if time.time()-last_log > 30:
                logger.info("Processing...")
                last_log = time.time()
            continue
        if item is None:
            break
        combined, _, bev, _ = item
        if async_writer is not None:
            async_writer.write(combined)
        if DISPLAY_ENABLED and not display_failed:
            try:
                disp = bev if DISPLAY_BEV_ONLY else combined
                if disp.shape[1] > 1600:
                    sc = 1600/disp.shape[1]
                    disp = cv2.resize(disp, (int(disp.shape[1]*sc), int(disp.shape[0]*sc)))
                window_title = "Lane Detection BEV" if DISPLAY_BEV_ONLY else "Lane Detection Dashboard"
                cv2.imshow(window_title, disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    SHUTDOWN_FLAG.set()
                    break
            except Exception as e:
                logger.error(f"Display error: {e}")
                display_failed = True


# ============ DECISION FUNCTIONS ============

def stabilize_recommendation(desired, state, params, force_immediate_stay=False):
    """Stabilize recommendation to avoid flicker.

    - Holds the last stable decision until a new candidate is confirmed for N frames.
    - Allows immediate switch to STAY when safety-critical (e.g., cut-in risk).

    Parameters
    ----------
    desired : str
        Raw decision from rules: "LEFT" | "STAY" | "RIGHT".
    state : dict
        Mutable state dict with keys: stable, candidate, count, hold.
    params : dict
        Expects optional keys: SWITCH_CONFIRM_FRAMES, HOLD_FRAMES.
    force_immediate_stay : bool
        If True and desired=="STAY", bypass confirmation and switch now.
    """
    if state is None:
        state = {}
    if not state.get("stable"):
        state["stable"] = "STAY"
        state["candidate"] = None
        state["count"] = 0
        state["hold"] = 0

    stable = state.get("stable", "STAY")
    desired = desired if desired in ("LEFT", "STAY", "RIGHT") else "STAY"

    if force_immediate_stay and desired == "STAY":
        state["stable"] = "STAY"
        state["candidate"] = None
        state["count"] = 0
        state["hold"] = 0
        return "STAY"

    # Optional hold: once we recommend a lane change, keep it for a few frames
    # unless a safety-critical STAY is triggered.
    hold_frames = int(params.get("HOLD_FRAMES", 0) or 0)
    if stable in ("LEFT", "RIGHT") and desired == "STAY" and int(state.get("hold", 0) or 0) > 0:
        state["hold"] = int(state.get("hold", 0)) - 1
        return stable

    if desired == stable:
        state["candidate"] = None
        state["count"] = 0
        return stable

    # Candidate confirmation
    if state.get("candidate") != desired:
        state["candidate"] = desired
        state["count"] = 1
    else:
        state["count"] = int(state.get("count", 0) or 0) + 1

    confirm = int(params.get("SWITCH_CONFIRM_FRAMES", 8) or 8)
    if state["count"] >= max(1, confirm):
        state["stable"] = desired
        state["candidate"] = None
        state["count"] = 0
        state["hold"] = hold_frames if desired in ("LEFT", "RIGHT") else 0

    return state.get("stable", "STAY")

def compute_lane_availability(side, *, yellow_edge=False, boundary_style=None,
                              outer_style=None, extra_markings=None,
                              at_edge_evidence=None, ego_position="CENTER",
                              vehicle_seen=False, edge_locked=False):
    """Tentukan apakah ada lajur tujuan yang VALID & boleh dilintasi di `side`.

    Ini adalah inti perbaikan fail-safe: rekomendasi pindah lajur HANYA boleh
    diberikan kalau ada BUKTI POSITIF bahwa lajur tujuan benar-benar ada dan
    batasnya boleh dilintasi. Tidak adanya kendaraan di suatu sisi TIDAK boleh
    dianggap sebagai "lajur kosong yang aman" -- bisa jadi itu median/bahu jalan.

    Sinyal yang dipakai semuanya SUDAH dihitung di pipeline (tanpa biaya
    komputasi tambahan yang berarti -> aman untuk Jetson Nano):

    Parameters
    ----------
    side : str
        "left" atau "right".
    yellow_edge : bool
        Garis kuning terdeteksi di sisi tsb (batas tepi jalan / median).
    boundary_style : str | None
        Tipe garis batas lajur ego di sisi tsb: "solid" | "dashed" | None.
    outer_style : str | None
        Tipe garis batas TERLUAR di sisi tsb: "solid" | "dashed" | None.
    extra_markings : bool | None
        True bila terdeteksi marka lajur tambahan DI LUAR batas lajur saat ini
        (indikasi kuat ada lajur tetangga).
    at_edge_evidence : bool | None
        Bukti frame-ini bahwa ego berada di lajur pinggir sisi tsb.
    ego_position : str
        Status state-machine: "LEFT_LANE" | "CENTER" | "RIGHT_LANE".
    vehicle_seen : bool
        True bila ada kendaraan SEARAH & DEKAT teramati di sisi tsb (sudah
        difilter dari lalu lintas seberang/jauh) -> bukti tambahan lajur ada.
    edge_locked : bool
        True bila latch edge-pinggir sisi tsb sedang aktif (bukti edge terlihat
        beberapa detik terakhir) -> paksa TIDAK tersedia.

    Returns
    -------
    dict : {available, crossable, lane_exists, not_at_edge, edge_locked}
    """
    target_state = "RIGHT_LANE" if side == "right" else "LEFT_LANE"

    # (1) Crossability dari tipe garis batas langsung:
    #     garis KUNING atau SOLID => dilarang dilintasi (apapun status ego).
    crossable = (not bool(yellow_edge)) and (boundary_style != "solid")

    # (2) Bukti POSITIF bahwa lajur tetangga memang ADA & boleh dilintasi.
    #     KUNCI: pakai tipe GARIS BATAS LANGSUNG di sisi tsb (garis yang akan
    #     diseberangi). Hanya garis DASHED terdeteksi yang membuktikan ada lajur
    #     tetangga yang boleh dimasuki. Marka jauh (outer/extra) TIDAK dipakai
    #     sendiri karena gampang keliru oleh guardrail / lajur seberang median.
    #     Alternatif: kendaraan SEARAH & DEKAT yang teramati di sisi tsb.
    boundary_is_dashed = (boundary_style == "dashed")
    #     Marka tambahan hanya jadi penguat BILA batas langsung juga dashed.
    extra_corroborates = bool(extra_markings) and boundary_is_dashed
    lane_exists = boundary_is_dashed or extra_corroborates or bool(vehicle_seen)

    # (3) Ego tidak sedang di lajur pinggir sisi tsb.
    not_at_edge = (ego_position != target_state) and (at_edge_evidence is not True)

    available = bool(crossable and lane_exists and not_at_edge and (not edge_locked))
    return {
        "available": available,
        "crossable": bool(crossable),
        "lane_exists": bool(lane_exists),
        "not_at_edge": bool(not_at_edge),
        "edge_locked": bool(edge_locked),
    }


def correct_side_labels_for_edge_lane(objects, ego_lane_index, same_lane_lat_m=2.2):
    """Koreksi label sisi kendaraan saat ego di lajur paling kanan/kiri.

    Saat ego di lajur paling KANAN tidak ada lajur kanan -> objek berlabel
    'right_side' dengan offset lateral KECIL (|offset_m| <= same_lane_lat_m)
    sebenarnya masih SATU LAJUR dengan ego (deteksi sering salah label) ->
    relabel ke 'center'. Objek jauh (mis. lalu lintas seberang median) dibiarkan.
    Mirror untuk lajur paling KIRI. Mengembalikan set track_id yang dikoreksi.
    """
    if ego_lane_index == 2:
        target = "right_side"
    elif ego_lane_index == 0:
        target = "left_side"
    else:
        return set()
    fixed = set()
    for o in objects:
        if o.get("position") != target:
            continue
        om = o.get("offset_m")
        if om is None:
            continue
        try:
            if abs(float(om)) <= float(same_lane_lat_m):
                o["position"] = "center"
                fixed.add(o.get("track_id"))
        except Exception:
            continue
    return fixed


class EgoLaneEstimator:
    """Estimasi indeks lajur ego (LEFT_LANE / CENTER / RIGHT_LANE) yang STABIL.

    Masalah inti: state-machine lama sering nyangkut di CENTER walau ego di lajur
    pinggir, sehingga rekomendasi salah. Kelas ini memfusikan beberapa isyarat
    klasik (murah, cocok untuk Jetson Nano) lewat integrator bocor + histeresis,
    karena perpindahan lajur adalah kejadian jarang & disengaja (jadi estimasi
    harus 'sticky', bukan ganti tiap frame).

    Isyarat per-sisi bahwa ego berada di TEPI sisi tsb:
      + garis batas sisi tsb KUNING / SOLID  (tepi jalan / median)
      + garis batas sisi tsb tidak ada padahal sisi seberang ada
      - garis batas sisi tsb DASHED            (ada lajur lagi -> bukan tepi)
      - ada kendaraan SEARAH & dekat di sisi tsb (lajur tetangga jelas ada)
    Akumulator di-EMA dengan tetapan waktu ~beberapa detik; belief berpindah
    hanya saat akumulasi melewati ambang (ENTER) dan dilepas pada ambang lebih
    rendah (EXIT) -> histeresis = anti-jitter.
    """

    def __init__(self, enter=0.60, exit=0.35, tau_sec=1.5):
        self.belief = "CENTER"
        self.r_acc  = 0.0     # akumulator bukti tepi-kanan [0..1]
        self.l_acc  = 0.0     # akumulator bukti tepi-kiri  [0..1]
        self.conf   = 0.0
        self.enter  = float(enter)
        self.exit   = float(exit)
        self.tau    = float(tau_sec)

    @staticmethod
    def _clip01(v):
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    @classmethod
    def _edge_target(cls, *, yellow, style, style_score, present, opposite_present, adj_vehicle):
        """Skor bukti 'sisi ini adalah TEPI' -> [0..1] (0.5 = netral).

        Prioritas: tipe GARIS BATAS LANGSUNG.
        - SOLID  -> garis tak boleh dilintasi (tepi jalan/median/garis tunggal) =
          bukti KUAT tepi. Karena solid memisahkan carriageway, kendaraan di
          seberang TIDAK dianggap bukti lajur tetangga (diabaikan).
        - DASHED -> ada lajur tetangga yang boleh dilintasi = bukti KUAT BUKAN tepi.
        - unknown/None -> pakai isyarat lemah (kuning, garis hilang, kendaraan).
        """
        if style == "solid":
            sc = 1.0 if style_score is None else cls._clip01(float(style_score))
            base = 0.55 + 0.40 * sc            # 0.55..0.95 sesuai keyakinan solid
            return cls._clip01(base + (0.05 if yellow else 0.0))
        if style == "dashed":
            return 0.15                         # ada lajur tetangga -> bukan tepi
        # unknown / tidak terklasifikasi
        s = 0.0
        if yellow:                              s += 0.45
        if (not present) and opposite_present:  s += 0.25
        if adj_vehicle:                         s -= 0.30
        return cls._clip01(0.5 + s)

    def update(self, *, left_style, right_style, left_style_score=None,
               right_style_score=None, yellow_left=False, yellow_right=False,
               left_present=True, right_present=True,
               left_adj_vehicle=False, right_adj_vehicle=False, fps=25.0):
        fps = max(5.0, float(fps))
        a = 1.0 - math.exp(-1.0 / max(1.0, self.tau * fps))  # alpha EMA per-frame

        r_t = self._edge_target(yellow=yellow_right, style=right_style,
                                style_score=right_style_score,
                                present=right_present, opposite_present=left_present,
                                adj_vehicle=right_adj_vehicle)
        l_t = self._edge_target(yellow=yellow_left, style=left_style,
                                style_score=left_style_score,
                                present=left_present, opposite_present=right_present,
                                adj_vehicle=left_adj_vehicle)
        self.r_acc += a * (r_t - self.r_acc)
        self.l_acc += a * (l_t - self.l_acc)

        # Histeresis + hanya transisi ke tetangga (RIGHT<->CENTER<->LEFT)
        if self.belief == "RIGHT_LANE":
            if self.r_acc < self.exit:
                self.belief = "CENTER"
        elif self.belief == "LEFT_LANE":
            if self.l_acc < self.exit:
                self.belief = "CENTER"
        else:  # CENTER
            if self.r_acc > self.enter and self.r_acc >= self.l_acc:
                self.belief = "RIGHT_LANE"
            elif self.l_acc > self.enter and self.l_acc > self.r_acc:
                self.belief = "LEFT_LANE"

        self.conf = max(self.r_acc, self.l_acc)
        return self.belief, float(self.conf)

def lane_change_decision(objects, ego_offset, params, buffer, ego_position, lane_avail=None):
    """Berikan rekomendasi perpindahan lajur untuk ego: LEFT / STAY / RIGHT.

    Alur keputusan:
    - Cek risiko cut-in ke lajur ego berdasarkan prediksi BI-LSTM.
    - Tentukan kebutuhan overtaking dari kendaraan depan (jarak dekat + ego closing).
    - Evaluasi lajur kiri/kanan dengan aturan safety yang sama.
    - Pilih lajur dengan peningkatan jarak depan (front gap) terbaik jika memenuhi ambang.
    - Selain itu STAY.

    """
    left, center, right = [], [], []
    for obj in objects:
        p = obj.get("position")
        if p == "center":
            center.append(obj)
        elif p == "left_side":
            left.append(obj)
        elif p == "right_side":
            right.append(obj)

    def nearest_front(objs):
        front = [o for o in objs if o.get("distance") is not None and float(o["distance"]) > 0]
        return min(front, key=lambda o: float(o["distance"])) if front else None

    front_c = nearest_front(center)
    front_l = nearest_front(left)
    front_r = nearest_front(right)

    def has_cutin_risk(objs, expected_class):
        # Adaptive cut-in confidence (Stage 1):
        # Kendaraan yang lebih dekat menjadi lebih berbahaya, sehingga ambang
        # confidence diturunkan untuk meningkatkan sensitivitas pada kondisi
        # urgent. Kendaraan jauh tetap memakai ambang penuh.
        base_thr = float(params.get("CUTIN_CONFIDENCE", 0.55))
        for o in objs:
            if o.get("lc_is_uncertain") is True:
                continue
            if o.get("lc_predicted_class") != expected_class:
                continue
            conf = float(o.get("lc_confidence") or 0.0)
            d = o.get("distance")
            # Distance-aware threshold:
            #   d < 10m  -> thr -= 0.15 (paling sensitif, urgent)
            #   d < 20m  -> thr -= 0.05 (sedang)
            #   d >= 20m -> thr penuh
            if d is not None:
                try:
                    d_abs = abs(float(d))
                    if d_abs < 10.0:
                        thr = max(0.35, base_thr - 0.15)
                    elif d_abs < 20.0:
                        thr = max(0.45, base_thr - 0.05)
                    else:
                        thr = base_thr
                except Exception:
                    thr = base_thr
            else:
                thr = base_thr
            if conf >= thr:
                return True
        return False

    cutin_from_left  = has_cutin_risk(left,  expected_class=2)  # kiri -> kanan menuju ego
    cutin_from_right = has_cutin_risk(right, expected_class=1)  # kanan -> kiri menuju ego

    def min_abs_distance(objs):
        vals = []
        for o in objs:
            if o.get("distance") is None:
                continue
            try:
                vals.append(abs(float(o["distance"])))
            except Exception:
                continue
        return min(vals) if vals else None

    debug_info = {
        "ego_position": ego_position,
        "ego_offset": float(ego_offset),
        "cutin_from_left": bool(cutin_from_left),
        "cutin_from_right": bool(cutin_from_right),
        "n_left": int(len(left)),
        "n_center": int(len(center)),
        "n_right": int(len(right)),
        "min_abs_left": min_abs_distance(left),
        "min_abs_center": min_abs_distance(center),
        "min_abs_right": min_abs_distance(right),
    }

    force_immediate_stay = False
    if cutin_from_left or cutin_from_right:
        desired = "STAY"
        force_immediate_stay = True
    elif abs(float(ego_offset)) > float(params["EGO_CHANGE_THRESHOLD"]):
        desired = "STAY"
        force_immediate_stay = True
        debug_info["ego_unstable"] = True
    else:
        close_dist  = float(params.get("OVERTAKE_CLOSE_DIST", 25.0))
        closing_thr = float(params.get("OVERTAKE_CLOSING_MPS", 2.0))
        min_improve = float(params.get("MIN_IMPROVEMENT", 0.0))
        ttc_thr     = float(params.get("TTC_THRESHOLD", 3.0))
        need_overtake = False
        ttc_front = float('inf')
        if front_c is not None:
            d = float(front_c.get("distance") or np.inf)
            v = float(front_c.get("rel_speed") or 0.0)
            # Stage 3 (TTC): kalau jarak mengecil cukup cepat, hitung waktu
            # tabrakan teoretis. Pemicu overtake jika:
            #   (a) trigger distance-based lama (d <= 25m AND closing >= 2m/s)
            #   (b) trigger TTC-based baru (TTC <= 3 detik) -- auto speed-adaptive
            # Sehingga sistem responsif pada kecepatan tinggi DAN tidak agresif
            # pada kecepatan rendah.
            if v < -0.1:  # ego sedang menutup jarak
                ttc_front = abs(d / v)
            dist_trigger = (d <= close_dist) and (v <= -closing_thr)
            ttc_trigger  = (ttc_front <= ttc_thr)
            need_overtake = dist_trigger or ttc_trigger
        debug_info["ttc_front"] = float(ttc_front) if np.isfinite(ttc_front) else None

        def is_danger(objs):
            dz = float(params["DANGER_ZONE"])
            for o in objs:
                if o.get("distance") is None:
                    continue
                if abs(float(o["distance"])) < dz:
                    return True
            return False

        def safe_front(front_obj):
            if front_obj is None:
                return np.inf
            return float(front_obj.get("distance") or np.inf)

        # Fail-safe gating: lajur tujuan harus TERBUKTI ada & boleh dilintasi.
        # Default True hanya bila info tidak diberikan (kompatibilitas pemanggil
        # lain); pipeline utama SELALU meneruskan lane_avail.
        la = lane_avail or {}
        left_available  = bool((la.get("left")  or {}).get("available", True))
        right_available = bool((la.get("right") or {}).get("available", True))

        # Catatan: TIDAK ada cek 'rear'/blind-spot. Kamera depan tidak bisa
        # melihat kendaraan dari belakang, jadi keamanan rekomendasi hanya
        # menilai jalur DEPAN di lajur tujuan -> bersifat ADVISORY (pengemudi
        # tetap wajib cek spion/blind-spot sendiri).
        left_safe = (
            left_available
            and ego_position != "LEFT_LANE"
            and (not is_danger(left))
            and safe_front(front_l) >= float(params["SAFE_FRONT"])
            and (not has_cutin_risk(left, expected_class=2))
        )
        right_safe = (
            right_available
            and ego_position != "RIGHT_LANE"
            and (not is_danger(right))
            and safe_front(front_r) >= float(params["SAFE_FRONT"])
            and (not has_cutin_risk(right, expected_class=1))
        )

        # Symmetric selection: choose the safer side with the best forward-gap improvement.
        # Improvement is measured against the current-lane nearest front vehicle.
        d_center = safe_front(front_c)
        d_left   = safe_front(front_l)
        d_right  = safe_front(front_r)

        improve_left  = d_left  - d_center
        improve_right = d_right - d_center

        cand_left  = left_safe and (improve_left >= min_improve)
        cand_right = right_safe and (improve_right >= min_improve)

        debug_info.update({
            "need_overtake": bool(need_overtake),
            "left_available": bool(left_available),
            "right_available": bool(right_available),
            "left_safe": bool(left_safe),
            "right_safe": bool(right_safe),
            "d_center": float(d_center) if np.isfinite(d_center) else None,
            "d_left": float(d_left) if np.isfinite(d_left) else None,
            "d_right": float(d_right) if np.isfinite(d_right) else None,
            "improve_left": float(improve_left) if np.isfinite(improve_left) else None,
            "improve_right": float(improve_right) if np.isfinite(improve_right) else None,
            "min_improvement": float(min_improve),
            "cand_left": bool(cand_left),
            "cand_right": bool(cand_right),
        })

        # If not overtaking, be conservative: only recommend a change when there's a clear improvement.
        if not need_overtake:
            if not (cand_left or cand_right):
                desired = "STAY"
            elif cand_left and not cand_right:
                desired = "LEFT"
            elif cand_right and not cand_left:
                desired = "RIGHT"
            else:
                # Both candidates: pick larger improvement; tie => STAY (avoid arbitrary bias).
                if improve_left > improve_right:
                    desired = "LEFT"
                elif improve_right > improve_left:
                    desired = "RIGHT"
                else:
                    desired = "STAY"
        else:
            # Overtake mode: allow either side if safe; prefer the side with larger forward gap.
            if not (left_safe or right_safe):
                desired = "STAY"
            elif left_safe and not right_safe:
                desired = "LEFT"
            elif right_safe and not left_safe:
                desired = "RIGHT"
            else:
                if d_left > d_right:
                    desired = "LEFT"
                elif d_right > d_left:
                    desired = "RIGHT"
                else:
                    desired = "STAY"

    debug_info["force_stay"] = bool(force_immediate_stay)
    debug_info["desired"] = desired
    return desired, force_immediate_stay, debug_info

def estimate_speed_bracket(objects, current_bracket="medium"):
    """Stage 4: Estimasi bracket kecepatan ego berbasis kondisi lalu lintas.

    Tidak perlu speedometer absolut. Logikanya: dari rel_speed kendaraan
    sekitar dan jumlah kendaraan terdeteksi, simpulkan ego di kondisi
    lambat (macet/kota), sedang, atau cepat (tol lengang).

    Returns: "slow" | "medium" | "fast"

    Heuristik:
      - Kalau ada banyak kendaraan dengan rel_speed mendekati 0 (semua
        bergerak bareng) -> traffic flow stabil, kemungkinan kondisi cepat
        kalau jaraknya jauh atau lambat kalau jaraknya dekat.
      - Kalau ada kendaraan dengan |rel_speed| besar (>5 m/s) -> mungkin
        ego sedang menutup kendaraan lambat (kondisi cepat dengan obstruksi).
      - Kalau kendaraan sangat sedikit atau tidak ada -> tidak yakin,
        pakai bracket saat ini (smoothing).
    """
    if not objects:
        return current_bracket

    rel_speeds = [float(o.get("rel_speed") or 0.0)
                  for o in objects
                  if o.get("rel_speed") is not None]
    distances = [abs(float(o.get("distance") or 0.0))
                 for o in objects
                 if o.get("distance") is not None]

    if len(rel_speeds) == 0 or len(distances) == 0:
        return current_bracket

    avg_abs_relspeed = float(np.mean([abs(v) for v in rel_speeds]))
    avg_distance = float(np.mean(distances))

    # Klasifikasi heuristik:
    # - kondisi cepat: kendaraan jauh rata-rata DAN sedikit closing
    # - kondisi lambat: kendaraan dekat rata-rata DAN sedikit closing
    # - kondisi sedang: di antara keduanya
    if avg_distance > 25.0 and avg_abs_relspeed < 3.0:
        return "fast"
    elif avg_distance < 12.0 and avg_abs_relspeed < 2.0:
        return "slow"
    else:
        return "medium"


def is_likely_at_edge(side, ego_position, edge_evidence,
                      vehicle_history, offset_history=None,
                      opposite_history=None,
                      history_threshold=0.05, min_history=30,
                      offset_threshold=15.0, offset_min_history=30,
                      asymmetry_threshold=0.5):
    """Stage 2: Multi-signal validasi posisi ego di lajur pinggir.

    Threshold dikalibrasi untuk MINIMASI false-positive: hanya memicu
    override pada kasus dengan bukti SANGAT KUAT bahwa ego di lajur pinggir
    walaupun modul deteksi lajur salah membaca scene.

    Empat sinyal independen, dengan voting 2-of-4:

    1. ego_position state machine
    2. edge_evidence (leftmost/rightmost_evidence)
    3. ASYMMETRY: sisi target hampir kosong DAN sisi sebaliknya aktif.
       Lebih spesifik dari sekadar "absence" karena menghindari kasus
       jalan sepi (ambigu).
    4. Offset history: rolling mean ego_offset_px konsisten ke arah
       tertentu DENGAN MAGNITUDO TINGGI (default 15 px).
    """
    target_state = "LEFT_LANE" if side == "left" else "RIGHT_LANE"

    # Sinyal 1: status ego_position dari state machine sistem
    sig_state = (ego_position == target_state)

    # Sinyal 2: bukti garis pinggir
    sig_edge = (edge_evidence is True)

    # Sinyal 3: ASYMMETRY (lebih spesifik dari absence murni).
    # Untuk LEFT_EDGE: sisi kiri target hampir kosong DAN sisi kanan punya
    # aktivitas konsisten -> bukti kuat ego di leftmost (karena kalau ego di
    # tengah, harusnya kedua sisi sama-sama ada kendaraan).
    sig_asymmetry = False
    if (len(vehicle_history) >= min_history and
        opposite_history is not None and len(opposite_history) >= min_history):
        avg_target = float(np.mean(vehicle_history))
        avg_opposite = float(np.mean(opposite_history))
        sig_asymmetry = (avg_target < history_threshold and
                         avg_opposite > asymmetry_threshold)

    # Sinyal 4: rolling mean offset menunjukkan ego konsisten di sisi
    # tertentu DENGAN MAGNITUDO TINGGI (threshold 15 px, bukan 5).
    # Threshold tinggi menghindari false-positive dari drift biasa.
    sig_offset = False
    if offset_history is not None and len(offset_history) >= offset_min_history:
        avg_offset = float(np.mean(offset_history))
        if side == "left" and avg_offset < -offset_threshold:
            sig_offset = True
        elif side == "right" and avg_offset > offset_threshold:
            sig_offset = True

    # Voting majority: minimal 2 dari 4 sinyal harus aktif
    return sum([sig_state, sig_edge, sig_asymmetry, sig_offset]) >= 2


def compute_ego_offset(left_line, right_line, frame_width, frame_height):
    if left_line is None or right_line is None:
        return 0.0
    y = int(frame_height*0.8)
    def gx(line):
        x1,y1,x2,y2 = line
        if y2==y1: return float(x1)
        return float(x1 + (y-y1)/(y2-y1)*(x2-x1))
    return float(frame_width)/2.0 - (gx(left_line)+gx(right_line))/2.0

def draw_recommendation(frame, decision):
    colors = {"LEFT":(255,0,255),"RIGHT":(0,165,255),"STAY":(0,255,0)}
    cv2.putText(frame, f"RECOMMENDATION: {decision}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors.get(decision,(255,255,255)), 2)


def draw_decision_debug(frame, info, desired, stable, ego_position):
    if not info:
        return
    def fmt(x, digits=1):
        if x is None:
            return "None"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)

    lines = [
        f"desired={desired} stable={stable} ego={ego_position}",
        f"bev_suppress_right={info.get('suppressed_bev_right')}",
        f"rightmost={info.get('rightmost_evidence')} leftmost={info.get('leftmost_evidence')} yellowR={info.get('yellow_right')} edgeY R/L={info.get('yellow_edge_r')}/{info.get('yellow_edge_l')} edgeW R/L={info.get('white_edge_r')}/{info.get('white_edge_l')} extraR/L={info.get('extra_r')}/{info.get('extra_l')}",
        f"rx2={fmt(info.get('rx2'),0)} lx2={fmt(info.get('lx2'),0)} laneWpx={fmt(info.get('lane_width_px'),0)} outerRx={fmt(info.get('outer_rx2'),0)} outerLx={fmt(info.get('outer_lx2'),0)}",
        f"style R/L={info.get('right_style')}/{info.get('left_style')} outerStyle R/L={info.get('outer_r_style')}/{info.get('outer_l_style')} align R/L={info.get('align_r')}/{info.get('align_l')}",
        f"cutin L={info.get('cutin_from_left')} R={info.get('cutin_from_right')} force_stay={info.get('force_stay')}",
        f"safe L={info.get('left_safe')} R={info.get('right_safe')} cand L={info.get('cand_left')} R={info.get('cand_right')}",
        f"dC={fmt(info.get('d_center'))} dL={fmt(info.get('d_left'))} dR={fmt(info.get('d_right'))}",
        f"impL={fmt(info.get('improve_left'))} impR={fmt(info.get('improve_right'))} minImp={fmt(info.get('min_improvement'))}",
        f"need_overtake={info.get('need_overtake')} ego_offset_px={fmt(info.get('ego_offset'),0)}",
        f"objs L/C/R={info.get('n_left')}/{info.get('n_center')}/{info.get('n_right')} minAbsDist L/C/R={fmt(info.get('min_abs_left'))}/{fmt(info.get('min_abs_center'))}/{fmt(info.get('min_abs_right'))}",
    ]

    x = 30
    y0 = 80
    dy = 18
    for i, txt in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)


# ============ MAIN PROCESSOR ============

class LaneOffsetProcessor:

    def __init__(self, video_path, output_dir):
        self.video_path  = video_path
        self.output_dir  = output_dir
        self.model       = None
        self.cap         = None
        self.async_writer = None
        self.csv_writer  = None
        self.decision_csv_writer = None
        self.perf_monitor = PerformanceMonitor()

        self.left_history  = []
        self.right_history = []
        self.last_known_boxes     = []
        self.consecutive_errors   = 0
        self.max_consecutive_errors = 10
        self.yolo_skip_frames     = YOLO_SKIP_FRAMES
        self.frame_counter        = 0
        self.last_yolo_results    = None

        self.stats = {'frames_processed':0, 'detections':0, 'offset_calculated':0}

        self.roi_mask      = None
        self.bev_template  = None
        self.combined_canvas = None
        self.roi_mask_lane = None
        self.lane_scale    = LANE_DOWNSCALE
        self.lane_scale_inv = 1.0/LANE_DOWNSCALE
        self.use_dashboard = DISPLAY_ENABLED or RECORDING_ENABLED

        # Recommendation stabilization state (prevents flicker)
        self.decision_buffer = {
            "stable": "STAY",
            "candidate": None,
            "count": 0,
            "hold": 0,
        }
        self._tracks         = {}
        self._next_track_id  = 1
        self.ego_position_state       = "CENTER"
        self._rightmost_true_streak   = 0
        self._rightmost_false_streak  = 0
        self._leftmost_true_streak    = 0
        self._leftmost_false_streak   = 0
        self._rightmost_last          = None
        self._rightmost_last_age      = 999
        self._leftmost_last           = None
        self._leftmost_last_age       = 999
        # Edge-lock latch: sekali ada bukti ego di lajur pinggir (state RIGHT/LEFT_LANE,
        # rightmost/leftmost_evidence, atau garis kuning/solid), kunci sisi tsb sebagai
        # "tidak ada lajur" selama beberapa detik agar tahan terhadap dropout deteksi.
        self._right_edge_lock         = 0
        self._left_edge_lock          = 0
        # [TRUNCATED] cooldown per track_id -- lihat is_bbox_truncated()
        self._truncated_cooldown      = {}
        # Hold pendek untuk bukti kendaraan searah di lajur tetangga (anti-flicker).
        self._right_adjveh_hold       = 0
        self._left_adjveh_hold        = 0
        # Estimator indeks lajur ego berbasis fusi klasik + temporal (anti-jitter).
        self.ego_lane_estimator       = EgoLaneEstimator()
        self._ego_lane_index          = 1
        # Kalibrasi jarak (penggaris lajur). _focal_px dihitung dari lebar frame
        # saat runtime; _cal_factor = rasio jarak-kalibrasi/raw utk fallback saat
        # garis lajur hilang (di-EMA dari frame yang lajurnya terdeteksi).
        self._focal_px                = None
        self._cal_factor              = 3.0    # default kasar (raw under-estimate ~3x)

        # Default parameter (base) -- akan dimodifikasi secara adaptif per frame
        # berdasarkan kondisi (kecepatan, jarak, urgensi) untuk pendekatan
        # "context-aware rule-based".
        self.decision_params = {
            # Base values (untuk kecepatan sedang)
            "SAFE_FRONT": 45, "DANGER_ZONE": 12,
            "MIN_IMPROVEMENT": 10, "EGO_CHANGE_THRESHOLD": 50,
            # Cut-in: dengan distance-aware adjustment di has_cutin_risk
            "CUTIN_CONFIDENCE": 0.55,
            # Overtake triggers (distance-based + TTC-based hybrid)
            "OVERTAKE_CLOSE_DIST": 35.0,
            "OVERTAKE_CLOSING_MPS": 2.0,
            "TTC_THRESHOLD": 3.0,   # Stage 3: kalau TTC <= 3s, picu need_overtake
            # Ego lane-state stabilization
            "EGO_LOCK_FRAMES": 4,
            "EGO_UNLOCK_FRAMES": 25,
            "EGO_EVIDENCE_HOLD_FRAMES": 5,
            # (Parameter 'rear' dihapus: kamera depan tidak bisa menilai
            #  kendaraan dari belakang -> rekomendasi bersifat advisory.)
            # Bukti lajur tetangga dari kendaraan: hanya kendaraan SEARAH & DEKAT
            # yang dianggap bukti (menyaring lalu lintas seberang median/jauh).
            "ADJ_LANE_MAX_DIST": 45.0,      # m, batas jarak longitudinal
            "ADJ_LANE_MAX_CLOSING": 7.0,    # m/s, closing > ini -> dianggap berlawanan arah
            "ADJ_VEHICLE_HOLD_FRAMES": 8,   # hold bukti kendaraan searah (~0.3s)
            "EDGE_LOCK_FRAMES": 75,         # kunci sisi-pinggir (~3s) setelah bukti edge
            "SAME_LANE_LAT_M": 2.2,         # batas lateral utk relabel objek 'sisi' -> ego-lane
            # Stabilizer params
            "SWITCH_CONFIRM_FRAMES": 8,
            "HOLD_FRAMES": 15,
        }

        # Stage 4: Speed-adaptive bracket presets.
        # Sistem akan otomatis pakai salah satu set ini per frame berdasarkan
        # estimasi kondisi kecepatan dari rel_speed kendaraan sekitar.
        self.speed_brackets = {
            "slow":   {"SAFE_FRONT": 28, "OVERTAKE_CLOSE_DIST": 22, "MIN_IMPROVEMENT": 7},
            "medium": {"SAFE_FRONT": 45, "OVERTAKE_CLOSE_DIST": 35, "MIN_IMPROVEMENT": 10},
            "fast":   {"SAFE_FRONT": 62, "OVERTAKE_CLOSE_DIST": 48, "MIN_IMPROVEMENT": 15},
        }
        self._speed_bracket_current = "medium"
        self._speed_bracket_streak = 0  # untuk smoothing transisi bracket

        # Stage 2: rolling history kendaraan per sisi (untuk multi-signal
        # validasi posisi ego). Window default 100 frame ≈ 4 detik pada 25 FPS,
        # tapi akan diadaptasi otomatis berdasarkan FPS aktual.
        self._left_vehicle_history  = []
        self._right_vehicle_history = []
        self._VEHICLE_HISTORY_LEN   = 100

        # Stage 2 enhancement: rolling history ego_offset_px untuk membantu
        # kasus deteksi lajur misalignment (barrier tidak terdeteksi sebagai tepi).
        self._ego_offset_history    = []
        self._OFFSET_HISTORY_LEN    = 50
        # Baseline panjang ego_offset (~10-12 detik) utk koreksi BIAS pemasangan
        # kamera / titik-tengah lajur. Cek 'ego_unstable' dipakai pada DEVIASI dari
        # baseline ini, bukan offset absolut (yang punya bias konstan).
        self._ego_offset_long       = []
        self._EGO_OFFSET_LONG_LEN   = 300

        # ──────────────────────────────────────────────────────────────────
        # FPS-adaptive temporal parameters.
        # ──────────────────────────────────────────────────────────────────
        self._time_based_params = {
            # Stabilizer
            "HOLD_TIME_SEC"             : 1.0,   # 1 detik penahanan keputusan LEFT/RIGHT
            "SWITCH_CONFIRM_TIME_SEC"   : 0.3,   # 0.3 detik konfirmasi keputusan baru
            # Ego lane-state
            "EGO_LOCK_TIME_SEC"         : 0.15,  # 0.15 detik untuk lock status pinggir
            "EGO_UNLOCK_TIME_SEC"       : 1.0,   # 1 detik untuk lepas status pinggir
            "EGO_EVIDENCE_HOLD_TIME_SEC": 0.2,   # 0.2 detik hold bukti pinggir
            # History windows
            "VEHICLE_HISTORY_TIME_SEC"  : 4.0,   # 4 detik window kendaraan per sisi
            "OFFSET_HISTORY_TIME_SEC"   : 2.0,   # 2 detik window offset history
            # Fail-safe rekomendasi
            "EDGE_LOCK_TIME_SEC"        : 3.0,   # 3 detik kunci sisi-pinggir
            "ADJ_VEHICLE_HOLD_TIME_SEC" : 0.3,   # 0.3 detik hold bukti kendaraan searah
        }
        self._fps_effective = 25.0  # FPS untuk konversi, di-update tiap frame

        # [LSTM] Predictor dan state kedip
        self.lc_predictor   = LaneChangePredictor()
        self.blink_counter  = 0
        self.BLINK_INTERVAL = 5          # lebih cepat 
        self.blink_visible  = True
        self.blink_phase    = 0          # phase counter untuk multi-layer animasi

    def _update_fps_adaptive_params(self, current_fps):
        # Smooth FPS effective dengan exponential moving average (alpha=0.1)
        if current_fps is not None and current_fps > 1.0:
            self._fps_effective = 0.9 * self._fps_effective + 0.1 * float(current_fps)
        fps = max(5.0, self._fps_effective)  # floor 5 FPS untuk hindari pembagian aneh
        tbp = self._time_based_params

        # Konversi time → frames untuk parameter stabilizer
        self.decision_params["HOLD_FRAMES"]              = max(1, int(round(tbp["HOLD_TIME_SEC"]               * fps)))
        self.decision_params["SWITCH_CONFIRM_FRAMES"]    = max(1, int(round(tbp["SWITCH_CONFIRM_TIME_SEC"]     * fps)))
        self.decision_params["EGO_LOCK_FRAMES"]          = max(1, int(round(tbp["EGO_LOCK_TIME_SEC"]           * fps)))
        self.decision_params["EGO_UNLOCK_FRAMES"]        = max(1, int(round(tbp["EGO_UNLOCK_TIME_SEC"]         * fps)))
        self.decision_params["EGO_EVIDENCE_HOLD_FRAMES"] = max(1, int(round(tbp["EGO_EVIDENCE_HOLD_TIME_SEC"]  * fps)))

        # Konversi time → frames untuk history windows
        self._VEHICLE_HISTORY_LEN = max(10, int(round(tbp["VEHICLE_HISTORY_TIME_SEC"] * fps)))
        self._OFFSET_HISTORY_LEN  = max(5,  int(round(tbp["OFFSET_HISTORY_TIME_SEC"]  * fps)))

        # Konversi time → frames untuk fail-safe rekomendasi
        self.decision_params["EDGE_LOCK_FRAMES"]         = max(1, int(round(tbp["EDGE_LOCK_TIME_SEC"]        * fps)))
        self.decision_params["ADJ_VEHICLE_HOLD_FRAMES"]  = max(1, int(round(tbp["ADJ_VEHICLE_HOLD_TIME_SEC"] * fps)))

    def _cleanup_tracks(self, now_t):
        stale = [k for k,v in self._tracks.items() if now_t - v.get("t",now_t) > TRACK_STALE_SEC]
        for k in stale:
            self._tracks.pop(k, None)

    def _assign_track_id(self, cx, cy, now_t):
        best_id, best_d2 = None, None
        for tid, st in self._tracks.items():
            dx = float(cx)-float(st.get("cx",cx))
            dy = float(cy)-float(st.get("cy",cy))
            d2 = dx*dx+dy*dy
            if d2 <= TRACK_MATCH_MAX_PX**2 and (best_d2 is None or d2 < best_d2):
                best_d2, best_id = d2, tid
        if best_id is None:
            best_id = self._next_track_id
            self._next_track_id += 1
        return best_id, self._tracks.get(best_id)

    def _build_roi_masks(self, w, h):
        self.roi_mask = np.zeros((h,w), dtype=np.uint8)
        # Slightly wider ROI helps detect outer-lane/shoulder markings (rightmost/leftmost cases)
        pts = np.array([[(int(0.01*w),h),(int(0.99*w),h),(int(0.75*w),int(0.45*h)),(int(0.25*w),int(0.45*h))]], np.int32)
        cv2.fillPoly(self.roi_mask, pts, 255)
        lw = max(1,int(w*self.lane_scale)); lh = max(1,int(h*self.lane_scale))
        pts_l = np.array([[(int(0.01*lw),lh),(int(0.99*lw),lh),(int(0.75*lw),int(0.45*lh)),(int(0.25*lw),int(0.45*lh))]], np.int32)
        self.roi_mask_lane = np.zeros((lh,lw), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask_lane, pts_l, 255)

    def find_working_codec(self, frame_size):
        # MP4 only (user request): try common MP4/H.264-ish FourCC first, then MPEG-4.
        # Note: availability depends on how OpenCV/FFmpeg was built on the machine.
        codecs = [
            ('mp4v', '.mp4'),
            ('avc1', '.mp4'),  # H.264 fallback
            ('FMP4', '.mp4'),
        ]
        td = os.path.join(self.output_dir,'codec_tests')
        os.makedirs(td, exist_ok=True)
        for cs, ext in codecs:
            try:
                fc = cv2.VideoWriter_fourcc(*cs)
                tp = os.path.join(td, f'test_{cs}{ext}')
                if test_video_codec(fc, self.fps, frame_size, tp):
                    return fc, ext
            except Exception:
                continue
        # Last-resort: still write .mp4 (even if codec probing failed).
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

    def initialize(self):
        logger.info("Initializing...")

        # [LSTM] Load predictor
        logger.info("[LSTM] Loading lane change predictor...")
        self.lc_predictor.load()
        logger.info("[LSTM] Predictor ready.")

        # YOLO
        self.device = _resolve_yolo_device()
        self.model  = YOLO(YOLO_MODEL_PATH)
        if self.device:
            self.model.to(self.device)
        _maybe_enable_cudnn_benchmark()

        # Video
        self._is_webcam_source = False
        if isinstance(self.video_path, str):
            source = self.video_path.strip()
            source_lower = source.lower()

            # Tandai sumber webcam/kamera (untuk raw recording + FPS measurement)
            if (source_lower.startswith('cam:') or source_lower.startswith('webcam:')
                    or source_lower.startswith('camera:') or source_lower.startswith('ac:')):
                self._is_webcam_source = True

            if source_lower.startswith('ac:'):
                # Assetto Corsa virtual cam (typically Windows-only)
                logger.info(f"Opening virtual cam index {VIRTUAL_CAM_INDEX}")
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(VIRTUAL_CAM_INDEX, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = cv2.VideoCapture(VIRTUAL_CAM_INDEX)
                else:
                    self.cap = cv2.VideoCapture(VIRTUAL_CAM_INDEX)

            elif source_lower.startswith('cam:'):
                # Webcam index (Lane_Change.py style)
                _, idx_str = source.split(':', 1)
                cam_index = int(idx_str) if idx_str.strip().isdigit() else 0
                logger.info(f"Opening webcam index {cam_index}")
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = cv2.VideoCapture(cam_index)
                else:
                    self.cap = cv2.VideoCapture(cam_index)

            else:
                # Video file path / stream URL
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps == 0 or math.isnan(self.fps):
            self.fps = 30.0

        # Untuk webcam: FPS yang dilaporkan sering bohong (30 padahal actual 15).
        # Ukur FPS asli dengan capture sample dulu — pakai nilai actual untuk
        # VideoWriter supaya video output tidak slow motion / fast forward.
        if self._is_webcam_source:
            try:
                logger.info(f"Reported FPS: {self.fps:.2f} — measuring actual FPS...")
                for _ in range(5):          # warmup buang 5 frame
                    self.cap.read()
                n_sample = 30
                t0 = time.time()
                got = 0
                for _ in range(n_sample):
                    ret, _ = self.cap.read()
                    if ret:
                        got += 1
                elapsed = time.time() - t0
                if got > 5 and elapsed > 0.1:
                    measured_fps = got / elapsed
                    logger.info(f"Measured FPS: {measured_fps:.2f} (over {got} frames in {elapsed:.2f}s)")
                    if abs(measured_fps - self.fps) / self.fps > 0.10:
                        logger.info(f"Using measured FPS {measured_fps:.2f} (reported was {self.fps:.2f})")
                        self.fps = measured_fps
                    else:
                        logger.info(f"Reported FPS akurat, tetap pakai {self.fps:.2f}")
            except Exception as e:
                logger.warning(f"FPS measurement failed: {e}, pakai reported FPS")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.orig_width, self.orig_height = self.width, self.height
        self.resize_scale = 1.0
        if TARGET_RESIZE_HEIGHT is not None and self.height != TARGET_RESIZE_HEIGHT:
            th = TARGET_RESIZE_HEIGHT
            self.resize_scale = float(th)/float(self.height)
            self.width  = int(self.width*self.resize_scale)
            self.height = th
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(total) if total and total > 0 else 0

        logger.info(f"Video input resolution: {self.orig_width}x{self.orig_height}")
        logger.info(f"Processing resolution: {self.width}x{self.height} (scaled: {self.resize_scale:.2f}x)")
        logger.info(f"Total frames: {self.total_frames} @ {self.fps:.2f} FPS")

        self.stats_panel_height = 80
        self.target_bev_height  = self.height
        self.target_bev_width   = int(BEV_WIDTH*(self.target_bev_height/BEV_HEIGHT))
        dw = self.width + self.target_bev_width
        dh = self.height + self.stats_panel_height
        self.dashboard_width  = dw
        self.dashboard_height = dh

        if RECORDING_ENABLED:
            fc, ext = self.find_working_codec((dw, dh))
            pv = get_next_filename(os.path.join(self.output_dir,"preview_combined"), ext)
            self.async_writer = AsyncVideoWriter(pv, fc, self.fps, (dw, dh))

        # Raw video recorder — hanya aktif untuk webcam/cam source.
        # Resolusi dipaksa max 720p (jaga aspect ratio) untuk hemat disk & CPU.
        self.raw_writer = None
        self.raw_size   = None
        if self._is_webcam_source:
            try:
                src_w = self.orig_width  if self.orig_width  > 0 else self.width
                src_h = self.orig_height if self.orig_height > 0 else self.height
                MAX_RAW_H = 720
                if src_h > MAX_RAW_H:
                    scale = MAX_RAW_H / float(src_h)
                    raw_size = (int(src_w * scale), MAX_RAW_H)
                else:
                    raw_size = (src_w, src_h)
                # width genap (sebagian codec butuh ini)
                raw_size = (raw_size[0] - (raw_size[0] % 2), raw_size[1] - (raw_size[1] % 2))
                self.raw_size = raw_size

                raw_fc, raw_ext = self.find_working_codec(raw_size)
                raw_path = get_next_filename(os.path.join(self.output_dir, "raw_webcam"), raw_ext)
                self.raw_writer = AsyncVideoWriter(raw_path, raw_fc, self.fps, raw_size)
                logger.info(f"Raw webcam recording: {raw_path} @ {raw_size[0]}x{raw_size[1]} {self.fps:.1f}fps")
            except Exception as e:
                logger.warning(f"Cannot start raw webcam recorder: {e}")
                self.raw_writer = None
                self.raw_size   = None

        csv_out = get_next_filename(os.path.join(self.output_dir,"detections"), ".csv")
        self.csv_writer = BatchCSVWriter(csv_out)

        # [EVAL] Logger keputusan per-frame untuk evaluasi model rekomendasi
        decision_csv_out = get_next_filename(os.path.join(self.output_dir,"decisions_log"), ".csv")
        self.decision_csv_writer = BatchCSVWriter(decision_csv_out)

        self._build_roi_masks(self.width, self.height)
        self.bev_template    = create_birdseye_view()
        self.combined_canvas = np.zeros((dh, dw, 3), dtype=np.uint8)
        logger.info("Initialization complete.")

    def process_frame(self, frame, frame_idx, current_fps=0.0):
        timestamp    = round(frame_idx/self.fps, 3)
        frame_height = frame.shape[0]
        objects      = []

        # FPS-adaptive parameter update. Parameter waktu (HOLD, CONFIRM, dst)
        # otomatis disesuaikan dengan FPS aktual sehingga behavior temporal
        # konsisten lintas hardware (Mac, Jetson, dll).
        # Pakai self.fps (target/file) sebagai fallback kalau current_fps masih 0
        # di awal video.
        fps_for_adapt = current_fps if current_fps and current_fps > 1.0 else self.fps
        self._update_fps_adaptive_params(fps_for_adapt)

        # [LSTM] Update kedip
        self.blink_counter += 1
        if self.blink_counter >= self.BLINK_INTERVAL:
            self.blink_counter = 0
            self.blink_visible = not self.blink_visible
            self.blink_phase   = (self.blink_phase + 1) % 4  # 4-phase untuk animasi lebih kaya

        should_run_yolo = (self.frame_counter % (self.yolo_skip_frames+1)) == 0
        self.frame_counter += 1

        # Lane detection
        lw = max(1, int(frame.shape[1]*self.lane_scale))
        lh = max(1, int(frame.shape[0]*self.lane_scale))
        lf = cv2.resize(frame, (lw,lh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        if self.roi_mask_lane is None or self.roi_mask_lane.shape != (lh,lw):
            self._build_roi_masks(frame.shape[1], frame.shape[0])
        me = cv2.bitwise_and(edges, self.roi_mask_lane)
        # Rescale masked edges to full resolution for line-style (solid/dashed) analysis
        me_full = cv2.resize(me, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        lp = cv2.HoughLinesP(me, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESH,
                              minLineLength=HOUGH_MIN_LINE_LEN, maxLineGap=HOUGH_MAX_LINE_GAP)

        left_lines, right_lines = [], []
        if lp is not None:
            for seg in lp:
                x1,y1,x2,y2 = seg[0]
                x1=int(x1*self.lane_scale_inv); y1=int(y1*self.lane_scale_inv)
                x2=int(x2*self.lane_scale_inv); y2=int(y2*self.lane_scale_inv)
                if x2==x1: continue
                s = (y2-y1)/(x2-x1)
                if not is_valid_slope(s): continue
                (left_lines if s<0 else right_lines).append((x1,y1,x2,y2))

        left_avg  = average_lane_lines(left_lines,  self.height, self.width, "left")
        right_avg = average_lane_lines(right_lines, self.height, self.width, "right")

        for la, hist in [(left_avg, 'left'), (right_avg, 'right')]:
            h = self.left_history if hist=='left' else self.right_history
            if la is not None:
                h.append(la)
                if len(h) > AVG_HISTORY: h.pop(0)
                setattr(self, f'{hist}_history', h)
        left_avg  = smooth_line(self.left_history)  or (self.left_history[-1]  if self.left_history  else None)
        right_avg = smooth_line(self.right_history) or (self.right_history[-1] if self.right_history else None)

        bev     = self.bev_template.copy()
        overlay = frame.copy()
        bev_ops = []   # (bev_x, bev_y, position, in_danger, track_id, lc_result)

        if left_avg and right_avg:
            x1_l, y1_l, x2_l, y2_l = left_avg
            x1_r, y1_r, x2_r, y2_r = right_avg
            dy_l = y2_l - y1_l
            dy_r = y2_r - y1_r
            if dy_l != 0 and dy_r != 0:
                inv_m_l = (x2_l - x1_l) / dy_l
                inv_m_r = (x2_r - x1_r) / dy_r
                if abs(inv_m_l - inv_m_r) > 1e-4:
                    y_int = (inv_m_l * y1_l - x1_l - inv_m_r * y1_r + x1_r) / (inv_m_l - inv_m_r)
                    if 0 < y_int < self.height:
                        x_int = int(inv_m_l * (y_int - y1_l) + x1_l)
                        y_int = int(y_int)
                        if y_int < y1_l and y_int < y1_r:
                            left_avg = (x1_l, y1_l, x_int, y_int)
                            right_avg = (x1_r, y1_r, x_int, y_int)

        if left_avg:
            cv2.line(overlay, (left_avg[0],left_avg[1]), (left_avg[2],left_avg[3]), (0,255,0), 1)
        if right_avg:
            cv2.line(overlay, (right_avg[0],right_avg[1]), (right_avg[2],right_avg[3]), (0,255,0), 1)

        # YOLO -- pakai .track() agar track_id persistent antar frame
        # Ini kunci agar buffer LSTM bisa terisi 45 frame per kendaraan
        if should_run_yolo:
            try:
                use_half = bool(YOLO_HALF and self.device and self.device.startswith("cuda"))
                res = self.model.track(
                    frame,
                    imgsz      = YOLO_IMGSZ,
                    device     = self.device if self.device else None,
                    classes    = [2, 3, 5, 7],
                    conf       = YOLO_CONF,
                    iou        = YOLO_IOU,
                    verbose    = False,
                    persist    = True,            # track_id konsisten antar frame
                    tracker    = "bytetrack.yaml" # ByteTrack lebih stabil dari BoT-SORT
                )
                if res[0].boxes is not None and len(res[0].boxes) > 0:
                    boxes       = res[0].boxes.xyxy.cpu().numpy()
                    confidences = res[0].boxes.conf.cpu().numpy()
                    class_ids   = res[0].boxes.cls.cpu().numpy().astype(int)
                    # Ekstrak track_id dari ByteTrack -- None jika tracker belum assign
                    if res[0].boxes.id is not None:
                        track_ids = res[0].boxes.id.cpu().numpy().astype(int).tolist()
                    else:
                        track_ids = [None] * len(boxes)
                    self.last_known_boxes   = (boxes, confidences, class_ids, track_ids)
                    self.last_yolo_results  = self.last_known_boxes
                    self.missed_frames      = 0
                    self.consecutive_errors = 0
                else:
                    self.missed_frames = getattr(self, 'missed_frames', 0) + 1
                    if self.last_known_boxes and self.missed_frames <= 6:
                        boxes, confidences, class_ids, track_ids = self.last_known_boxes
                    else:
                        boxes, confidences, class_ids, track_ids = [], [], [], []
                    self.last_yolo_results = (boxes, confidences, class_ids, track_ids)
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"YOLO error frame {frame_idx}: {e}")
                if self.consecutive_errors > self.max_consecutive_errors:
                    raise
                boxes, confidences, class_ids, track_ids = (
                    self.last_known_boxes if self.last_known_boxes else ([], [], [], [])
                )
        else:
            boxes,confidences,class_ids,track_ids = (self.last_yolo_results or self.last_known_boxes or ([],[],[],[]))

        # --------------------------------------------------------
        # Loop deteksi
        # --------------------------------------------------------
        active_track_ids = set()
        lc_predictions   = {}

        for box, conf, cls_id, yolo_track_id in zip(boxes, confidences, class_ids, track_ids):
            if conf < YOLO_CONF: continue
            cn = self.model.names.get(cls_id, str(cls_id))
            if cn not in DETECT_CLASSES: continue
            self.stats['detections'] += 1

            x1,y1,x2,y2 = box
            cx = int((x1+x2)/2)
            cy = int(y2)
            lon_dist = calculate_longitudinal_distance(cy, frame_height)   # RAW: fitur BiLSTM & filter oncoming (JANGAN diubah)
            # Jarak TERKALIBRASI (penggaris lajur) utk keputusan/tampilan/BEV.
            if self._focal_px is None:
                self._focal_px = focal_px_from_hfov(self.width, CAMERA_HFOV_DEG)
            _zc = distance_from_lane_width(cy, left_avg, right_avg, self._focal_px)
            if _zc is not None and lon_dist > 0.1:
                ratio = float(np.clip(_zc / lon_dist, 0.3, 10.0))
                self._cal_factor = 0.98 * self._cal_factor + 0.02 * ratio   # EMA halus
                lon_dist_cal = _zc
            else:
                lon_dist_cal = float(np.clip(lon_dist * self._cal_factor, 0.5, 200.0))

            # Gunakan track_id dari ByteTrack jika tersedia,
            # fallback ke tracker internal untuk kendaraan yang belum di-assign
            self._cleanup_tracks(float(timestamp))
            if yolo_track_id is not None:
                tid = int(yolo_track_id)
                prev_track = self._tracks.get(tid)
            else:
                tid, prev_track = self._assign_track_id(cx, cy, float(timestamp))

            rel_speed = None
            if prev_track is not None and prev_track.get("t") is not None:
                dt = float(timestamp) - float(prev_track["t"])
                # rel_speed dari jarak TERKALIBRASI agar TTC = d/v konsisten skalanya.
                pdc = prev_track.get("dist_cal", prev_track.get("dist"))
                if dt > 1e-3 and pdc is not None:
                    rel_speed = (float(lon_dist_cal) - float(pdc)) / dt

            active_track_ids.add(tid)

            # ── [TRUNCATED] Cek apakah bbox terjepit ke tepi frame ──────────
            # is_truncated_now  : kondisi frame ini saja.
            # is_truncated      : is_truncated_now ATAU masih dalam cooldown
            #                     (bbox baru saja lepas dari tepi, posisinya
            #                     belum tentu stabil lagi).
            is_truncated_now = is_bbox_truncated(x1, x2, self.width)
            _prev_cd = self._truncated_cooldown.get(tid, 0)
            if is_truncated_now:
                self._truncated_cooldown[tid] = TRUNCATED_COOLDOWN_FRAMES
            elif _prev_cd > 0:
                self._truncated_cooldown[tid] = _prev_cd - 1
            is_truncated = bool(is_truncated_now or self._truncated_cooldown.get(tid, 0) > 0)

            offset_data = None
            if left_avg is not None and right_avg is not None:
                yref = int(self.height*0.8)
                def gx(line):
                    x1l,y1l,x2l,y2l=line
                    if y2l==y1l: return float(x1l)
                    return float(x1l+(yref-y1l)/(y2l-y1l)*(x2l-x1l))
                lx_r = gx(left_avg); rx_r = gx(right_avg)
                lw_lane = rx_r - lx_r
                if not np.isfinite(lw_lane): continue
                if lw_lane <= LANE_WIDTH_MIN_PX or lw_lane >= self.width*LANE_WIDTH_MAX_FRAC: continue
                if cx < lx_r-lw_lane or cx > rx_r+lw_lane: continue

                offset_data = calculate_lateral_offset(cx, cy, left_avg, right_avg)

                # Filter oncoming
                if FILTER_ONCOMING and offset_data is not None:
                    pd_dist = prev_track.get("dist") if prev_track else None
                    pd_t    = prev_track.get("t")    if prev_track else None
                    if pd_dist is not None and pd_t is not None:
                        dt = float(timestamp)-float(pd_t)
                        if dt > 1e-3:
                            cs = (float(pd_dist)-float(lon_dist))/dt
                            ln = float(abs(offset_data.get("offset_normalized",0)))
                            nc = offset_data.get("position")!="center" or ln>=ONCOMING_LATERAL_NORM_MIN
                            if cs >= ONCOMING_CLOSING_SPEED_MPS and nc:
                                self._tracks[tid] = {"cx":cx,"cy":cy,"t":float(timestamp),"dist":float(lon_dist),"dist_cal":float(lon_dist_cal)}
                                continue

                if FILTER_ONCOMING:
                    self._tracks[tid] = {"cx":cx,"cy":cy,"t":float(timestamp),"dist":float(lon_dist),"dist_cal":float(lon_dist_cal)}

                if offset_data is not None:
                    self.stats['offset_calculated'] += 1
                    objects.append({
                        "track_id": int(tid),
                        "position": offset_data['position'],
                        "distance": float(lon_dist_cal),
                        "rel_speed": float(rel_speed) if rel_speed is not None else None,
                        "offset_m": float(offset_data['offset_m']),
                        "offset_normalized": float(offset_data['offset_normalized']),
                        "lc_predicted_class": None,
                        "lc_confidence": None,
                        "lc_is_uncertain": None,
                    })

            # Update track history unconditionally (for rel_speed / closing speed)
            self._tracks[tid] = {"cx": cx, "cy": cy, "t": float(timestamp), "dist": float(lon_dist), "dist_cal": float(lon_dist_cal)}

            # [LSTM] Update buffer dan prediksi
            od = offset_data

            # [TRUNCATED] Saat bbox terjepit tepi frame, offset_normalized /
            # lane_width_px yang dihitung dari cx TIDAK bisa dipercaya --
            # jangan diberikan ke predictor sebagai observasi lateral yang
            # valid (kirim None = "celah data", pola yang sudah didukung
            # predictor untuk kasus od is None). longitudinal_distance_m &
            # timestamp tetap dikirim agar buffer waktu tidak bolong.
            if is_truncated:
                lc_result = self.lc_predictor.update_and_predict(
                    track_id                = int(tid),
                    offset_normalized       = None,
                    lane_width_px           = None,
                    longitudinal_distance_m = lon_dist,
                    timestamp               = timestamp,
                )
            else:
                lc_result = self.lc_predictor.update_and_predict(
                    track_id                = int(tid),
                    offset_normalized       = od['offset_normalized'] if od else None,
                    lane_width_px           = od['lane_width_px']     if od else None,
                    longitudinal_distance_m = lon_dist,
                    timestamp               = timestamp,
                )

            # [TRUNCATED] Jaga-jaga tambahan: walau predictor masih sempat
            # menghasilkan prediksi dari sisa buffer sebelumnya, paksa jadi
            # "uncertain" selama bbox dianggap terpotong -- supaya label
            # lane-change TIDAK ditampilkan/dipercaya untuk kondisi ini
            # (lihat gate `not lc_r['is_uncertain']` di bagian rendering &
            # bounding-box-color di bawah).
            if lc_result is not None and is_truncated:
                lc_result = dict(lc_result)
                lc_result["is_uncertain"] = True

            if lc_result:
                lc_predictions[tid] = lc_result

            # Attach BI-LSTM prediction to the latest decision-object for this track
            if lc_result is not None and objects:
                try:
                    if objects[-1].get("track_id") == int(tid):
                        objects[-1]["lc_predicted_class"] = lc_result.get("predicted_class")
                        objects[-1]["lc_confidence"] = lc_result.get("confidence")
                        objects[-1]["lc_is_uncertain"] = lc_result.get("is_uncertain")
                except Exception:
                    pass

            # BEV op -- dikumpulkan dulu, digambar setelah highlight.
            # bx (lateral) dari map_to_birdseye; by (kedalaman) dari jarak
            # TERKALIBRASI agar SEJAJAR dengan gridline jarak BEV (d/100*H).
            if od is not None and abs(od['offset_m']) <= 5.0:
                bx, _ = map_to_birdseye(cx, cy, od['offset_normalized'], self.height, od['position'])
                by = int(np.clip(bev_y_from_distance(lon_dist_cal), 12, BEV_HEIGHT - 1))
                in_d   = by > (BEV_HEIGHT - BEV_DANGER_ZONE)
                bev_ops.append((bx, by, od['position'], in_d, tid, lc_predictions.get(tid)))

            # Visualisasi bounding box
            lc_r = lc_predictions.get(tid)
            if lc_r and not lc_r['is_uncertain'] and lc_r['predicted_class'] != 0:
                box_color = LABEL_COLORS[lc_r['predicted_class']]
            elif od:
                box_color = {'center':(0,255,0),'right_side':(0,165,255),'left_side':(255,0,255)}.get(od['position'],(128,128,128))
            else:
                box_color = (128,128,128)

            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), box_color, 2)
            cv2.circle(overlay, (cx,cy), 5, (0,0,255), -1)

            # Track ID di pojok kiri atas bounding box
            id_text = f"ID:{tid}"
            (tw_id, th_id), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty_id = max(th_id + 4, int(y1) - 2)
            cv2.rectangle(overlay, (int(x1), ty_id - th_id - 3),
                          (int(x1) + tw_id + 4, ty_id + 2), (0,0,0), -1)
            cv2.putText(overlay, id_text, (int(x1)+2, ty_id),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # [TRUNCATED] Indikator visual kecil supaya kondisi ini terlihat
            # jelas saat debugging (bbox dianggap terpotong tepi frame ->
            # prediksi lane-change untuk kendaraan ini sedang ditahan).
            if is_truncated:
                edge_text = "EDGE"
                (tw_e, th_e), _ = cv2.getTextSize(edge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                ex = int(x1) + tw_id + 8
                cv2.rectangle(overlay, (ex, ty_id - th_e - 3), (ex + tw_e + 4, ty_id + 2), (0,0,0), -1)
                cv2.putText(overlay, edge_text, (ex + 2, ty_id),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,140,255), 1)

            # [LSTM] Label prediksi di atas bounding box
            if lc_r and not lc_r['is_uncertain'] and lc_r['predicted_class'] != 0:
                lt = f"{lc_r['predicted_label']} {lc_r['confidence']:.0%}"
                (tw,th),_ = cv2.getTextSize(lt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                tx = int(x1); ty = max(20, int(y1)-25)
                cv2.rectangle(overlay, (tx, ty-th-4), (tx+tw+4, ty+2), (0,0,0), -1)
                cv2.putText(overlay, lt, (tx+2,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Info lat/lon
            if od:
                cv2.putText(overlay, f"Lat:{od['offset_m']:+.2f}m", (cx+10,cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                cv2.putText(overlay, f"Lon:{lon_dist_cal:.1f}m",     (cx+10,cy-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255),   1)
                cv2.putText(overlay, od['position'],                  (cx+10,cy-46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color,     1)
            else:
                cv2.putText(overlay, f"Lon:{lon_dist_cal:.1f}m", (cx+10,cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                cv2.putText(overlay, "Lat: N/A",              (cx+10,cy-28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            # CSV
            row = {
                "frame":int(frame_idx),"timestamp":float(timestamp),"fps":round(current_fps,2),
                "track_id":int(tid),"class":cn,"conf":float(conf),
                "x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
                "cx":int(cx),"cy":int(cy),"longitudinal_distance_m":round(lon_dist,2),
                # [LSTM] kolom prediksi
                "lc_predicted_class" : lc_r['predicted_class']  if lc_r else None,
                "lc_predicted_label" : lc_r['predicted_label']  if lc_r else None,
                "lc_confidence"      : lc_r['confidence']        if lc_r else None,
                "lc_is_uncertain"    : lc_r['is_uncertain']      if lc_r else None,
                "lc_frames_buffered" : lc_r['frames_buffered']   if lc_r else None,
                # [TRUNCATED] kolom evaluasi: True bila bbox dianggap
                # terjepit tepi frame pada baris ini (lihat is_bbox_truncated)
                "bbox_truncated"     : bool(is_truncated),
            }
            if od:
                row.update({"lane_left_x":od['lane_left_x'],"lane_right_x":od['lane_right_x'],
                            "lane_center_x":od['lane_center_x'],"lane_width_px":od['lane_width_px'],
                            "offset_px":od['offset_px'],"offset_m":od['offset_m'],
                            "offset_normalized":od['offset_normalized'],"position":od['position']})
            else:
                row.update({k:None for k in ["lane_left_x","lane_right_x","lane_center_x","lane_width_px",
                                              "offset_px","offset_m","offset_normalized","position"]})
            self.csv_writer.add_row(row)

        # [LSTM] Cleanup buffer SETELAH loop selesai (fix bug track 187)
        self.lc_predictor.cleanup_stale_tracks(active_track_ids)

        # [TRUNCATED] Buang entri cooldown milik track yang sudah tidak aktif
        # agar dict tidak membesar terus sepanjang video.
        if self._truncated_cooldown:
            for _tid in list(self._truncated_cooldown.keys()):
                if _tid not in active_track_ids:
                    del self._truncated_cooldown[_tid]

        # Ego position & recommendation
        ego_offset = compute_ego_offset(left_avg, right_avg, self.width, self.height)

        # (Blok deteksi posisi ego identik dengan aslinya -- dikopi langsung)
        rightmost_evidence = None
        leftmost_evidence  = None
        # Inisialisasi sinyal batas lajur agar SELALU terdefinisi (dipakai untuk
        # fail-safe lane availability di bawah, walau salah satu garis hilang).
        yellow_right = False
        yellow_edge_r = yellow_edge_l = False
        white_edge_r  = white_edge_l  = False
        outer_r = outer_l = None
        right_style = left_style = None
        right_style_s = left_style_s = None        # versi STRING ternormalisasi
        right_style_score = left_style_score = None
        extra_r = extra_l = None
        if left_avg is not None and right_avg is not None:
            yref = int(self.height*0.8)
            def gx2(line):
                x1l,y1l,x2l,y2l=line
                if y2l==y1l: return float(x1l)
                return float(x1l+(yref-y1l)/(y2l-y1l)*(x2l-x1l))
            lx2 = gx2(left_avg); rx2 = gx2(right_avg)
            if rx2 < lx2: lx2,rx2 = rx2,lx2
            lw2 = rx2-lx2
            lw_valid = 40 <= lw2 <= self.width*0.9
            yellow_right = detect_yellow_on_right_lane(frame, right_avg, self.height)
            yellow_edge_r = detect_yellow_on_edge(frame, side="right", frame_height=self.height)
            yellow_edge_l = detect_yellow_on_edge(frame, side="left",  frame_height=self.height)
            white_edge_r  = detect_white_on_edge(frame, side="right", frame_height=self.height)
            white_edge_l  = detect_white_on_edge(frame, side="left",  frame_height=self.height)

            # Solid/dashed + outer boundary inference from Hough segments
            outer_r = select_outer_boundary_from_segments(right_lines, side="right", yref=yref, edge_img=me_full)
            outer_l = select_outer_boundary_from_segments(left_lines,  side="left",  yref=yref, edge_img=me_full)

            right_style = classify_line_style_from_edges(right_avg, me_full)
            left_style  = classify_line_style_from_edges(left_avg,  me_full)
            # NORMALISASI: classify_line_style_from_edges mengembalikan DICT
            # {'style','score',...}. Ekstrak string + score agar perbandingan
            # '== solid/dashed' di hilir benar (bug sebelumnya: dict != "solid").
            right_style_s = (right_style or {}).get("style") if isinstance(right_style, dict) else right_style
            left_style_s  = (left_style  or {}).get("style") if isinstance(left_style,  dict) else left_style
            right_style_score = (right_style or {}).get("score") if isinstance(right_style, dict) else None
            left_style_score  = (left_style  or {}).get("score") if isinstance(left_style,  dict) else None
            extra_r, extra_l = None, None
            if lw_valid:
                # Detect additional lane markings beyond the current lane boundaries.
                # Use ALL line segments and a less aggressive threshold to reduce false RIGHT_LANE locks.
                all_segs = (left_lines or []) + (right_lines or [])
                rth = rx2 + 0.35*lw2
                lth = lx2 - 0.35*lw2
                rh = sum(1 for s in all_segs if (get_x_at_y((*s[:2],*s[2:]), yref) or -1e9) > rth)
                lh2= sum(1 for s in all_segs if (get_x_at_y((*s[:2],*s[2:]), yref) or  1e9) < lth)
                extra_r = rh >= 2
                extra_l = lh2 >= 2
            if lw_valid:
                # If we can detect an outer boundary, decide rightmost/leftmost by alignment:
                # In the rightmost lane, right_avg should align with the outer boundary (often solid).
                tol = 0.25 * float(lw2)
                if outer_r is not None and outer_r.get("x") is not None and outer_r.get("style") == "solid":
                    align_r = abs(float(rx2) - float(outer_r["x"])) <= tol
                    if align_r and (yellow_right or yellow_edge_r or white_edge_r or float(outer_r["x"]) >= self.width*0.92):
                        rightmost_evidence = True
                    elif not align_r:
                        rightmost_evidence = False

                if outer_l is not None and outer_l.get("x") is not None and outer_l.get("style") == "solid":
                    align_l = abs(float(lx2) - float(outer_l["x"])) <= tol
                    if align_l and (yellow_edge_l or white_edge_l or float(outer_l["x"]) <= self.width*0.08):
                        leftmost_evidence = True
                    elif not align_l:
                        leftmost_evidence = False

                if extra_r is True:
                    rightmost_evidence = False
                elif (rightmost_evidence is None) and (yellow_right or yellow_edge_r or white_edge_r):
                    rightmost_evidence = True
                rm_r = (self.width-rx2)/lw2 if lw2>0 else 1
                lm_l = lx2/lw2 if lw2>0 else 1
                if rightmost_evidence is None:
                    rightmost_evidence = rm_r<=0.20 or rx2>=self.width*0.94
                if leftmost_evidence is None:
                    leftmost_evidence  = (lm_l<=0.20 or lx2<=self.width*0.06 or yellow_edge_l or white_edge_l)
            else:
                if extra_r is False or yellow_right or yellow_edge_r or white_edge_r: rightmost_evidence = True
                elif extra_r is True: rightmost_evidence = False

        # Counter-evidence from detections: if we can see vehicles on the side,
        # that strongly suggests there IS an adjacent lane.
        # Apply this BEFORE streak/lock logic to avoid false RIGHT_LANE/LEFT_LANE when markings are occluded.
        if any((o.get("position") == "right_side") for o in objects):
            rightmost_evidence = False
        if any((o.get("position") == "left_side") for o in objects):
            leftmost_evidence = False

        # Evidence hold: if evidence momentarily disappears (None), reuse the last non-None value
        # for a few frames to prevent ego lane-state flicker.
        hold_n = int(self.decision_params.get("EGO_EVIDENCE_HOLD_FRAMES", 5) or 5)
        if rightmost_evidence is None:
            if self._rightmost_last is not None and int(self._rightmost_last_age) < hold_n:
                rightmost_evidence = self._rightmost_last
                self._rightmost_last_age += 1
        else:
            self._rightmost_last = bool(rightmost_evidence)
            self._rightmost_last_age = 0

        if leftmost_evidence is None:
            if self._leftmost_last is not None and int(self._leftmost_last_age) < hold_n:
                leftmost_evidence = self._leftmost_last
                self._leftmost_last_age += 1
        else:
            self._leftmost_last = bool(leftmost_evidence)
            self._leftmost_last_age = 0

        LOCK_F = int(self.decision_params.get("EGO_LOCK_FRAMES", 4) or 4)
        UNLOCK_F = int(self.decision_params.get("EGO_UNLOCK_FRAMES", 12) or 12)
        if rightmost_evidence is True:
            self._rightmost_true_streak  += 1; self._rightmost_false_streak = 0
        elif rightmost_evidence is False:
            self._rightmost_false_streak += 1; self._rightmost_true_streak  = 0
        if leftmost_evidence is True:
            self._leftmost_true_streak  += 1; self._leftmost_false_streak = 0
        elif leftmost_evidence is False:
            self._leftmost_false_streak += 1; self._leftmost_true_streak  = 0

        if self.ego_position_state == "RIGHT_LANE":
            if rightmost_evidence is False and self._rightmost_false_streak >= UNLOCK_F:
                self.ego_position_state = "CENTER"
        else:
            if rightmost_evidence is True and self._rightmost_true_streak >= LOCK_F:
                self.ego_position_state = "RIGHT_LANE"
        if self.ego_position_state == "LEFT_LANE":
            if leftmost_evidence is False and self._leftmost_false_streak >= UNLOCK_F:
                self.ego_position_state = "CENTER"
        else:
            if leftmost_evidence is True and self._leftmost_true_streak >= LOCK_F:
                self.ego_position_state = "LEFT_LANE"
        if rightmost_evidence is True and leftmost_evidence is True:
            self.ego_position_state = "CENTER"

        ego_position = self.ego_position_state
        # Suppress BEV right-side (visual-only) dipindah ke SETELAH koreksi label
        # + estimator (lihat di bawah) agar konsisten dgn posisi ego frame ini.
        suppressed_bev_right = 0

        # ──────────────────────────────────────────────────────────────────
        # Stage 4: Update bracket kecepatan adaptif berdasarkan kondisi
        # lalu lintas, dengan smoothing untuk hindari osilasi antar bracket.
        # ──────────────────────────────────────────────────────────────────
        new_bracket = estimate_speed_bracket(objects, self._speed_bracket_current)
        if new_bracket == self._speed_bracket_current:
            self._speed_bracket_streak += 1
        else:
            self._speed_bracket_streak = 1
        # Hanya pindah bracket setelah 10 frame konsisten (smoothing)
        if self._speed_bracket_streak >= 10 and new_bracket != self._speed_bracket_current:
            self._speed_bracket_current = new_bracket
            self._speed_bracket_streak = 0

        # Inject parameter dari bracket aktif ke decision_params (override base)
        bracket_params = self.speed_brackets[self._speed_bracket_current]
        adaptive_params = dict(self.decision_params)
        adaptive_params.update(bracket_params)

        # ──────────────────────────────────────────────────────────────────
        # Stage 2: Update rolling history kendaraan per sisi
        # ──────────────────────────────────────────────────────────────────
        n_left_now  = sum(1 for o in objects if o.get("position") == "left_side")
        n_right_now = sum(1 for o in objects if o.get("position") == "right_side")
        self._left_vehicle_history.append(n_left_now)
        self._right_vehicle_history.append(n_right_now)
        if len(self._left_vehicle_history) > self._VEHICLE_HISTORY_LEN:
            self._left_vehicle_history.pop(0)
        if len(self._right_vehicle_history) > self._VEHICLE_HISTORY_LEN:
            self._right_vehicle_history.pop(0)

        # Rolling history ego_offset_px (untuk sinyal ke-4 di is_likely_at_edge)
        self._ego_offset_history.append(float(ego_offset))
        if len(self._ego_offset_history) > self._OFFSET_HISTORY_LEN:
            self._ego_offset_history.pop(0)

        # Baseline panjang + offset terkoreksi-bias untuk cek 'ego_unstable'.
        self._ego_offset_long.append(float(ego_offset))
        if len(self._ego_offset_long) > self._EGO_OFFSET_LONG_LEN:
            self._ego_offset_long.pop(0)
        offset_baseline = float(np.median(self._ego_offset_long)) if self._ego_offset_long else 0.0
        ego_offset_corr = float(ego_offset) - offset_baseline

        # ──────────────────────────────────────────────────────────────────
        # Fail-safe lane availability: tentukan apakah lajur kiri/kanan benar2
        # ADA & boleh dilintasi sebelum rekomendasi pindah. Pakai sinyal yang
        # sudah dihitung (tipe garis, marka tambahan) + riwayat kendaraan sisi.
        # Biaya komputasi ~0 -> aman untuk Jetson Nano.
        # ──────────────────────────────────────────────────────────────────
        # Bukti kendaraan lajur-tetangga yang KETAT: hanya kendaraan SEARAH &
        # DEKAT pada frame ini (menyaring lalu lintas seberang median yang jauh /
        # closing cepat). Pakai hold pendek agar tidak flicker.
        adj_max_dist    = float(self.decision_params.get("ADJ_LANE_MAX_DIST", 45.0))
        adj_max_closing = float(self.decision_params.get("ADJ_LANE_MAX_CLOSING", 7.0))
        def _adj_vehicle_now(side_pos):
            for o in objects:
                if o.get("position") != side_pos:
                    continue
                d = o.get("distance")
                if d is None:
                    continue
                try:
                    if abs(float(d)) > adj_max_dist:
                        continue
                except Exception:
                    continue
                v = o.get("rel_speed")
                if v is not None:
                    try:
                        if float(v) < -adj_max_closing:   # closing cepat -> kemungkinan berlawanan arah
                            continue
                    except Exception:
                        pass
                return True
            return False
        adj_hold = int(self.decision_params.get("ADJ_VEHICLE_HOLD_FRAMES", 8) or 8)
        self._right_adjveh_hold = adj_hold if _adj_vehicle_now("right_side") else max(0, self._right_adjveh_hold - 1)
        self._left_adjveh_hold  = adj_hold if _adj_vehicle_now("left_side")  else max(0, self._left_adjveh_hold - 1)
        right_vehicle_seen = self._right_adjveh_hold > 0
        left_vehicle_seen  = self._left_adjveh_hold  > 0
        right_adj_now = _adj_vehicle_now("right_side")
        left_adj_now  = _adj_vehicle_now("left_side")

        # ──────────────────────────────────────────────────────────────────
        # Estimasi indeks lajur ego (fusi klasik + temporal, sticky/anti-jitter).
        # Outputnya MENGGANTI ego_position dari state-machine lama yang sering
        # nyangkut di CENTER. State-machine lama tetap dipakai sbg salah satu
        # isyarat (lewat rightmost/leftmost_evidence di edge-lock di bawah).
        # ──────────────────────────────────────────────────────────────────
        ego_belief, ego_conf = self.ego_lane_estimator.update(
            left_style=left_style_s, right_style=right_style_s,
            left_style_score=left_style_score, right_style_score=right_style_score,
            yellow_left=bool(yellow_edge_l),
            yellow_right=bool(yellow_right or yellow_edge_r),
            left_present=(left_avg is not None),
            right_present=(right_avg is not None),
            left_adj_vehicle=left_adj_now, right_adj_vehicle=right_adj_now,
            fps=fps_for_adapt)
        ego_position = ego_belief
        self.ego_position_state = ego_belief  # sinkronkan agar konsisten di hilir

        # Indeks lajur ego (0=kiri,1=tengah,2=kanan), murni dari posisi ego.
        ego_lane_index = {"RIGHT_LANE": 2, "LEFT_LANE": 0}.get(ego_position, 1)
        self._ego_lane_index = ego_lane_index

        # Koreksi label sisi: saat ego di lajur paling kanan/kiri, kendaraan
        # 'sisi' yang lateralnya kecil sebenarnya SATU LAJUR dgn ego -> 'center'.
        # Dilakukan SEBELUM lane_change_decision & BEV supaya keputusan (front_c
        # untuk overtake) dan tampilan ikut terkoreksi.
        fixed_ids = correct_side_labels_for_edge_lane(
            objects, ego_lane_index,
            same_lane_lat_m=float(self.decision_params.get("SAME_LANE_LAT_M", 2.2)))
        if fixed_ids and bev_ops:
            bev_ops = [
                ((BEV_WIDTH // 2 if btid in fixed_ids else bx), by,
                 ("center" if btid in fixed_ids else pos), ind, btid, blc)
                for (bx, by, pos, ind, btid, blc) in bev_ops
            ]
        # Visual-only: sisa objek 'right_side'/'left_side' di sisi yang TAK ada
        # lajurnya (mis. lalu lintas seberang median) disembunyikan dari BEV.
        if bev_ops and ego_lane_index in (0, 2):
            drop = "right_side" if ego_lane_index == 2 else "left_side"
            before = len(bev_ops)
            bev_ops = [op for op in bev_ops if op[2] != drop]
            suppressed_bev_right = before - len(bev_ops)

        # Edge-lock latch: HANYA dipicu oleh BELIEF estimator (ego benar2 di lajur
        # pinggir). Sebelumnya ikut dipicu right_style=="solid"/kuning sehingga
        # mengunci sisi kanan ~93% frame walau ego di lajur tengah -> RIGHT nyaris
        # tak pernah tersedia. Estimator sudah memfusikan style/kuning untuk
        # menentukan RIGHT_LANE, jadi cukup ikut belief-nya.
        edge_lock_f = int(self.decision_params.get("EDGE_LOCK_FRAMES", 75) or 75)
        right_edge_now = (ego_position == "RIGHT_LANE") or (rightmost_evidence is True)
        left_edge_now  = (ego_position == "LEFT_LANE")  or (leftmost_evidence is True)
        self._right_edge_lock = edge_lock_f if right_edge_now else max(0, self._right_edge_lock - 1)
        self._left_edge_lock  = edge_lock_f if left_edge_now  else max(0, self._left_edge_lock - 1)

        lane_avail = {
            "right": compute_lane_availability(
                "right",
                yellow_edge=bool(yellow_right or yellow_edge_r),
                boundary_style=right_style_s,
                outer_style=(outer_r or {}).get("style") if outer_r else None,
                extra_markings=extra_r,
                at_edge_evidence=rightmost_evidence,
                ego_position=ego_position,
                vehicle_seen=right_vehicle_seen,
                edge_locked=self._right_edge_lock > 0),
            "left": compute_lane_availability(
                "left",
                yellow_edge=bool(yellow_edge_l),
                boundary_style=left_style_s,
                outer_style=(outer_l or {}).get("style") if outer_l else None,
                extra_markings=extra_l,
                at_edge_evidence=leftmost_evidence,
                ego_position=ego_position,
                vehicle_seen=left_vehicle_seen,
                edge_locked=self._left_edge_lock > 0),
        }

        # ──────────────────────────────────────────────────────────────────
        # Pemanggilan logika rekomendasi dengan parameter adaptif
        # ──────────────────────────────────────────────────────────────────
        desired, force_stay, decision_debug = lane_change_decision(
            objects, ego_offset_corr, adaptive_params, self.decision_buffer, ego_position,
            lane_avail=lane_avail)
        decision = stabilize_recommendation(
            desired, self.decision_buffer, adaptive_params, force_immediate_stay=force_stay)

        # ──────────────────────────────────────────────────────────────────
        # Stage 2: Validasi akhir dengan multi-signal (BUKAN hanya ego_position).
        # Sistem dianggap "di pinggir" jika minimal 2 dari 3 sinyal aktif:
        #   1. ego_position state machine
        #   2. leftmost/rightmost_evidence frame ini
        #   3. absensi kendaraan di sisi tsb selama window panjang
        # ──────────────────────────────────────────────────────────────────
        likely_at_right = is_likely_at_edge(
            "right", ego_position, rightmost_evidence,
            vehicle_history=self._right_vehicle_history,
            offset_history=self._ego_offset_history,
            opposite_history=self._left_vehicle_history)
        likely_at_left = is_likely_at_edge(
            "left", ego_position, leftmost_evidence,
            vehicle_history=self._left_vehicle_history,
            offset_history=self._ego_offset_history,
            opposite_history=self._right_vehicle_history)
        if decision == "RIGHT" and likely_at_right:
            decision = "STAY"
        if decision == "LEFT" and likely_at_left:
            decision = "STAY"

        # GATE KERAS: keputusan akhir tidak boleh mengarah ke sisi yang tidak
        # tersedia. Ini menangkap kasus stabilizer masih MENAHAN RIGHT/LEFT lama
        # walau availability sudah berubah jadi False (sumber sisa kebocoran).
        _ra = bool((lane_avail.get("right") or {}).get("available", True))
        _la = bool((lane_avail.get("left")  or {}).get("available", True))
        if decision == "RIGHT" and not _ra:
            decision = "STAY"
            self.decision_buffer["stable"] = "STAY"
            self.decision_buffer["candidate"] = None
            self.decision_buffer["count"] = 0
            self.decision_buffer["hold"] = 0
        if decision == "LEFT" and not _la:
            decision = "STAY"
            self.decision_buffer["stable"] = "STAY"
            self.decision_buffer["candidate"] = None
            self.decision_buffer["count"] = 0
            self.decision_buffer["hold"] = 0

        # Indeks lajur ego untuk BEV (model 3 lajur: 0=kiri,1=tengah,2=kanan),
        # MURNI dari posisi lajur ego hasil estimator (bukan dari availability),
        # supaya pewarnaan BEV mencerminkan "ego di lajur mana" apa adanya.

        # Catat info adaptif untuk logging/debugging
        if decision_debug is not None:
            decision_debug["speed_bracket"] = self._speed_bracket_current
            decision_debug["likely_at_left"] = bool(likely_at_left)
            decision_debug["likely_at_right"] = bool(likely_at_right)

        # [EVAL] Catat satu baris keputusan per-frame untuk bahan evaluasi model rekomendasi
        if self.decision_csv_writer is not None:
            di = decision_debug or {}
            db = self.decision_buffer or {}
            decision_row = {
                # Identifier
                "frame": int(frame_idx), "timestamp": float(timestamp), "fps": round(current_fps, 2),
                # Keputusan
                "desired": desired, "decision": decision, "force_stay": bool(force_stay),
                # Konteks ego
                "ego_position": ego_position, "ego_offset_px": float(ego_offset),
                # Pemicu cut-in
                "cutin_from_left": di.get("cutin_from_left"), "cutin_from_right": di.get("cutin_from_right"),
                # Pemicu overtake
                "need_overtake": di.get("need_overtake"),
                # Ketersediaan lajur (fail-safe gating)
                "left_available": di.get("left_available"), "right_available": di.get("right_available"),
                "right_lane_exists": (lane_avail.get("right") or {}).get("lane_exists"),
                "left_lane_exists":  (lane_avail.get("left")  or {}).get("lane_exists"),
                "right_edge_locked": (lane_avail.get("right") or {}).get("edge_locked"),
                "left_edge_locked":  (lane_avail.get("left")  or {}).get("edge_locked"),
                # Estimator indeks lajur ego (audit fusi klasik+temporal)
                "ego_lane_index": self._ego_lane_index,
                "ego_lane_conf": round(float(ego_conf), 3),
                "ego_r_acc": round(float(self.ego_lane_estimator.r_acc), 3),
                "ego_l_acc": round(float(self.ego_lane_estimator.l_acc), 3),
                "left_style": left_style_s, "right_style": right_style_s,
                "right_style_score": right_style_score, "left_style_score": left_style_score,
                "yellow_right": bool(yellow_right or yellow_edge_r),
                "left_adj_veh": bool(left_adj_now), "right_adj_veh": bool(right_adj_now),
                # Status keamanan
                "left_safe": di.get("left_safe"), "right_safe": di.get("right_safe"),
                "cand_left": di.get("cand_left"), "cand_right": di.get("cand_right"),
                # Data primitif (input rule simplifikasi)
                "n_left": di.get("n_left"), "n_center": di.get("n_center"), "n_right": di.get("n_right"),
                "d_center": di.get("d_center"), "d_left": di.get("d_left"), "d_right": di.get("d_right"),
                # Improvement (tie-breaking)
                "improve_left": di.get("improve_left"), "improve_right": di.get("improve_right"),
                "min_improvement": di.get("min_improvement"),
                # Bukti lajur (validasi posisi pinggir)
                "rightmost_evidence": rightmost_evidence, "leftmost_evidence": leftmost_evidence,
                # State stabilisasi (flicker & timing)
                "stable": db.get("stable"), "candidate": db.get("candidate"),
                "confirm_count": db.get("count"), "hold": db.get("hold"),
            }
            self.decision_csv_writer.add_row(decision_row)

        # BEV: highlight rekomendasi (gaya semula) + posisi ego & blok sisi (merah)
        apply_bev_recommendation_highlight(bev, decision)
        draw_bev_lane_state(bev, ego_lane_index, _la, _ra, decision)
        for bx, by, pos, ind, btid, blc in bev_ops:
            draw_vehicle_on_birdseye(
                bev, bx, by, pos, in_danger_zone=ind,
                lc_class       = blc['predicted_class'] if blc else None,
                lc_confidence  = blc['confidence']       if blc else 0.0,
                lc_is_uncertain= blc['is_uncertain']     if blc else True,
                blink_visible  = self.blink_visible,
                blink_phase    = self.blink_phase,
                track_id       = btid,
            )

        draw_recommendation(overlay, decision)
        if DEBUG_DECISION:
            try:
                if decision_debug is not None:
                    decision_debug.update({
                        "suppressed_bev_right": int(suppressed_bev_right),
                        "rightmost_evidence": rightmost_evidence,
                        "leftmost_evidence": leftmost_evidence,
                        "yellow_right": bool(yellow_right) if 'yellow_right' in locals() else None,
                        "yellow_edge_r": bool(yellow_edge_r) if 'yellow_edge_r' in locals() else None,
                        "yellow_edge_l": bool(yellow_edge_l) if 'yellow_edge_l' in locals() else None,
                        "white_edge_r": bool(white_edge_r) if 'white_edge_r' in locals() else None,
                        "white_edge_l": bool(white_edge_l) if 'white_edge_l' in locals() else None,
                        "lane_width_px": float(lw2) if 'lw2' in locals() else None,
                        "rx2": float(rx2) if 'rx2' in locals() else None,
                        "lx2": float(lx2) if 'lx2' in locals() else None,
                        "right_style": (right_style.get('style') if 'right_style' in locals() and right_style else None),
                        "left_style": (left_style.get('style') if 'left_style' in locals() and left_style else None),
                        "outer_rx2": (float(outer_r.get('x')) if 'outer_r' in locals() and outer_r and outer_r.get('x') is not None else None),
                        "outer_lx2": (float(outer_l.get('x')) if 'outer_l' in locals() and outer_l and outer_l.get('x') is not None else None),
                        "outer_r_style": (outer_r.get('style') if 'outer_r' in locals() and outer_r else None),
                        "outer_l_style": (outer_l.get('style') if 'outer_l' in locals() and outer_l else None),
                        "align_r": (abs(float(rx2)-float(outer_r.get('x'))) <= 0.25*float(lw2) if 'outer_r' in locals() and outer_r and outer_r.get('x') is not None and 'lw2' in locals() and lw2 else None),
                        "align_l": (abs(float(lx2)-float(outer_l.get('x'))) <= 0.25*float(lw2) if 'outer_l' in locals() and outer_l and outer_l.get('x') is not None and 'lw2' in locals() and lw2 else None),
                        "extra_r": extra_r if 'extra_r' in locals() else None,
                        "extra_l": extra_l if 'extra_l' in locals() else None,
                    })
            except Exception:
                pass
            draw_decision_debug(
                overlay,
                decision_debug,
                desired=desired,
                stable=self.decision_buffer.get("stable", decision),
                ego_position=ego_position,
            )

        # Dashboard
        avg_fps       = self.perf_monitor.get_average_fps()
        lane_detected = left_avg is not None and right_avg is not None
        det_count     = len(boxes) if boxes is not None else 0

        if not self.use_dashboard:
            return overlay, overlay, bev

        self.combined_canvas[:] = 0
        self.combined_canvas[0:self.height, 0:self.width] = overlay
        bev_s = cv2.resize(bev, (self.target_bev_width, self.height))
        self.combined_canvas[0:self.height, self.width:self.width+self.target_bev_width] = bev_s
        cv2.line(self.combined_canvas, (self.width,0), (self.width,self.height), (80,80,80), 1)

        sy = self.height
        cv2.line(self.combined_canvas, (0,sy), (self.dashboard_width,sy), (80,80,80), 1)
        cv2.rectangle(self.combined_canvas, (0,sy), (self.dashboard_width,self.dashboard_height), (25,25,25), -1)

        W  = self.dashboard_width
        sw = W//4   # 4 section
        ty = sy+50
        dv = (60,60,60); mg = 15

        def stat(label, value, x, color=(200,200,200)):
            cv2.putText(self.combined_canvas, label, (x+20,sy+25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
            cv2.putText(self.combined_canvas, str(value), (x+20,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 1)

        fc = (0,255,255) if avg_fps>=25 else (0,165,255) if avg_fps>=15 else (0,0,255)
        stat("FPS",      f"{avg_fps:.1f}",                     0,     fc)
        cv2.line(self.combined_canvas,(sw,sy+mg),(sw,self.dashboard_height-mg),dv,1)
        stat("LANES",    "OK" if lane_detected else "LOST",    sw,    (0,255,0) if lane_detected else (0,0,255))
        cv2.line(self.combined_canvas,(sw*2,sy+mg),(sw*2,self.dashboard_height-mg),dv,1)
        stat("VEHICLES", str(det_count),                        sw*2,  (200,200,200))
        cv2.line(self.combined_canvas,(sw*3,sy+mg),(sw*3,self.dashboard_height-mg),dv,1)
        # [LSTM] Panel ke-4: jumlah lane change alert
        n_alert = sum(1 for r in lc_predictions.values() if r and r['predicted_class']!=0 and not r['is_uncertain'])
        ac = (0,100,255) if n_alert>0 else (100,200,100)
        stat("LC ALERT", f"{n_alert} vehicle(s)", sw*3, ac)

        return self.combined_canvas, overlay, bev

    def run(self):
        logger.info("Starting pipeline...")
        SHUTDOWN_FLAG.clear()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        reader = threading.Thread(target=frame_reader,     args=(self.cap, FRAME_QUEUE, self.raw_writer, self.raw_size), daemon=True)
        proc_t = threading.Thread(target=processor_thread, args=(self, FRAME_QUEUE, RESULT_QUEUE), daemon=True)
        reader.start(); proc_t.start()
        try:
            consume_results_main_thread(RESULT_QUEUE, self.async_writer)
        except KeyboardInterrupt:
            SHUTDOWN_FLAG.set()
        SHUTDOWN_FLAG.set()
        try:
            reader.join(timeout=2); proc_t.join(timeout=2)
        except Exception:
            pass
        self._drain_queues()
        self.cleanup()

    def _drain_queues(self):
        for q in [FRAME_QUEUE, RESULT_QUEUE]:
            while not q.empty():
                try: q.get_nowait()
                except: break

    def cleanup(self):
        if self.cap:          self.cap.release()
        if self.async_writer: self.async_writer.release()
        if getattr(self, 'raw_writer', None): self.raw_writer.release()
        if self.csv_writer:   self.csv_writer.close()
        if self.decision_csv_writer: self.decision_csv_writer.close()
        cv2.destroyAllWindows()
        logger.info(f"Done. Detections: {self.stats['detections']} | FPS avg: {self.perf_monitor.get_average_fps():.2f}")


# ============ MAIN ============

def main():
    global DISPLAY_ENABLED, DISPLAY_BEV_ONLY, DEFAULT_AC_FPS, RECORDING_ENABLED, YOLO_SKIP_FRAMES, VIRTUAL_CAM_INDEX, TARGET_RESIZE_HEIGHT, DEBUG_DECISION, FLIP_CAMERA

    p = argparse.ArgumentParser(description="Lane offset processor (webcam or video file)")
    p.add_argument('--source',        '-s', default=VIDEO_PATH,
                   help='Video file path, "webcam", or "cam[:index]"')
    p.add_argument('--title',         '-t', default='Assetto Corsa')
    p.add_argument('--ac-fps',        type=int, default=DEFAULT_AC_FPS)
    p.add_argument('--virtual-index', type=int, default=VIRTUAL_CAM_INDEX)
    p.add_argument('--resize',        type=int, default=None, help='Target height to resize video to (e.g., 720 or 480). If not set, keeps original resolution.')
    p.add_argument('--yolo-skip',     type=int, default=0, help='Skip N frames between YOLO detections (0=no skip, 1=every other, 2=every third, etc.)')
    p.add_argument('--no-display',    action='store_true', help='Disable display windows')
    p.add_argument('--bev',            action='store_true', help='Show BEV only in the display window')
    p.add_argument('--no-record',     action='store_true', help='Disable video recording (CSV only)')
    p.add_argument('--flip',          action='store_true', help='Flip camera 180 degrees (upside down)')
    p.add_argument('--debug',         action='store_true', help='Enable debug logging')
    p.add_argument('--debug-decision', action='store_true', help='Show decision debug overlay (LEFT/RIGHT/STAY reasons)')
    args = p.parse_args()

    DISPLAY_ENABLED   = not args.no_display
    DISPLAY_BEV_ONLY  = bool(args.bev and DISPLAY_ENABLED)
    RECORDING_ENABLED = not args.no_record
    DEFAULT_AC_FPS    = int(args.ac_fps)
    VIRTUAL_CAM_INDEX = int(args.virtual_index)
    TARGET_RESIZE_HEIGHT = args.resize
    FLIP_CAMERA       = bool(args.flip)
    YOLO_SKIP_FRAMES  = max(0, int(args.yolo_skip))
    DEBUG_DECISION    = bool(args.debug_decision or args.debug)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    source = str(args.source)
    if source.lower() == 'ac':
        source = f"ac:{args.title}"
    else:
        source = normalize_source(source)

    try:
        proc = LaneOffsetProcessor(source, OUTPUT_DIR)
        proc.initialize()
        proc.run()
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())