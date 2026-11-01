"""
lane_change_predictor_BILSTM.py
Predictor real-time lane change berbasis BiLSTM untuk integrasi ke Program_Update.py.

Cara kerja:
- Setiap frame, update_and_predict() dipanggil per track_id
- Fitur dikumpulkan dalam sliding window (deque) sepanjang SEQUENCE_LEN frame
- Begitu buffer penuh, model di-inference dan hasil dikembalikan
- Scaler yang sama dengan preprocessing training dipakai untuk normalisasi

File yang dibutuhkan (taruh di folder yang sama):
    models/bilstm/bilstm_best.pt     <- model weights
    training_data/scaler.pkl         <- StandardScaler dari preprocessing

Ekspor dari notebook training:
    LABEL_COLORS  : dict {class_id: BGR color} untuk bounding box
    LABEL_MAP     : dict {class_id: label_string}
    LaneChangePredictor : class utama
"""

import os
import math
import pickle
import logging
import warnings
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Label & warna (diimport oleh Program_Update.py) ──────────────────────────
LABEL_MAP = {
    0: "Keep Lane",
    1: "Lane Change Left",
    2: "Lane Change Right",
}

# BGR colors untuk bounding box overlay
LABEL_COLORS = {
    0: (100, 200, 100),   # hijau  — Keep Lane
    1: (255,   0, 255),   # magenta — Lane Change Left
    2: (  0, 165, 255),   # oranye  — Lane Change Right
}

# ── Konfigurasi (sesuaikan dengan hasil training) ─────────────────────────────
MODEL_PATH   = os.path.join("models", "bilstm", "bilstm_best.pt")
SCALER_PATH  = os.path.join("training_data", "scaler.pkl")

SEQUENCE_LEN  = 30      # harus sama dengan preprocessing
INPUT_SIZE    = 6       # jumlah fitur
HIDDEN_SIZE   = 32      # sesuai bilstm_best.pt
NUM_LAYERS    = 2
NUM_CLASSES   = 3
DROPOUT       = 0.4

CONFIDENCE_THRESHOLD = 0.70   # di bawah ini → is_uncertain = True
MIN_FRAMES_BEFORE_PREDICT = 15  # buffer minimal sebelum mulai prediksi


# ── Arsitektur BiLSTM (harus identik dengan training notebook) ────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.classifier(out)


# ── Predictor utama ───────────────────────────────────────────────────────────
class LaneChangePredictor:
    """
    Predictor real-time lane change yang di-call per frame per track_id.

    Penggunaan di Program_Update.py:
        predictor = LaneChangePredictor()
        predictor.load()

        # Per frame, per kendaraan:
        result = predictor.update_and_predict(
            track_id                = tid,
            offset_normalized       = od['offset_normalized'],
            lane_width_px           = od['lane_width_px'],
            longitudinal_distance_m = lon_dist,
            timestamp               = timestamp,
        )
        if result:
            print(result['predicted_label'], result['confidence'])
    """

    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.device  = None
        self.ready   = False

        # Buffer per track_id: deque of raw feature vectors
        # raw features: [offset_normalized, longitudinal_distance_m, bbox_w_norm]
        # derived akan dihitung dari history saat inference
        self._buffers: dict[int, deque] = {}

        # Simpan nilai sebelumnya untuk hitung lat_vel, lat_acc, long_vel
        self._prev: dict[int, dict] = {}

        # Cache prediksi terakhir per track (untuk frame di mana buffer belum penuh)
        self._last_result: dict[int, dict] = {}

    def load(self):
        """Load model weights dan scaler. Dipanggil sekali saat initialize()."""
        if not TORCH_AVAILABLE:
            logger.warning("[Predictor] PyTorch tidak tersedia. Predictor dinonaktifkan.")
            return

        # Load scaler
        if not os.path.exists(SCALER_PATH):
            logger.warning(f"[Predictor] Scaler tidak ditemukan: {SCALER_PATH}. "
                           f"Prediksi akan menggunakan fitur tanpa normalisasi.")
            self.scaler = None
        else:
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"[Predictor] Scaler loaded dari {SCALER_PATH}")

        # Load model
        if not os.path.exists(MODEL_PATH):
            logger.error(f"[Predictor] Model tidak ditemukan: {MODEL_PATH}")
            logger.error(f"[Predictor] Pastikan file ada di: {os.path.abspath(MODEL_PATH)}")
            return

        self.device = torch.device('cpu')
        self.model  = BiLSTM(
            input_size  = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            num_classes = NUM_CLASSES,
            dropout     = DROPOUT
        ).to(self.device)

        try:
            state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            # weights_only belum tersedia di PyTorch lama
            state = torch.load(MODEL_PATH, map_location=self.device)

        self.model.load_state_dict(state)
        self.model.eval()
        self.ready = True
        logger.info(f"[Predictor] Model loaded dari {MODEL_PATH} | device={self.device}")

    def reset_track(self, track_id: int):
        """Hapus buffer dan state untuk track yang sudah tidak aktif."""
        self._buffers.pop(track_id, None)
        self._prev.pop(track_id, None)
        self._last_result.pop(track_id, None)

    def cleanup_stale_tracks(self, active_track_ids: set):
        """Hapus buffer track yang tidak lagi aktif di frame ini."""
        stale = [tid for tid in self._buffers if tid not in active_track_ids]
        for tid in stale:
            self.reset_track(tid)

    def _extract_features(self, track_id: int,
                           offset_normalized: float,
                           longitudinal_distance_m: float,
                           bbox_w_norm: float) -> np.ndarray:
        """
        Hitung 6 fitur dari nilai saat ini + history sebelumnya.
        Urutan fitur harus sama persis dengan FEATURE_COLS di preprocessing:
            [offset_normalized, lat_vel, lat_acc, longitudinal_distance_m, long_vel, bbox_w_norm]
        """
        prev = self._prev.get(track_id, {})

        # Lateral velocity
        prev_offset = prev.get('offset_normalized', offset_normalized)
        lat_vel     = offset_normalized - prev_offset

        # Lateral acceleration
        prev_lat_vel = prev.get('lat_vel', 0.0)
        lat_acc      = lat_vel - prev_lat_vel

        # Longitudinal velocity
        prev_long = prev.get('longitudinal_distance_m', longitudinal_distance_m)
        long_vel  = longitudinal_distance_m - prev_long

        # Simpan state untuk frame berikutnya
        self._prev[track_id] = {
            'offset_normalized'      : offset_normalized,
            'lat_vel'                : lat_vel,
            'longitudinal_distance_m': longitudinal_distance_m,
        }

        return np.array([
            offset_normalized,
            lat_vel,
            lat_acc,
            longitudinal_distance_m,
            long_vel,
            bbox_w_norm,
        ], dtype=np.float32)

    def update_and_predict(self,
                           track_id: int,
                           offset_normalized,
                           lane_width_px,
                           longitudinal_distance_m: float,
                           timestamp: float,
                           bbox_width_px: float = None) -> dict | None:
        """
        Update buffer track dan lakukan prediksi jika buffer sudah cukup penuh.

        Args:
            track_id               : ID track dari ByteTrack/YOLO
            offset_normalized      : posisi lateral (-1.0 sampai +1.0), atau None jika lane N/A
            lane_width_px          : lebar lajur dalam pixel, untuk normalisasi bbox
            longitudinal_distance_m: jarak longitudinal ke kamera (meter)
            timestamp              : timestamp frame (detik)
            bbox_width_px          : lebar bounding box dalam pixel (opsional)

        Returns:
            dict dengan keys:
                predicted_class  : int (0=KL, 1=LLC, 2=RLC)
                predicted_label  : str
                confidence       : float
                is_uncertain     : bool
                frames_buffered  : int
            atau None jika buffer belum cukup dan tidak ada cached result
        """
        if not self.ready:
            return None

        # Tidak bisa prediksi tanpa offset
        if offset_normalized is None or not math.isfinite(float(offset_normalized)):
            return self._last_result.get(track_id)

        offset_normalized       = float(offset_normalized)
        longitudinal_distance_m = float(longitudinal_distance_m) if longitudinal_distance_m is not None else 0.0

        # Normalisasi bbox width
        if bbox_width_px is not None and lane_width_px and float(lane_width_px) > 0:
            bbox_w_norm = float(bbox_width_px) / float(lane_width_px)
        else:
            bbox_w_norm = 0.5  # default jika tidak tersedia

        bbox_w_norm = float(np.clip(bbox_w_norm, 0.0, 2.0))

        # Ekstrak fitur frame ini
        feat = self._extract_features(
            track_id, offset_normalized, longitudinal_distance_m, bbox_w_norm
        )

        # Init buffer jika track baru
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=SEQUENCE_LEN)

        self._buffers[track_id].append(feat)
        n_frames = len(self._buffers[track_id])

        # Belum cukup frame untuk prediksi
        if n_frames < MIN_FRAMES_BEFORE_PREDICT:
            result = {
                'predicted_class': 0,
                'predicted_label': LABEL_MAP[0],
                'confidence'     : 0.0,
                'is_uncertain'   : True,
                'frames_buffered': n_frames,
            }
            self._last_result[track_id] = result
            return result

        # Build sequence — pad dengan frame pertama kalau buffer belum penuh
        buf = list(self._buffers[track_id])
        if len(buf) < SEQUENCE_LEN:
            pad_len = SEQUENCE_LEN - len(buf)
            buf     = [buf[0]] * pad_len + buf

        seq = np.stack(buf, axis=0).astype(np.float32)  # (SEQUENCE_LEN, INPUT_SIZE)

        # Normalisasi dengan scaler
        if self.scaler is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    seq = self.scaler.transform(seq)
            except Exception as e:
                logger.debug(f"[Predictor] Scaler error track {track_id}: {e}")

        # Inference
        try:
            x      = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            pred_class  = int(np.argmax(probs))
            confidence  = float(probs[pred_class])
            is_uncertain = confidence < CONFIDENCE_THRESHOLD

            result = {
                'predicted_class': pred_class,
                'predicted_label': LABEL_MAP[pred_class],
                'confidence'     : confidence,
                'is_uncertain'   : is_uncertain,
                'frames_buffered': n_frames,
            }
            self._last_result[track_id] = result
            return result

        except Exception as e:
            logger.error(f"[Predictor] Inference error track {track_id}: {e}")
            return self._last_result.get(track_id)
