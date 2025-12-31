import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import warnings
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class BISINDOPreprocessor:
    """Modul Preprocessing: Mengubah Video Mentah menjadi Koordinat Landmark (Landmark Extraction)."""
    
    def __init__(self, raw_path, out_path, target_frames=40):
        self.raw_path = Path(raw_path)
        self.out_path = Path(out_path)
        self.target_frames = target_frames
        
        # Inisialisasi MediaPipe Holistic (Pose + Tangan)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Konfigurasi Fitur: 8 titik pose + 21 titik tangan kiri + 21 titik tangan kanan
        self.POSE_POINTS = [11, 12, 13, 14, 15, 16, 23, 24] # Bahu, Siku, Pergelangan, Pinggul
        self.feature_dim = (len(self.POSE_POINTS) + 21 + 21) * 3 # Total 150 Fitur (x, y, z)
        
        self.out_path.mkdir(parents=True, exist_ok=True)

    def extract_landmarks(self, frame):
        """Mengekstrak koordinat x, y, z dari satu frame video."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)
            
            data = np.zeros(self.feature_dim, dtype=np.float32)
            idx = 0
            
            # 1. Ekstraksi Pose (Subset)
            if res.pose_landmarks:
                for i in self.POSE_POINTS:
                    lm = res.pose_landmarks.landmark[i]
                    data[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
            else: return None # Jika badan tidak terdeteksi, abaikan frame
            
            # 2. Ekstraksi Tangan Kiri
            if res.left_hand_landmarks:
                for lm in res.left_hand_landmarks.landmark:
                    data[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
            else:
                data[idx:idx+63] = np.nan; idx += 63 # Gunakan NaN jika tangan hilang
                
            # 3. Ekstraksi Tangan Kanan
            if res.right_hand_landmarks:
                for lm in res.right_hand_landmarks.landmark:
                    data[idx:idx+3] = [lm.x, lm.y, lm.z]; idx += 3
            else:
                data[idx:idx+63] = np.nan
                
            return data
        except: return None

    def refine_sequence(self, sequence):
        """Pembersihan data: Interpolasi nilai hilang, resampling durasi, dan smoothing."""
        # 1. Interpolasi (Mengisi koordinat tangan yang hilang/NaN)
        for i in range(sequence.shape[1]):
            col = sequence[:, i]
            nan_mask = np.isnan(col)
            if np.any(nan_mask) and not np.all(nan_mask):
                col[nan_mask] = np.interp(np.where(nan_mask)[0], np.where(~nan_mask)[0], col[~nan_mask])
            elif np.all(nan_mask): col.fill(0)
            
        # 2. Resampling (Menyamakan semua durasi video menjadi 40 frame)
        x_old = np.linspace(0, 1, len(sequence))
        x_new = np.linspace(0, 1, self.target_frames)
        resampled = np.array([np.interp(x_new, x_old, sequence[:, i]) for i in range(sequence.shape[1])]).T
        
        # 3. Smoothing (Mengurangi 'getaran' pada deteksi landmark)
        for i in range(resampled.shape[1]):
            resampled[:, i] = gaussian_filter1d(resampled[:, i], sigma=1.0)
            
        return resampled

    def normalize(self, sequence):
        """Normalisasi: Menjadikan titik bahu sebagai pusat (0,0) dan skala berdasarkan lebar bahu."""
        res = sequence.reshape(self.target_frames, -1, 3)
        for t in range(self.target_frames):
            shoulder_center = (res[t, 0] + res[t, 1]) / 2 # Rata-rata bahu kiri & kanan
            res[t] -= shoulder_center
            dist = np.linalg.norm(res[t, 0] - res[t, 1])
            if dist > 0.01: res[t] /= dist
        return res.reshape(self.target_frames, -1)

    def process_all(self):
        """Looping utama untuk memproses seluruh folder video."""
        folders = sorted([f for f in self.raw_path.iterdir() if f.is_dir()])
        label_map = {f.name: i for i, f in enumerate(folders)}
        X, y = [], []

        for folder in folders:
            v_files = list(folder.glob('*.mp4')) + list(folder.glob('*.avi'))
            print(f"📂 Processing {folder.name}: {len(v_files)} video")
            
            for v_path in v_files:
                cap = cv2.VideoCapture(str(v_path))
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    lm = self.extract_landmarks(frame)
                    if lm is not None: frames.append(lm)
                cap.release()

                if len(frames) > 10: # Abaikan video jika deteksi terlalu sedikit
                    seq = self.refine_sequence(np.array(frames))
                    seq = self.normalize(seq)
                    X.append(seq)
                    y.append(label_map[folder.name])

        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
        
        # Simpan Dataset
        np.savez_compressed(self.out_path/"bisindo_dataset.npz", X=X, y=y, label_map=label_map)
        with open(self.out_path/"label_map.json", "w") as f: json.dump(label_map, f, indent=2)
        
        print(f"✅ Selesai! Total sampel: {len(X)}")

if __name__ == "__main__":
    RAW = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video"
    OUT = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed"
    
    pre = BISINDOPreprocessor(RAW, OUT)
    pre.process_all()