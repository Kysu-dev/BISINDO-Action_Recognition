import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # Hide TF warnings
# HAPUS: os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # ⚠️ JANGAN DISABLE GPU!

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class BISINDODataPreprocessor:
    """
    PREPROCESSOR MURNI: Convert videos → NPZ format
    NO TRAIN/VAL/TEST SPLIT - itu tugas training script
    """
    
    def __init__(
        self,
        dataset_path,
        output_path,
        target_frames=30,
        min_frames=10,
        smoothing=True
    ):
        """
        Args:
            dataset_path: Path ke folder video mentah
            output_path: Path untuk save dataset.npz
            target_frames: Frame tetap untuk semua video
            min_frames: Minimum frames untuk video valid
            smoothing: Enable temporal smoothing
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path tidak ditemukan: {self.dataset_path}")
        
        self.target_frames = target_frames
        self.min_frames = min_frames
        self.smoothing = smoothing
        
        # Initialize MediaPipe dengan GPU SUPPORT
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Bisa 2 untuk akurasi lebih tinggi
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmarks untuk bahasa isyarat (upper body focus)
        self.POSE_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, elbows, wrists, hips
        self.HAND_LANDMARKS = 21  # MediaPipe hand landmarks
        
        self.n_pose = len(self.POSE_LANDMARKS)
        self.n_hand = self.HAND_LANDMARKS
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3  # (x, y, z)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("BISINDO PREPROCESSOR - VIDEO TO NPZ CONVERTER")
        print("="*60)
        print(f"Input:  {self.dataset_path}")
        print(f"Output: {self.output_path}")
        print(f"Target: {self.target_frames} frames")
        print(f"Features: {self.feature_dim} dimensions")
        print("="*60 + "\n")
    
    def extract_frame_landmarks(self, frame):
        """
        Extract landmarks dari single frame
        Returns: landmarks array atau None jika gagal
        """
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            
            # Process dengan MediaPipe
            results = self.holistic.process(rgb)
            
            # Initialize array untuk landmarks
            landmarks = np.zeros(self.feature_dim, dtype=np.float32)
            idx = 0
            
            # 1. Pose landmarks
            if results.pose_landmarks:
                for lm_idx in self.POSE_LANDMARKS:
                    lm = results.pose_landmarks.landmark[lm_idx]
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                # Jika pose tidak terdeteksi, skip frame ini
                return None
            
            # 2. Left hand landmarks
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                # Isi dengan NaN untuk interpolasi nanti
                landmarks[idx:idx+(self.n_hand*3)] = np.nan
                idx += self.n_hand * 3
            
            # 3. Right hand landmarks
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
            else:
                landmarks[idx:idx+(self.n_hand*3)] = np.nan
                idx += self.n_hand * 3
            
            return landmarks
            
        except Exception as e:
            print(f"⚠️ Error ekstraksi: {e}")
            return None
    
    def interpolate_missing_landmarks(self, sequence):
        """
        Interpolasi landmarks yang missing (NaN)
        """
        if sequence.size == 0:
            return sequence
        
        seq_interp = sequence.copy()
        n_frames, n_features = seq_interp.shape
        
        for i in range(n_features):
            col = seq_interp[:, i]
            nan_mask = np.isnan(col)
            
            # Jika ada NaN
            if np.any(nan_mask) and not np.all(nan_mask):
                # Indeks yang valid
                valid_idx = np.where(~nan_mask)[0]
                valid_vals = col[valid_idx]
                
                # Interpolasi semua frame
                all_idx = np.arange(n_frames)
                col_interp = np.interp(all_idx, valid_idx, valid_vals)
                seq_interp[:, i] = col_interp
            elif np.all(nan_mask):
                # Jika semua NaN, set ke 0
                seq_interp[:, i] = 0.0
        
        return seq_interp
    
    def normalize_sequence(self, sequence):
        """
        Normalize sequence: center dan scale
        """
        n_frames = sequence.shape[0]
        n_landmarks = self.feature_dim // 3
        
        # Reshape ke (frames, landmarks, 3)
        reshaped = sequence.reshape(n_frames, n_landmarks, 3)
        normalized = np.zeros_like(reshaped)
        
        for t in range(n_frames):
            frame = reshaped[t].copy()
            
            # 1. Center berdasarkan hip (indeks 6 & 7 di pose landmarks)
            if self.n_pose >= 8:
                hip_center = (frame[6] + frame[7]) / 2
                frame -= hip_center
            
            # 2. Scale berdasarkan shoulder width
            if self.n_pose >= 2:
                shoulder_width = np.linalg.norm(frame[0] - frame[1])
                if shoulder_width > 0.01:  # Hindari division by zero
                    frame /= shoulder_width
            
            # 3. Rotation alignment (opsional)
            if self.n_pose >= 2:
                shoulder_vec = frame[1] - frame[0]
                shoulder_vec[2] = 0  # Ignore z-axis
                
                if np.linalg.norm(shoulder_vec) > 0.01:
                    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
                    c, s = np.cos(-angle), np.sin(-angle)
                    
                    # Rotasi hanya x dan y
                    x_rot = c * frame[:, 0] - s * frame[:, 1]
                    y_rot = s * frame[:, 0] + c * frame[:, 1]
                    frame[:, 0] = x_rot
                    frame[:, 1] = y_rot
            
            normalized[t] = frame
        
        return normalized.reshape(n_frames, -1)
    
    def process_video(self, video_path):
        """
        Process single video menjadi sequence
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Tidak bisa buka: {video_path.name}")
            return None
        
        frames_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_frame_landmarks(frame)
            if landmarks is not None:
                frames_data.append(landmarks)
        
        cap.release()
        
        # Validasi jumlah frame
        if len(frames_data) < self.min_frames:
            print(f"⚠️ Video {video_path.name}: Hanya {len(frames_data)} frames (min: {self.min_frames})")
            return None
        
        # Convert ke numpy array
        sequence = np.array(frames_data, dtype=np.float32)
        
        # Interpolasi missing landmarks
        sequence = self.interpolate_missing_landmarks(sequence)
        
        # Resample ke fixed length
        if len(sequence) != self.target_frames:
            x_old = np.linspace(0, 1, len(sequence))
            x_new = np.linspace(0, 1, self.target_frames)
            
            resampled = np.zeros((self.target_frames, self.feature_dim), dtype=np.float32)
            for i in range(self.feature_dim):
                resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])
            sequence = resampled
        
        # Smoothing temporal
        if self.smoothing:
            for i in range(sequence.shape[1]):
                sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=1.0)
        
        # Normalize
        sequence = self.normalize_sequence(sequence)
        
        return sequence
    
    def process_dataset(self):
        """
        Process semua video di dataset
        Returns: X, y, label_map saved as .npz
        """
        # Cari semua class folder
        class_folders = sorted([f for f in self.dataset_path.iterdir() if f.is_dir()])
        
        if not class_folders:
            raise ValueError(f"Tidak ada folder kelas di: {self.dataset_path}")
        
        # Buat label mapping
        label_map = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        X_all, y_all = [], []
        
        print("🎬 Memulai preprocessing video...")
        
        for class_name, class_idx in label_map.items():
            class_path = self.dataset_path / class_name
            
            # Cari semua video files
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(class_path.glob(ext))
            
            if not video_files:
                print(f"⚠️ Tidak ada video di folder: {class_name}")
                continue
            
            print(f"\n📂 Kelas: {class_name} ({len(video_files)} video)")
            
            for i, video_path in enumerate(video_files, 1):
                print(f"   [{i}/{len(video_files)}] Processing {video_path.name}...", end="\r")
                
                sequence = self.process_video(video_path)
                if sequence is not None:
                    X_all.append(sequence)
                    y_all.append(class_idx)
            
            print(f"   ✅ {class_name}: {len([x for x in y_all if x == class_idx])} video berhasil")
        
        if not X_all:
            raise ValueError("❌ Tidak ada video yang berhasil diproses!")
        
        # Convert ke numpy arrays
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)
        
        # Simpan sebagai single .npz file
        output_file = self.output_path / "bisindo_dataset.npz"
        np.savez_compressed(
            output_file,
            X=X_all,
            y=y_all,
            label_map=label_map,
            feature_dim=self.feature_dim,
            target_frames=self.target_frames
        )
        
        # Simpan label map sebagai json juga
        with open(self.output_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING SELESAI!")
        print("="*60)
        print(f"Total samples: {X_all.shape[0]}")
        print(f"Sequence shape: {X_all.shape[1:]} (frames × features)")
        print(f"Number of classes: {len(label_map)}")
        print(f"Output file: {output_file}")
        print("="*60)
        
        return X_all, y_all, label_map


def main():
    """Main function untuk preprocessing standalone"""
    # Konfigurasi
    DATASET_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video"
    OUTPUT_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed"
    
    # Buat preprocessor
    preprocessor = BISINDODataPreprocessor(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        target_frames=30,
        min_frames=10,
        smoothing=True
    )
    
    # Jalankan preprocessing
    try:
        X, y, label_map = preprocessor.process_dataset()
        print(f"\n🎉 Dataset siap untuk training!")
        print(f"   Gunakan file: {OUTPUT_PATH}/bisindo_dataset.npz")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()