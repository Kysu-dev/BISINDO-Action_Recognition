import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import sys
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class BISINDOPreprocessor:
    def __init__(self, 
                 raw_video_path,
                 output_path,
                 target_frames=40,
                 min_confidence=0.3,
                 enable_smoothing=True,
                 quality_check=True):
        
        self.raw_video_path = Path(raw_video_path)
        self.output_path = Path(output_path)
        
        if not self.raw_video_path.exists():
            raise FileNotFoundError(f"❌ Folder tidak ditemukan: {self.raw_video_path}")
        
        self.target_frames = target_frames
        self.min_confidence = min_confidence
        self.enable_smoothing = enable_smoothing
        self.quality_check = quality_check
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark configuration
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Source: {self.raw_video_path}")
        print(f"Output: {self.output_path}")
        print(f"Target frames: {self.target_frames}")
        print(f"Features: {self.feature_dim}")

    def extract_landmarks(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            
            results = self.holistic.process(rgb)
            
            landmarks = np.zeros(self.feature_dim, dtype=np.float32)
            confidences = []
            idx = 0
            
            # 1. POSE
            if results.pose_landmarks:
                for lm_idx in self.POSE_SUBSET:
                    lm = results.pose_landmarks.landmark[lm_idx]
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    confidences.append(lm.visibility)
                    idx += 3
            else:
                return None, 0.0
            
            # 2. LEFT HAND
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                idx += self.n_hand * 3
                confidences.extend([0.0] * self.n_hand)
            
            # 3. RIGHT HAND
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                idx += self.n_hand * 3
                confidences.extend([0.0] * self.n_hand)
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            return landmarks, avg_confidence
            
        except Exception as e:
            return None, 0.0

    def interpolate_missing_values(self, sequence):
        if sequence.size == 0:
            return sequence
        
        seq_clean = sequence.copy()
        n_frames, n_features = seq_clean.shape
        
        for i in range(n_features):
            col = seq_clean[:, i]
            nan_mask = np.isnan(col)
            
            if np.any(nan_mask) and not np.all(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                valid_vals = col[valid_idx]
                all_idx = np.arange(n_frames)
                seq_clean[:, i] = np.interp(all_idx, valid_idx, valid_vals)
            elif np.all(nan_mask):
                seq_clean[:, i] = 0.0
        
        return seq_clean

    def resample_to_fixed_length(self, sequence):
        if sequence.shape[0] == self.target_frames:
            return sequence
        
        n_frames_orig = sequence.shape[0]
        n_features = sequence.shape[1]
        
        x_old = np.linspace(0, 1, n_frames_orig)
        x_new = np.linspace(0, 1, self.target_frames)
        
        resampled = np.zeros((self.target_frames, n_features), dtype=np.float32)
        
        for i in range(n_features):
            resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])
        
        return resampled

    def normalize_sequence(self, sequence):
        n_frames = sequence.shape[0]
        n_landmarks = self.feature_dim // 3
        
        reshaped = sequence.reshape(n_frames, n_landmarks, 3)
        normalized = np.zeros_like(reshaped)
        
        for t in range(n_frames):
            frame = reshaped[t].copy()
            
            left_shoulder = frame[0]
            right_shoulder = frame[1]
            shoulder_center = (left_shoulder + right_shoulder) / 2
            
            frame -= shoulder_center
            
            shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
            
            if shoulder_dist > 0.01:
                frame /= shoulder_dist
            else:
                pose_points = frame[:self.n_pose]
                norm = np.linalg.norm(pose_points)
                if norm > 0.01:
                    frame /= norm
            
            normalized[t] = frame
        
        return normalized.reshape(n_frames, -1)

    def validate_sequence(self, sequence):
        if sequence.shape[0] == 0:
            return 0.0
        
        nan_ratio = np.isnan(sequence).sum() / sequence.size
        if nan_ratio > 0.3:
            return 0.0
        
        variance = np.var(sequence, axis=0).mean()
        if variance < 0.00001:
            return 0.0
        
        outliers = np.sum(np.abs(sequence) > 10)
        if outliers > sequence.shape[0] * sequence.shape[1] * 0.2:
            return 0.0
        
        quality = 1.0 - (nan_ratio * 0.5)
        return quality

    def process_single_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        frames_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, confidence = self.extract_landmarks(frame)
            
            if landmarks is not None and confidence >= self.min_confidence:
                frames_data.append(landmarks)
        
        cap.release()
        
        if len(frames_data) < 8:
            return None
        
        sequence = np.array(frames_data, dtype=np.float32)
        sequence = self.interpolate_missing_values(sequence)
        sequence = self.resample_to_fixed_length(sequence)
        
        if self.enable_smoothing:
            from scipy.ndimage import gaussian_filter1d
            for i in range(sequence.shape[1]):
                sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=1.0)
        
        sequence = self.normalize_sequence(sequence)
        
        if self.quality_check:
            quality = self.validate_sequence(sequence)
            if quality < 0.5:
                return None
        
        return sequence

    def process_all_videos(self):
        print("🔍 Mencari folder kelas...")
        
        class_folders = sorted([
            f for f in self.raw_video_path.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        if not class_folders:
            raise ValueError("❌ Tidak ada folder kelas ditemukan")
        
        label_map = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        X_all = []
        y_all = []
        video_info = []
        
        print(f"📁 Ditemukan {len(class_folders)} kelas")
        
        for class_name, class_id in label_map.items():
            class_path = self.raw_video_path / class_name
            
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(sorted(class_path.glob(ext)))
            
            if not video_files:
                continue
            
            print(f"📂 Processing: {class_name} ({len(video_files)} video)")
            
            processed_count = 0
            for i, video_path in enumerate(video_files, 1):
                print(f"   [{i:3d}/{len(video_files):3d}] {video_path.name:<30}", end="\r")
                
                sequence = self.process_single_video(video_path)
                if sequence is not None:
                    X_all.append(sequence)
                    y_all.append(class_id)
                    video_info.append({
                        'filename': video_path.name,
                        'class': class_name,
                        'class_id': class_id,
                        'frames': sequence.shape[0]
                    })
                    processed_count += 1
            
            print(f"   ✅ {class_name}: {processed_count}/{len(video_files)} video berhasil      ")
        
        if not X_all:
            raise ValueError("❌ Tidak ada video yang berhasil diproses!")
        
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)
        
        return X_all, y_all, label_map, video_info

    def save_as_npz(self, X, y, label_map, video_info=None):
        output_file = self.output_path / "bisindo_dataset.npz"
        
        print(f"\n💾 Saving dataset to: {output_file}")
        
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            label_map=label_map,
            feature_dim=self.feature_dim,
            target_frames=self.target_frames,
            video_info=video_info if video_info else [],
            normalization_method="shoulder_centered"
        )
        
        with open(self.output_path / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dataset saved successfully!")
        return output_file

    def run(self):
        try:
            X, y, label_map, video_info = self.process_all_videos()
            self.save_as_npz(X, y, label_map, video_info)
            return X, y, label_map
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

def main():
    # Path configuration
    RAW_VIDEO_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video"
    OUTPUT_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed"
    
    TARGET_FRAMES = 40  # 40 frames recommended
    
    preprocessor = BISINDOPreprocessor(
        raw_video_path=RAW_VIDEO_PATH,
        output_path=OUTPUT_PATH,
        target_frames=TARGET_FRAMES,
        min_confidence=0.3,
        enable_smoothing=True,
        quality_check=True
    )
    
    X, y, label_map = preprocessor.run()
    
    if X is not None:
        print(f"✅ Preprocessing complete. Ready for training.")
        print(f"   Samples: {X.shape[0]}")

if __name__ == "__main__":
    main()