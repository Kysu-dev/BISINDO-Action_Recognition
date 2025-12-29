import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# HAPUS: os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class BISINDODataPreprocessor:
    """
    PREPROCESSOR: Convert semua video → SINGLE .npz file
    TIDAK ADA SPLIT: Split dilakukan di training script
    """
    
    def __init__(
        self,
        raw_video_path,
        output_path,
        target_frames=30,
        min_confidence=0.4,
        enable_smoothing=True
    ):
        """
        Args:
            raw_video_path: Path ke folder video mentah
            output_path: Path untuk save SINGLE .npz file
            target_frames: Jumlah frame tetap per video
            min_confidence: Minimum confidence untuk menerima frame
            enable_smoothing: Enable temporal smoothing
        """
        self.raw_video_path = Path(raw_video_path)
        self.output_path = Path(output_path)
        
        if not self.raw_video_path.exists():
            raise FileNotFoundError(f"❌ Folder video tidak ditemukan: {self.raw_video_path}")
        
        self.target_frames = target_frames
        self.min_confidence = min_confidence
        self.enable_smoothing = enable_smoothing
        
        # Initialize MediaPipe (biarkan GPU aktif)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark configuration untuk BISINDO
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]  # Upper body
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        
        # Feature dimension calculation
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3  # x, y, z
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("🎬 BISINDO VIDEO PREPROCESSOR")
        print("="*60)
        print(f"Source: {self.raw_video_path}")
        print(f"Output: {self.output_path}")
        print(f"Target: {self.target_frames} frames per video")
        print(f"Features: {self.feature_dim} per frame")
        print("="*60 + "\n")
    
    def extract_landmarks(self, frame):
        """
        Extract landmarks dari satu frame
        Returns: landmarks array, confidence score
        """
        try:
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            
            # Process with MediaPipe
            results = self.holistic.process(rgb)
            
            # Initialize landmarks array
            landmarks = np.zeros(self.feature_dim, dtype=np.float32)
            confidences = []
            idx = 0
            
            # 1. Extract POSE landmarks
            if results.pose_landmarks:
                for lm_idx in self.POSE_SUBSET:
                    lm = results.pose_landmarks.landmark[lm_idx]
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    confidences.append(lm.visibility)
                    idx += 3
            else:
                return None, 0.0
            
            # 2. Extract LEFT HAND landmarks
            left_hand_detected = False
            if results.left_hand_landmarks:
                left_hand_detected = True
                for lm in results.left_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                # Fill with NaN for later interpolation
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                idx += self.n_hand * 3
                confidences.extend([0.0] * self.n_hand)
            
            # 3. Extract RIGHT HAND landmarks
            right_hand_detected = False
            if results.right_hand_landmarks:
                right_hand_detected = True
                for lm in results.right_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                idx += self.n_hand * 3
                confidences.extend([0.0] * self.n_hand)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return landmarks, avg_confidence
            
        except Exception as e:
            print(f"⚠️ Error extracting landmarks: {str(e)[:100]}")
            return None, 0.0
    
    def interpolate_missing_values(self, sequence):
        """
        Interpolasi nilai NaN dalam sequence
        """
        if sequence.size == 0:
            return sequence
        
        seq_clean = sequence.copy()
        n_frames, n_features = seq_clean.shape
        
        for i in range(n_features):
            col = seq_clean[:, i]
            nan_mask = np.isnan(col)
            
            # Jika ada NaN yang perlu diinterpolasi
            if np.any(nan_mask) and not np.all(nan_mask):
                # Linear interpolation
                valid_idx = np.where(~nan_mask)[0]
                valid_vals = col[valid_idx]
                all_idx = np.arange(n_frames)
                seq_clean[:, i] = np.interp(all_idx, valid_idx, valid_vals)
            elif np.all(nan_mask):
                # Jika semua NaN, set ke 0
                seq_clean[:, i] = 0.0
        
        return seq_clean
    
    def resample_to_fixed_length(self, sequence):
        """
        Resample sequence ke target_frames menggunakan linear interpolation
        """
        if sequence.shape[0] == self.target_frames:
            return sequence
        
        n_frames_orig = sequence.shape[0]
        n_features = sequence.shape[1]
        
        # Linear interpolation untuk resampling
        x_old = np.linspace(0, 1, n_frames_orig)
        x_new = np.linspace(0, 1, self.target_frames)
        
        resampled = np.zeros((self.target_frames, n_features), dtype=np.float32)
        
        for i in range(n_features):
            resampled[:, i] = np.interp(x_new, x_old, sequence[:, i])
        
        return resampled
    
    def normalize_sequence(self, sequence):
        """
        Normalize sequence: center dan scale
        """
        n_frames = sequence.shape[0]
        n_landmarks = self.feature_dim // 3
        
        # Reshape to (frames, landmarks, 3)
        reshaped = sequence.reshape(n_frames, n_landmarks, 3)
        normalized = np.zeros_like(reshaped)
        
        for t in range(n_frames):
            frame = reshaped[t].copy()
            
            # Center berdasarkan hip (indeks 6 & 7 dalam pose)
            hip_center = (frame[6] + frame[7]) / 2  # Hip landmarks
            frame -= hip_center
            
            # Scale berdasarkan shoulder distance
            shoulder_dist = np.linalg.norm(frame[0] - frame[1])  # Shoulders
            if shoulder_dist > 0.01:
                frame /= shoulder_dist
            
            normalized[t] = frame
        
        return normalized.reshape(n_frames, -1)
    
    def process_single_video(self, video_path):
        """
        Process satu video file
        Returns: sequence atau None jika gagal
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Tidak bisa buka video: {video_path.name}")
            return None
        
        frames_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks, confidence = self.extract_landmarks(frame)
            
            # Terima frame jika confidence cukup atau landmarks tidak None
            if landmarks is not None and confidence >= self.min_confidence:
                frames_data.append(landmarks)
        
        cap.release()
        
        # Validasi: minimal 10 frames
        if len(frames_data) < 10:
            print(f"⚠️ Video terlalu pendek: {video_path.name} ({len(frames_data)} frames)")
            return None
        
        # Convert to numpy array
        sequence = np.array(frames_data, dtype=np.float32)
        
        # Interpolasi missing values
        sequence = self.interpolate_missing_values(sequence)
        
        # Resample ke fixed length
        sequence = self.resample_to_fixed_length(sequence)
        
        # Temporal smoothing
        if self.enable_smoothing:
            for i in range(sequence.shape[1]):
                sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=1.0)
        
        # Normalize
        sequence = self.normalize_sequence(sequence)
        
        return sequence
    
    def process_all_videos(self):
        """
        Process semua video dalam dataset
        Returns: X, y, label_map
        """
        print("🔍 Mencari folder kelas...")
        
        # Cari semua class folders
        class_folders = sorted([
            f for f in self.raw_video_path.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ])
        
        if not class_folders:
            raise ValueError(f"❌ Tidak ada folder kelas ditemukan di: {self.raw_video_path}")
        
        # Buat label mapping
        label_map = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        X_all = []  # Features
        y_all = []  # Labels
        video_info = []  # Informasi video
        
        print(f"📁 Ditemukan {len(class_folders)} kelas:")
        for class_name, class_id in label_map.items():
            print(f"   {class_id}: {class_name}")
        
        print("\n🎬 Memulai preprocessing video...")
        
        for class_name, class_id in label_map.items():
            class_path = self.raw_video_path / class_name
            
            # Cari semua video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(class_path.glob(ext))
            
            if not video_files:
                print(f"⚠️ Tidak ada video di kelas: {class_name}")
                continue
            
            print(f"\n📂 Processing kelas: {class_name} ({len(video_files)} video)")
            
            processed_count = 0
            for i, video_path in enumerate(video_files, 1):
                print(f"   [{i:3d}/{len(video_files):3d}] {video_path.name}", end="\r")
                
                sequence = self.process_single_video(video_path)
                if sequence is not None:
                    X_all.append(sequence)
                    y_all.append(class_id)
                    video_info.append({
                        'filename': video_path.name,
                        'class': class_name,
                        'class_id': class_id
                    })
                    processed_count += 1
            
            print(f"   ✅ {class_name}: {processed_count}/{len(video_files)} video berhasil")
        
        # Validasi: minimal ada data
        if not X_all:
            raise ValueError("❌ Tidak ada video yang berhasil diproses!")
        
        # Convert to numpy arrays
        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.int32)
        
        print(f"\n📊 Dataset summary:")
        print(f"   Total samples: {X_all.shape[0]}")
        print(f"   Sequence shape: {X_all.shape[1:]} (frames × features)")
        print(f"   Classes: {len(np.unique(y_all))}")
        
        return X_all, y_all, label_map, video_info
    
    def save_as_single_npz(self, X, y, label_map, video_info=None):
        """
        Save semua data ke SATU file .npz
        """
        output_file = self.output_path / "bisindo_dataset.npz"
        
        print(f"\n💾 Menyimpan dataset ke: {output_file}")
        
        # Save sebagai single compressed .npz file
        np.savez_compressed(
            output_file,
            X=X,          # Features
            y=y,          # Labels
            label_map=label_map,  # Label mapping
            feature_dim=self.feature_dim,
            target_frames=self.target_frames,
            video_info=video_info if video_info else []
        )
        
        # Also save label map as JSON for readability
        with open(self.output_path / "label_map.json", "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        
        # Print file size
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Dataset saved! Size: {file_size_mb:.2f} MB")
        
        return output_file
    
    def run(self):
        """
        Run full preprocessing pipeline
        """
        print("\n" + "="*60)
        print("🚀 STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        try:
            # 1. Process semua video
            X, y, label_map, video_info = self.process_all_videos()
            
            # 2. Save sebagai single .npz file
            output_file = self.save_as_single_npz(X, y, label_map, video_info)
            
            print("\n" + "="*60)
            print("🎉 PREPROCESSING COMPLETE!")
            print("="*60)
            print(f"📁 Output file: {output_file}")
            print(f"📊 Dataset shape: {X.shape}")
            print(f"🎯 Classes: {len(label_map)}")
            print("="*60)
            
            return X, y, label_map
            
        except Exception as e:
            print(f"\n❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """
    Main function untuk preprocessing
    """
    # ⚙️ CONFIGURATION
    RAW_VIDEO_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video"
    OUTPUT_PATH = r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed"
    
    # Create preprocessor
    preprocessor = BISINDODataPreprocessor(
        raw_video_path=RAW_VIDEO_PATH,
        output_path=OUTPUT_PATH,
        target_frames=30,      # 30 frames per video
        min_confidence=0.3,    # Minimum confidence threshold
        enable_smoothing=True  # Enable temporal smoothing
    )
    
    # Run preprocessing
    X, y, label_map = preprocessor.run()
    
    if X is not None:
        print(f"\n🎯 Dataset siap untuk training!")
        print(f"   File: {OUTPUT_PATH}/bisindo_dataset.npz")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Gunakan file ini di training script Anda.")


if __name__ == "__main__":
    main()