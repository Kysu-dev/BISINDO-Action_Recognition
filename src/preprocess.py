import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split


class ImprovedBISINDOPreprocessor:

    def __init__(
        self,
        dataset_path,
        output_path,
        target_frames=30,
        use_rotation_alignment=True,
        confidence_threshold=0.5,
        smoothing_sigma=1.0,
        add_velocity=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ Dataset path not found: {self.dataset_path}")

        self.target_frames = target_frames
        self.use_rotation_alignment = use_rotation_alignment
        self.confidence_threshold = confidence_threshold
        self.smoothing_sigma = smoothing_sigma
        self.add_velocity = add_velocity
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total_videos": 0,
            "processed_videos": 0,
            "failed_videos": 0,
            "low_confidence_frames": 0,
            "interpolated_frames": 0,
        }

        print("\n" + "=" * 70)
        print("IMPROVED BISINDO Preprocessor Initialized")
        print("=" * 70)
        print(f"Dataset   : {self.dataset_path}")
        print(f"Output    : {self.output_path}")
        print(f"Frames    : {self.target_frames}")
        print(f"Features  : {self.feature_dim}")
        print(f"Velocity  : {self.add_velocity}")
        print(f"Split     : {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        print("=" * 70 + "\n")

    def extract_landmarks(self, frame):
        """Extract landmarks dengan handling yang lebih baik"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            if not results.pose_landmarks:
                return None, 0.0

            coords = []
            conf = []

            # Pose landmarks
            for i in self.POSE_SUBSET:
                lm = results.pose_landmarks.landmark[i]
                coords.extend([lm.x, lm.y, lm.z])
                conf.append(lm.visibility)

            # Left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                conf.extend([1.0] * self.n_hand)
            else:
                coords.extend([None] * self.n_hand * 3)  # ✅ Use None instead of 0
                conf.extend([0.0] * self.n_hand)

            # Right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                conf.extend([1.0] * self.n_hand)
            else:
                coords.extend([None] * self.n_hand * 3)  # ✅ Use None instead of 0
                conf.extend([0.0] * self.n_hand)

            return np.array(coords, dtype=object), float(np.mean(conf))

        except Exception as e:
            return None, 0.0

    def interpolate_missing(self, frames):
        """✅ Interpolasi untuk missing hand landmarks"""
        frames = np.array(frames, dtype=float)  # Convert to float
        n_frames, n_features = frames.shape
        
        for i in range(n_features):
            col = frames[:, i]
            mask = ~np.isnan(col)
            
            if mask.sum() == 0:  # Semua missing
                frames[:, i] = 0.0
            elif mask.sum() < n_frames:  # Ada yang missing
                indices = np.arange(n_frames)
                frames[:, i] = np.interp(indices, indices[mask], col[mask])
                self.stats["interpolated_frames"] += 1
        
        return frames

    def uniform_sampling(self, seq):
        """Uniform sampling dengan validasi"""
        n, f = seq.shape
        if n == self.target_frames:
            return seq
        x_old = np.arange(n)
        x_new = np.linspace(0, n - 1, self.target_frames)
        out = np.zeros((self.target_frames, f))
        for i in range(f):
            out[:, i] = np.interp(x_new, x_old, seq[:, i])
        return out

    def compute_velocity(self, seq):
        """✅ Hitung velocity features"""
        velocity = np.diff(seq, axis=0)
        velocity = np.vstack([velocity[0], velocity])  # Pad first frame
        return velocity

    def normalize(self, seq):
        """Normalisasi dengan validasi yang lebih baik"""
        n_frames = seq.shape[0]
        n_landmarks = self.feature_dim // 3
        reshaped = seq.reshape(n_frames, n_landmarks, 3)
        out = np.zeros_like(reshaped)

        for t in range(n_frames):
            frame = reshaped[t].copy()
            pose = frame[:self.n_pose]

            # Hip center (indices 6,7 dalam POSE_SUBSET adalah hips)
            hip = (pose[6] + pose[7]) / 2
            frame -= hip

            # Shoulder normalization
            shoulder_width = np.linalg.norm(pose[0] - pose[1])
            if shoulder_width < 1e-6:
                shoulder_width = 1.0
            frame /= shoulder_width

            # Rotation alignment
            if self.use_rotation_alignment:
                v = pose[1] - pose[0]
                angle = np.arctan2(v[1], v[0])
                c, s = np.cos(-angle), np.sin(-angle)
                x, y = frame[:, 0].copy(), frame[:, 1].copy()
                frame[:, 0] = c * x - s * y
                frame[:, 1] = s * x + c * y

            out[t] = frame

        return out.reshape(n_frames, -1)

    def process_video(self, video_path):
        """Process video dengan handling yang lebih robust"""
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                lm, conf = self.extract_landmarks(frame)
                if lm is not None:
                    if conf >= self.confidence_threshold:
                        frames.append(lm)
                    else:
                        self.stats["low_confidence_frames"] += 1

            # ✅ Validasi minimum frames (50% dari target)
            min_frames = max(10, self.target_frames // 2)
            if len(frames) < min_frames:
                return None

            # ✅ Interpolasi missing values
            seq = self.interpolate_missing(frames)
            
            # Uniform sampling
            seq = self.uniform_sampling(seq)
            
            # Smoothing
            if self.smoothing_sigma > 0:
                for i in range(seq.shape[1]):
                    seq[:, i] = gaussian_filter1d(seq[:, i], self.smoothing_sigma)
            
            # Normalization
            seq = self.normalize(seq)

            # ✅ Add velocity features
            if self.add_velocity:
                vel = self.compute_velocity(seq)
                seq = np.concatenate([seq, vel], axis=1)

            return seq

        finally:
            if cap is not None:
                cap.release()

    def process_dataset(self):
        """Process dataset dengan train/val/test split"""
        class_folders = sorted([f for f in self.dataset_path.iterdir() if f.is_dir()])
        label_map = {c.name: i for i, c in enumerate(class_folders)}

        with open(self.output_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        X, y, filenames = [], [], []

        for cname, label in label_map.items():
            videos = list((self.dataset_path / cname).glob("*.mp4"))
            print(f"\n📁 {cname} ({len(videos)} videos)")

            for i, v in enumerate(videos, 1):
                print(f"   [{i}/{len(videos)}] {v.name}", end="\r")
                self.stats["total_videos"] += 1
                seq = self.process_video(v)
                if seq is not None:
                    X.append(seq)
                    y.append(label)
                    filenames.append(v.name)
                    self.stats["processed_videos"] += 1
                else:
                    self.stats["failed_videos"] += 1
            print()  # New line after class

        X = np.array(X)
        y = np.array(y)

        # ✅ Train/Val/Test Split
        X_temp, X_test, y_temp, y_test, fn_temp, fn_test = train_test_split(
            X, y, filenames, 
            test_size=self.test_ratio, 
            stratify=y,
            random_state=self.random_seed
        )

        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val, fn_train, fn_val = train_test_split(
            X_temp, y_temp, fn_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=self.random_seed
        )

        # Save splits
        np.savez_compressed(
            self.output_path / "bisindo_train.npz",
            X=X_train, y=y_train, filenames=fn_train
        )
        np.savez_compressed(
            self.output_path / "bisindo_val.npz",
            X=X_val, y=y_val, filenames=fn_val
        )
        np.savez_compressed(
            self.output_path / "bisindo_test.npz",
            X=X_test, y=y_test, filenames=fn_test
        )

        # Save statistics
        stats_summary = {
            **self.stats,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_dim": X_train.shape[2]
        }
        
        with open(self.output_path / "preprocessing_stats.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE")
        print("=" * 70)
        print(f"Train: {X_train.shape}")
        print(f"Val  : {X_val.shape}")
        print(f"Test : {X_test.shape}")
        print(f"\nProcessed: {self.stats['processed_videos']}/{self.stats['total_videos']}")
        print(f"Failed: {self.stats['failed_videos']}")
        print(f"Interpolated frames: {self.stats['interpolated_frames']}")
        print("=" * 70)


if __name__ == "__main__":
    preprocessor = ImprovedBISINDOPreprocessor(
        dataset_path=r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video",
        output_path=r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed",
        target_frames=30,
        add_velocity=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    preprocessor.process_dataset()