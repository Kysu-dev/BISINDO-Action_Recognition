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


class BISINDOPreprocessor:

    def __init__(
        self,
        dataset_path,
        output_path,
        target_frames=30,
        use_rotation_alignment=True,
        confidence_threshold=0.5,
        smoothing_sigma=1.0
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

        # ✅ VALIDASI PATH (FIX WinError 3)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"❌ Dataset path not found: {self.dataset_path}")

        self.target_frames = target_frames
        self.use_rotation_alignment = use_rotation_alignment
        self.confidence_threshold = confidence_threshold
        self.smoothing_sigma = smoothing_sigma

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
        }

        print("\n" + "=" * 70)
        print("BISINDO Preprocessor Initialized")
        print("=" * 70)
        print(f"Dataset   : {self.dataset_path}")
        print(f"Output    : {self.output_path}")
        print(f"Frames    : {self.target_frames}")
        print(f"Features  : {self.feature_dim}")
        print("=" * 70 + "\n")

    def extract_landmarks(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            if not results.pose_landmarks:
                return None, 0.0

            coords = []
            conf = []

            for i in self.POSE_SUBSET:
                lm = results.pose_landmarks.landmark[i]
                coords.extend([lm.x, lm.y, lm.z])
                conf.append(lm.visibility)

            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                conf.extend([1.0] * self.n_hand)
            else:
                coords.extend([0.0] * self.n_hand * 3)
                conf.extend([0.0] * self.n_hand)

            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                conf.extend([1.0] * self.n_hand)
            else:
                coords.extend([0.0] * self.n_hand * 3)
                conf.extend([0.0] * self.n_hand)

            return np.array(coords), float(np.mean(conf))

        except Exception:
            return None, 0.0

    def uniform_sampling(self, seq):
        n, f = seq.shape
        if n == self.target_frames:
            return seq
        x_old = np.arange(n)
        x_new = np.linspace(0, n - 1, self.target_frames)
        out = np.zeros((self.target_frames, f))
        for i in range(f):
            out[:, i] = np.interp(x_new, x_old, seq[:, i])
        return out

    def normalize(self, seq):
        n_frames = seq.shape[0]
        n_landmarks = self.feature_dim // 3
        reshaped = seq.reshape(n_frames, n_landmarks, 3)
        out = np.zeros_like(reshaped)

        for t in range(n_frames):
            frame = reshaped[t]
            pose = frame[:self.n_pose]

            hip = (pose[6] + pose[7]) / 2
            frame -= hip

            shoulder_width = np.linalg.norm(pose[0] - pose[1])
            if shoulder_width < 1e-6:
                shoulder_width = 1.0
            frame /= shoulder_width

            if self.use_rotation_alignment:
                v = pose[1] - pose[0]
                angle = np.arctan2(v[1], v[0])
                c, s = np.cos(-angle), np.sin(-angle)
                x, y = frame[:, 0], frame[:, 1]
                frame[:, 0] = c * x - s * y
                frame[:, 1] = s * x + c * y

            out[t] = frame

        return out.reshape(n_frames, -1)

    def process_video(self, video_path):
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                lm, conf = self.extract_landmarks(frame)
                if lm is not None and conf >= self.confidence_threshold:
                    frames.append(lm)
                elif conf < self.confidence_threshold:
                    self.stats["low_confidence_frames"] += 1

            if len(frames) < 10:
                return None

            seq = np.array(frames)
            seq = self.uniform_sampling(seq)
            if self.smoothing_sigma > 0:
                for i in range(seq.shape[1]):
                    seq[:, i] = gaussian_filter1d(seq[:, i], self.smoothing_sigma)
            seq = self.normalize(seq)

            return seq

        finally:
            if cap is not None:
                cap.release()

    def process_dataset(self):
        class_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
        label_map = {c.name: i for i, c in enumerate(class_folders)}

        with open(self.output_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        X, y = [], []

        for cname, label in label_map.items():
            videos = list((self.dataset_path / cname).glob("*.mp4"))
            print(f"\n📁 {cname} ({len(videos)} videos)")

            for i, v in enumerate(videos, 1):
                print(f"   [{i}/{len(videos)}] {v.name}")
                self.stats["total_videos"] += 1
                seq = self.process_video(v)
                if seq is not None:
                    X.append(seq)
                    y.append(label)
                    self.stats["processed_videos"] += 1
                else:
                    self.stats["failed_videos"] += 1

        X = np.array(X)
        y = np.array(y)

        np.savez_compressed(
            self.output_path / "bisindo_dataset.npz",
            X=X,
            y=y,
            label_map=label_map
        )

        print("\n✅ DONE")
        print("Shape X:", X.shape)
        print("Shape y:", y.shape)


if __name__ == "__main__":
    pre = BISINDOPreprocessor(
        dataset_path=r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\raw_video",
        output_path=r"D:\SEMESTER 5\Computer Visual\baruu\BISINDO-Action_Recognition\data\processed"
    )
    pre.process_dataset()
