# preprocess_bisindo_final.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import pickle

print("🚀 BISINDO Preprocessing - FINAL VERSION")

# ================= CONFIG =================
DATASET_PATH = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\raw_video"
OUTPUT_PATH = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\processed"

SEQUENCE_LENGTH = 30
MAX_VIDEOS = 40
IMAGE_SIZE = 256

# =========================================

mp_holistic = mp.solutions.holistic

classes = sorted([
    c for c in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, c))
])[:13]

print(f"📁 Total classes: {len(classes)}")

os.makedirs(OUTPUT_PATH, exist_ok=True)

all_data, all_labels = [], []

# ================= UTILITIES =================

def uniform_sampling(total_frames, num_samples):
    if total_frames <= num_samples:
        return np.linspace(0, total_frames - 1, total_frames, dtype=int)
    return np.linspace(0, total_frames - 1, num_samples, dtype=int)

def normalize_landmarks(landmarks, origin):
    return landmarks - origin

# =============================================

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(DATASET_PATH, class_name)
        videos = [v for v in os.listdir(class_path) if v.endswith(".mp4")]
        videos = videos[:MAX_VIDEOS]

        print(f"\n📂 {class_name} | {len(videos)} videos")

        for vid_idx, video in enumerate(videos):
            video_path = os.path.join(class_path, video)
            cap = cv2.VideoCapture(video_path)

            frames_raw = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_raw.append(frame)

            cap.release()

            if len(frames_raw) < 10:
                continue

            sampled_idx = uniform_sampling(len(frames_raw), SEQUENCE_LENGTH)
            sequence = []

            for idx in sampled_idx:
                frame = cv2.resize(frames_raw[idx], (IMAGE_SIZE, IMAGE_SIZE))
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                # ================= POSE =================
                if results.pose_landmarks:
                    pose = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.pose_landmarks.landmark
                    ])

                    # landmark penting saja
                    pose = pose[[0, 11, 12, 13, 14, 15, 16, 23, 24]]

                    # normalisasi ke bahu kiri
                    origin = pose[1]
                    pose = normalize_landmarks(pose, origin).flatten()
                else:
                    pose = np.zeros(9 * 3)

                # ================= HANDS =================
                if results.left_hand_landmarks:
                    lh = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.left_hand_landmarks.landmark
                    ])
                    lh = normalize_landmarks(lh, lh[0]).flatten()
                else:
                    lh = np.zeros(21 * 3)

                if results.right_hand_landmarks:
                    rh = np.array([
                        [lm.x, lm.y, lm.z]
                        for lm in results.right_hand_landmarks.landmark
                    ])
                    rh = normalize_landmarks(rh, rh[0]).flatten()
                else:
                    rh = np.zeros(21 * 3)

                # skip frame kosong total
                if np.sum(pose) == 0 and np.sum(lh) == 0 and np.sum(rh) == 0:
                    continue

                keypoints = np.concatenate([pose, lh, rh])
                sequence.append(keypoints)

            # Padding akhir
            if len(sequence) == 0:
                continue

            while len(sequence) < SEQUENCE_LENGTH:
                sequence.append(sequence[-1])

            all_data.append(sequence[:SEQUENCE_LENGTH])
            all_labels.append(class_idx)

            if (vid_idx + 1) % 5 == 0:
                print(f"   ✔ Processed {vid_idx+1}/{len(videos)}")

# ================= SAVE =================

X = np.array(all_data)
y = np.array(all_labels)

np.savez_compressed(
    os.path.join(OUTPUT_PATH, "bisindo_dataset_final.npz"),
    X=X,
    y=y
)

metadata = {
    "classes": classes,
    "sequence_length": SEQUENCE_LENGTH,
    "features_per_frame": X.shape[2],
    "total_samples": len(X)
}

with open(os.path.join(OUTPUT_PATH, "metadata.pkl"), "wb") as f:
    pickle.dump(metadata, f)

print("\n✅ PREPROCESS SELESAI")
print(f"📊 Shape X : {X.shape}")
print(f"📊 Shape y : {y.shape}")
print(f"💾 Output : bisindo_dataset_final.npz")
