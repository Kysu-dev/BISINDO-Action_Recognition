# preprocess_fixed.py - WITH BETTER PROGRESS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

print("🚀 BISINDO Preprocessing - Fixed Version")

# Setup MediaPipe
mp_holistic = mp.solutions.holistic

# Config
DATASET_PATH = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\raw_video"
OUTPUT_PATH = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\processed"
SEQUENCE_LENGTH = 30
MAX_VIDEOS = 40  # Max videos per class

# Get classes
classes = [c for c in os.listdir(DATASET_PATH) 
           if os.path.isdir(os.path.join(DATASET_PATH, c))]
classes = sorted(classes)[:13]  # Take first 13

print(f"📁 Processing {len(classes)} classes")

# Create output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

all_data = []
all_labels = []

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(DATASET_PATH, class_name)
        video_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith('.mp4')]
        video_files = video_files[:MAX_VIDEOS]
        
        print(f"\n📂 {class_name} ({len(video_files)} videos)")
        
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(class_path, video_file)
            
            # Progress
            if (i + 1) % 5 == 0:
                print(f"   Video {i+1}/{len(video_files)}")
            
            try:
                # Process video
                cap = cv2.VideoCapture(video_path)
                frames = []
                
                while len(frames) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize
                    frame = cv2.resize(frame, (256, 256))
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    
                    # MediaPipe
                    results = holistic.process(image)
                    
                    # Extract keypoints
                    keypoints = []
                    
                    # Pose
                    if results.pose_landmarks:
                        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                                        for lm in results.pose_landmarks.landmark])
                        pose = pose[[0, 11, 12, 13, 14, 15, 16, 23, 24]].flatten()
                    else:
                        pose = np.zeros(9 * 4)
                    
                    # Left hand
                    if results.left_hand_landmarks:
                        lh = np.array([[lm.x, lm.y, lm.z] 
                                      for lm in results.left_hand_landmarks.landmark]).flatten()
                    else:
                        lh = np.zeros(21 * 3)
                    
                    # Right hand
                    if results.right_hand_landmarks:
                        rh = np.array([[lm.x, lm.y, lm.z] 
                                      for lm in results.right_hand_landmarks.landmark]).flatten()
                    else:
                        rh = np.zeros(21 * 3)
                    
                    keypoints = np.concatenate([pose, lh, rh])
                    frames.append(keypoints)
                
                cap.release()
                
                # Padding
                if len(frames) < SEQUENCE_LENGTH:
                    frames.extend([frames[-1]] * (SEQUENCE_LENGTH - len(frames)))
                
                all_data.append(frames[:SEQUENCE_LENGTH])
                all_labels.append(class_idx)
                
            except Exception as e:
                print(f"   ❌ Error: {video_file} - {e}")
                continue
    
    # Convert to numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    # Save
    npz_path = os.path.join(OUTPUT_PATH, "bisindo_dataset.npz")
    np.savez_compressed(npz_path, X=X, y=y)
    
    # Metadata
    metadata = {
        'classes': classes,
        'sequence_length': SEQUENCE_LENGTH,
        'total_samples': len(X),
        'features_per_frame': X.shape[2]
    }
    
    with open(os.path.join(OUTPUT_PATH, "metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ COMPLETED!")
    print(f"📊 Total samples: {len(X)}")
    print(f"📐 Data shape: {X.shape}")
    print(f"💾 Saved to: {npz_path}")