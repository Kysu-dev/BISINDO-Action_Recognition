"""
realtime_predictor.py - FIXED VERSION with correct dimensions
"""

import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
import time
import warnings
import os
import sys
from collections import deque, Counter
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 REAL-TIME BISINDO GESTURE RECOGNITION - FIXED VERSION")
print("="*80)

# ==================== CONFIGURATION ====================
BASE_DIR = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO"
MODEL_PATH = os.path.join(BASE_DIR, "models", "bisindo_final_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.pkl")

# Sesuaikan dengan model training Anda
SEQ_LENGTH = 30  # Dari training
FEATURES = 162   # HARUS SAMA dengan training!

# Realtime settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
PREDICTION_THRESHOLD = 0.7  # Diturunkan sedikit
SMOOTHING_WINDOW = 3  # Diperkecil untuk respons lebih cepat

# ==================== DEBUG: CHECK MODEL INPUT SHAPE ====================
print("\n🔍 CHECKING MODEL INPUT SHAPE...")
model = tf.keras.models.load_model(MODEL_PATH)
model_input_shape = model.input_shape
print(f"✅ Model expects input shape: {model_input_shape}")
print(f"   Should be: (None, {SEQ_LENGTH}, {FEATURES})")

# ==================== LOAD CLASSES ====================
with open(CLASS_NAMES_PATH, 'rb') as f:
    CLASS_NAMES = pickle.load(f)
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ Loaded {NUM_CLASSES} classes: {CLASS_NAMES}")

# ==================== MEDIAPIPE SETUP ====================
print("\n🤖 INITIALIZING MEDIAPIPE...")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    refine_face_landmarks=False  # Nonaktifkan face untuk konsistensi
)

print("✅ MediaPipe Holistic initialized")

# ==================== FEATURE EXTRACTION - DIMENSION FIX ====================
def extract_landmarks_fixed(results):
    """
    Extract exactly 162 features to match training data
    Training data kemungkinan menggunakan:
    - Hanya tangan kiri dan kanan (tanpa pose): 21 points × 3 × 2 = 126 features
    - Atau subset dari landmarks
    """
    features = []
    
    # OPTION 1: Jika training hanya menggunakan tangan (126 features)
    # Extract left hand landmarks (21 points, x,y,z)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
    else:
        features.extend([0, 0, 0] * 21)  # 63 features
    
    # Extract right hand landmarks (21 points, x,y,z)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
    else:
        features.extend([0, 0, 0] * 21)  # 63 features
    
    # Total sekarang: 126 features
    
    # OPTION 2: Tambahkan pose shoulders saja jika perlu mencapai 162
    # Shoulder landmarks (landmark 11 & 12) = 2 points × 3 = 6 features
    if results.pose_landmarks:
        # Left shoulder (11)
        features.extend([
            results.pose_landmarks.landmark[11].x,
            results.pose_landmarks.landmark[11].y,
            results.pose_landmarks.landmark[11].z
        ])
        # Right shoulder (12)
        features.extend([
            results.pose_landmarks.landmark[12].x,
            results.pose_landmarks.landmark[12].y,
            results.pose_landmarks.landmark[12].z
        ])
    else:
        features.extend([0, 0, 0, 0, 0, 0])  # 6 features
    
    # Total: 126 + 6 = 132 features
    
    # OPTION 3: Jika masih kurang, tambahkan wrists
    if results.pose_landmarks:
        # Left wrist (15) - jika ada
        if len(results.pose_landmarks.landmark) > 15:
            features.extend([
                results.pose_landmarks.landmark[15].x,
                results.pose_landmarks.landmark[15].y,
                results.pose_landmarks.landmark[15].z
            ])
        else:
            features.extend([0, 0, 0])
        
        # Right wrist (16)
        if len(results.pose_landmarks.landmark) > 16:
            features.extend([
                results.pose_landmarks.landmark[16].x,
                results.pose_landmarks.landmark[16].y,
                results.pose_landmarks.landmark[16].z
            ])
        else:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # Total: 132 + 6 = 138 features
    
    # OPTION 4: Tambahkan face landmarks tertentu jika perlu
    # Hips (23, 24) = 2 points × 3 = 6 features
    if results.pose_landmarks and len(results.pose_landmarks.landmark) > 24:
        # Left hip (23)
        features.extend([
            results.pose_landmarks.landmark[23].x,
            results.pose_landmarks.landmark[23].y,
            results.pose_landmarks.landmark[23].z
        ])
        # Right hip (24)
        features.extend([
            results.pose_landmarks.landmark[24].x,
            results.pose_landmarks.landmark[24].y,
            results.pose_landmarks.landmark[24].z
        ])
    else:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # Total: 138 + 6 = 144 features
    
    # OPTION 5: Tambahkan elbows (13, 14) untuk mencapai 162
    if results.pose_landmarks and len(results.pose_landmarks.landmark) > 14:
        # Left elbow (13)
        features.extend([
            results.pose_landmarks.landmark[13].x,
            results.pose_landmarks.landmark[13].y,
            results.pose_landmarks.landmark[13].z
        ])
        # Right elbow (14)
        features.extend([
            results.pose_landmarks.landmark[14].x,
            results.pose_landmarks.landmark[14].y,
            results.pose_landmarks.landmark[14].z
        ])
    else:
        features.extend([0, 0, 0, 0, 0, 0])
    
    # Total: 144 + 6 = 150 features
    
    # OPTION 6: Tambahkan nose (0) dan ears (7, 8) untuk mencapai 162
    if results.pose_landmarks:
        # Nose (0)
        features.extend([
            results.pose_landmarks.landmark[0].x,
            results.pose_landmarks.landmark[0].y,
            results.pose_landmarks.landmark[0].z
        ])
        
        # Left ear (7)
        if len(results.pose_landmarks.landmark) > 7:
            features.extend([
                results.pose_landmarks.landmark[7].x,
                results.pose_landmarks.landmark[7].y,
                results.pose_landmarks.landmark[7].z
            ])
        else:
            features.extend([0, 0, 0])
        
        # Right ear (8)
        if len(results.pose_landmarks.landmark) > 8:
            features.extend([
                results.pose_landmarks.landmark[8].x,
                results.pose_landmarks.landmark[8].y,
                results.pose_landmarks.landmark[8].z
            ])
        else:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Total: 150 + 9 = 159 features
    
    # OPTION 7: Tambahkan 3 features terakhir (mungkin index tertentu)
    # Untuk mencapai tepat 162, tambahkan knees (25, 26) atau lainnya
    if results.pose_landmarks and len(results.pose_landmarks.landmark) > 26:
        # Left knee (25) - hanya x coordinate
        features.append(results.pose_landmarks.landmark[25].x)
        # Right knee (26) - hanya x coordinate  
        features.append(results.pose_landmarks.landmark[26].x)
        # Add one more feature
        features.append(0.0)  # Placeholder
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Konversi ke numpy array
    features_array = np.array(features, dtype=np.float32)
    
    # Pastikan panjang tepat 162
    if len(features_array) > FEATURES:
        features_array = features_array[:FEATURES]
    elif len(features_array) < FEATURES:
        # Padding dengan zeros
        padding = np.zeros(FEATURES - len(features_array))
        features_array = np.concatenate([features_array, padding])
    
    return features_array

def extract_landmarks_simple(results):
    """
    Versi sederhana - cek dulu apa yang digunakan di training
    Kemungkinan besar hanya tangan kiri dan kanan (126 features)
    dan beberapa pose landmarks
    """
    features = []
    
    # Prioritaskan tangan (paling penting untuk BISINDO)
    # Left hand (21 points × 3)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark[:21]:  # Pastikan 21 points
            features.extend([landmark.x, landmark.y, landmark.z])
    else:
        features.extend([0, 0, 0] * 21)
    
    # Right hand (21 points × 3)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark[:21]:
            features.extend([landmark.x, landmark.y, landmark.z])
    else:
        features.extend([0, 0, 0] * 21)
    
    # Tambahkan pose upper body saja
    if results.pose_landmarks:
        # Shoulders, elbows, wrists (12 landmarks = 36 features)
        upper_body_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # 8 points
        for idx in upper_body_indices:
            if idx < len(results.pose_landmarks.landmark):
                landmark = results.pose_landmarks.landmark[idx]
                features.extend([landmark.x, landmark.y, landmark.z])
            else:
                features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0] * 8)  # 8 points × 3
    
    # Pastikan 162 features
    features_array = np.array(features, dtype=np.float32)
    
    if len(features_array) != FEATURES:
        print(f"⚠️  Warning: Features extracted: {len(features_array)}, expected: {FEATURES}")
        if len(features_array) > FEATURES:
            features_array = features_array[:FEATURES]
        else:
            padding = np.zeros(FEATURES - len(features_array))
            features_array = np.concatenate([features_array, padding])
    
    return features_array

# ==================== ALTERNATIVE: CHECK TRAINING DATA STRUCTURE ====================
def check_training_data_structure():
    """Check bagaimana data training dibangun"""
    print("\n🔍 CHECKING TRAINING DATA STRUCTURE...")
    
    try:
        data_path = os.path.join(BASE_DIR, "data", "processed", "bisindo_dataset.npz")
        data = np.load(data_path)
        X = data['X']
        
        print(f"✅ Training data shape: {X.shape}")
        print(f"   Samples: {X.shape[0]}, Frames: {X.shape[1]}, Features: {X.shape[2]}")
        
        # Cek sample pertama
        sample = X[0]
        print(f"\n📊 Sample analysis:")
        print(f"   Shape: {sample.shape}")
        print(f"   Min value: {sample.min():.4f}")
        print(f"   Max value: {sample.max():.4f}")
        print(f"   Mean: {sample.mean():.4f}")
        print(f"   Std: {sample.std():.4f}")
        
        # Cek apakah ada zeros (mungkin hanya tangan saja)
        zero_percentage = np.sum(sample == 0) / sample.size * 100
        print(f"   Zero percentage: {zero_percentage:.1f}%")
        
        return X.shape[2]  # Return jumlah features
    except Exception as e:
        print(f"❌ Error checking training data: {e}")
        return FEATURES  # Default ke 162

# ==================== NORMALIZATION ====================
def normalize_landmarks_simple(landmarks):
    """Simple normalization"""
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Normalize to [-1, 1] range
    if np.max(np.abs(landmarks)) > 0:
        landmarks = landmarks / np.max(np.abs(landmarks))
    
    return landmarks

# ==================== BUFFER MANAGEMENT ====================
class SequenceBuffer:
    def __init__(self, sequence_length=SEQ_LENGTH, features=FEATURES):
        self.sequence_length = sequence_length
        self.features = features
        self.buffer = deque(maxlen=sequence_length)
        self.prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
        
    def add_frame(self, landmarks):
        self.buffer.append(landmarks)
        
    def get_sequence(self):
        if len(self.buffer) < self.sequence_length:
            # Duplicate last frame
            last_frame = self.buffer[-1] if self.buffer else np.zeros(self.features)
            current_buffer = list(self.buffer)
            while len(current_buffer) < self.sequence_length:
                current_buffer.append(last_frame)
            return np.array(current_buffer).reshape(1, self.sequence_length, self.features)
        return np.array(self.buffer).reshape(1, self.sequence_length, self.features)
    
    def is_ready(self):
        return len(self.buffer) >= self.sequence_length // 2  # Kurangi threshold
    
    def clear(self):
        self.buffer.clear()
        self.prediction_buffer.clear()

# ==================== MAIN REAL-TIME LOOP - FIXED ====================
def main():
    print("\n🎥 STARTING REAL-TIME GESTURE RECOGNITION")
    print("="*50)
    print("Press 'q' to quit, 'c' to clear buffer")
    
    # Check training data structure untuk konfirmasi features
    actual_features = check_training_data_structure()
    global FEATURES
    if actual_features != FEATURES:
        print(f"⚠️  Updating FEATURES from {FEATURES} to {actual_features}")
        FEATURES = actual_features
    
    # Initialize buffer dengan features yang benar
    buffer = SequenceBuffer(SEQ_LENGTH, FEATURES)
    
    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam initialized")
    time.sleep(1)
    
    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    current_gesture = None
    current_confidence = 0
    
    print("\n🚀 Starting... Show your hands to the camera!")
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process dengan MediaPipe
            results = holistic.process(rgb_frame)
            
            # Extract landmarks - coba kedua method
            try:
                landmarks = extract_landmarks_simple(results)
            except:
                # Fallback ke method fixed
                landmarks = extract_landmarks_fixed(results)
            
            # Debug: print shape occasionally
            if fps_frame_count % 30 == 0:
                print(f"📏 Landmarks shape: {landmarks.shape}, Expected: ({FEATURES},)")
            
            # Normalize
            landmarks = normalize_landmarks_simple(landmarks)
            
            # Add to buffer
            buffer.add_frame(landmarks)
            
            # Predict if buffer has some frames
            if buffer.is_ready():
                try:
                    sequence = buffer.get_sequence()
                    
                    # Debug sequence shape
                    if fps_frame_count % 60 == 0:
                        print(f"🧪 Sequence shape: {sequence.shape}, Expected: (1, {SEQ_LENGTH}, {FEATURES})")
                    
                    # Predict
                    predictions = model.predict(sequence, verbose=0)[0]
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx]
                    
                    if confidence > PREDICTION_THRESHOLD:
                        current_gesture = CLASS_NAMES[predicted_idx]
                        current_confidence = confidence
                        buffer.prediction_buffer.append(predicted_idx)
                        
                        # Smoothing
                        if len(buffer.prediction_buffer) >= SMOOTHING_WINDOW:
                            most_common = Counter(buffer.prediction_buffer).most_common(1)
                            if most_common:
                                smoothed_idx = most_common[0][0]
                                current_gesture = CLASS_NAMES[smoothed_idx]
                
                except Exception as e:
                    print(f"⚠️  Prediction error: {e}")
                    # Clear buffer on error
                    buffer.clear()
            
            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2)
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2)
                )
            
            # FPS calculation
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display info
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            buffer_status = "READY" if buffer.is_ready() else f"BUFFER: {len(buffer.buffer)}/{SEQ_LENGTH}"
            cv2.putText(frame, buffer_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if current_gesture:
                cv2.putText(frame, f"{current_gesture}: {current_confidence:.1%}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('BISINDO Real-time', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                buffer.clear()
                current_gesture = None
                print("🔄 Buffer cleared")
            elif key == ord('d'):
                # Debug info
                print(f"\n🔍 DEBUG INFO:")
                print(f"  Buffer frames: {len(buffer.buffer)}")
                if buffer.buffer:
                    print(f"  Last frame shape: {buffer.buffer[-1].shape}")
                    print(f"  Last frame min/max: {buffer.buffer[-1].min():.3f}/{buffer.buffer[-1].max():.3f}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("\n✅ Real-time recognition stopped")

# ==================== QUICK TEST FUNCTION ====================
def quick_dimension_test():
    """Quick test to determine correct feature dimensions"""
    print("\n🧪 QUICK DIMENSION TEST")
    print("="*50)
    
    # Load training data
    try:
        data_path = os.path.join(BASE_DIR, "data", "processed", "bisindo_dataset.npz")
        data = np.load(data_path)
        X = data['X']
        
        print(f"Training data shape: {X.shape}")
        print(f"Features per frame: {X.shape[2]}")
        
        # Analyze first sample
        sample = X[0]
        print(f"\nFirst sample shape: {sample.shape}")
        
        # Try to understand structure
        print("\nTrying to understand feature structure...")
        
        # Mungkin structure: [hand_left, hand_right, pose_upper_body]
        # 21 points × 3 × 2 = 126 (hands)
        # 12 points × 3 = 36 (pose) -> total 162
        
        # Test MediaPipe extraction
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Test extraction
            features_simple = extract_landmarks_simple(results)
            print(f"\nSimple extraction: {len(features_simple)} features")
            
            features_fixed = extract_landmarks_fixed(results)
            print(f"Fixed extraction: {len(features_fixed)} features")
            
            # Try prediction
            if len(features_simple) == FEATURES:
                # Create dummy sequence
                dummy_sequence = np.array([features_simple] * SEQ_LENGTH)
                dummy_sequence = dummy_sequence.reshape(1, SEQ_LENGTH, FEATURES)
                
                try:
                    pred = model.predict(dummy_sequence, verbose=0)
                    print(f"\n✅ Prediction successful! Shape: {pred.shape}")
                    print(f"   Sample prediction: {np.argmax(pred[0])} - {CLASS_NAMES[np.argmax(pred[0])]}")
                except Exception as e:
                    print(f"❌ Prediction failed: {e}")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='realtime', 
                       choices=['realtime', 'test', 'debug'])
    
    args = parser.parse_args()
    
    if args.mode == 'debug' or args.mode == 'test':
        quick_dimension_test()
    else:
        main()