import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from pathlib import Path
import json
from scipy.ndimage import gaussian_filter1d


class BISINDORealtimeRecognition:
    """
    Real-time BISINDO Sign Language Recognition
    Menggunakan webcam untuk detect gesture secara live
    """
    
    def __init__(
        self,
        model_path,
        label_map_path,
        target_frames=30,
        confidence_threshold=0.5,
        smoothing_window=3
    ):
        """
        Args:
            model_path: Path ke trained model (.h5)
            label_map_path: Path ke label_map.json
            target_frames: Jumlah frame untuk prediction
            confidence_threshold: Minimum confidence untuk show prediction
            smoothing_window: Window untuk smoothing predictions
        """
        print("\n" + "="*70)
        print("🎬 BISINDO REAL-TIME RECOGNITION SYSTEM")
        print("="*70)
        
        # Load model
        print("📦 Loading model...")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Load label map
        print("📋 Loading label map...")
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        print(f"✅ {len(self.label_map)} classes loaded")
        
        # Configuration
        self.target_frames = target_frames
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Initialize MediaPipe
        print("🤖 Initializing MediaPipe...")
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe initialized")
        
        # Landmark configuration
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3
        
        # Frame buffer untuk collect sequence
        self.frame_buffer = deque(maxlen=target_frames * 2)
        self.landmark_buffer = []
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # Recording state
        self.is_recording = False
        self.recording_countdown = 0
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'confident_predictions': 0,
            'fps': 0
        }
        
        print("="*70)
        print("🎯 System ready!")
        print("="*70 + "\n")
    
    def extract_landmarks(self, frame):
        """Extract landmarks dari frame"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.holistic.process(rgb)
            
            if not results.pose_landmarks:
                return None, None, 0.0
            
            # Initialize landmarks array
            landmarks = np.zeros(self.feature_dim, dtype=np.float32)
            confidences = []
            idx = 0
            
            # Pose landmarks
            for lm_idx in self.POSE_SUBSET:
                lm = results.pose_landmarks.landmark[lm_idx]
                landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                confidences.append(lm.visibility)
                idx += 3
            
            # Left hand
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                idx += self.n_hand * 3
                confidences.extend([0.0] * self.n_hand)
            
            # Right hand
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3
                confidences.extend([1.0] * self.n_hand)
            else:
                landmarks[idx:idx + (self.n_hand * 3)] = np.nan
                confidences.extend([0.0] * self.n_hand)
            
            avg_confidence = np.mean(confidences)
            
            return results, landmarks, avg_confidence
            
        except Exception as e:
            return None, None, 0.0
    
    def interpolate_missing(self, sequence):
        """Interpolasi missing values"""
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
    
    def resample_sequence(self, sequence):
        """Resample sequence ke target frames"""
        if len(sequence) == self.target_frames:
            return np.array(sequence)
        
        n_frames_orig = len(sequence)
        n_features = sequence[0].shape[0]
        
        x_old = np.linspace(0, 1, n_frames_orig)
        x_new = np.linspace(0, 1, self.target_frames)
        
        sequence_array = np.array(sequence)
        resampled = np.zeros((self.target_frames, n_features), dtype=np.float32)
        
        for i in range(n_features):
            resampled[:, i] = np.interp(x_new, x_old, sequence_array[:, i])
        
        return resampled
    
    def normalize_sequence(self, sequence):
        """Normalize sequence (sama seperti preprocessing)"""
        n_frames = sequence.shape[0]
        n_landmarks = self.feature_dim // 3
        
        reshaped = sequence.reshape(n_frames, n_landmarks, 3)
        normalized = np.zeros_like(reshaped)
        
        for t in range(n_frames):
            frame = reshaped[t].copy()
            pose = frame[:self.n_pose]
            
            # Hip-centered
            hip_center = (pose[6] + pose[7]) / 2
            frame -= hip_center
            
            # Shoulder scaling
            shoulder_dist = np.linalg.norm(pose[0] - pose[1])
            if shoulder_dist > 1e-6:
                frame /= shoulder_dist
            
            # Rotation alignment
            shoulder_vec = pose[1] - pose[0]
            angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            x, y = frame[:, 0].copy(), frame[:, 1].copy()
            frame[:, 0] = cos_a * x - sin_a * y
            frame[:, 1] = sin_a * x + cos_a * y
            
            normalized[t] = frame
        
        return normalized.reshape(n_frames, -1)
    
    def preprocess_sequence(self, landmarks_list):
        """Preprocess sequence untuk prediction"""
        if len(landmarks_list) < 10:
            return None
        
        # Convert to numpy
        sequence = np.array(landmarks_list, dtype=np.float32)
        
        # Interpolate missing values
        sequence = self.interpolate_missing(sequence)
        
        # Resample to target frames
        sequence = self.resample_sequence(landmarks_list)
        
        # Smoothing
        for i in range(sequence.shape[1]):
            sequence[:, i] = gaussian_filter1d(sequence[:, i], sigma=1.0)
        
        # Normalize
        sequence = self.normalize_sequence(sequence)
        
        return sequence
    
    def predict(self, sequence):
        """Predict gesture dari sequence"""
        if sequence is None:
            return None, 0.0
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence, axis=0)
        
        # Predict
        predictions = self.model.predict(sequence_batch, verbose=0)[0]
        
        # Get top prediction
        pred_class = np.argmax(predictions)
        pred_confidence = predictions[pred_class]
        pred_label = self.id_to_label[pred_class]
        
        self.stats['total_predictions'] += 1
        if pred_confidence >= self.confidence_threshold:
            self.stats['confident_predictions'] += 1
        
        return pred_label, pred_confidence
    
    def smooth_prediction(self, label, confidence):
        """Smooth predictions menggunakan voting"""
        self.prediction_history.append((label, confidence))
        
        if len(self.prediction_history) < self.smoothing_window:
            return label, confidence
        
        # Voting dari recent predictions
        recent_labels = [p[0] for p in self.prediction_history]
        recent_confidences = [p[1] for p in self.prediction_history]
        
        # Find most common label
        unique_labels = list(set(recent_labels))
        label_counts = [recent_labels.count(l) for l in unique_labels]
        most_common_idx = np.argmax(label_counts)
        smoothed_label = unique_labels[most_common_idx]
        
        # Average confidence for that label
        label_confidences = [
            c for l, c in self.prediction_history 
            if l == smoothed_label
        ]
        smoothed_confidence = np.mean(label_confidences)
        
        return smoothed_label, smoothed_confidence
    
    def draw_landmarks(self, frame, results):
        """Draw landmarks pada frame"""
        if results is None:
            return frame
        
        # Draw pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw hands
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        return frame
    
    def draw_ui(self, frame, prediction=None, confidence=0.0, fps=0):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay untuk info
        overlay = frame.copy()
        
        # Recording indicator
        if self.is_recording:
            if self.recording_countdown > 0:
                # Countdown
                cv2.circle(overlay, (w - 50, 50), 30, (0, 165, 255), -1)
                cv2.putText(
                    overlay, str(self.recording_countdown),
                    (w - 65, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
                )
            else:
                # Recording
                cv2.circle(overlay, (w - 50, 50), 25, (0, 0, 255), -1)
                cv2.putText(
                    overlay, "REC",
                    (w - 90, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
        
        # Info panel
        info_height = 180
        cv2.rectangle(overlay, (10, 10), (400, info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Buffer status
        buffer_pct = len(self.landmark_buffer) / self.target_frames * 100
        buffer_color = (0, 255, 0) if len(self.landmark_buffer) >= self.target_frames else (0, 165, 255)
        cv2.putText(
            frame, f"Buffer: {len(self.landmark_buffer)}/{self.target_frames} ({buffer_pct:.0f}%)",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2
        )
        
        # Prediction
        if prediction and confidence >= self.confidence_threshold:
            # Prediction panel
            pred_panel_h = 120
            cv2.rectangle(frame, (10, h - pred_panel_h - 10), (w - 10, h - 10), (0, 0, 0), -1)
            
            # Confidence bar
            bar_width = int((w - 40) * confidence)
            bar_color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255) if confidence > 0.6 else (0, 255, 255)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), bar_color, -1)
            cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (255, 255, 255), 2)
            
            # Prediction text
            cv2.putText(
                frame, f"Prediction: {prediction}",
                (20, h - pred_panel_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            
            cv2.putText(
                frame, f"Confidence: {confidence:.2%}",
                (20, h - pred_panel_h + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop Recording",
            "R: Reset Buffer",
            "Q: Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(
                frame, text,
                (20, 100 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
        
        return frame
    
    def run(self):
        """Main loop untuk real-time recognition"""
        print("\n🎥 Starting webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Webcam started")
        print("\n" + "="*70)
        print("CONTROLS:")
        print("  SPACE : Start/Stop recording gesture")
        print("  R     : Reset buffer")
        print("  Q     : Quit")
        print("="*70 + "\n")
        
        # Variables
        fps_list = deque(maxlen=30)
        current_prediction = None
        current_confidence = 0.0
        
        try:
            while True:
                start_time = cv2.getTickCount()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Extract landmarks
                results, landmarks, conf = self.extract_landmarks(frame)
                
                # Draw landmarks
                frame = self.draw_landmarks(frame, results)
                
                # Recording logic
                if self.is_recording:
                    if self.recording_countdown > 0:
                        self.recording_countdown -= 1
                    elif landmarks is not None:
                        self.landmark_buffer.append(landmarks)
                        
                        # Auto predict ketika buffer penuh
                        if len(self.landmark_buffer) >= self.target_frames:
                            sequence = self.preprocess_sequence(self.landmark_buffer)
                            if sequence is not None:
                                pred_label, pred_conf = self.predict(sequence)
                                current_prediction, current_confidence = self.smooth_prediction(
                                    pred_label, pred_conf
                                )
                            
                            # Keep last 50% of buffer for continuous prediction
                            keep_frames = self.target_frames // 2
                            self.landmark_buffer = list(self.landmark_buffer)[-keep_frames:]
                
                # Calculate FPS
                end_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (end_time - start_time)
                fps_list.append(fps)
                avg_fps = np.mean(fps_list)
                
                # Draw UI
                frame = self.draw_ui(
                    frame, 
                    current_prediction, 
                    current_confidence, 
                    avg_fps
                )
                
                # Show frame
                cv2.imshow('BISINDO Real-time Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord(' '):
                    # Toggle recording
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("🔴 Recording started (3 seconds countdown)...")
                        self.recording_countdown = int(avg_fps * 3)  # 3 seconds countdown
                        self.landmark_buffer = []
                        current_prediction = None
                        current_confidence = 0.0
                    else:
                        print("⏸️  Recording stopped")
                elif key == ord('r'):
                    # Reset buffer
                    print("🔄 Buffer reset")
                    self.landmark_buffer = []
                    current_prediction = None
                    current_confidence = 0.0
                    self.is_recording = False
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            
            # Print statistics
            print("\n" + "="*70)
            print("📊 SESSION STATISTICS")
            print("="*70)
            print(f"Total predictions      : {self.stats['total_predictions']}")
            print(f"Confident predictions  : {self.stats['confident_predictions']}")
            if self.stats['total_predictions'] > 0:
                conf_rate = self.stats['confident_predictions'] / self.stats['total_predictions'] * 100
                print(f"Confidence rate        : {conf_rate:.1f}%")
            print("="*70)


def main():
    """Main function"""
    # Configuration
    MODEL_PATH = "models/best_model.h5"
    LABEL_MAP_PATH = "../data/processed/label_map.json"
    
    # Check if files exist
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please train the model first!")
        return
    
    if not Path(LABEL_MAP_PATH).exists():
        print(f"❌ Label map not found: {LABEL_MAP_PATH}")
        print("Please run preprocessing first!")
        return
    
    # Create recognition system
    recognizer = BISINDORealtimeRecognition(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        target_frames=30,
        confidence_threshold=0.5,
        smoothing_window=3
    )
    
    # Run real-time recognition
    recognizer.run()


if __name__ == "__main__":
    main()