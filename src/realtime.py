import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
import json
from collections import deque
import time

class GestureFilter:
    """Post-processing filter untuk menghilangkan false positives"""
    def __init__(self):
        self.last_gestures = deque(maxlen=10)
        self.last_confidences = deque(maxlen=10)
        self.neutral_count = 0
    
    def filter(self, gesture, confidence):
        self.last_gestures.append(gesture)
        self.last_confidences.append(confidence)
        
        # Jika gesture adalah neutral, hitung
        if gesture == "neutral":
            self.neutral_count = min(self.neutral_count + 1, 10)
        else:
            self.neutral_count = max(self.neutral_count - 1, 0)
        
        # Rule 1: Jika confidence rendah (< 0.7), cenderung ke neutral
        if confidence < 0.7:
            return "neutral", 1.0
        
        # Rule 2: Jika 7 dari 10 prediksi terakhir adalah "neutral", force neutral
        if list(self.last_gestures).count("neutral") >= 7:
            return "neutral", 1.0
        
        # Rule 3: Jika neutral_count tinggi, tapi dapat prediksi lain, cek konsistensi
        if self.neutral_count >= 8 and gesture != "neutral":
            # Cek apakah prediksi ini konsisten dalam 5 frame terakhir
            recent_gestures = list(self.last_gestures)[-5:]
            if recent_gestures.count(gesture) < 3:
                return "neutral", 1.0
        
        return gesture, confidence

class RealtimeBISINDO:
    def __init__(self, model_path, label_map_path, target_frames=40, flip_mode=False):
        print("🔄 Loading model...")
        self.model = load_model(model_path)
        
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.target_frames = target_frames
        self.flip_mode = flip_mode  # ✅ Mode flip bisa diatur
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark config
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=target_frames)
        self.is_recording = False
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=7)
        
        # Gesture history untuk logging
        self.gesture_history = []
        self.last_stable_gesture = None
        self.gesture_hold_frames = 0
        self.gesture_hold_threshold = 10
        
        # Neutral detection
        self.last_prediction_time = time.time()
        self.neutral_timeout = 1.5  # 1.5 detik tanpa gerakan -> neutral
        self.last_gesture_time = time.time()
        self.static_frame_count = 0
        self.static_threshold = 15  # 15 frame static -> neutral
        
        # Filter
        self.gesture_filter = GestureFilter()
        
        # Performance metrics
        self.total_predictions = 0
        self.start_time = time.time()
        
        # Debug counter
        self.debug_counter = 0
        self.last_motion_level = 0
        
        # Pastikan "neutral" ada di label_map
        if "neutral" not in self.label_map:
            print("⚠️  WARNING: 'neutral' not in label map! Adding it...")
            # Tambahkan neutral jika tidak ada
            neutral_id = len(self.label_map)
            self.label_map["neutral"] = neutral_id
            self.reverse_label_map[neutral_id] = "neutral"
        
        print(f"✅ Model loaded: {len(self.label_map)} classes")
        print(f"✅ Neutral detection: ENABLED (timeout: {self.neutral_timeout}s)")
        print(f"✅ Flip mode: {'ENABLED' if flip_mode else 'DISABLED'}")
    
    def extract_landmarks(self, frame, flip_horizontal=False):
        """Extract landmarks dengan mirror correction"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        
        landmarks = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0
        detected = False
        
        # POSE
        if results.pose_landmarks:
            for lm_idx in self.POSE_SUBSET:
                lm = results.pose_landmarks.landmark[lm_idx]
                x, y, z = lm.x, lm.y, lm.z
                
                if flip_horizontal:
                    x = 1.0 - x
                
                landmarks[idx:idx+3] = [x, y, z]
                idx += 3
            detected = True
        else:
            landmarks[idx:idx + (self.n_pose * 3)] = np.nan
            idx += self.n_pose * 3
        
        # Swap hands untuk mirror mode
        left_hand = results.right_hand_landmarks if flip_horizontal else results.left_hand_landmarks
        right_hand = results.left_hand_landmarks if flip_horizontal else results.right_hand_landmarks
        
        # LEFT HAND
        if left_hand:
            for lm in left_hand.landmark:
                x, y, z = lm.x, lm.y, lm.z
                if flip_horizontal:
                    x = 1.0 - x
                landmarks[idx:idx+3] = [x, y, z]
                idx += 3
        else:
            landmarks[idx:idx + (self.n_hand * 3)] = np.nan
            idx += self.n_hand * 3
        
        # RIGHT HAND
        if right_hand:
            for lm in right_hand.landmark:
                x, y, z = lm.x, lm.y, lm.z
                if flip_horizontal:
                    x = 1.0 - x
                landmarks[idx:idx+3] = [x, y, z]
                idx += 3
        else:
            landmarks[idx:idx + (self.n_hand * 3)] = np.nan
            idx += self.n_hand * 3
        
        return landmarks, results, detected
    
    def interpolate_missing_values(self, sequence):
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
    
    def calculate_motion_level(self, sequence):
        """
        Hitung level gerakan dalam sequence.
        Return nilai antara 0 (tidak bergerak) sampai 1 (bergerak banyak).
        """
        if sequence.shape[0] < 2:
            return 0.0
        
        motion_energy = 0
        
        # Fokus pada landmark tangan (indeks setelah pose landmarks)
        hand_start_idx = self.n_pose * 3
        hand_end_idx = hand_start_idx + (2 * self.n_hand * 3)
        
        for i in range(1, sequence.shape[0]):
            # Ambil hanya landmark tangan
            current_hands = sequence[i, hand_start_idx:hand_end_idx]
            prev_hands = sequence[i-1, hand_start_idx:hand_end_idx]
            
            # Hitung perbedaan Euclidean
            diff = current_hands - prev_hands
            # Reshape ke (n_landmarks, 3) untuk menghitung jarak per landmark
            diff_reshaped = diff.reshape(-1, 3)
            distances = np.linalg.norm(diff_reshaped, axis=1)
            
            motion_energy += np.mean(distances)
        
        # Normalisasi: rata-rata motion per frame
        avg_motion = motion_energy / (sequence.shape[0] - 1)
        
        # Simpan untuk debug
        self.last_motion_level = avg_motion
        
        return avg_motion
    
    def is_motionless(self, sequence, threshold=0.01):
        """
        Deteksi apakah sequence memiliki gerakan yang signifikan.
        threshold: nilai ambang batas untuk menentukan "tidak bergerak"
        """
        motion_level = self.calculate_motion_level(sequence)
        return motion_level < threshold
    
    def preprocess_sequence(self, sequence):
        sequence = self.interpolate_missing_values(sequence)
        sequence = self.resample_to_fixed_length(sequence)
        sequence = self.normalize_sequence(sequence)
        return sequence
    
    def predict(self, sequence):
        if len(sequence) < self.target_frames // 2:
            return "neutral", 1.0
        
        sequence = np.array(list(sequence))
        sequence = self.preprocess_sequence(sequence)
        
        # Debug: print motion level setiap 30 prediksi
        self.debug_counter += 1
        if self.debug_counter % 30 == 0:
            motion_level = self.calculate_motion_level(sequence)
            print(f"🔍 Motion level: {motion_level:.6f} {'(neutral)' if motion_level < 0.01 else '(moving)'}")
        
        # Deteksi 1: Jika gerakan sangat minimal, langsung return neutral
        if self.is_motionless(sequence, threshold=0.008):
            return "neutral", 1.0
        
        # Deteksi 2: Jika ada banyak NaN values (tangan tidak terdeteksi)
        nan_ratio = np.sum(np.isnan(sequence)) / sequence.size
        if nan_ratio > 0.3:  # Jika lebih dari 30% data hilang
            return "neutral", 1.0
        
        sequence = np.expand_dims(sequence, axis=0)
        
        prediction = self.model.predict(sequence, verbose=0)[0]
        class_id = np.argmax(prediction)
        confidence = prediction[class_id]
        
        self.total_predictions += 1
        
        # Deteksi 3: Jika confidence rendah, return neutral
        if confidence < 0.8:
            return "neutral", 1.0
        
        gesture = self.reverse_label_map[class_id]
        
        # Deteksi 4: Jika bukan neutral tapi motion level rendah
        if gesture != "neutral" and self.is_motionless(sequence[0], threshold=0.015):
            return "neutral", 1.0
        
        # Smoothing dengan buffer yang lebih ketat
        self.prediction_buffer.append((class_id, confidence))
        
        # Hanya return gesture jika konsisten dalam buffer
        if len(self.prediction_buffer) >= 5:
            recent_predictions = [p[0] for p in list(self.prediction_buffer)[-5:]]
            recent_confidences = [p[1] for p in list(self.prediction_buffer)[-5:]]
            
            # Cek konsistensi dan confidence
            if (recent_predictions.count(class_id) >= 4 and 
                np.mean(recent_confidences[-3:]) > 0.8):
                
                # Apply filter
                filtered_gesture, filtered_confidence = self.gesture_filter.filter(gesture, confidence)
                return filtered_gesture, filtered_confidence
        
        return "neutral", 1.0
    
    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
    
    def draw_info(self, frame, gesture, confidence, fps):
        h, w = frame.shape[:2]
        
        # Status box
        status_text = "RECORDING" if self.is_recording else "READY"
        status_color = (0, 255, 0) if self.is_recording else (255, 255, 255)
        cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 60), status_color, 2)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, status_color, 2)
        
        # Buffer progress
        buffer_filled = len(self.frame_buffer)
        progress = int((buffer_filled / self.target_frames) * 180)
        cv2.rectangle(frame, (10, 70), (10 + progress, 90), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 70), (190, 90), (255, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {buffer_filled}/{self.target_frames}", 
                   (200, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Motion level indicator
        motion_width = int(180 * min(self.last_motion_level * 50, 1.0))
        motion_color = (0, 255, 0) if self.last_motion_level < 0.01 else (0, 255, 255) if self.last_motion_level < 0.02 else (0, 165, 255)
        cv2.rectangle(frame, (10, 100), (10 + motion_width, 115), motion_color, -1)
        cv2.rectangle(frame, (10, 100), (190, 115), (255, 255, 255), 2)
        cv2.putText(frame, f"Motion: {self.last_motion_level:.4f}", 
                   (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Stability indicator
        if gesture and gesture != "neutral" and self.gesture_hold_frames > 0:
            stability = min(self.gesture_hold_frames / self.gesture_hold_threshold, 1.0)
            stab_width = int(180 * stability)
            cv2.rectangle(frame, (10, 120), (10 + stab_width, 135), (0, 255, 255), -1)
            cv2.rectangle(frame, (10, 120), (190, 135), (255, 255, 255), 2)
            cv2.putText(frame, f"Stability: {int(stability*100)}%", 
                       (200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Prediction result - NEUTRAL
        if gesture == "neutral" and confidence > 0.5:
            # Tampilkan status neutral dengan warna biru
            cv2.rectangle(frame, (10, h-100), (w-10, h-10), (50, 50, 100), -1)
            cv2.rectangle(frame, (10, h-100), (w-10, h-10), (100, 200, 255), 3)
            
            cv2.putText(frame, "NEUTRAL", (w//2 - 60, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 200, 255), 3)
            cv2.putText(frame, "No gesture detected", (w//2 - 100, h-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 230, 255), 2)
        
        # Prediction result - GESTURE
        elif gesture and confidence > 0.5 and gesture != "neutral":
            if confidence > 0.85:
                box_color = (0, 255, 0)
            elif confidence > 0.70:
                box_color = (0, 255, 255)
            else:
                box_color = (0, 165, 255)
            
            cv2.rectangle(frame, (10, h-100), (w-10, h-10), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, h-100), (w-10, h-10), box_color, 3)
            
            cv2.putText(frame, f"Gesture: {gesture}", (20, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, h-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Stats
        runtime = time.time() - self.start_time
        pred_rate = self.total_predictions / runtime if runtime > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Mode: {'FLIP' if self.flip_mode else 'NORMAL'}", (w-150, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Pred/s: {pred_rate:.1f}", (w-150, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Uptime: {int(runtime)}s", (w-150, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Gesture history
        if len(self.gesture_history) > 0:
            y_start = h - 200
            cv2.putText(frame, "Recent Gestures:", (w-280, y_start), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            for i, (hist_gesture, hist_conf, hist_time) in enumerate(self.gesture_history[-3:]):
                if hist_gesture != "neutral":  # Hanya tampilkan yang bukan neutral
                    time_ago = int(time.time() - hist_time)
                    text = f"{i+1}. {hist_gesture} ({hist_conf:.0%}) - {time_ago}s ago"
                    cv2.putText(frame, text, (w-280, y_start + 25 + i*20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # Instructions
        instructions = [
            "SPACE: Start/Stop",
            "R: Reset",
            "S: Save Log",
            "Q: Quit"
        ]
        y_offset = h - 150
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, y_offset + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Neutral status indicator (di pojok kiri atas)
        if gesture == "neutral":
            cv2.circle(frame, (30, h-180), 15, (100, 200, 255), -1)
            cv2.circle(frame, (30, h-180), 15, (255, 255, 255), 2)
            cv2.putText(frame, "N", (25, h-177), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def save_gesture_log(self):
        if not self.gesture_history:
            print("⚠️  No gestures to save")
            return
        
        # Filter hanya gesture yang bukan neutral
        non_neutral_history = [(g, c, t) for g, c, t in self.gesture_history if g != "neutral"]
        
        if not non_neutral_history:
            print("⚠️  Only neutral gestures detected, nothing to save")
            return
        
        log_dir = Path("../logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"gesture_log_{timestamp}.json"
        
        log_data = {
            'session_start': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
            'total_gestures': len(non_neutral_history),
            'neutral_detections': len([g for g, _, _ in self.gesture_history if g == "neutral"]),
            'gestures': [
                {
                    'gesture': g,
                    'confidence': float(c),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
                }
                for g, c, t in non_neutral_history
            ]
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Gesture log saved: {log_file}")
        print(f"   - Total gestures: {len(non_neutral_history)}")
        print(f"   - Neutral detections: {log_data['neutral_detections']}")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ Tidak dapat membuka webcam")
            return
        
        print("\n" + "="*70)
        print("🎥 BISINDO REALTIME RECOGNITION")
        print("="*70)
        print("Controls:")
        print("  SPACE - Start/Stop recording")
        print("  R     - Reset buffer")
        print("  S     - Save gesture log")
        print("  Q     - Quit")
        print("="*70)
        print("🟦 NEUTRAL mode akan aktif ketika:")
        print("   - Tidak ada gerakan tangan terdeteksi")
        print("   - Gerakan sangat minimal")
        print("   - Confidence rendah (< 80%)")
        print("="*70 + "\n")
        
        fps = 0
        prev_time = time.time()
        current_gesture = "neutral"  # Default ke neutral
        current_confidence = 1.0
        last_gesture_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip untuk display
            frame_display = cv2.flip(frame, 1)
            
            # ✅ Extract dengan flip sesuai mode
            landmarks, results, detected = self.extract_landmarks(frame, flip_horizontal=self.flip_mode)
            
            # Deteksi apakah ada tangan yang terlihat
            has_hands = results.left_hand_landmarks or results.right_hand_landmarks
            
            # Reset static counter jika ada tangan terdeteksi
            if has_hands:
                self.static_frame_count = 0
            else:
                self.static_frame_count += 1
            
            # Auto-reset ke neutral jika tidak ada tangan yang terlihat
            if self.static_frame_count > self.static_threshold:
                if current_gesture != "neutral":
                    current_gesture = "neutral"
                    current_confidence = 1.0
                    self.frame_buffer.clear()
                    self.prediction_buffer.clear()
                    self.gesture_hold_frames = 0
                    self.last_stable_gesture = None
            
            # Auto-reset ke neutral berdasarkan waktu
            if current_gesture != "neutral":
                time_since_last_gesture = time.time() - last_gesture_time
                if time_since_last_gesture > self.neutral_timeout:
                    current_gesture = "neutral"
                    current_confidence = 1.0
                    self.frame_buffer.clear()
                    self.prediction_buffer.clear()
                    self.gesture_hold_frames = 0
                    self.last_stable_gesture = None
                    print("🔄 Auto-reset to neutral (timeout)")
            
            # Jika recording dan terdeteksi tangan
            if self.is_recording and has_hands and self.static_frame_count <= self.static_threshold:
                self.frame_buffer.append(landmarks)
                
                # Hanya prediksi jika buffer cukup penuh
                if len(self.frame_buffer) >= self.target_frames // 2:
                    predicted_gesture, predicted_confidence = self.predict(self.frame_buffer)
                    
                    if predicted_gesture and predicted_gesture != "neutral":
                        last_gesture_time = time.time()  # Reset timer
                        
                        # Gesture stability tracking
                        if predicted_gesture == self.last_stable_gesture:
                            self.gesture_hold_frames += 1
                        else:
                            self.gesture_hold_frames = 1
                            self.last_stable_gesture = predicted_gesture
                        
                        # Log gesture jika stabil
                        if self.gesture_hold_frames == self.gesture_hold_threshold:
                            self.gesture_history.append((
                                predicted_gesture,
                                predicted_confidence,
                                time.time()
                            ))
                            print(f"✅ Detected: {predicted_gesture} ({predicted_confidence:.2%})")
                        
                        current_gesture = predicted_gesture
                        current_confidence = predicted_confidence
                    
                    elif predicted_gesture == "neutral":
                        current_gesture = "neutral"
                        current_confidence = 1.0
                        self.gesture_hold_frames = 0
                        self.last_stable_gesture = None
            
            # Jika tidak recording, tetap reset ke neutral
            elif not self.is_recording:
                current_gesture = "neutral"
                current_confidence = 1.0
                self.frame_buffer.clear()
                self.prediction_buffer.clear()
                self.gesture_hold_frames = 0
                self.last_stable_gesture = None
            
            # Draw landmarks
            self.draw_landmarks(frame_display, results)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            # Draw UI
            self.draw_info(frame_display, current_gesture, current_confidence, fps)
            
            cv2.imshow('BISINDO Realtime Recognition', frame_display)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.is_recording = not self.is_recording
                if not self.is_recording:
                    self.frame_buffer.clear()
                    self.prediction_buffer.clear()
                    current_gesture = "neutral"
                    current_confidence = 1.0
                    self.gesture_hold_frames = 0
                    self.last_stable_gesture = None
                print(f"{'🔴 Recording...' if self.is_recording else '⏸️  Paused (NEUTRAL)'}")
            elif key == ord('r'):
                self.frame_buffer.clear()
                self.prediction_buffer.clear()
                current_gesture = "neutral"
                current_confidence = 1.0
                self.gesture_hold_frames = 0
                self.last_stable_gesture = None
                self.static_frame_count = 0
                print("🔄 Buffer reset (NEUTRAL)")
            elif key == ord('s'):
                self.save_gesture_log()
            elif key == ord('d'):  # Debug key
                print(f"\n=== DEBUG INFO ===")
                print(f"Current gesture: {current_gesture}")
                print(f"Confidence: {current_confidence:.2%}")
                print(f"Buffer size: {len(self.frame_buffer)}")
                print(f"Static frames: {self.static_frame_count}")
                print(f"Motion level: {self.last_motion_level:.6f}")
                print(f"Has hands: {has_hands}")
                print(f"Hold frames: {self.gesture_hold_frames}")
                print(f"=================\n")
        
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()
        
        # Auto-save on exit
        if self.gesture_history:
            self.save_gesture_log()
        
        print("\n✅ Program selesai")
        print(f"📊 Total predictions: {self.total_predictions}")
        print(f"📊 Total gestures detected: {len([g for g, _, _ in self.gesture_history if g != 'neutral'])}")
        print(f"📊 Neutral detections: {len([g for g, _, _ in self.gesture_history if g == 'neutral'])}")

def main():
    MODEL_DIR = Path("../models")
    
    # ✅ Cari model terbaru (bukan hardcode)
    model_folders = sorted(MODEL_DIR.glob("bisindo_model_*"))
    
    if not model_folders:
        print("❌ Tidak ada model yang ditemukan di folder models/")
        print("   Jalankan 'python train.py' terlebih dahulu!")
        return
    
    latest_model = model_folders[-1]
    model_path = latest_model / "final_model.h5"
    label_map_path = latest_model / "label_map.json"
    
    if not model_path.exists() or not label_map_path.exists():
        print(f"❌ Model atau label_map tidak ditemukan di: {latest_model}")
        return
    
    print(f"✅ Using model: {latest_model.name}")
    
    # ✅ PILIH MODE
    print("\n" + "="*70)
    print("PILIH MODE:")
    print("="*70)
    print("1. NORMAL MODE (training tanpa flip)")
    print("2. FLIP MODE (training dengan flip)")
    print("="*70)
    
    while True:
        choice = input("Pilih mode (1/2): ").strip()
        if choice == '1':
            flip_mode = False
            print("✅ Mode: NORMAL (tangan kanan = right, tangan kiri = left)")
            break
        elif choice == '2':
            flip_mode = True
            print("✅ Mode: FLIP (mirror correction enabled)")
            break
        else:
            print("❌ Pilihan tidak valid, coba lagi.")
    
    recognizer = RealtimeBISINDO(
        model_path=str(model_path),
        label_map_path=str(label_map_path),
        target_frames=40,
        flip_mode=flip_mode  # ✅ Pass mode ke class
    )
    
    recognizer.run()

if __name__ == "__main__":
    main() 