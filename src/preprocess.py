import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['DISABLE_TFLITE_XNNPACK'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import sys
from scipy.ndimage import gaussian_filter1d

# Redirect stderr untuk avoid WinError 6
try:
    getattr(sys.stderr, 'flush', lambda: None)()
except Exception:
    sys.stderr = sys.stdout


class BISINDOPreprocessor:
    """
    Production-ready BISINDO preprocessing dengan normalisasi robust.
    
    Features:
    - Uniform temporal sampling (bukan FPS-based)
    - Multi-layer normalization (translation, scale, rotation)
    - Noise reduction (Gaussian smoothing)
    - Confidence-based filtering
    - Missing landmark handling
    """
    
    def __init__(self, 
                 dataset_path='dataset',
                 output_path='processed_data',
                 target_frames=30,
                 use_rotation_alignment=True,
                 confidence_threshold=0.5,
                 smoothing_sigma=1.0):
        """
        Args:
            dataset_path: Path ke folder dataset video
            output_path: Path output untuk .npz file
            target_frames: Target sequence length (default: 30)
            use_rotation_alignment: Enable rotation normalization
            confidence_threshold: Minimum confidence untuk landmark (0.0-1.0)
            smoothing_sigma: Gaussian smoothing parameter (0=off, 1.0=moderate)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.target_frames = target_frames
        self.use_rotation_alignment = use_rotation_alignment
        self.confidence_threshold = confidence_threshold
        self.smoothing_sigma = smoothing_sigma
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark configuration
        # Pose: Shoulders, Elbows, Wrists, Hips (8 landmarks)
        # Exclude: Nose (unstable), Wrist duplicates (17-22), Knees (irrelevant)
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        
        # Feature dimensions
        # Total: pose(8×3) + left_hand(21×3) + right_hand(21×3) = 150
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'processed_videos': 0,
            'failed_videos': 0,
            'low_confidence_frames': 0,
            'errors': []
        }
        
        print(f"\n{'='*70}")
        print(f"BISINDO Preprocessor Initialized")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  Target frames: {self.target_frames}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Pose landmarks: {self.n_pose} (shoulders, elbows, wrists, hips)")
        print(f"  Hand landmarks: {self.n_hand} × 2 (left + right)")
        print(f"  Rotation alignment: {'ON' if self.use_rotation_alignment else 'OFF'}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Smoothing sigma: {self.smoothing_sigma}")
        print(f"{'='*70}\n")
    
    def extract_landmarks(self, frame, results=None):
        """
        Extract landmarks dari single frame dengan confidence filtering.
        
        Args:
            frame: Video frame (BGR)
            results: Pre-computed MediaPipe results (optional)
        
        Returns:
            landmarks: (50, 3) array atau None jika gagal
            confidence_score: Average confidence (0.0-1.0)
        """
        try:
            # Process frame jika results belum ada
            if results is None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(frame_rgb)
            
            landmarks_list = []
            confidence_scores = []
            
            # === POSE LANDMARKS (subset) ===
            if results.pose_landmarks:
                pose_coords = []
                pose_conf = []
                
                for i in self.POSE_SUBSET:
                    lm = results.pose_landmarks.landmark[i]
                    pose_coords.extend([lm.x, lm.y, lm.z])
                    pose_conf.append(lm.visibility)  # Confidence score
                
                landmarks_list.append(np.array(pose_coords))
                confidence_scores.extend(pose_conf)
            else:
                # No pose detected → return None (reject frame)
                return None, 0.0
            
            # === LEFT HAND LANDMARKS ===
            if results.left_hand_landmarks:
                left_hand_coords = []
                for lm in results.left_hand_landmarks.landmark:
                    left_hand_coords.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(np.array(left_hand_coords))
                confidence_scores.extend([1.0] * self.n_hand)  # Hand detection = high conf
            else:
                # Missing hand → zero padding (acceptable untuk BISINDO)
                landmarks_list.append(np.zeros(self.n_hand * 3))
                confidence_scores.extend([0.0] * self.n_hand)
            
            # === RIGHT HAND LANDMARKS ===
            if results.right_hand_landmarks:
                right_hand_coords = []
                for lm in results.right_hand_landmarks.landmark:
                    right_hand_coords.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(np.array(right_hand_coords))
                confidence_scores.extend([1.0] * self.n_hand)
            else:
                landmarks_list.append(np.zeros(self.n_hand * 3))
                confidence_scores.extend([0.0] * self.n_hand)
            
            # Concatenate all landmarks
            landmarks = np.concatenate(landmarks_list)  # (150,)
            avg_confidence = np.mean(confidence_scores)
            
            return landmarks, avg_confidence
        
        except Exception as e:
            return None, 0.0
    
    def uniform_temporal_sampling(self, landmarks_sequence):
        """
        Uniform sampling ke target_frames dengan linear interpolation.
        
        CRITICAL: Ini menggantikan FPS-based sampling untuk konsistensi temporal.
        
        Args:
            landmarks_sequence: (n_frames, feature_dim)
        
        Returns:
            sampled: (target_frames, feature_dim)
        """
        n_frames, n_features = landmarks_sequence.shape
        
        if n_frames == self.target_frames:
            return landmarks_sequence
        
        # Create interpolation indices
        old_indices = np.arange(n_frames)
        new_indices = np.linspace(0, n_frames - 1, self.target_frames)
        
        # Interpolate each feature independently
        sampled = np.zeros((self.target_frames, n_features))
        for i in range(n_features):
            sampled[:, i] = np.interp(new_indices, old_indices, landmarks_sequence[:, i])
        
        return sampled
    
    def normalize_skeleton(self, landmarks_sequence):
        """
        Multi-layer normalization untuk invariance terhadap:
        1. Translation (posisi dalam frame)
        2. Scale (jarak dari kamera)
        3. Rotation (orientasi body) [optional]
        
        Args:
            landmarks_sequence: (n_frames, 150)
        
        Returns:
            normalized: (n_frames, 150)
        """
        n_frames = landmarks_sequence.shape[0]
        normalized = np.zeros_like(landmarks_sequence)
        
        # Reshape untuk processing: (n_frames, n_landmarks, 3)
        n_landmarks = self.feature_dim // 3
        reshaped = landmarks_sequence.reshape(n_frames, n_landmarks, 3)
        
        for t in range(n_frames):
            frame_lms = reshaped[t].copy()  # (50, 3)
            
            # Extract pose landmarks (first 8 landmarks)
            pose = frame_lms[:self.n_pose]  # (8, 3)
            
            # Indices dalam POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
            # Mapping: L_shoulder=0, R_shoulder=1, ..., L_hip=6, R_hip=7
            left_shoulder_idx = 0
            right_shoulder_idx = 1
            left_hip_idx = 6
            right_hip_idx = 7
            
            # === LAYER 1: TRANSLATION INVARIANCE ===
            # Center: Hip midpoint (paling stabil)
            left_hip = pose[left_hip_idx]
            right_hip = pose[right_hip_idx]
            hip_center = (left_hip + right_hip) / 2.0
            
            # Translate semua landmarks
            frame_lms = frame_lms - hip_center
            pose = frame_lms[:self.n_pose]
            
            # === LAYER 2: SCALE INVARIANCE ===
            # Reference: Shoulder width (proporsional dengan body size)
            left_shoulder = pose[left_shoulder_idx]
            right_shoulder = pose[right_shoulder_idx]
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            
            # Avoid division by zero
            if shoulder_width < 1e-6:
                shoulder_width = 1.0
            
            # Scale normalization
            frame_lms = frame_lms / shoulder_width
            pose = frame_lms[:self.n_pose]
            
            # === LAYER 3: ROTATION INVARIANCE (Optional) ===
            if self.use_rotation_alignment:
                # Align coordinate system dengan shoulder axis
                shoulder_vector = right_shoulder - left_shoulder
                shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector) + 1e-6)
                
                # Compute rotation angle di XY plane (2D projection)
                angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
                
                # Rotation matrix (2D, around Z-axis)
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                
                # Apply rotation ke X dan Y coordinates (Z tetap)
                rotated = frame_lms.copy()
                rotated[:, 0] = cos_a * frame_lms[:, 0] - sin_a * frame_lms[:, 1]
                rotated[:, 1] = sin_a * frame_lms[:, 0] + cos_a * frame_lms[:, 1]
                # Z coordinate tidak berubah
                
                frame_lms = rotated
            
            # Store normalized landmarks
            normalized[t] = frame_lms.flatten()
        
        return normalized
    
    def remove_noise(self, landmarks_sequence):
        """
        Gaussian smoothing untuk reduce jitter MediaPipe.
        
        Args:
            landmarks_sequence: (n_frames, feature_dim)
        
        Returns:
            smoothed: (n_frames, feature_dim)
        """
        if self.smoothing_sigma <= 0:
            return landmarks_sequence
        
        smoothed = np.zeros_like(landmarks_sequence)
        
        # Apply Gaussian filter per feature (temporal axis)
        for i in range(landmarks_sequence.shape[1]):
            smoothed[:, i] = gaussian_filter1d(
                landmarks_sequence[:, i],
                sigma=self.smoothing_sigma,
                mode='nearest'
            )
        
        return smoothed
    
    def process_video(self, video_path, verbose=False):
        """
        Process single video dengan full preprocessing pipeline.
        
        Pipeline:
        1. Extract all frames → landmarks
        2. Confidence filtering
        3. Uniform temporal sampling
        4. Noise reduction
        5. Multi-layer normalization
        
        Args:
            video_path: Path ke video file
            verbose: Print debug info
        
        Returns:
            features: (target_frames, feature_dim) atau None
            error_msg: Error message (jika gagal)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return None, "Cannot open video"
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0 or total_frames == 0:
                cap.release()
                return None, "Invalid video properties"
            
            if verbose:
                print(f"    📹 {video_path.name} (FPS={fps:.1f}, Frames={total_frames})")
            
            # === STEP 1: Extract landmarks dari semua frame ===
            landmarks_list = []
            confidence_list = []
            frame_idx = 0
            
            # Optional: Skip frames untuk speed (tetap lebih baik dari FPS-based)
            read_interval = 1  # Set ke 2-3 untuk video sangat panjang
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % read_interval == 0:
                    landmarks, confidence = self.extract_landmarks(frame)
                    
                    if landmarks is not None and confidence >= self.confidence_threshold:
                        landmarks_list.append(landmarks)
                        confidence_list.append(confidence)
                    else:
                        if confidence < self.confidence_threshold:
                            self.stats['low_confidence_frames'] += 1
                
                frame_idx += 1
            
            cap.release()
            
            # Check minimal frames
            if len(landmarks_list) < 10:  # Minimal 10 frame untuk gesture
                return None, f"Too few valid frames: {len(landmarks_list)}"
            
            if verbose:
                print(f"    ✓ Extracted {len(landmarks_list)} valid frames (avg conf: {np.mean(confidence_list):.2f})")
            
            # Convert to numpy
            landmarks_sequence = np.array(landmarks_list)  # (n_frames, 150)
            
            # === STEP 2: Uniform temporal sampling ===
            landmarks_sequence = self.uniform_temporal_sampling(landmarks_sequence)
            
            # === STEP 3: Noise reduction ===
            landmarks_sequence = self.remove_noise(landmarks_sequence)
            
            # === STEP 4: Normalization ===
            landmarks_sequence = self.normalize_skeleton(landmarks_sequence)
            
            return landmarks_sequence, None
        
        except Exception as e:
            if verbose:
                print(f"    ❌ Exception: {video_path.name} - {e}")
            return None, str(e)
    
    def process_dataset(self, verbose=True):
        """
        Process entire dataset dengan batch processing.
        
        Returns:
            X: (n_samples, target_frames, feature_dim)
            y: (n_samples,)
            label_map: Dict[class_name -> label_idx]
        """
        # Scan class folders
        class_folders = sorted([f for f in self.dataset_path.iterdir() if f.is_dir()])
        
        if len(class_folders) == 0:
            raise ValueError(f"❌ No class folders found in {self.dataset_path}")
        
        # Create label mapping
        label_map = {folder.name: idx for idx, folder in enumerate(class_folders)}
        
        # Save label map
        with open(self.output_path / 'label_map.json', 'w') as f:
            json.dump(label_map, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"BISINDO Dataset Preprocessing")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output: {self.output_path}")
        print(f"Classes: {len(label_map)}")
        print(f"\nClass mapping:")
        for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            print(f"  [{idx:2d}] {name}")
        print(f"{'='*70}\n")
        
        X_data = []
        y_data = []
        metadata = []
        
        # Process each class
        for class_folder in class_folders:
            class_name = class_folder.name
            class_label = label_map[class_name]
            
            # Get video files
            video_files = sorted([
                p for p in class_folder.iterdir()
                if p.is_file() and p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
            ])
            
            if len(video_files) == 0:
                print(f"⚠️  No videos in: {class_name}")
                continue
            
            print(f"\n📁 Class: {class_name} ({len(video_files)} videos)")
            self.stats['total_videos'] += len(video_files)
            
            class_success = 0
            class_failed = 0
            
            for video_file in tqdm(video_files, desc=f"  {class_name}", leave=False):
                features, error = self.process_video(video_file, verbose=False)
                
                if features is not None:
                    X_data.append(features)
                    y_data.append(class_label)
                    metadata.append({
                        'filename': video_file.name,
                        'class': class_name,
                        'label': class_label
                    })
                    class_success += 1
                    self.stats['processed_videos'] += 1
                else:
                    class_failed += 1
                    self.stats['failed_videos'] += 1
                    self.stats['errors'].append({
                        'file': str(video_file),
                        'class': class_name,
                        'error': error
                    })
            
            print(f"  ✓ Success: {class_success}/{len(video_files)}")
            if class_failed > 0:
                print(f"  ❌ Failed: {class_failed}")
        
        # Validate dataset
        if len(X_data) == 0:
            raise ValueError("❌ No videos processed successfully!")
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Preprocessing Complete!")
        print(f"{'='*70}")
        print(f"Total videos: {self.stats['total_videos']}")
        print(f"Processed: {self.stats['processed_videos']}")
        print(f"Failed: {self.stats['failed_videos']}")
        print(f"Low confidence frames: {self.stats['low_confidence_frames']}")
        print(f"\nDataset shape:")
        print(f"  X: {X_data.shape}")
        print(f"  y: {y_data.shape}")
        print(f"  Avg samples/class: {len(X_data) / len(label_map):.1f}")
        print(f"{'='*70}\n")
        
        # Class distribution
        print("Class distribution:")
        unique, counts = np.unique(y_data, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = [k for k, v in label_map.items() if v == label][0]
            bar = '█' * int(count / max(counts) * 40)
            print(f"  {class_name:20s} [{count:3d}] {bar}")
        
        # Save dataset
        output_file = self.output_path / 'bisindo_dataset.npz'
        print(f"\n💾 Saving: {output_file}")
        
        np.savez_compressed(
            output_file,
            X=X_data,
            y=y_data,
            metadata=metadata,
            label_map=label_map,
            feature_dim=self.feature_dim,
            target_frames=self.target_frames
        )
        
        print(f"✅ Saved! Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Save error log
        if len(self.stats['errors']) > 0:
            with open(self.output_path / 'error_log.json', 'w') as f:
                json.dump(self.stats['errors'], f, indent=2)
            print(f"⚠️  Error log: {self.output_path / 'error_log.json'}")
        
        self.holistic.close()
        
        return X_data, y_data, label_map


# ============================================================================
# REALTIME INFERENCE CLASS
# ============================================================================

class BISINDORealtimeProcessor:
    """
    Realtime processor untuk inference dengan normalisasi yang sama.
    
    PENTING: Gunakan normalisasi IDENTIK dengan training!
    """
    
    def __init__(self, target_frames=30, use_rotation_alignment=True):
        self.target_frames = target_frames
        self.use_rotation_alignment = use_rotation_alignment
        
        # MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Config (HARUS SAMA dengan training!)
        self.POSE_SUBSET = [11, 12, 13, 14, 15, 16, 23, 24]
        self.n_pose = len(self.POSE_SUBSET)
        self.n_hand = 21
        self.feature_dim = (self.n_pose + 2 * self.n_hand) * 3
        
        # Buffer untuk frame collection
        self.frame_buffer = []
        self.max_buffer_size = 120  # 4 detik @ 30 fps
    
    def extract_landmarks(self, frame):
        """Extract landmarks (sama dengan training)"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)
            
            landmarks_list = []
            
            # Pose
            if results.pose_landmarks:
                pose = np.array([
                    [results.pose_landmarks.landmark[i].x,
                     results.pose_landmarks.landmark[i].y,
                     results.pose_landmarks.landmark[i].z]
                    for i in self.POSE_SUBSET
                ]).flatten()
            else:
                return None
            landmarks_list.append(pose)
            
            # Left hand
            if results.left_hand_landmarks:
                left_hand = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in results.left_hand_landmarks.landmark
                ]).flatten()
            else:
                left_hand = np.zeros(self.n_hand * 3)
            landmarks_list.append(left_hand)
            
            # Right hand
            if results.right_hand_landmarks:
                right_hand = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in results.right_hand_landmarks.landmark
                ]).flatten()
            else:
                right_hand = np.zeros(self.n_hand * 3)
            landmarks_list.append(right_hand)
            
            return np.concatenate(landmarks_list)
        
        except Exception:
            return None
    
    def normalize_skeleton(self, landmarks_sequence):
        """Normalisasi (IDENTIK dengan training)"""
        n_frames = landmarks_sequence.shape[0]
        normalized = np.zeros_like(landmarks_sequence)
        
        n_landmarks = self.feature_dim // 3
        reshaped = landmarks_sequence.reshape(n_frames, n_landmarks, 3)
        
        for t in range(n_frames):
            frame_lms = reshaped[t].copy()
            pose = frame_lms[:self.n_pose]
            
            # Translation
            left_hip = pose[6]
            right_hip = pose[7]
            hip_center = (left_hip + right_hip) / 2.0
            frame_lms = frame_lms - hip_center
            pose = frame_lms[:self.n_pose]
            
            # Scale
            left_shoulder = pose[0]
            right_shoulder = pose[1]
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            if shoulder_width < 1e-6:
                shoulder_width = 1.0
            frame_lms = frame_lms / shoulder_width
            pose = frame_lms[:self.n_pose]
            
            # Rotation (optional)
            if self.use_rotation_alignment:
                shoulder_vector = right_shoulder - left_shoulder
                shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector) + 1e-6)
                angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
                
                cos_a = np.cos(-angle)
                sin_a = np.sin(-angle)
                
                rotated = frame_lms.copy()
                rotated[:, 0] = cos_a * frame_lms[:, 0] - sin_a * frame_lms[:, 1]
                rotated[:, 1] = sin_a * frame_lms[:, 0] + cos_a * frame_lms[:, 1]
                frame_lms = rotated
            
            normalized[t] = frame_lms.flatten()
        
        return normalized
    
    def process_frame(self, frame):
        """
        Process single frame untuk realtime.
        
        Returns:
            success: Boolean
        """
        landmarks = self.extract_landmarks(frame)
        
        if landmarks is not None:
            self.frame_buffer.append(landmarks)
            
            # Limit buffer size
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            return True
        return False
    
    def get_sequence(self):
        """
        Get processed sequence untuk inference.
        
        Returns:
            sequence: (target_frames, feature_dim) atau None
        """
        if len(self.frame_buffer) < 10:  # Minimal frames
            return None
        
        # Convert to numpy
        sequence = np.array(self.frame_buffer)
        
        # Uniform sampling
        n_frames = len(sequence)
        if n_frames != self.target_frames:
            old_indices = np.arange(n_frames)
            new_indices = np.linspace(0, n_frames - 1, self.target_frames)
            
            sampled = np.zeros((self.target_frames, sequence.shape[1]))
            for i in range(sequence.shape[1]):
                sampled[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
            sequence = sampled
        
        # Normalize
        sequence = self.normalize_skeleton(sequence)
        
        return sequence
    
    def reset(self):
        """Reset buffer untuk gesture baru"""
        self.frame_buffer = []


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 BISINDO Preprocessing - Production Ready\n")
    
    # Initialize preprocessor
    preprocessor = BISINDOPreprocessor(
        dataset_path='raw_video',
        output_path='processed_data',
        target_frames=30,
        use_rotation_alignment=True,  # Enable rotation normalization
        confidence_threshold=0.5,
        smoothing_sigma=1.0  # Gaussian smoothing
    )
    
    # Process dataset
    try:
        X, y, label_map = preprocessor.process_dataset(verbose=True)
        
        print(f"\n{'='*70}")
        print("✅ PREPROCESSING COMPLETE!")
        print(f"{'='*70}")
        print("\nOutput files:")
        print(f"  📦 Dataset: processed_data/bisindo_dataset.npz")