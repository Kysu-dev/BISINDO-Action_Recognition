"""
quick_test.py - Quick test for real-time system
"""

import cv2
import mediapipe as mp
import time

print("Testing webcam and MediaPipe...")

# Test webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
else:
    print("✅ Webcam accessible")
    
    # Test frame capture
    success, frame = cap.read()
    if success:
        print(f"✅ Frame captured: {frame.shape}")
        
        # Test MediaPipe
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            if results.pose_landmarks:
                print(f"✅ Pose landmarks detected: {len(results.pose_landmarks.landmark)} points")
            if results.left_hand_landmarks:
                print(f"✅ Left hand detected: {len(results.left_hand_landmarks.landmark)} points")
            if results.right_hand_landmarks:
                print(f"✅ Right hand detected: {len(results.right_hand_landmarks.landmark)} points")
    
    cap.release()

print("\n🎯 System ready for real-time testing!")