# visualize_select_video.py - CHOOSE VIDEO INTERACTIVELY
import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def find_all_videos():
    """Scan dataset and return all videos with class info"""
    dataset_path = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\raw_video"
    videos = []
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return videos
    
    # Walk through all folders
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            # Find all MP4 files in this class
            for file in os.listdir(class_path):
                if file.lower().endswith('.mp4'):
                    full_path = os.path.join(class_path, file)
                    videos.append({
                        'path': full_path,
                        'class': class_name,
                        'filename': file,
                        'size_mb': os.path.getsize(full_path) / (1024*1024)
                    })
    
    return videos

def display_video_list(videos):
    """Display videos in a nice numbered list"""
    print("\n" + "="*80)
    print("📁 AVAILABLE VIDEOS IN DATASET")
    print("="*80)
    
    # Group by class
    classes = {}
    for i, video in enumerate(videos, 1):
        class_name = video['class']
        if class_name not in classes:
            classes[class_name] = []
        classes[class_name].append((i, video))
    
    # Display by class
    for class_name in sorted(classes.keys()):
        print(f"\n📂 CLASS: {class_name}")
        print("-" * 40)
        for i, video in classes[class_name]:
            print(f"  [{i:3d}] {video['filename']:30} ({video['size_mb']:.1f} MB)")
    
    print("\n" + "="*80)
    return len(videos)

def select_video_interactive(videos):
    """Let user select a video"""
    total = len(videos)
    
    while True:
        try:
            print(f"\nSelect video [1-{total}], or:")
            print("  'r' - Random video")
            print("  'c' - Choose another class")
            print("  'q' - Quit")
            
            choice = input("Your choice: ").strip().lower()
            
            if choice == 'q':
                return None
            elif choice == 'r':
                import random
                selected = random.choice(videos)
                print(f"🎲 Randomly selected: {selected['class']}/{selected['filename']}")
                return selected
            elif choice == 'c':
                # Show classes
                classes = sorted(set(v['class'] for v in videos))
                print("\nAvailable classes:")
                for i, cls in enumerate(classes, 1):
                    count = sum(1 for v in videos if v['class'] == cls)
                    print(f"  {i}. {cls} ({count} videos)")
                
                class_choice = input(f"Select class [1-{len(classes)}]: ").strip()
                if class_choice.isdigit():
                    idx = int(class_choice) - 1
                    if 0 <= idx < len(classes):
                        selected_class = classes[idx]
                        class_videos = [v for v in videos if v['class'] == selected_class]
                        
                        print(f"\nVideos in {selected_class}:")
                        for i, v in enumerate(class_videos, 1):
                            print(f"  {i}. {v['filename']}")
                        
                        video_choice = input(f"Select video [1-{len(class_videos)}]: ").strip()
                        if video_choice.isdigit():
                            vid_idx = int(video_choice) - 1
                            if 0 <= vid_idx < len(class_videos):
                                return class_videos[vid_idx]
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < total:
                    return videos[idx]
                else:
                    print(f"❌ Please enter number between 1 and {total}")
            else:
                print("❌ Invalid choice")
        
        except (ValueError, IndexError):
            print("❌ Invalid input")

def visualize_video(video_info):
    """Visualize selected video with MediaPipe"""
    video_path = video_info['path']
    class_name = video_info['class']
    filename = video_info['filename']
    
    print(f"\n🎬 VISUALIZING: {class_name}/{filename}")
    print(f"📁 Path: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video!")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"📊 Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f} seconds")
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        print("\n🎮 Controls:")
        print("  Spacebar - Pause/Resume")
        print("  → (Right arrow) - Next frame (when paused)")
        print("  ← (Left arrow) - Previous frame (when paused)")
        print("  'q' - Quit")
        print("  's' - Save current frame as image")
        
        paused = False
        current_frame = 0
        saved_frames = []
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("🎬 End of video reached")
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frame = 0
                    continue
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Process frame
            display_frame = frame.copy()
            
            # MediaPipe processing
            image_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            
            # Draw landmarks with different colors
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                )
            
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2)
                )
            
            # Add overlay information
            overlay_height = 200
            overlay = np.zeros((overlay_height, display_frame.shape[1], 3), dtype=np.uint8)
            
            # Text information
            info_lines = [
                f"CLASS: {class_name}",
                f"FILE: {filename}",
                f"FRAME: {current_frame}/{frame_count} ({current_frame/frame_count*100:.1f}%)",
                f"TIME: {current_frame/fps:.1f}s / {duration:.1f}s",
                "",
                "DETECTION STATUS:",
                f"  Pose: {'✅ DETECTED' if results.pose_landmarks else '❌ NOT DETECTED'}",
                f"  Left Hand: {'✅ DETECTED' if results.left_hand_landmarks else '❌ NOT DETECTED'}",
                f"  Right Hand: {'✅ DETECTED' if results.right_hand_landmarks else '❌ NOT DETECTED'}",
                "",
                f"PAUSED: {'YES' if paused else 'NO'}",
                f"SAVED FRAMES: {len(saved_frames)}"
            ]
            
            for i, line in enumerate(info_lines):
                y = 30 + i * 20
                cv2.putText(overlay, line, (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Combine video and overlay
            display_frame = np.vstack([display_frame, overlay])
            
            # Show window
            cv2.imshow(f'BISINDO MediaPipe - {class_name}/{filename}', display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(0 if paused else 25) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord(' '):  # Pause/Resume
                paused = not paused
                print(f"⏸️ {'PAUSED' if paused else '▶️ RESUMED'}")
            elif key == ord('s'):  # Save frame
                timestamp = cv2.getTickCount()
                save_path = f"frame_{class_name}_{timestamp}.jpg"
                cv2.imwrite(save_path, display_frame)
                saved_frames.append(save_path)
                print(f"💾 Frame saved: {save_path}")
            elif key == 83 and paused:  # Right arrow (next frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if ret:
                    current_frame += 1
                    print(f"⏭️ Next frame: {current_frame}")
            elif key == 81 and paused:  # Left arrow (previous frame)
                if current_frame > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 2)
                    ret, frame = cap.read()
                    current_frame -= 1
                    print(f"⏮️ Previous frame: {current_frame}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n✅ Visualization finished")
    if saved_frames:
        print(f"📸 Saved {len(saved_frames)} frames:")
        for f in saved_frames:
            print(f"  - {f}")

def main():
    print("="*80)
    print("🎬 BISINDO MEDIAPIPE VIDEO SELECTOR")
    print("="*80)
    
    # Scan for videos
    print("🔍 Scanning dataset for videos...")
    videos = find_all_videos()
    
    if not videos:
        print("❌ No videos found in dataset!")
        print(f"   Check path: D:\\Kuliah\\Semester 5\\Computer_Vision\\BISINDO\\data\\raw_video")
        return
    
    total_videos = len(videos)
    total_classes = len(set(v['class'] for v in videos))
    
    print(f"✅ Found {total_videos} videos in {total_classes} classes")
    
    # Display list
    display_video_list(videos)
    
    # Let user select
    while True:
        selected = select_video_interactive(videos)
        if selected is None:
            print("\n👋 Exiting...")
            break
        
        # Visualize selected video
        visualize_video(selected)
        
        # Ask to continue
        continue_choice = input("\nVisualize another video? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()