
import os
#Path settings
BASE_DIR = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO"
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw_video")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed")

# Read classes from dataset folder
def get_classes():
    """Baca otomatis dari folder"""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Folder tidak ditemukan: {DATASET_PATH}")
        print(f"   Buat folder: {DATASET_PATH}")
        print(f"   Taruh dataset di dalamnya")
        return []
    
    classes = []
    for item in os.listdir(DATASET_PATH):
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            # Cek apakah ada file video di dalamnya
            videos = [f for f in os.listdir(item_path) 
                     if f.lower().endswith('.mp4')]
            if videos:  # Hanya tambah jika ada video
                classes.append(item)
    
    print(f"📁 Ditemukan {len(classes)} kelas:")
    for cls in sorted(classes):
        class_path = os.path.join(DATASET_PATH, cls)
        videos = [f for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
        print(f"   • {cls}: {len(videos)} video")
    
    return sorted(classes)

SELECTED_CLASSES = get_classes()

# ==================== SETTINGS ====================
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (256, 256)
MAX_VIDEOS_PER_CLASS = 40  # Ambil 40 dari 50 video

MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# ==================== VALIDASI ====================
if __name__ == "__main__":
    print("="*50)
    print("VALIDASI KONFIGURASI")
    print("="*50)
    print(f"Base dir: {BASE_DIR}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Kelas: {len(SELECTED_CLASSES)} kelas")
    
    total_videos = 0
    for cls in SELECTED_CLASSES:
        class_path = os.path.join(DATASET_PATH, cls)
        videos = [f for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
        total_videos += min(len(videos), MAX_VIDEOS_PER_CLASS)
    
    print(f"Total video yang akan diproses: ~{total_videos}")
    print("="*50)