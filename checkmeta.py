import numpy as np
import pickle
import os

# Path
data_path = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\processed\bisindo_dataset.npz"
meta_path = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO\data\processed\metadata.pkl"

# Load data
data = np.load(data_path)
X = data['X']
y = data['Y'] if 'Y' in data else data['y']  # bisa 'Y' atau 'y'

# Load metadata
with open(meta_path, 'rb') as f:
    metadata = pickle.load(f)

print("="*50)
print("DATA VERIFICATION")
print("="*50)
print(f"✅ Data loaded from: {data_path}")
print(f"📊 Data shape: {X.shape}")
print(f"🏷️  Labels shape: {y.shape}")
print(f"🎯 Number of classes: {len(metadata['classes'])}")
print(f"📋 Classes: {metadata['classes']}")
print(f"🎬 Sequence length: {metadata['sequence_length']}")
print(f"🔢 Features per frame: {metadata['features_per_frame']}")
print(f"📈 Total samples: {metadata['total_samples']}")

# Check class distribution
print(f"\n📊 Class distribution:")
for i, cls in enumerate(metadata['classes']):
    count = np.sum(y == i)
    print(f"  {cls}: {count} samples")

print("="*50)
print("✅ DATA READY FOR TRAINING!")