"""
check_data_structure.py - Check training data structure
"""

import numpy as np
import os

BASE_DIR = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bisindo_dataset.npz")

print("🔍 CHECKING TRAINING DATA STRUCTURE")
print("="*50)

data = np.load(DATA_PATH)
X = data['X']
y = data['y']

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"\nSamples: {X.shape[0]}")
print(f"Frames per sample: {X.shape[1]}")
print(f"Features per frame: {X.shape[2]}")

# Analyze first sample
sample = X[0]
print(f"\n📊 First sample analysis:")
print(f"  Shape: {sample.shape}")
print(f"  Data type: {sample.dtype}")
print(f"  Min value: {sample.min():.4f}")
print(f"  Max value: {sample.max():.4f}")
print(f"  Mean: {sample.mean():.4f}")
print(f"  Std: {sample.std():.4f}")

# Check for zeros (might indicate missing landmarks)
zero_mask = sample == 0
zero_percentage = np.sum(zero_mask) / sample.size * 100
print(f"  Zero values: {zero_percentage:.1f}%")

# Try to guess structure
print(f"\n🤔 Trying to guess feature structure...")
print(f"  162 features could be:")

options = [
    "21 hand points × 3 coordinates × 2 hands = 126 + 12 pose points × 3 = 162",
    "21 hand points × 3 × 2 = 126 + 21 face points × 3 = 189 (too many)",
    "Maybe only hand coordinates (x,y) without z: 21 × 2 × 2 = 84 + pose",
    "Could include face mesh points (468 × 3 = 1404, too many)"
]

for opt in options:
    print(f"  - {opt}")

# Save a sample for comparison
np.save("sample_training_frame.npy", sample[0])
print(f"\n💾 Saved first frame to 'sample_training_frame.npy'")
print("  You can load and inspect it with: np.load('sample_training_frame.npy')")