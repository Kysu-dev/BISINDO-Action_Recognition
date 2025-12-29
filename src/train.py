import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LSTM, Bidirectional, Dense, Dropout, 
    BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2


# ==================================================
# 🎯 DATA AUGMENTATION CLASS
# ==================================================
class TemporalAugmentor:
    """Data augmentation untuk time series"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def time_warp(self, sequence, sigma=0.2):
        """Random time warping"""
        if not self.enabled or np.random.rand() > 0.5:
            return sequence
        
        n_frames = sequence.shape[0]
        # Generate smooth random warping
        warp = np.cumsum(np.random.randn(n_frames) * sigma)
        warp = (warp - warp.min()) / (warp.max() - warp.min())
        warp = warp * (n_frames - 1)
        
        # Interpolate
        old_indices = np.arange(n_frames)
        warped = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            warped[:, i] = np.interp(old_indices, warp, sequence[:, i])
        
        return warped
    
    def jitter(self, sequence, sigma=0.03):
        """Add random noise"""
        if not self.enabled or np.random.rand() > 0.5:
            return sequence
        
        noise = np.random.randn(*sequence.shape) * sigma
        return sequence + noise
    
    def scaling(self, sequence, sigma=0.1):
        """Random scaling"""
        if not self.enabled or np.random.rand() > 0.5:
            return sequence
        
        scale = 1 + np.random.randn() * sigma
        return sequence * scale
    
    def rotation(self, sequence, max_angle=15):
        """Random rotation (for x,y coordinates)"""
        if not self.enabled or np.random.rand() > 0.5:
            return sequence
        
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        augmented = sequence.copy()
        n_landmarks = sequence.shape[1] // 3
        
        for t in range(sequence.shape[0]):
            for i in range(n_landmarks):
                x_idx = i * 3
                y_idx = i * 3 + 1
                
                x = sequence[t, x_idx]
                y = sequence[t, y_idx]
                
                augmented[t, x_idx] = cos_a * x - sin_a * y
                augmented[t, y_idx] = sin_a * x + cos_a * y
        
        return augmented
    
    def augment(self, sequence):
        """Apply random augmentations"""
        if not self.enabled:
            return sequence
        
        seq = sequence.copy()
        seq = self.time_warp(seq)
        seq = self.jitter(seq)
        seq = self.scaling(seq)
        seq = self.rotation(seq)
        return seq


# ==================================================
# 🎯 DATA GENERATOR
# ==================================================
class DataGenerator(tf.keras.utils.Sequence):
    """Generator dengan augmentation on-the-fly"""
    
    def __init__(self, X, y, batch_size=64, augmentor=None, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentor = augmentor
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
        ]
        
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            seq = self.X[idx]
            
            # Apply augmentation
            if self.augmentor is not None:
                seq = self.augmentor.augment(seq)
            
            X_batch.append(seq)
            y_batch.append(self.y[idx])
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================================================
# 1. LOAD DATASET
# ==================================================
print("\n" + "="*70)
print("📦 LOADING DATASET")
print("="*70)

data = np.load(
    "../data/processed/bisindo_dataset.npz",
    allow_pickle=True
)

X = data["X"].astype("float32")
y = data["y"]
label_map = data["label_map"].item()
num_classes = len(label_map)

print(f"X shape       : {X.shape}")
print(f"y shape       : {y.shape}")
print(f"Num classes   : {num_classes}")
print(f"Classes       : {list(label_map.keys())}")
print(f"Samples/class : {len(X) // num_classes} (avg)")
print("="*70)


# ==================================================
# 2. TRAIN / VALIDATION SPLIT
# ==================================================
print("\n📊 SPLITTING DATASET...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train samples: {len(X_train)}")
print(f"Val samples  : {len(X_val)}")


# ==================================================
# 3. DATA AUGMENTATION SETUP
# ==================================================
print("\n🎨 SETTING UP DATA AUGMENTATION...")

train_augmentor = TemporalAugmentor(enabled=True)
train_generator = DataGenerator(
    X_train, y_train,
    batch_size=64,
    augmentor=train_augmentor,
    shuffle=True
)

val_generator = DataGenerator(
    X_val, y_val,
    batch_size=64,
    augmentor=None,  # No augmentation for validation
    shuffle=False
)

print("✅ Augmentation enabled for training")
print("   - Time warping")
print("   - Jittering (noise)")
print("   - Scaling")
print("   - Rotation")


# ==================================================
# 4. BUILD IMPROVED MODEL (ANTI-OVERFITTING)
# ==================================================
print("\n🏗️ BUILDING MODEL...")

model = Sequential([
    # ✅ Conv Block 1 (Reduced filters)
    Conv1D(
        filters=32,  # ⬇️ Reduced from 64
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(0.01),  # ✅ L2 regularization
        input_shape=(X.shape[1], X.shape[2])
    ),
    BatchNormalization(),
    SpatialDropout1D(0.2),  # ✅ Dropout setelah conv
    MaxPooling1D(pool_size=2),
    
    # ✅ Conv Block 2
    Conv1D(
        filters=64,  # ⬇️ Reduced from 128
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(0.01)
    ),
    BatchNormalization(),
    SpatialDropout1D(0.2),
    MaxPooling1D(pool_size=2),
    
    # ✅ Optional Conv Block 3 (untuk pattern yang lebih complex)
    Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=l2(0.01)
    ),
    BatchNormalization(),
    SpatialDropout1D(0.3),
    
    # ✅ Bidirectional LSTM (reduced units)
    Bidirectional(LSTM(64, return_sequences=False)),  # ⬇️ 128→64
    Dropout(0.5),
    
    # ✅ Dense layers dengan regularization
    Dense(
        64,  # ⬇️ Reduced from 128
        activation="relu",
        kernel_regularizer=l2(0.01)
    ),
    Dropout(0.5),
    
    # ✅ Output layer
    Dense(num_classes, activation="softmax")
])

model.summary()

# Print total parameters
total_params = model.count_params()
print(f"\n📊 Total parameters: {total_params:,}")


# ==================================================
# 5. COMPILE MODEL
# ==================================================
print("\n⚙️ COMPILING MODEL...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-4  # ⬇️ Reduced from 1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model compiled")


# ==================================================
# 6. CALLBACKS
# ==================================================
print("\n🔔 SETTING UP CALLBACKS...")

Path("models").mkdir(exist_ok=True)

callbacks = [
    # ✅ Early stopping dengan patience lebih besar
    EarlyStopping(
        monitor="val_loss",
        patience=15,  # ⬆️ Increased from 10
        restore_best_weights=True,
        verbose=1
    ),
    
    # ✅ Save best model
    ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    
    # ✅ Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

print("✅ Callbacks configured")


# ==================================================
# 7. TRAIN MODEL
# ==================================================
print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)


# ==================================================
# 8. SAVE FINAL MODEL
# ==================================================
model.save("models/final_model.h5")
print("\n✅ Model saved to models/final_model.h5")


# ==================================================
# 9. PLOT TRAINING HISTORY
# ==================================================
print("\n📊 GENERATING PLOTS...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
axes[0].set_title("Model Accuracy", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history["loss"], label="Train", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
axes[1].set_title("Model Loss", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate (if available)
if 'lr' in history.history:
    axes[2].plot(history.history["lr"], linewidth=2, color='orange')
    axes[2].set_title("Learning Rate", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'LR history not available', 
                 ha='center', va='center', fontsize=12)
    axes[2].set_title("Learning Rate", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
plt.show()


# ==================================================
# 10. EVALUATE ON VALIDATION SET
# ==================================================
print("\n📈 EVALUATING MODEL...")

y_pred_prob = model.predict(X_val, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_map.keys(),
    yticklabels=label_map.keys(),
    cbar_kws={'label': 'Count'}
)
plt.title("Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


# ==================================================
# 11. CLASSIFICATION REPORT
# ==================================================
print("\n" + "="*70)
print("📋 CLASSIFICATION REPORT")
print("="*70)
print(
    classification_report(
        y_val,
        y_pred,
        target_names=list(label_map.keys()),
        digits=4
    )
)


# ==================================================
# 12. CONFIDENCE DISTRIBUTION
# ==================================================
confidence = np.max(y_pred_prob, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(confidence, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(confidence.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {confidence.mean():.3f}')
plt.title("Prediction Confidence Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Confidence", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# ==================================================
# 13. PER-CLASS ACCURACY
# ==================================================
class_accuracy = {}
for class_name, class_id in label_map.items():
    mask = y_val == class_id
    if mask.sum() > 0:
        acc = (y_pred[mask] == class_id).mean()
        class_accuracy[class_name] = acc

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
classes = list(class_accuracy.keys())
accuracies = list(class_accuracy.values())

bars = plt.bar(classes, accuracies, color='steelblue', alpha=0.8)
plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
            linewidth=2, label=f'Average: {np.mean(accuracies):.3f}')
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 1.1])
plt.title("Per-Class Accuracy", fontsize=14, fontweight='bold')
plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()


# ==================================================
# 14. MODEL DIAGNOSIS
# ==================================================
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
gap = train_acc - val_acc

print("\n" + "="*70)
print("🔍 MODEL DIAGNOSIS")
print("="*70)
print(f"Final Train Accuracy : {train_acc:.4f}")
print(f"Final Val Accuracy   : {val_acc:.4f}")
print(f"Accuracy Gap         : {gap:.4f}")

# Diagnosis
if train_acc < 0.7 and val_acc < 0.7:
    print("\n⚠️  UNDERFITTING DETECTED")
    print("💡 Suggestions:")
    print("   - Increase model complexity")
    print("   - Train longer (more epochs)")
    print("   - Reduce regularization")
elif gap > 0.15:
    print("\n⚠️  OVERFITTING DETECTED")
    print("💡 Suggestions:")
    print("   - Add more dropout")
    print("   - Increase L2 regularization")
    print("   - Add more data augmentation")
    print("   - Reduce model complexity")
elif gap > 0.10:
    print("\n⚠️  SLIGHT OVERFITTING")
    print("💡 Model is acceptable but can be improved")
else:
    print("\n✅ MODEL IS WELL-FITTED!")
    print("🎉 Good balance between train and validation")

print("="*70)


# ==================================================
# 15. SUMMARY
# ==================================================
print("\n" + "="*70)
print("📁 OUTPUT FILES")
print("="*70)
print("✅ models/best_model.h5")
print("✅ models/final_model.h5")
print("✅ training_history.png")
print("✅ confusion_matrix.png")
print("✅ confidence_distribution.png")
print("✅ per_class_accuracy.png")
print("="*70)

# Save training history
np.save("training_history.npy", history.history)
print("✅ training_history.npy (for later analysis)")

print("\n🎯 Training complete!")
print("="*70)