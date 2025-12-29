# train_improved_final.py - COMPLETE IMPROVED VERSION
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 BISINDO TRAINING - IMPROVED VERSION WITH CROSS VALIDATION")
print("="*80)

# ==================== CONFIGURATION ====================
BASE_DIR = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bisindo_dataset.npz")
METADATA_PATH = os.path.join(BASE_DIR, "data", "processed", "metadata.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_final")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==================== LOAD AND VERIFY DATA ====================
print("\n📊 LOADING AND VERIFYING DATA")
print("-" * 50)

# Load data
data = np.load(DATA_PATH)
X = data['X']  # Shape: (samples, frames, features)
y = data['y']  # Shape: (samples,)

print(f"✅ Dataset loaded: {X.shape}")
print(f"✅ Samples: {X.shape[0]}, Frames: {X.shape[1]}, Features: {X.shape[2]}")

# Load metadata
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)

CLASS_NAMES = metadata['classes']
NUM_CLASSES = len(CLASS_NAMES)
SEQ_LENGTH = X.shape[1]
FEATURES = X.shape[2]

print(f"✅ Classes: {NUM_CLASSES}")
print(f"✅ Class names: {CLASS_NAMES}")

# ==================== DATA AUGMENTATION FUNCTIONS ====================
def augment_sequence(sequence):
    """Apply augmentation to keypoint sequence - FIXED VERSION"""
    aug = sequence.copy()
    
    # 1. Add Gaussian noise (reduced intensity)
    noise = np.random.normal(0, 0.01, sequence.shape)  # ↓ dari 0.02 ke 0.01
    aug += noise
    
    # 2. Random scaling (simulate distance changes)
    scale = np.random.uniform(0.95, 1.05)  # ↓ dari 0.9-1.1 ke 0.95-1.05
    aug *= scale
    
    # 3. Random time shift (temporal augmentation)
    if np.random.random() > 0.7:  # ↓ dari 0.5 ke 0.7 (kurang sering)
        shift = np.random.randint(-2, 3)  # ↓ dari -3,4 ke -2,3
        if shift != 0:
            aug = np.roll(aug, shift, axis=0)
            # Fill rolled values
            if shift > 0:
                aug[:shift] = aug[shift:shift+1]
            elif shift < 0:
                aug[shift:] = aug[shift-1:shift]
    
    # 4. Random axis scaling - FIXED VERSION
    # Untuk setiap frame, scale features secara independen
    axis_scale = np.random.uniform(0.98, 1.02, size=(1, FEATURES))  # Shape: (1, 162)
    aug *= axis_scale  # Broadcast dari (1,162) ke (30,162) → OK!
    
    # 5. Add random bias per feature (opsional)
    if np.random.random() > 0.5:
        bias = np.random.uniform(-0.02, 0.02, size=(1, FEATURES))
        aug += bias
    
    return aug

def create_augmented_dataset(X_data, y_data, augment_factor=2):
    """Create augmented dataset"""
    X_augmented = []
    y_augmented = []
    
    for i in range(len(X_data)):
        # Original sample
        X_augmented.append(X_data[i])
        y_augmented.append(y_data[i])
        
        # Augmented samples
        for _ in range(augment_factor):
            X_augmented.append(augment_sequence(X_data[i]))
            y_augmented.append(y_data[i])
    
    return np.array(X_augmented), np.array(y_augmented)

# ==================== CREATE MODEL FUNCTION ====================
def create_regularized_model(input_shape, num_classes):
    """Create 1D CNN + LSTM model with strong regularization"""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # ===== ENHANCED CNN BLOCKS WITH REGULARIZATION =====
        # Block 1
        layers.Conv1D(64, 5, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),  # Increased dropout
        
        # Block 2
        layers.Conv1D(128, 3, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),  # Increased dropout
        
        # Block 3
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.6),  # Increased dropout
        
        # ===== LSTM WITH REGULARIZATION =====
        layers.Reshape((-1, 256)),
        layers.LSTM(128, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        
        layers.LSTM(64, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.6),
        
        # ===== CLASSIFIER WITH REGULARIZATION =====
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.7),  # High dropout before output
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with lower learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# ==================== STRATIFIED K-FOLD CROSS VALIDATION ====================
print("\n🔬 STRATIFIED 5-FOLD CROSS VALIDATION")
print("-" * 50)

# Setup cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store results
cv_train_accuracies = []
cv_val_accuracies = []
cv_histories = []
best_model = None
best_val_accuracy = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔄 FOLD {fold}/{n_splits}")
    print("=" * 40)
    
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    print(f"   Training samples: {len(X_train_fold)}")
    print(f"   Validation samples: {len(X_val_fold)}")
    
    # Data augmentation for training fold
    print("   Applying data augmentation...")
    X_train_aug, y_train_aug = create_augmented_dataset(X_train_fold, y_train_fold, augment_factor=2)
    
    print(f"   Augmented training samples: {len(X_train_aug)}")
    
    # One-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train_aug, NUM_CLASSES)
    y_val_onehot = tf.keras.utils.to_categorical(y_val_fold, NUM_CLASSES)
    
    # Create model for this fold
    model = create_regularized_model((SEQ_LENGTH, FEATURES), NUM_CLASSES)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # Train
    print("   Training model...")
    history = model.fit(
        X_train_aug, y_train_onehot,
        validation_data=(X_val_fold, y_val_onehot),
        epochs=60,
        batch_size=16,
        callbacks=callbacks,
        verbose=0
    )
    
    # Store history
    cv_histories.append(history.history)
    
    # Get best validation accuracy
    best_epoch = np.argmin(history.history['val_loss'])
    train_acc = history.history['accuracy'][best_epoch]
    val_acc = history.history['val_accuracy'][best_epoch]
    
    cv_train_accuracies.append(train_acc)
    cv_val_accuracies.append(val_acc)
    
    print(f"   ✓ Best epoch: {best_epoch + 1}")
    print(f"   ✓ Train accuracy: {train_acc:.4f}")
    print(f"   ✓ Val accuracy: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model = model
        print(f"   🏆 New best model! (val_acc: {val_acc:.4f})")

# ==================== CROSS-VALIDATION RESULTS ====================
print("\n📈 CROSS-VALIDATION RESULTS")
print("=" * 50)

print(f"\nFold-wise results:")
for i in range(n_splits):
    print(f"  Fold {i+1}: Train={cv_train_accuracies[i]:.4f}, Val={cv_val_accuracies[i]:.4f}")

print(f"\n📊 Summary Statistics:")
print(f"  Mean Train Accuracy: {np.mean(cv_train_accuracies):.4f} ± {np.std(cv_train_accuracies):.4f}")
print(f"  Mean Val Accuracy:   {np.mean(cv_val_accuracies):.4f} ± {np.std(cv_val_accuracies):.4f}")
print(f"  Accuracy Gap:        {np.mean(cv_train_accuracies) - np.mean(cv_val_accuracies):.4f}")

# ==================== FINAL TRAINING ON FULL DATA ====================
print("\n🚀 FINAL TRAINING ON FULL DATASET")
print("-" * 50)

# Final split (80% train, 20% test)
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Final training samples: {len(X_train_full)}")
print(f"Final test samples: {len(X_test_final)}")

# Augment full training data
X_train_full_aug, y_train_full_aug = create_augmented_dataset(
    X_train_full, y_train_full, augment_factor=3
)

print(f"Augmented training samples: {len(X_train_full_aug)}")

# One-hot encoding
y_train_full_onehot = tf.keras.utils.to_categorical(y_train_full_aug, NUM_CLASSES)
y_test_final_onehot = tf.keras.utils.to_categorical(y_test_final, NUM_CLASSES)

# Create and train final model
final_model = create_regularized_model((SEQ_LENGTH, FEATURES), NUM_CLASSES)

final_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, 'best_final_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

# Split training data for validation during final training
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_full_aug, y_train_full_aug, 
    test_size=0.15, random_state=42, stratify=y_train_full_aug
)

y_train_final_onehot = tf.keras.utils.to_categorical(y_train_final, NUM_CLASSES)
y_val_final_onehot = tf.keras.utils.to_categorical(y_val_final, NUM_CLASSES)

print("\nTraining final model...")
final_history = final_model.fit(
    X_train_final, y_train_final_onehot,
    validation_data=(X_val_final, y_val_final_onehot),
    epochs=80,
    batch_size=16,
    callbacks=final_callbacks,
    verbose=1
)

print("✅ Final training completed!")

# ==================== EVALUATION ====================
print("\n📊 FINAL EVALUATION")
print("-" * 50)

# Load best model
final_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_final_model.h5'))

# Evaluate on test set
test_results = final_model.evaluate(X_test_final, y_test_final_onehot, verbose=0)

print(f"📈 Test Set Performance:")
print(f"  Loss:      {test_results[0]:.4f}")
print(f"  Accuracy:  {test_results[1]:.4f}")
print(f"  Precision: {test_results[2]:.4f}")
print(f"  Recall:    {test_results[3]:.4f}")
print(f"  AUC:       {test_results[4]:.4f}")

# Predictions
y_pred = final_model.predict(X_test_final, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_final_onehot, axis=1)

# Classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                           target_names=CLASS_NAMES))

# ==================== VISUALIZATIONS ====================
print("\n🎨 GENERATING VISUALIZATIONS")
print("-" * 50)

# 1. Cross-Validation Results
plt.figure(figsize=(15, 10))

# CV Accuracy Plot
plt.subplot(2, 3, 1)
x_pos = np.arange(n_splits)
plt.bar(x_pos - 0.15, cv_train_accuracies, width=0.3, label='Train', alpha=0.8)
plt.bar(x_pos + 0.15, cv_val_accuracies, width=0.3, label='Validation', alpha=0.8)
plt.axhline(y=np.mean(cv_train_accuracies), color='blue', linestyle='--', alpha=0.5)
plt.axhline(y=np.mean(cv_val_accuracies), color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross Validation Results')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Final Training History
plt.subplot(2, 3, 2)
plt.plot(final_history.history['accuracy'], label='Train', linewidth=2)
plt.plot(final_history.history['val_accuracy'], label='Validation', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Final Model Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.plot(final_history.history['loss'], label='Train', linewidth=2)
plt.plot(final_history.history['val_loss'], label='Validation', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Accuracy Gap Analysis
plt.subplot(2, 3, 4)
train_acc = final_history.history['accuracy']
val_acc = final_history.history['val_accuracy']
acc_gap = [train_acc[i] - val_acc[i] for i in range(len(train_acc))]
plt.plot(acc_gap, color='red', linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.fill_between(range(len(acc_gap)), 0, acc_gap, 
                 where=np.array(acc_gap) > 0, color='red', alpha=0.3, label='Overfitting')
plt.fill_between(range(len(acc_gap)), 0, acc_gap,
                 where=np.array(acc_gap) < 0, color='blue', alpha=0.3, label='Underfitting')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Gap (Train - Val)')
plt.title('Overfitting Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Confusion Matrix
plt.subplot(2, 3, 5)
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')

# 5. Class-wise Performance
plt.subplot(2, 3, 6)
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_classes, y_pred_classes, average=None
)
x = np.arange(NUM_CLASSES)
width = 0.25
plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
plt.bar(x, recall, width, label='Recall', alpha=0.8)
plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Class-wise Performance')
plt.xticks(x, CLASS_NAMES, rotation=45, ha='right')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

# ==================== DIAGNOSIS AND SAVING ====================
print("\n🔍 FINAL DIAGNOSIS")
print("=" * 50)

final_train_acc = final_history.history['accuracy'][-1]
final_val_acc = final_history.history['val_accuracy'][-1]
accuracy_gap = final_train_acc - final_val_acc

print(f"📊 Performance Summary:")
print(f"  Cross-Validation Mean Val Accuracy: {np.mean(cv_val_accuracies):.4f}")
print(f"  Final Test Accuracy:                {test_results[1]:.4f}")
print(f"  Final Train Accuracy:               {final_train_acc:.4f}")
print(f"  Final Val Accuracy:                 {final_val_acc:.4f}")
print(f"  Accuracy Gap:                       {accuracy_gap:.4f}")

print(f"\n🎯 Diagnosis:")

if accuracy_gap > 0.15:
    print("  ❌ SEVERE OVERFITTING - Need more regularization/data")
elif accuracy_gap > 0.08:
    print("  ⚠️  MODERATE OVERFITTING - Acceptable but can improve")
elif accuracy_gap > 0.03:
    print("  ✅ MINIMAL OVERFITTING - Good generalization")
elif accuracy_gap < -0.05:
    print("  ⚠️  UNDERFITTING - Model too simple")
else:
    print("  🎉 EXCELLENT - Well balanced model")

if test_results[1] >= 0.85:
    print(f"  🏆 EXCELLENT ACCURACY: {test_results[1]:.1%}")
elif test_results[1] >= 0.75:
    print(f"  📈 GOOD ACCURACY: {test_results[1]:.1%}")
elif test_results[1] >= 0.65:
    print(f"  📊 MODERATE ACCURACY: {test_results[1]:.1%}")
else:
    print(f"  ⚠️  LOW ACCURACY: {test_results[1]:.1%}")

# Save final model
final_model.save(os.path.join(MODELS_DIR, 'bisindo_final_model.h5'))

# Save class names
with open(os.path.join(MODELS_DIR, 'class_names.pkl'), 'wb') as f:
    pickle.dump(CLASS_NAMES, f)

# Save training history
with open(os.path.join(MODELS_DIR, 'training_history.pkl'), 'wb') as f:
    pickle.dump(final_history.history, f)

print(f"\n💾 Model saved: {MODELS_DIR}/bisindo_final_model.h5")
print(f"📈 Plots saved: {PLOTS_DIR}/")
print(f"📊 CV Mean Accuracy: {np.mean(cv_val_accuracies):.2%}")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)