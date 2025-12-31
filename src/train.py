import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, 
    Dropout, BatchNormalization, SpatialDropout1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2

np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✅ GPU Terdeteksi: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("\n⚠️ Training berjalan di CPU")

class AggressiveAugmentor:
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def time_warp(self, seq, sigma=0.2):
        if not self.enabled or np.random.rand() > 0.5:
            return seq
        n_frames = seq.shape[0]
        warp = np.cumsum(np.random.randn(n_frames) * sigma)
        warp = (warp - warp.min()) / (warp.max() - warp.min() + 1e-6)
        warp = warp * (n_frames - 1)
        old_idx = np.arange(n_frames)
        warped = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            warped[:, i] = np.interp(old_idx, warp, seq[:, i])
        return warped
    
    def speed_variation(self, seq, min_speed=0.6, max_speed=1.4):
        if not self.enabled or np.random.rand() > 0.5:
            return seq
        speed = np.random.uniform(min_speed, max_speed)
        n_frames = seq.shape[0]
        x_old = np.linspace(0, 1, n_frames)
        n_new = max(int(n_frames / speed), 10)
        x_new = np.linspace(0, 1, n_new)
        resampled = np.zeros((n_new, seq.shape[1]))
        for i in range(seq.shape[1]):
            resampled[:, i] = np.interp(x_new, x_old, seq[:, i])
        if n_new < n_frames:
            padded = np.zeros((n_frames, seq.shape[1]))
            padded[:n_new] = resampled
            return padded
        else:
            return resampled[:n_frames]
    
    def frame_dropout(self, seq, drop_rate=0.15):
        if not self.enabled or np.random.rand() > 0.5:
            return seq
        n_frames = seq.shape[0]
        keep_prob = 1 - drop_rate
        mask = np.random.binomial(1, keep_prob, n_frames)
        while mask.sum() < n_frames * 0.5:
            mask = np.random.binomial(1, keep_prob, n_frames)
        kept_indices = np.where(mask > 0)[0]
        if len(kept_indices) < 2:
            return seq
        kept_data = seq[kept_indices]
        result = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            x_old = np.linspace(0, 1, len(kept_indices))
            x_new = np.linspace(0, 1, n_frames)
            result[:, i] = np.interp(x_new, x_old, kept_data[:, i])
        return result
    
    def augment(self, seq):
        if not self.enabled:
            return seq
        result = seq.copy()
        if np.random.rand() > 0.4: result = self.time_warp(result)
        if np.random.rand() > 0.4: result = self.speed_variation(result)
        if np.random.rand() > 0.5: result = self.frame_dropout(result)
        return result

def build_lightweight_model(max_frames, feature_dim, num_classes):
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, padding="same", activation="relu", 
               kernel_regularizer=l2(0.001), input_shape=(max_frames, feature_dim)),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        MaxPooling1D(pool_size=2),
        
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model

class SmallDatasetTrainer:
    def __init__(self, data_path, batch_size=16, epochs=200):
        print("\n" + "="*70)
        print("📦 LOADING DATASET")
        print("="*70)
        
        data = np.load(data_path, allow_pickle=True)
        self.X = data['X'].astype('float32')
        self.y = data['y']
        self.label_map = data['label_map'].item()
        self.feature_dim = self.X.shape[2] 
        self.max_frames = self.X.shape[1]
        self.num_classes = len(self.label_map)
        
        print(f"Dataset Shape: {self.X.shape}")
        print(f"Total Kelas:   {self.num_classes}")
        print(f"Durasi Frame:  {self.max_frames}")
        
        self.batch_size = batch_size
        self.epochs = epochs

    def train_final(self):
        print("\n" + "="*70)
        print("🎯 TRAINING FINAL MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
        )
        
        print("🛠️ Melakukan Data Augmentation...")
        augmentor = AggressiveAugmentor(enabled=True)
        X_train_aug = []
        y_train_aug = []
        
        for i in range(len(X_train)):
            X_train_aug.append(X_train[i])
            y_train_aug.append(y_train[i])
            
            for _ in range(2):
                aug_seq = augmentor.augment(X_train[i].copy())
                if not np.isnan(aug_seq).any() and not np.isinf(aug_seq).any():
                    X_train_aug.append(aug_seq)
                    y_train_aug.append(y_train[i])
        
        X_train = np.array(X_train_aug)
        y_train = np.array(y_train_aug)
        print(f"Total Data Training: {X_train.shape[0]} sampel")
        
        model = build_lightweight_model(self.max_frames, self.feature_dim, self.num_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"../models/bisindo_model_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(model_dir / "best_model.h5"), monitor='val_accuracy', 
                          save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                            min_lr=1e-6, verbose=1)
        ]
        
        print("🚀 Mulai Training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print("\n" + "="*70)
        print("📊 HASIL TRAINING")
        print("="*70)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy:   {val_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        
        # Save model
        model.save(str(model_dir / "final_model.h5"))
        
        # Save label map
        with open(model_dir / "label_map.json", "w", encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            'model_info': {
                'architecture': 'Conv1D + BiLSTM',
                'num_classes': self.num_classes,
                'feature_dim': self.feature_dim,
                'max_frames': self.max_frames,
                'total_params': int(model.count_params())
            },
            'dataset_split': {
                'total_samples': len(self.X),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            },
            'performance': {
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc)
            },
            'label_map': self.label_map,
            'training_date': timestamp
        }
        
        with open(model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(model_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Confusion matrix
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        sorted_labels = sorted(self.label_map.items(), key=lambda x: x[1])
        class_names = [k for k, v in sorted_labels]
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Classification report
        print("\n" + "="*70)
        print("📋 CLASSIFICATION REPORT")
        print("="*70)
        all_labels = list(range(len(class_names)))
        print(classification_report(y_test, y_pred, target_names=class_names, 
                                   labels=all_labels, zero_division=0, digits=4))
        
        print(f"\n✅ Model tersimpan di: {model_dir}")
        print(f"✅ Training history: {model_dir / 'training_history.png'}")
        print(f"✅ Confusion matrix: {model_dir / 'confusion_matrix.png'}")
        print(f"✅ Metadata: {model_dir / 'metadata.json'}")
        
        return model, history

def main():
    DATA_PATH = Path("../data/processed/bisindo_dataset.npz")
    
    if not DATA_PATH.exists():
        print(f"\n❌ Dataset tidak ditemukan: {DATA_PATH}")
        print("   Jalankan 'python preprocess.py' terlebih dahulu!")
        return
    
    trainer = SmallDatasetTrainer(
        data_path=DATA_PATH,
        batch_size=16,
        epochs=200
    )
    
    trainer.train_final()

if __name__ == "__main__":
    main()