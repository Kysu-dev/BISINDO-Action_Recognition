import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

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

# --- KONFIGURASI ENVIRONTMENT ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)
tf.random.set_seed(42)

# Optimasi Memori GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

class AggressiveAugmentor:
    """Modul Augmentasi Data untuk mengatasi dataset skala kecil."""
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def time_warp(self, seq, sigma=0.2):
        if not self.enabled or np.random.rand() > 0.5: return seq
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
        if not self.enabled or np.random.rand() > 0.5: return seq
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
        return resampled[:n_frames]
    
    def frame_dropout(self, seq, drop_rate=0.15):
        if not self.enabled or np.random.rand() > 0.5: return seq
        n_frames = seq.shape[0]
        mask = np.random.binomial(1, 1 - drop_rate, n_frames)
        while mask.sum() < 2: mask = np.random.binomial(1, 1 - drop_rate, n_frames)
        
        kept_data = seq[np.where(mask > 0)[0]]
        result = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            result[:, i] = np.interp(np.linspace(0, 1, n_frames), np.linspace(0, 1, len(kept_data)), kept_data[:, i])
        return result
    
    def augment(self, seq):
        if not self.enabled: return seq
        res = self.time_warp(seq)
        res = self.speed_variation(res)
        res = self.frame_dropout(res)
        return res

def build_lightweight_model(max_frames, feature_dim, num_classes):
    """Arsitektur Hybrid: CNN (Spatial) + BiLSTM (Temporal)."""
    model = Sequential([
        # Layer 1: CNN untuk ekstraksi fitur koordinat lokal
        Conv1D(16, 3, padding="same", activation="relu", kernel_regularizer=l2(0.001), input_shape=(max_frames, feature_dim)),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        MaxPooling1D(2),
        
        # Layer 2: CNN pendalaman fitur
        Conv1D(32, 3, padding="same", activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        SpatialDropout1D(0.15),
        MaxPooling1D(2),
        
        # Layer 3: Bidirectional LSTM untuk memproses urutan gerakan (maju-mundur)
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        
        # Layer 4: Fully Connected & Output
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model

class SmallDatasetTrainer:
    def __init__(self, data_path, batch_size=16, epochs=200):
        data = np.load(data_path, allow_pickle=True)
        self.X, self.y = data['X'].astype('float32'), data['y']
        self.label_map = data['label_map'].item()
        self.feature_dim, self.max_frames = self.X.shape[2], self.X.shape[1]
        self.num_classes = len(self.label_map)
        self.batch_size, self.epochs = batch_size, epochs

    def train_final(self):
        # 1. Data Splitting (Train 68%, Val 12%, Test 20%)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
        
        # 2. Augmentasi Data (Menggandakan data training x3)
        augmentor = AggressiveAugmentor()
        X_t_aug, y_t_aug = [], []
        for i in range(len(X_train)):
            X_t_aug.append(X_train[i])
            y_t_aug.append(y_train[i])
            for _ in range(2):
                seq = augmentor.augment(X_train[i].copy())
                if not np.isnan(seq).any():
                    X_t_aug.append(seq); y_t_aug.append(y_train[i])
        
        X_train, y_train = np.array(X_t_aug), np.array(y_t_aug)
        
        # 3. Model Setup
        model = build_lightweight_model(self.max_frames, self.feature_dim, self.num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # 4. Folder Management
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"../models/bisindo_model_{ts}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 5. Training Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ModelCheckpoint(str(model_dir / "best_model.h5"), monitor='val_accuracy', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # 6. Execution
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks)
        
        # 7. Post-Training Evaluation
        self.evaluate_and_save(model, history, X_test, y_test, model_dir, ts)

    def evaluate_and_save(self, model, history, X_test, y_test, model_dir, ts):
        """Menghasilkan log laporan, visualisasi history, dan Confusion Matrix."""
        # Simpan Aset
        model.save(str(model_dir / "final_model.h5"))
        with open(model_dir / "label_map.json", "w", encoding='utf-8') as f:
            json.dump(self.label_map, f, indent=2, ensure_ascii=False)
            
        # Visualisasi History Training
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['accuracy'], label='Train'); axes[0].plot(history.history['val_accuracy'], label='Val')
        axes[0].set_title('Accuracy'); axes[0].legend()
        axes[1].plot(history.history['loss'], label='Train'); axes[1].plot(history.history['val_loss'], label='Val')
        axes[1].set_title('Loss'); axes[1].legend()
        plt.savefig(model_dir / 'training_history.png')
        
        # Confusion Matrix
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        names = [k for k, v in sorted(self.label_map.items(), key=lambda x: x[1])]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
        plt.title('Confusion Matrix'); plt.tight_layout()
        plt.savefig(model_dir / 'confusion_matrix.png')
        
        # Print Final Report
        print(classification_report(y_test, y_pred, target_names=names, zero_division=0, digits=4))

def main():
    path = Path("../data/processed/bisindo_dataset.npz")
    if not path.exists(): return print("Dataset tidak ditemukan!")
    
    trainer = SmallDatasetTrainer(data_path=path)
    trainer.train_final()

if __name__ == "__main__":
    main()