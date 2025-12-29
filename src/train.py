# train_bisindo_1dcnn_lstm_final.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ================= SEED =================
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("🎯 BISINDO ACTION RECOGNITION")
print("📌 1D CNN + LSTM (FINAL & CORRECT VERSION)")
print("="*80)

# ================= PATH =================
BASE_DIR = r"D:\Kuliah\Semester 5\Computer_Vision\BISINDO"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bisindo_dataset_final.npz")
META_PATH = os.path.join(BASE_DIR, "data", "processed", "metadata.pkl")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ================= LOAD DATA =================
data = np.load(DATA_PATH)
X = data["X"]      # (N, 30, 153)
y = data["y"]      # (N,)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

CLASS_NAMES = metadata["classes"]
NUM_CLASSES = len(CLASS_NAMES)
SEQ_LEN = X.shape[1]
FEATURES = X.shape[2]

print(f"✅ Data shape : {X.shape}")
print(f"✅ Classes    : {CLASS_NAMES}")

# ================= AUGMENTATION =================
def augment_sequence(seq):
    aug = seq.copy()

    # Gaussian noise
    aug += np.random.normal(0, 0.01, aug.shape)

    # Scaling
    aug *= np.random.uniform(0.95, 1.05)

    # Temporal shift
    if np.random.rand() > 0.7:
        shift = np.random.randint(-2, 3)
        aug = np.roll(aug, shift, axis=0)

    return aug

def augment_dataset(X, y, factor=2):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        X_aug.append(X[i])
        y_aug.append(y[i])
        for _ in range(factor):
            X_aug.append(augment_sequence(X[i]))
            y_aug.append(y[i])
    return np.array(X_aug), np.array(y_aug)

# ================= MODEL =================
def build_1dcnn_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # ===== 1D CNN =====
        layers.Conv1D(64, 5, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),

        # ===== LSTM =====
        layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),

        layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),

        # ===== CLASSIFIER =====
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.6),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ================= CROSS VALIDATION =================
print("\n🔬 5-FOLD STRATIFIED CROSS VALIDATION")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n🔁 Fold {fold}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, factor=1)

    y_train_oh = tf.keras.utils.to_categorical(y_train_aug, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    model = build_1dcnn_lstm((SEQ_LEN, FEATURES), NUM_CLASSES)

    history = model.fit(
        X_train_aug, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=50,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=0
    )

    val_acc = max(history.history['val_accuracy'])
    cv_acc.append(val_acc)
    print(f"   ✅ Val Accuracy: {val_acc:.4f}")

print("\n📊 CV Mean Accuracy:", np.mean(cv_acc), "±", np.std(cv_acc))

# ================= FINAL TRAIN =================
print("\n🚀 FINAL TRAINING")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_aug, y_train_aug = augment_dataset(X_train, y_train, factor=2)

y_train_oh = tf.keras.utils.to_categorical(y_train_aug, NUM_CLASSES)
y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

final_model = build_1dcnn_lstm((SEQ_LEN, FEATURES), NUM_CLASSES)

history = final_model.fit(
    X_train_aug, y_train_oh,
    validation_split=0.15,
    epochs=80,
    batch_size=16,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "bisindo_1dcnn_lstm_best.h5"),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ],
    verbose=1
)

# ================= EVALUATION =================
final_model.load_weights(os.path.join(MODEL_DIR, "bisindo_1dcnn_lstm_best.h5"))

y_pred = final_model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report")
print(classification_report(y_test, y_pred_cls, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred_cls)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.show()

# ================= SAVE =================
final_model.save(os.path.join(MODEL_DIR, "bisindo_1dcnn_lstm_final.h5"))
with open(os.path.join(MODEL_DIR, "class_names.pkl"), "wb") as f:
    pickle.dump(CLASS_NAMES, f)

print("\n🎉 TRAINING SELESAI & MODEL TERSIMPAN")
print("="*80)
