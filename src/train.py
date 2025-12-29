import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# 🔥 TAMBAHKAN IMPORTS INI! 🔥
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D,
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==================================================
# 1. LOAD DATASET
# ==================================================
data = np.load(
    "data/processed/bisindo_dataset.npz",
    allow_pickle=True
)

X = data["X"].astype("float32")   # (N, 30, 150)
y = data["y"]                     # (N,)
label_map = data["label_map"].item()

num_classes = len(label_map)

print("="*60)
print("DATASET INFO")
print("="*60)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", num_classes)
print("Classes:", list(label_map.keys()))
print("="*60)


# ==================================================
# 2. TRAIN / VALIDATION SPLIT
# ==================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==================================================
# 3. MODEL: CNN 1D + LSTM
# ==================================================
model = Sequential([
    Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=(X.shape[1], X.shape[2])
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(128),
    Dropout(0.5),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.summary()


# ==================================================
# 4. COMPILE MODEL
# ==================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==================================================
# 5. CALLBACKS
# ==================================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ==================================================
# 6. TRAIN MODEL
# ==================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ==================================================
# 7. SAVE FINAL MODEL
# ==================================================
model.save("models/final_model.h5")
print("\n✅ Model saved to models/final_model.h5")


# ==================================================
# 8. TRAINING CURVE
# ==================================================
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=300)
plt.show()


# ==================================================
# 9. CONFUSION MATRIX
# ==================================================
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_map.keys(),
    yticklabels=label_map.keys()
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ==================================================
# 10. CLASSIFICATION REPORT
# ==================================================
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(
    classification_report(
        y_val,
        y_pred,
        target_names=list(label_map.keys())
    )
)


# ==================================================
# 11. CONFIDENCE DISTRIBUTION
# ==================================================
confidence = np.max(y_pred_prob, axis=1)

plt.figure(figsize=(6,4))
plt.hist(confidence, bins=20, edgecolor='black', alpha=0.7)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=300)
plt.show()


# ==================================================
# 12. QUICK DIAGNOSIS
# ==================================================
train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]

print("\n" + "="*60)
print("MODEL DIAGNOSIS")
print("="*60)

print(f"Final Train Accuracy : {train_acc:.4f}")
print(f"Final Val Accuracy   : {val_acc:.4f}")

if train_acc < 0.7 and val_acc < 0.7:
    print("⚠️  UNDERFITTING detected")
elif train_acc - val_acc > 0.15:
    print("⚠️  OVERFITTING detected")
else:
    print("✅ Model is WELL-FITTED")

print("="*60)
print("\n🎯 Output files:")
print("   - models/best_model.h5")
print("   - models/final_model.h5")
print("   - training_history.png")
print("   - confusion_matrix.png")
print("   - confidence_distribution.png")