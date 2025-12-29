import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ==================== 1. LOAD DATA ====================
print("📂 Loading dataset...")
data = np.load("data/processed/bisindo_dataset.npz", allow_pickle=True)
X = data["X"].astype("float32")
y = data["y"]
label_map = data["label_map"].item()
num_classes = len(label_map)
print(f"✓ Data shape: {X.shape}")
print(f"✓ Classes: {num_classes}")

# ==================== 2. SPLIT DATA ====================
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
)
print(f"✓ Train: {X_train.shape[0]}")
print(f"✓ Val:   {X_val.shape[0]}")
print(f"✓ Test:  {X_test.shape[0]}")

# ==================== 3. BUILD MODEL ====================
print("\n🏗️ Building model...")
model = tf.keras.Sequential([
    # CNN Layers
    tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu', 
                          input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Dropout(0.3),
    
    # LSTM Layers
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.4),
    
    # Dense Layers
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ==================== 4. CALLBACKS ====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'models/best_model.h5', monitor='val_accuracy', save_best_only=True
    )
]

# ==================== 5. TRAIN MODEL ====================
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('models/final_model.h5')
print("✓ Model saved: models/final_model.h5")

# ==================== 6. EVALUATE ====================
print("\n📊 Evaluating...")
best_model = tf.keras.models.load_model('models/best_model.h5')

# Test set evaluation
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"✓ Test Accuracy: {test_acc:.4f}")

# Predictions
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# ==================== 7. CONFUSION MATRIX ====================
print("\n📈 Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ==================== 8. CLASSIFICATION REPORT ====================
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=list(label_map.keys())))

# Save report
report = classification_report(y_test, y_pred_classes, 
                              target_names=list(label_map.keys()),
                              output_dict=True)
with open('classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ==================== 9. TRAINING HISTORY PLOT ====================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
plt.show()

# ==================== 10. FINAL DIAGNOSIS ====================
print("\n" + "="*50)
print("🎯 TRAINING COMPLETE")
print("="*50)
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

if test_acc > 0.8:
    print("✅ Excellent! Model performs well.")
elif test_acc > 0.7:
    print("⚠️ Good, but could be improved.")
else:
    print("❌ Needs improvement. Consider:")
    print("   - More training data")
    print("   - Data augmentation")
    print("   - Hyperparameter tuning")

print("\n📁 Outputs:")
print("   - models/best_model.h5 (best model)")
print("   - models/final_model.h5 (final model)")
print("   - confusion_matrix.png")
print("   - training_history.png")
print("   - classification_report.json")
print("="*50)