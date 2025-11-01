# train.py - CORRECTED VERSION - Full Dataset Training

import pandas as pd
import numpy as np
import re
import string
import pickle
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import tensorflow as tf

# CORRECT IMPORTS FOR TensorFlow 2.20.0
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Optimize for Ryzen 5 7430U
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(6)

print("\n" + "="*80)
print("🚀 FULL DATASET TRAINING - Production Model")
print("="*80)
print("Dataset Size: ALL 44,000+ articles")
print("Estimated time: 60-90 minutes on Ryzen 5 7430U")
print("="*80 + "\n")

start_time = time.time()

# ==================== CONFIGURATION ====================
DATASET_SIZE = None  # None = USE ALL DATA
MAX_WORDS = 10000
MAX_LEN = 500
EPOCHS = 10
BATCH_SIZE = 128

print("Configuration:")
print(f"  MAX_WORDS: {MAX_WORDS}")
print(f"  MAX_LEN: {MAX_LEN}")
print(f"  EPOCHS: {EPOCHS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}\n")

# ==================== LOAD DATA ====================
print("Loading full datasets...")
print("-" * 80)

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

print(f"✓ Fake news articles: {len(fake):,}")
print(f"✓ Real news articles: {len(true):,}")
print(f"✓ Total articles: {len(fake) + len(true):,}")

# Assign labels
fake["class"] = 0
true["class"] = 1

# Keep last samples for final validation
fake = fake.iloc[:-10]
true = true.iloc[:-10]

# Merge and shuffle all data
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Total training samples: {len(data):,}\n")

# ==================== TEXT PREPROCESSING ====================
print("Preprocessing text (this may take a few minutes)...")
print("-" * 80)

def clean_text(text):
    """Exact same preprocessing as app.py - CRITICAL FOR CONSISTENCY"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing
data['text'] = data['text'].apply(clean_text)

print("✓ Text preprocessing complete\n")

# ==================== TRAIN-TEST SPLIT ====================
print("Creating train-test split...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], 
    data['class'], 
    test_size=0.2, 
    random_state=42, 
    stratify=data['class']
)

print(f"✓ Training samples: {len(X_train):,}")
print(f"✓ Test samples: {len(X_test):,}")
print(f"✓ Train/Test ratio: 80/20\n")

# ==================== TOKENIZATION & PADDING ====================
print("Tokenizing and padding sequences...")
print("-" * 80)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

vocab_size = len(tokenizer.word_index)

print(f"✓ Vocabulary size: {vocab_size:,} unique words")
print(f"✓ Sequence length: {MAX_LEN} tokens")
print(f"✓ Training shape: {X_train_pad.shape}")
print(f"✓ Test shape: {X_test_pad.shape}\n")

# ==================== BUILD MODEL ====================
print("Building Bidirectional LSTM model...")
print("-" * 80)

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
print("-" * 80)
model.summary()
print("-" * 80 + "\n")

# ==================== CALLBACKS ====================
print("Setting up callbacks...")
print("-" * 80)

# And in model.fit
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[reduce_lr],  
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("✓ Early stopping enabled (patience=3)")
print("✓ Learning rate reduction enabled\n")

# ==================== TRAIN MODEL ====================
print("="*80)
print("TRAINING STARTED")
print("="*80)
print("Monitor the accuracy below. It should INCREASE over epochs.\n")

training_start = time.time()

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

training_time = time.time() - training_start

print("\n" + "="*80)
print(f"✓ Training completed in {training_time/60:.1f} minutes ({training_time:.0f} seconds)")
print("="*80 + "\n")

# ==================== EVALUATE MODEL ====================
print("Evaluating model on test set...")
print("-" * 80)

y_pred_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc = roc_auc_score(y_test, y_pred_prob)

print("\n" + "="*80)
print("📊 FINAL MODEL PERFORMANCE")
print("="*80)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print("="*80 + "\n")

# Validation
if accuracy < 0.85:
    print("⚠️  WARNING: Model accuracy is lower than expected!")
    print("This might indicate training issues.\n")
elif accuracy >= 0.95:
    print("✅ EXCELLENT: Model is performing great!\n")
else:
    print("✅ GOOD: Model performance is acceptable.\n")

# ==================== SAVE MODEL & ARTIFACTS ====================
print("Saving model and artifacts...")
print("-" * 80)

os.makedirs('models', exist_ok=True)

# Save model
model.save('models/model.keras')
print("✓ Model saved: models/model.keras")

# Save tokenizer
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved: models/tokenizer.pkl")

# Save configuration
config = {
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc_roc': float(auc),
    'vocab_size': vocab_size,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'total_articles': len(data),
    'epochs_trained': len(history.history['loss']),
    'training_time_minutes': float(training_time/60)
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("✓ Config saved: models/config.json")

# Save training history
history_data = {
    'train_loss': [float(x) for x in history.history['loss']],
    'train_accuracy': [float(x) for x in history.history['accuracy']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'val_accuracy': [float(x) for x in history.history['val_accuracy']]
}

with open('models/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=4)
print("✓ Training history saved: models/training_history.json")

total_time = time.time() - start_time

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Model accuracy: {accuracy*100:.2f}%")
print(f"F1-Score: {f1:.4f}")
print(f"Total articles trained on: {len(data):,}")

print("\nYour production-ready model is saved in models/")
print("\n✨ Next steps:")
print("  1. Restart your Flask app: python app.py")
print("  2. Test at: http://localhost:5000")
print("  3. Upload to production!")

print("="*80 + "\n")
