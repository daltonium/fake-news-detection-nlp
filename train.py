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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

# Optimize for your CPU
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(6)

print("\n" + "="*80)
print("🚀 FULL DATASET TRAINING - Production Model")
print("="*80)
print("Dataset Size: ALL 44,000+ articles")
print("Estimated time: 100-120 minutes (10 full epochs)")
print("="*80 + "\n")

start_time = time.time()

# ==================== CONFIGURATION ====================
DATASET_SIZE = None
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

# CRITICAL: MATCH FRIEND'S LABEL CONVENTION
fake["class"] = 1  # FAKE = 1 (like your friend)
true["class"] = 0  # REAL = 0 (like your friend)

print(f"\n✓ Label Encoding (matching friend's model):")
print(f"  - Real news: 0")
print(f"  - Fake news: 1")
print(f"  - Prediction logic: prediction == 0 means REAL\n")

# Keep last samples for validation
fake = fake.iloc[:-10]
true = true.iloc[:-10]

# Merge and shuffle
data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Total training samples: {len(data):,}\n")

# ==================== TEXT PREPROCESSING ====================
print("Preprocessing text...")
print("-" * 80)

def clean_text(text):
    """Match preprocessing from app.py"""
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
print(f"✓ Test samples: {len(X_test):,}\n")

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

print(f"✓ Vocabulary size: {vocab_size:,}")
print(f"✓ Sequence length: {MAX_LEN}")
print(f"✓ Training shape: {X_train_pad.shape}\n")

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

# ==================== TRAIN MODEL ====================
print("="*80)
print("TRAINING STARTED - FULL 10 EPOCHS")
print("="*80)

training_start = time.time()

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[],
    verbose=1
)

training_time = time.time() - training_start

print("\n" + "="*80)
print(f"✓ Training completed in {training_time/60:.1f} minutes")
print("="*80 + "\n")

# ==================== EVALUATE MODEL ====================
print("Evaluating model...")
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

# ==================== SAVE MODEL ====================
print("Saving model...")
print("-" * 80)

os.makedirs('models', exist_ok=True)

model.save('models/model.keras')
print("✓ Model saved: models/model.keras")

with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved: models/tokenizer.pkl")

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
    'epochs_trained': EPOCHS,
    'training_time_minutes': float(training_time/60),
    'label_convention': {
        'real': 0,
        'fake': 1
    }
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("✓ Config saved: models/config.json")

print("\n✅ TRAINING COMPLETE!")
print("="*80 + "\n")
