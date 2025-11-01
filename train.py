# train.py - Train and save your model

import pandas as pd
import numpy as np
import re
import string
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Fix the imports for Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

print("🚀 Starting Training...\n")

# Settings
DATASET_SIZE = 3000
MAX_WORDS = 5000
MAX_LEN = 200
EPOCHS = 3

# Load data
print("Loading datasets...")
fake = pd.read_csv('Fake.csv').sample(n=DATASET_SIZE//2, random_state=42)
true = pd.read_csv('True.csv').sample(n=DATASET_SIZE//2, random_state=42)
fake["class"] = 0
true["class"] = 1
data = pd.concat([fake.iloc[:-5], true.iloc[:-5]], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"✓ Loaded {len(data)} samples\n")

# Preprocess
def clean(text):
    if pd.isna(text): 
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing text...")
data['text'] = data['text'].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['class'], test_size=0.2, random_state=42
)
print("✓ Data split complete\n")

# Tokenize
print("Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')
print(f"✓ Vocabulary size: {len(tokenizer.word_index):,}\n")

# Build model
print("Building model...")
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)
print("✓ Model built\n")

# Train
print("Training model...")
print("="*70)
history = model.fit(
    X_train_pad, y_train, 
    validation_split=0.2, 
    epochs=EPOCHS, 
    batch_size=64, 
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)], 
    verbose=1
)

# Evaluate
print("\n" + "="*70)
print("Evaluating model...")
y_pred_prob = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*70)

# Save everything
print("\nSaving model and artifacts...")
os.makedirs('models', exist_ok=True)

# Save model
model.save('models/model.keras')
print("✓ Model saved: models/model.keras")

# Save tokenizer
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✓ Tokenizer saved: models/tokenizer.pkl")

# Save config
config = {
    'max_words': MAX_WORDS, 
    'max_len': MAX_LEN, 
    'accuracy': float(acc),
    'f1_score': float(f1),
    'vocab_size': len(tokenizer.word_index)
}
with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("✓ Config saved: models/config.json")

print("\n" + "="*70)
print("💾 ALL FILES SAVED SUCCESSFULLY!")
print("="*70)
print("\nYou can now use the model:")
print("  python predict.py")
print("\nOr import in other scripts:")
print("  from predict import FakeNewsDetector")
print("="*70)
