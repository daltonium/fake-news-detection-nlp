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
fake = pd.read_csv('Fake.csv').sample(n=DATASET_SIZE//2, random_state=42)
true = pd.read_csv('True.csv').sample(n=DATASET_SIZE//2, random_state=42)
fake["class"] = 0
true["class"] = 1
data = pd.concat([fake.iloc[:-5], true.iloc[:-5]], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocess
def clean(text):
    if pd.isna(text): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['text'] = data['text'].apply(clean)
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.2, random_state=42)

# Tokenize
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

# Build model
model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("Training model...")
model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=64, 
          callbacks=[EarlyStopping(patience=2, restore_best_weights=True)], verbose=1)

# Evaluate
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f"\n✅ Training Complete!")
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"F1-Score: {f1:.4f}\n")

# Save
os.makedirs('models', exist_ok=True)
model.save('models/model.keras')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('models/config.json', 'w') as f:
    json.dump({'max_words': MAX_WORDS, 'max_len': MAX_LEN, 'accuracy': float(acc)}, f)

print("💾 Saved:")
print("  ✓ models/model.keras")
print("  ✓ models/tokenizer.pkl")
print("  ✓ models/config.json")
print("\nRun: python predict.py")
