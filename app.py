# type: ignore

# app.py - Simple Fake News Detector

from flask import Flask, render_template, request, jsonify
import pickle
import json
import re
import string
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
print("Loading model...")
model = keras.models.load_model('models/model.keras')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/config.json', 'r') as f:
    config = json.load(f)
print("✓ Model loaded\n")

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text', '').strip()
    
    if len(text) < 20:
        return jsonify({'error': 'Please enter at least 20 characters'})
    
    # Clean and predict
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config['max_len'], padding='post')
    score = float(model.predict(padded, verbose=0)[0][0])
    
    is_real = score > 0.5
    confidence = score if is_real else (1 - score)
    
    return jsonify({
        'prediction': 'REAL NEWS ✓' if is_real else 'FAKE NEWS ⚠️',
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    print("🚀 Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)
