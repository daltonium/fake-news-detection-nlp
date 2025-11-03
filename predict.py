import pickle
import json
import re
import string
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

class FakeNewsDetector:
    def __init__(self):
        print("Loading model...")
        self.model = keras.models.load_model('models/model.keras')
        with open('models/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open('models/config.json', 'r') as f:
            self.config = json.load(f)
        self.max_len = self.config['max_len']
        print(f"✓ Model loaded (Accuracy: {self.config['accuracy']*100:.2f}%)\n")
    
    def clean(self, text):
        if pd.isna(text): return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict(self, text):
        cleaned = self.clean(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        score = self.model.predict(padded, verbose=0)[0][0]
        
        is_real = score > 0.5
        confidence = score if is_real else (1 - score)
        
        return {
            'prediction': 'REAL NEWS ✓' if is_real else 'FAKE NEWS ⚠️',
            'confidence': f'{confidence*100:.2f}%',
            'score': float(score)
        }

# Run interactive mode
if __name__ == "__main__":
    detector = FakeNewsDetector()
    
    print("="*70)
    print("🔍 FAKE NEWS DETECTOR")
    print("="*70)
    print("Enter news text (or 'quit' to exit)\n")
    
    while True:
        text = input("News: ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if len(text.strip()) < 20:
            print("⚠️  Enter at least 20 characters\n")
            continue
        
        result = detector.predict(text)
        print(f"\n{'='*70}")
        print(f"Result:     {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"{'='*70}\n")
