from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load friend's model and vectorizer
print("Loading model...")
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
print("✓ Model loaded\n")

def clean_text(text):
    """Match friend's preprocessing"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get text from JSON request (your current frontend)
    text = request.json.get('text', '').strip()
    
    if len(text) < 20:
        return jsonify({'error': 'Please enter at least 20 characters'})
    
    # Clean and predict
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Friend's convention: 0 = REAL, 1 = FAKE
    if prediction == 0:
        result = "REAL NEWS ✓"
        confidence = probability[0] * 100
    else:
        result = "FAKE NEWS ⚠️"
        confidence = probability[1] * 100
    
    return jsonify({
        'prediction': result,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    print("🚀 Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)
