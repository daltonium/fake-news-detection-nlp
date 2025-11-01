# fake-news-detection-nlp-enhanced.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

# =============== NLTK INSTALLATIONS & IMPORTS ===============
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

# Download required NLTK data (FIXED - added punkt_tab and omw-1.4)
nltk.download('punkt')
nltk.download('punkt_tab')  # FIX: Added this
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # FIX: Added this for better lemmatization
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # FIX: Added this
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')  # FIX: Added this
nltk.download('words')

# =============== SENTIMENT ANALYSIS IMPORT ===============
from textblob import TextBlob

# =============== READABILITY IMPORTS ===============
import textstat

# Load the datasets
print("Loading datasets...")
fake_data = pd.read_csv('Fake.csv')
true_data = pd.read_csv('True.csv')

# Assign class labels
fake_data["class"] = 0
true_data["class"] = 1

# Create manual testing datasets with .copy() to avoid warnings
fake_data_manual_testing = fake_data.tail().copy()
for i in range(23480, 23470, -1):
    fake_data.drop([i], axis=0, inplace=True)
    
true_data_manual_testing = true_data.tail().copy()
for i in range(21416, 21406, -1):
    true_data.drop([i], axis=0, inplace=True)

fake_data_manual_testing["class"] = 0
true_data_manual_testing["class"] = 1

# Merge datasets
data_merge = pd.concat([fake_data, true_data], axis=0)

# =============== FEATURE 3: KEEP METADATA COLUMNS ===============
# Don't drop title, author, date - we'll use them as features
data = data_merge.copy()

# Shuffle the data
data = data.sample(frac=1)

# Reset index
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

print(f"Total dataset size: {len(data)} articles")


# =============== FEATURE 1: ENHANCED TEXT PREPROCESSING WITH NLTK ===============
def advanced_text_preprocessing(text):
    """
    Advanced text preprocessing using NLTK
    Includes: tokenization, stopword removal, stemming/lemmatization
    """
    try:
        # Handle NaN or non-string values
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'\[.*?\]','',text)
        text = re.sub(r"\\W"," ",text)
        text = re.sub(r'https?://\S+|www\.\S+','',text)
        text = re.sub(r'<.*?>+','',text)
        text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
        text = re.sub(r'\w*\d\w*','',text)
        text = re.sub(r'\s+',' ',text).strip()  # Remove extra whitespace
        
        if not text:  # If text is empty after cleaning
            return ""
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Stopword removal
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Lemmatization (more accurate than stemming)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back to text
        return ' '.join(tokens)
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return ""


# =============== FEATURE 4: READABILITY SCORE FUNCTIONS ===============
def calculate_flesch_kincaid(text):
    """Calculate Flesch-Kincaid readability score"""
    try:
        if not text or len(text.strip()) < 10:
            return 0
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0

def calculate_smog_index(text):
    """Calculate SMOG (Simple Measure of Gobbledygook) index"""
    try:
        if not text or len(text.strip()) < 10:
            return 0
        return textstat.smog_index(text)
    except:
        return 0

def calculate_readability_features(text):
    """Calculate multiple readability metrics"""
    try:
        if not text or len(text.strip()) < 10:
            return {
                'flesch_kincaid': 0,
                'smog': 0,
                'flesch_reading_ease': 0
            }
        return {
            'flesch_kincaid': textstat.flesch_kincaid_grade(text),
            'smog': textstat.smog_index(text),
            'flesch_reading_ease': textstat.flesch_reading_ease(text)
        }
    except:
        return {
            'flesch_kincaid': 0,
            'smog': 0,
            'flesch_reading_ease': 0
        }


# =============== FEATURE 5: SENTIMENT ANALYSIS ===============
def analyze_sentiment(text):
    """
    Analyze sentiment using TextBlob
    Returns: polarity (-1 to 1) and subjectivity (0 to 1)
    """
    try:
        if not text or len(text.strip()) < 3:
            return {'polarity': 0, 'subjectivity': 0}
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except:
        return {'polarity': 0, 'subjectivity': 0}


# =============== FEATURE 6: NAMED ENTITY RECOGNITION (NER) ===============
def extract_named_entities(text):
    """
    Extract named entities: PERSON, ORGANIZATION, LOCATION
    """
    try:
        if not text or len(text.strip()) < 3:
            return {
                'num_persons': 0,
                'num_organizations': 0,
                'num_locations': 0,
                'total_entities': 0
            }
        
        # Tokenize and POS tagging
        tokens = word_tokenize(text)
        
        # Limit to first 100 tokens for performance
        if len(tokens) > 100:
            tokens = tokens[:100]
        
        pos_tags = pos_tag(tokens)
        
        # Named entity chunking
        chunks = ne_chunk(pos_tags, binary=False)
        
        # Extract entities
        entities = {
            'PERSON': [],
            'ORGANIZATION': [],
            'GPE': [],  # Geo-Political Entity (location)
            'LOCATION': []
        }
        
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_name = ' '.join(c[0] for c in chunk)
                if entity_type in entities:
                    entities[entity_type].append(entity_name)
        
        return {
            'num_persons': len(entities['PERSON']),
            'num_organizations': len(entities['ORGANIZATION']),
            'num_locations': len(entities['GPE']) + len(entities['LOCATION']),
            'total_entities': sum(len(v) for v in entities.values())
        }
    
    except Exception as e:
        print(f"Error in NER: {e}")
        return {
            'num_persons': 0,
            'num_organizations': 0,
            'num_locations': 0,
            'total_entities': 0
        }


# =============== FEATURE 3: METADATA FEATURE EXTRACTION ===============
def extract_metadata_features(row):
    """Extract features from title, author, and date"""
    features = {}
    
    # Title features
    if pd.notna(row.get('title', '')):
        features['title_length'] = len(str(row['title']))
        features['title_word_count'] = len(str(row['title']).split())
    else:
        features['title_length'] = 0
        features['title_word_count'] = 0
    
    # Author features
    if pd.notna(row.get('subject', '')):
        features['has_author'] = 1
    else:
        features['has_author'] = 0
    
    # Date features
    if pd.notna(row.get('date', '')):
        features['has_date'] = 1
        try:
            date_str = str(row['date'])
            features['date_length'] = len(date_str)
        except:
            features['date_length'] = 0
    else:
        features['has_date'] = 0
        features['date_length'] = 0
    
    return features


# =============== APPLY ALL PREPROCESSING AND FEATURE EXTRACTION ===============
print("\nApplying advanced text preprocessing...")
data['text_processed'] = data['text'].apply(advanced_text_preprocessing)
print("✓ Text preprocessing completed")

print("\nExtracting readability features...")
readability_features = data['text'].apply(calculate_readability_features)
data['flesch_kincaid'] = readability_features.apply(lambda x: x['flesch_kincaid'])
data['smog'] = readability_features.apply(lambda x: x['smog'])
data['flesch_reading_ease'] = readability_features.apply(lambda x: x['flesch_reading_ease'])
print("✓ Readability features extracted")

print("\nAnalyzing sentiment...")
sentiment_features = data['text'].apply(analyze_sentiment)
data['sentiment_polarity'] = sentiment_features.apply(lambda x: x['polarity'])
data['sentiment_subjectivity'] = sentiment_features.apply(lambda x: x['subjectivity'])
print("✓ Sentiment analysis completed")

print("\nExtracting named entities (this may take a while)...")
ner_features = data['text_processed'].apply(extract_named_entities)
data['num_persons'] = ner_features.apply(lambda x: x['num_persons'])
data['num_organizations'] = ner_features.apply(lambda x: x['num_organizations'])
data['num_locations'] = ner_features.apply(lambda x: x['num_locations'])
data['total_entities'] = ner_features.apply(lambda x: x['total_entities'])
print("✓ Named entity recognition completed")

print("\nExtracting metadata features...")
metadata_features = data.apply(extract_metadata_features, axis=1)
data['title_length'] = metadata_features.apply(lambda x: x['title_length'])
data['title_word_count'] = metadata_features.apply(lambda x: x['title_word_count'])
data['has_author'] = metadata_features.apply(lambda x: x['has_author'])
data['has_date'] = metadata_features.apply(lambda x: x['has_date'])
data['date_length'] = metadata_features.apply(lambda x: x['date_length'])
print("✓ Metadata features extracted")


# =============== FEATURE 2: TF-IDF VECTORIZATION ===============
# Define features and target
x_text = data['text_processed']
y = data['class']

# Additional numeric features
additional_features = data[[
    'flesch_kincaid', 'smog', 'flesch_reading_ease',
    'sentiment_polarity', 'sentiment_subjectivity',
    'num_persons', 'num_organizations', 'num_locations', 'total_entities',
    'title_length', 'title_word_count', 'has_author', 'has_date', 'date_length'
]]

# Train-test split
x_text_train, x_text_test, y_train, y_test = train_test_split(
    x_text, y, test_size=0.25, random_state=42
)
additional_train, additional_test = train_test_split(
    additional_features, test_size=0.25, random_state=42
)

# =============== TF-IDF Vectorization ===============
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nApplying TF-IDF Vectorization...")
vectorization = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
xv_train_tfidf = vectorization.fit_transform(x_text_train)
xv_test_tfidf = vectorization.transform(x_text_test)

print(f"✓ TF-IDF Feature count: {xv_train_tfidf.shape[1]} features")

# Combine TF-IDF features with additional features
from scipy.sparse import hstack, csr_matrix

xv_train = hstack([xv_train_tfidf, csr_matrix(additional_train.values)])
xv_test = hstack([xv_test_tfidf, csr_matrix(additional_test.values)])

print(f"✓ Total features: {xv_train.shape[1]} (TF-IDF + NLP enhancements)")
print(f"✓ Training samples: {xv_train.shape[0]}, Test samples: {xv_test.shape[0]}\n")


# =============== Logistic Regression ===============
from sklearn.linear_model import LogisticRegression

print("="*80)
print("TRAINING MODELS")
print("="*80)

print('\n1. Training Logistic Regression...')
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)

print('   Accuracy:', LR.score(xv_test, y_test))
print(classification_report(y_test, pred_lr))


# =============== Decision Tree Classifier ===============
from sklearn.tree import DecisionTreeClassifier

print('2. Training Decision Tree Classifier...')
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)

print('   Accuracy:', DT.score(xv_test, y_test))
print(classification_report(y_test, pred_dt))


# =============== HistGradientBoosting Classifier ===============
from sklearn.ensemble import HistGradientBoostingClassifier

print('3. Training HistGradientBoosting Classifier...')
GB = HistGradientBoostingClassifier(random_state=0)
GB.fit(xv_train.toarray(), y_train)
pred_gb = GB.predict(xv_test.toarray())

print('   Accuracy:', GB.score(xv_test.toarray(), y_test))
print(classification_report(y_test, pred_gb))


# =============== Random Forest Classifier ===============
from sklearn.ensemble import RandomForestClassifier

print('4. Training Random Forest Classifier...')
RF = RandomForestClassifier(random_state=0, n_estimators=100)
RF.fit(xv_train, y_train)
pred_rf = RF.predict(xv_test)

print('   Accuracy:', RF.score(xv_test, y_test))
print(classification_report(y_test, pred_rf))


# =============== Enhanced Manual Testing Function ===============
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing_enhanced(news, title="", author="", date=""):
    """
    Enhanced manual testing with all NLP features
    """
    # Create test dataframe
    testing_news = {
        "text": [news],
        "title": [title],
        "subject": [author],
        "date": [date]
    }
    new_def_test = pd.DataFrame(testing_news)
    
    # Apply text preprocessing
    new_def_test['text_processed'] = new_def_test["text"].apply(advanced_text_preprocessing)
    
    # Extract all features
    readability = calculate_readability_features(news)
    sentiment = analyze_sentiment(news)
    entities = extract_named_entities(new_def_test['text_processed'].iloc[0])
    metadata = extract_metadata_features(new_def_test.iloc[0])
    
    # Create feature vector
    additional_features_test = pd.DataFrame([{
        'flesch_kincaid': readability['flesch_kincaid'],
        'smog': readability['smog'],
        'flesch_reading_ease': readability['flesch_reading_ease'],
        'sentiment_polarity': sentiment['polarity'],
        'sentiment_subjectivity': sentiment['subjectivity'],
        'num_persons': entities['num_persons'],
        'num_organizations': entities['num_organizations'],
        'num_locations': entities['num_locations'],
        'total_entities': entities['total_entities'],
        'title_length': metadata['title_length'],
        'title_word_count': metadata['title_word_count'],
        'has_author': metadata['has_author'],
        'has_date': metadata['has_date'],
        'date_length': metadata['date_length']
    }])
    
    # TF-IDF vectorization
    new_xv_test_tfidf = vectorization.transform(new_def_test['text_processed'])
    
    # Combine features
    new_xv_test = hstack([new_xv_test_tfidf, csr_matrix(additional_features_test.values)])
    
    # Predictions
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test.toarray())
    pred_RF = RF.predict(new_xv_test)
    
    # Display enhanced analysis
    print("\n" + "="*80)
    print("ENHANCED NLP ANALYSIS")
    print("="*80)
    print(f"\n📊 Readability Scores:")
    print(f"  • Flesch-Kincaid Grade Level: {readability['flesch_kincaid']:.2f}")
    print(f"  • SMOG Index: {readability['smog']:.2f}")
    print(f"  • Flesch Reading Ease: {readability['flesch_reading_ease']:.2f}")
    
    print(f"\n💭 Sentiment Analysis:")
    print(f"  • Polarity: {sentiment['polarity']:.2f} ({'Positive' if sentiment['polarity'] > 0 else 'Negative' if sentiment['polarity'] < 0 else 'Neutral'})")
    print(f"  • Subjectivity: {sentiment['subjectivity']:.2f} ({'Subjective' if sentiment['subjectivity'] > 0.5 else 'Objective'})")
    
    print(f"\n🏷️ Named Entity Recognition:")
    print(f"  • Persons mentioned: {entities['num_persons']}")
    print(f"  • Organizations mentioned: {entities['num_organizations']}")
    print(f"  • Locations mentioned: {entities['num_locations']}")
    print(f"  • Total entities: {entities['total_entities']}")
    
    print(f"\n📝 Metadata Features:")
    print(f"  • Title length: {metadata['title_length']} characters")
    print(f"  • Has author: {'Yes' if metadata['has_author'] else 'No'}")
    print(f"  • Has date: {'Yes' if metadata['has_date'] else 'No'}")
    
    print("\n" + "="*80)
    print("MODEL PREDICTIONS")
    print("="*80)
    return print("\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(pred_DT[0]),
        output_lable(pred_GB[0]),
        output_lable(pred_RF[0])
    ))


# =============== Test the enhanced model ===============
print('\n' + '='*80)
print('ENHANCED MANUAL TESTING WITH NLP FEATURES')
print('='*80)
news = str(input("Enter news text: "))
title = str(input("Enter title (optional): "))
author = str(input("Enter author (optional): "))
date = str(input("Enter date (optional): "))

manual_testing_enhanced(news, title, author, date)
