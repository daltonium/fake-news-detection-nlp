# Fake News Detection NLP

Fake News Detection NLP is a machine learning project designed to classify news articles as **fake** or **real**, using Natural Language Processing (NLP) techniques. It includes a simple Flask-based HTML interface for real-time predictions.

---

## 🚀 Features

* Classify news articles: **Real vs. Fake**
* User-friendly web interface
* Text preprocessing (tokenization, stopword removal)
* Vectorization using **TF-IDF** or **CountVectorizer**
* Multiple ML models for performance comparison
* Performance metrics display (Accuracy, Precision, Recall, F1)

---

## 🛠️ Technologies Used

* **Python**
* **scikit-learn**
* **pandas**
* **Flask** (web backend)
* **HTML/CSS**
* **NLTK**

---

## 📥 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/daltonium/fake-news-detection-nlp.git
cd fake-news-detection-nlp
```

### 2. Install Dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install essential packages manually:

```bash
pip install pandas scikit-learn flask nltk
```

### 3. Prepare the Dataset

Place your labeled dataset inside the `data/` folder or update the configuration paths.

---

## ▶️ Usage

### Run Preprocessing & Training

```bash
python main.py
```

Or run:

```bash
python train.py
```

### Start the Web App

```bash
python app.py
```

Open your browser at:

```
http://localhost:5000
```

Paste a news article → Click **Predict**.

---

## 📂 File Structure

```
├── app.py           # Flask backend
├── main.py          # Core logic
├── train.py         # Model training
├── preprocessing.py # NLP preprocessing
├── templates/       # HTML templates
├── static/          # CSS/JS files
├── data/            # Dataset folder
├── requirements.txt # Python dependencies
```

---

## 📊 Model Performance

Common evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Results are displayed in the terminal or in the app interface.

---

## 🤝 Contributing

Pull requests and issues are welcome!

1. Fork the repository
2. Create a new branch
3. Submit a pull request with a clear description

---

## 📄 License

This project is licensed under the **MIT License**.


---

### ✔️ Quick Review

* Combines Python ML with a web interface
* Easy setup and modular structure
* Add screenshots or sample predictions to improve the README visually
