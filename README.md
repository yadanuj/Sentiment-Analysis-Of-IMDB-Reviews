# IMDB Sentiment Analysis - Modern Full-Stack Application

This project is a modernized version of a sentiment analysis tool for IMDB reviews. It features a robust Machine Learning model capable of correctly identifying sentiment even in complex sentences with negations (e.g., "not good") and contrasts (e.g., "good but...").

## 🚀 Features

- **Robust ML Model**: Trained with N-Grams (unigrams and bigrams) and custom negation scope tagging to handle complex linguistic patterns.
- **Glassmorphism UI**: A high-fidelity, modern React frontend built with Vite.
- **Fast Backend**: A Flask-based Python API for real-time sentiment prediction.
- **Quick Test Buttons**: Pre-written sample reviews to quickly test the model's accuracy.

## 📁 Project Structure

- `/frontend`: React application (Vite, CSS Modules).
- `/backend`: Flask API (`api.py`) and model loading logic.
- `optimized_imdb_model.pkl`: The trained Logistic Regression model.
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer with bigram support.
- `extracted_code.py`: The core NLP cleaning and retraining logic.

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.x
- Node.js & npm

### Backend Setup
1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install flask flask-cors joblib scikit-learn pandas
   ```
4. Run the API:
   ```bash
   python api.py
   ```

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## 🧠 How it works

The model solves the common "negation problem" using **Negation Scope Tagging**. Incoming text is processed to append a `_neg` suffix to words following a negation (like "not", "isn't", etc.) until punctuation is reached. This ensures that "not awesome" is recognized differently from "awesome".

---
Built with ❤️ by [yadanuj](https://github.com/yadanuj)
