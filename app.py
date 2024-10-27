from flask import Flask, render_template, request, jsonify
import joblib
import os
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

app = Flask(__name__)

# Load the trained SVM model and vectorizer
model = joblib.load('svm_model.pkl')  # Load your saved SVM model
vectorizer = joblib.load('vectorizer.pkl')  # Load your saved TfidfVectorizer

def preprocess_email(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_spam', methods=['POST'])
def check_spam():
    email_text = request.form['email_text']  # Get input from the form
    email_text = preprocess_email(email_text)  # Preprocess email
    email_vectorized = vectorizer.transform([email_text])  # Vectorize
    prediction = model.predict(email_vectorized)  # Predict spam or ham

    result = 'This email is classified as Spam!' if prediction[0] == 1 else 'This email is classified as Not Spam!'
    return jsonify({'result': result})  # Return the result as JSON


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)

