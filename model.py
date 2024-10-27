import os
import numpy as np
import pandas as pd
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

train_folder_path = 'train-mails'  # Path to the folder containing training emails

# Function to read emails from folder
def read_emails_from_folder(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='latin-1') as file:
            emails.append(file.read())
    return emails

# Preprocess emails (basic)
def preprocess_email(text):
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load and preprocess emails
emails = read_emails_from_folder(train_folder_path)
emails = [preprocess_email(email) for email in emails]

# Simulate labels for now (for illustration; normally you'd have proper labels)
# Label 0 for ham, 1 for spam (manually split your training data or label it)
# Let's assume the first half is ham and the second half is spam (adjust as needed)
labels = np.array([0] * (len(emails) // 2) + [1] * (len(emails) // 2))

# Vectorize the email content
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Test the model
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
