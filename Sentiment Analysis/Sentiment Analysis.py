import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)


df = pd.read_csv("C:/Users/krsar/Downloads/Internship - YBI Foundation/Sentiment Analysis/archive/Dataset-SA.csv")

df['cleaned_review'] = df['Review'].apply(preprocess_text)
print("Cleaned Data Head:")
print(df[['Review', 'cleaned_review']].head())

X = df['cleaned_review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
y_pred = nb_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_reviews = ["I love this product, it's amazing!", "This is a terrible purchase."]
cleaned_new_reviews = [preprocess_text(review) for review in new_reviews]
new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_new_reviews)
predictions = nb_classifier.predict(new_reviews_tfidf)
print("\nPredictions on new reviews:")
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: '{review}' -> Predicted Sentiment: {sentiment}")

