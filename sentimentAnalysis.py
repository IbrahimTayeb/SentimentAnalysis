import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import string

# Load data
data = pd.read_csv('data.csv')
texts = data['text_column']

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word.lower() for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

data['cleaned_text'] = data['text_column'].apply(preprocess_text)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

data['sentiment_textblob'] = data['cleaned_text'].apply(get_sentiment)


# Label encoding: Assuming your data has a 'label' column with sentiments as 'positive', 'negative', 'neutral'
data['label'] = data['label'].map({'positive': 1, 'neutral': 0, 'negative': -1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train the model
text_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = text_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['cleaned_text'])
X = tokenizer.texts_to_sequences(data['cleaned_text'])
X = pad_sequences(X, maxlen=100)

# Convert labels to categorical
y = pd.get_dummies(data['label']).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)


class SentimentAnalyzer:
    def __init__(self):
        self.model = None

    def preprocess(self, text):
        # Your preprocessing code here
        return preprocess_text(text)

    def analyze_with_textblob(self, text):
        text = self.preprocess(text)
        return get_sentiment(text)

    def train_scikit(self, X_train, y_train):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        self.model.fit(X_train, y_train)

    def analyze_with_scikit(self, text):
        text = self.preprocess(text)
        return self.model.predict([text])[0]

    def train_tensorflow(self, X_train, y_train):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    def analyze_with_tensorflow(self, text):
        text = self.preprocess(text)
        text_seq = tokenizer.texts_to_sequences([text])
        text_pad = pad_sequences(text_seq, maxlen=100)
        return self.model.predict(text_pad)[0]


# Example usage:
analyzer = SentimentAnalyzer()
analyzer.train_scikit(X_train, y_train)
print(analyzer.analyze_with_scikit("This is a great product!"))
