# SentimentAnalysis

This project implements a sentiment analysis pipeline using both traditional machine learning and deep learning techniques. The code leverages Python libraries like NLTK, TextBlob, scikit-learn, and TensorFlow to preprocess text data, analyze sentiment, and classify text into sentiment categories. 

Key Features:
1. Text Preprocessing:
   - Tokenizes text, removes punctuation, and filters out stopwords.
   - Calculates sentiment polarity using TextBlob.

2. Sentiment Classification:
   - Naive Bayes Classifier:
     - Uses scikit-learn's `Pipeline` for text vectorization (CountVectorizer and TF-IDF) and classification.
   - Deep Learning Model:
     - Uses TensorFlow/Keras to build an LSTM-based neural network for sentiment classification.

3. Data Handling:
   - Automatically loads data from a CSV file and preprocesses it.
   - Supports label encoding for sentiment categories (`positive`, `neutral`, `negative`).

4. Custom SentimentAnalyzer Class:
   - Provides reusable methods for preprocessing, training, and sentiment prediction.
   - Supports both scikit-learn and TensorFlow models for flexibility.

Tools & Libraries Used:
- Data Preprocessing: Pandas, NLTK, TextBlob
- Machine Learning: scikit-learn (Naive Bayes Classifier)
- Deep Learning: TensorFlow/Keras (LSTM Model)
- Metrics: Accuracy, classification report

Usage:
1. Prepare Data:
   - Place your dataset (`data.csv`) in the working directory with at least two columns: `text_column` (text data) and `label` (sentiment).

2. Run the Code:
   - Preprocess text data.
   - Train and evaluate models using both scikit-learn and TensorFlow.

3. Example:
   - Instantiate the `SentimentAnalyzer` class, train a model, and analyze sentiment for new inputs:
     ```python
     analyzer = SentimentAnalyzer()
     analyzer.train_scikit(X_train, y_train)
     print(analyzer.analyze_with_scikit("This is a fantastic product!"))
     ```

Results:
- Evaluates sentiment prediction accuracy using machine learning and deep learning models.
- Provides classification reports for detailed analysis of model performance.
