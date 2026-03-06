# Arabic Sentiment Analysis using Classical Machine Learning

## Overview

This project builds a sentiment analysis system for Arabic tweets using classical machine learning models.
The system classifies tweets into three sentiment categories:

* **POS** – Positive
* **NEG** – Negative
* **OBJ** – Objective / Neutral

The project includes preprocessing tailored for Arabic text, feature engineering, and training multiple machine learning models.

---

## Features

* Arabic text normalization
* Emoji sentiment handling
* Negation detection
* Stopword removal and stemming
* TF-IDF feature extraction
* Arabic-specific handcrafted features
* Multiple ML models with hyperparameter tuning

---

## Models Used

* Decision Tree
* Random Forest
* Naïve Bayes
* Multilayer Perceptron (MLP)
* Dummy Classifier (baseline)

---

## Dataset

The dataset contains Arabic tweets labeled with sentiment categories.

Initial format:

```
tweet_text \t label
```

Labels:

* POS
* NEG
* OBJ
* NEUTRAL (converted to OBJ during preprocessing)

---

## Preprocessing Pipeline

The preprocessing pipeline includes:

1. Arabic letter normalization
2. Removing elongation
3. Cleaning URLs, numbers, and HTML
4. Emoji replacement with sentiment tokens
5. Negation handling
6. Stopword removal
7. Stemming using ISRI Stemmer

---

## Feature Extraction

Two feature groups are combined:

### 1. TF-IDF Features

* Unigrams and bigrams
* Maximum 5000 features

### 2. Handcrafted Features

* Negation count
* Dialect word detection
* Emoji sentiment indicators

These are combined into a single feature matrix using `scipy.sparse.hstack`.

---

## Dataset Split

The dataset is split using stratified sampling:

* **60%** Training
* **20%** Validation
* **20%** Testing

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/arabic-sentiment-analysis.git
cd arabic-sentiment-analysis
```

Install dependencies:

```
pip install -r requirements.txt
```

Download required NLTK resources:

```
python -m nltk.downloader stopwords
```

---

## Running the Project

Run the notebook:

```
notebooks/arabic_sentiment_analysis.ipynb
```

Or run scripts in the `src` folder.

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score (macro average)

---

## Visualization

Exploratory Data Analysis includes:

* Class distribution
* Tweet length distribution
* Negation counts per sentiment

---

## Future Improvements

* Use Arabic transformer models (AraBERT)
* Improve dialect detection
* Expand emoji sentiment mapping
* Deploy as an API

---

## Author

Batol Abu Samhadana
