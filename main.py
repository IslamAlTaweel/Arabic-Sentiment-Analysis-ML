"""
Arabic Sentiment ML Analyzer:
Develops and systematically evaluates multiple ML models for sentiment
classification in Arabic social media data.

METHODOLOGIES
1. Dataset is distributed amongst 4 sentiment classes
	    - POS, NEG, OBJ, NEUTRAL

2. Data preprocessing (clean/standardize data prior to feature extraction)
	a. text cleaning operations:
		-converting all chars to lowercase
		-remove HTML and URL tags
		-eliminate special chars and numerical tokens
	b. noise reduction:
		-correct common misspellings
		-expand contractions
	c. text normalization:
		-stemming or lemmatization
			-reduce words to their canonical form
		-removal of stop words irrelevant to semantic information
		-standardize text variations
			-dates
			-emojis

3. Feature Extraction:
	a. Automated Text Representations
		-TF-IDF: Counts how often words appear.
		 It gives more importance to unique, meaningful words
		 and less to common ones (like "the" or "and").

		-Word Embeddings (ex.,Word2Vec/FastText): Group words with similar meanings together.
		 words are related because they appear in similar contexts.

		-Contextual embeddings from Transformer Based Language Models (AI): The most advanced version.
		 These look at the whole sentence to understand the specific meaning of a word based on the words around it.

	b. Hand-Engineered (Human-Made) Features
	    Since Arabic is a complex language, manually tell the computer to look for specific clues to
	    capture linguistic characteristics specific to arabic

		-Dialect Indicators: Identifying if the tweet is in Modern Standard Arabic or a local dialect

		-Character Normalization: Cleaning up the text

		-Negation Handling: Marking words like "not" or "never," which completely change the meaning of a sentence.


	c. Extra Social Media "Clues"
	    These are the "bonus" details that help the AI understand the tone or intent of a tweet:
        -tweet length
        - word repetition patterns
        -punctuation usage
        -presence of emoticons or hashtags
		-Visuals: Looking at emojis/emoticons.
		-Structure: Checking the length of the tweet or if the user is repeating words for emphasis.
		-Metadata: Looking at hashtags and punctuation (like using lots of exclamation marks).

4. Model Training : training and evaluation of the following 3 classifiers:
	a. Decision trees (including random forests)
	b. Naive Bayes
	c. Neural Networks

5. Dataset splitting and Parameter Optimization:
	a. 60% of the dataset reserved for training
	b. 20% of the dataset reserved for validation
	c. 20% of the dataset reserved for testing
	d. parameter optimization for each classifier
	e. discuss challenges such as class imbalance and dataset sparsity
	f. propose methods such as resampling, weighting etc.)

6. Model Evaluation:
    a. accuracy
	b. precision
	c. recall
	d. F1-score
			"""

import pandas as pd
import matplotlib.pyplot as plt # Import pandas library for data manipulation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Name of the input file consisting of the tweets in arabic
file_name = "Arabic.txt"

# Define the set of valid sentiment LABELS
LABELS = {"POS", "NEG", "OBJ", "NEUTRAL"}


# Arabic stopwords from NLTK
nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
stemmer = ISRIStemmer()




def main():
    df = convert_to_csv_file()
    if df is None:
        return

    # Show first 5 rows fully (no truncation)
    pd.set_option("display.max_colwidth", None)  # Don't truncate text

    print("=== BEFORE CLEANING ===")
    print(df[["text", "label"]].head())
    data_analysis(df)


    # Apply preprocessing
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    print("\n=== AFTER CLEANING ===")
    print(df[["cleaned_text", "label"]].head())
    data_analysis(df)

    # save preprocessed CSV
    df.to_csv(f"{file_name}_preprocessed.csv", index=False, encoding="utf-8-sig")
    print(f"\nPreprocessed data saved to '{file_name}_preprocessed.csv'")


    tfidf_features, labels, vectorizer = extract_features(df)   # Generate TF–IDF vectors

    handEngineered_features = extract_handEngineered_features(df)

    final_features = combine_features(tfidf_features, handEngineered_features)

    features_training, features_validation, features_testing, labels_training, labels_validation, labels_testing = split_dataset(final_features, labels)

    print("Train size:", features_training.shape)
    print("Validation size:", features_validation.shape)
    print("Test size:", features_testing.shape)

    print("TF-IDF shape:", tfidf_features.shape)
    print("Handcrafted shape:", handEngineered_features.shape)
    print("Final shape:", final_features.shape)

    # --------------------------
    # Train & evaluate models
    # --------------------------
    decisiontree_model, randomforest_model = train_and_evaluate_models(features_training, labels_training, features_validation, labels_validation)
    evaluate_on_test_set(decisiontree_model, randomforest_model,
                         features_testing, labels_testing)




def convert_to_csv_file():
    # Create an empty list to store processed data
    data = []

    # ensure that the input file exists and was found by the program
    try:
        # Open the text file in UTF-8 encoding
        with open(file_name, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip() # Remove leading/trailing spaces and newline characters
                if not line:
                    continue        # Skip empty lines

                # split the line from the right on the last tab character to separate text and label
                parts = line.rsplit("\t", 1)

                # Check if the split produced exactly two parts and the label is valid
                if len(parts) == 2 and parts[1] in LABELS:
                    text, label = parts         # Assign text and label
                    data.append([text, label])  # Add them to our data list
                else:
                    # skip malformed lines safely
                    print("Skipped:", line)

        # Convert the list of lists into a pandas DataFrame with columns "text" and "label"
        df = pd.DataFrame(data, columns=["text", "label"])

        # Save the structured DataFrame to a CSV file
        # utf-8-sig ensures Arabic characters are properly saved and can be opened in Excel
        df.to_csv(f"{file_name}.csv", index=False, encoding="utf-8-sig")

        return df
    # in case the input file doesn't exist or error arrises
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found. Please ensure it exists in the same directory as the Arabic Sentiment Analysis script.")




def data_analysis(df):

    print(df.shape)
    # Print the number of samples per sentiment label (class distribution)
    print(df["label"].value_counts())

    # Plot class distribution
    df["label"].value_counts().plot(kind="bar")
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.show()




# Function to clean each tweet
def clean_text(text):
    # Perform lowercase, remove URLs, HTML, numbers, punctuation, and extra spaces.
    text = str(text)
    # make lowercase
    text = normalize_arabic(text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation and special characters (keep Arabic letters)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove repeating chars while safely keeping two to emphasize emotion
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text




def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub(r"ي\b", "ى", text)  # only end of word
    text = re.sub(r"ه\b", "ة", text)  # common normalization direction
    return text




def remove_stopwords(text):
    # Remove common Arabic stopwords.
    words = text.split()                # Split sentence into a list of words
    words = [w for w in words if w not in arabic_stopwords] # keep w in words only if it isnt a stopword
    return " ".join(words)




def stem_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)




def preprocess_text(text):
    # Apply basic cleaning
    text = clean_text(text)
    # Stopword removal
    text = remove_stopwords(text)
    text = stem_text(text)

    return text




# Feature Extraction:
#   1. Automated Text Representations
#       a. -TF-IDF: Counts how often words appear.
#           It gives more importance to unique, meaningful words and less to common ones.
def extract_features(df):
    """
    Converts cleaned Arabic text into numerical features using TF-IDF
    and separates the sentiment labels.

    Returns:
    - text_features: numerical representation of text (for ML models)
    - sentiment_labels: target classes (POS, NEG, OBJ, NEUTRAL)
    - tfidf_vectorizer: the trained TF-IDF object (needed later for testing)
    """

    # Create a TF-IDF vectorizer
    # ngram_range=(1,2) --> uses single words (unigrams) and word pairs (bigrams)
    # max_features=5000 --> limits vocabulary size to most important 5000 features
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_features=5000)

    # Convert cleaned text into numerical feature vectors
    # Each tweet becomes a row of numbers
    text_features = tfidf_vectorizer.fit_transform(df["cleaned_text"])

    # Extract the sentiment labels (targets)
    sentiment_labels = df["label"]

    return text_features, sentiment_labels, tfidf_vectorizer




# TWEET LENGTH FEATURE: The model learns that length can act as a weak proxy for emotional engagement
#       - Sentiment intensity correlation: Strong emotions often produce longer tweets (rants, praise, complaints)
#       - Neutral vs. opinionated separation: Very short tweets are more likely to be neutral or factual (“OK”, “Thanks”, “Update posted”)
#       - Signal amplification: Longer tweets provide more opportunities for sentiment-bearing words
def tweet_length_feature(text):
    return len(text.split())




# PUNCTUATION USAGE FEATURE: Punctuation acts as a paralinguistic cue (a substitute for tone of voice in text)
#       - Exclamation marks : excitement, anger, enthusiasm
#       - Question marks : doubt, disbelief, sarcasm (context-dependent)
#       - Multiple punctuation : emotional intensity or exaggeration
import string
ARABIC_PUNCTUATION = "؟،؛ـ"
def punctuation_count(text):
    count = 0
    for char in text:
        if char in string.punctuation or char in ARABIC_PUNCTUATION:
            count += 1
    return count



# REPEATED PUNCTUATION FEATURE : This captures emotional emphasis like:
def repeated_punctuation_count(text):
    return len(re.findall(r'([!?؟])\1+', text))




# REPEATED WORDS FEATURE (word elongation or duplication): A classifier learns that repetition increases confidence in sentiment polarity,
#                                                          not necessarily changing the direction but strengthening it.
#       - Emphasis and intensity: Repetition is a linguistic signal of strong emotion.
#       - Common in informal text: Social media users amplify sentiment through repetition rather than formal modifiers.
#       - Polarity strengthening: Repetition rarely occurs in neutral statements.
def char_elongation_count(text):
    return len(re.findall(r'(.)\1{2,}', text))

def word_repetition_count(text):
    return len(re.findall(r'\b(\w+)\s+\1\b', text))



# PRESENCE OF HASHTAGS FEATURE: The model learns that hashtags modify the interpretation of the sentence
#       - Explicit sentiment labeling : Users often encode sentiment directly in hashtags:  #loveit, #hateit, #fail, #awesome
#       - Topic–sentiment coupling : Hashtags bind sentiment to a specific entity or event: #iPhone, #WorldCup, #CustomerService
#       - Emphasis and stance : ex., Repeated hashtags (#fail #fail #fail) (These signal strong emotional stance)
#       - Sarcasm and meta-commentary : Some hashtags frame the entire tweet sarcastically: ex., #blessed (often ironic) ,#justsaying
def hashtag_count(text):
    return len(re.findall(r"#\w+", text))




# EMOJIS / EMOTICONS FEATURE:
#   - Direct expression of emotion (paralinguistic signals)
#   - Visual substitutes for facial expressions and tone
#   - Strong sentiment polarity: Many emojis have clear, dominant sentiment: (positive or negative)
#                                This provides high-confidence sentiment cues, especially when text is short or ambiguous.
#                                The model learns that emojis can override or disambiguate text sentiment.
#   - Sentiment intensity: Multiple emojis (Strong emotion amplification). Emoji frequency can function as an intensity scalar.
#   - Sarcasm and irony signals (Certain emojis often signal sarcasm), critical because sarcasm is difficult to detect from words alone.
def emoji_count(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def emoticon_count(text):
    emoticons = r'(:\)|:\(|:D|<3|;\)|:-\)|:-\(|:-D)'
    return len(re.findall(emoticons, text))



# Combine handcrafted features
def extract_handEngineered_features(df):
    df["tweet_length"] = df["cleaned_text"].apply(tweet_length_feature)

    # Use RAW text for social signals
    df["punctuation_count"] = df["text"].apply(punctuation_count)
    df["char_elongation_count"] = df["text"].apply(char_elongation_count)
    df["word_repetition_count"] = df["text"].apply(word_repetition_count)
    df["hashtag_count"] = df["text"].apply(hashtag_count)
    df["emoji_count"] = df["text"].apply(emoji_count)
    df["emoticon_count"] = df["text"].apply(emoticon_count)
    df["repeated_punct_count"] = df["text"].apply(repeated_punctuation_count)

    return df[[
        "tweet_length",
        "punctuation_count",
        "char_elongation_count",
        "word_repetition_count",
        "hashtag_count",
        "emoji_count",
        "emoticon_count",
        "repeated_punct_count"
    ]]




# Combine TF-IDF + handcrafted features
from scipy.sparse import hstack
def combine_features(tfidf_features, handcrafted_features):
    return hstack([tfidf_features, handcrafted_features.values])




# 60% of the dataset reserved for training
# 20% of the dataset reserved for validation
# 20% of the dataset reserved for testing
def split_dataset(features, labels):

    # First split: 60% training, 40% temp (later to be split into validation and testing)
    features_training, features_temp, labels_training, labels_temp = train_test_split(features, labels,
                                                                test_size=0.4, random_state=42, stratify=labels)

    # Second split: 20% validation, 20% testing
    features_validation, features_testing, labels_validation, labels_testing = train_test_split(features_temp, labels_temp,
                                                                test_size=0.5, random_state=42, stratify=labels_temp)

    return features_training, features_validation, features_testing, labels_training, labels_validation, labels_testing




def train_and_evaluate_models(features_training, labels_training,
                              features_validation, labels_validation):
    # --------------------------
    # 1. Decision Tree
    # --------------------------
    decisiontree_parameters = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }

    # Handle class imbalance directly here
    decisiontree = DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    )

    decisiontree_grid = GridSearchCV(
        decisiontree,
        decisiontree_parameters,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )

    decisiontree_grid.fit(features_training, labels_training)

    print("=== Decision Tree Best Parameters ===")
    print(decisiontree_grid.best_params_)

    decisiontree_predictions = decisiontree_grid.predict(features_validation)
    print("\n--- Decision Tree Evaluation ---")
    print(classification_report(labels_validation, decisiontree_predictions))
    print("Confusion Matrix:\n", confusion_matrix(labels_validation, decisiontree_predictions))

    # --------------------------
    # 2. Random Forest
    # --------------------------
    randomforest_parameters = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    # Handle class imbalance directly
    randomforest = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    randomforest_grid = GridSearchCV(
        randomforest,
        randomforest_parameters,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )

    randomforest_grid.fit(features_training, labels_training)

    print("\n=== Random Forest Best Parameters ===")
    print(randomforest_grid.best_params_)

    randomforest_predictions = randomforest_grid.predict(features_validation)
    print("\n--- Random Forest Evaluation ---")
    print(classification_report(labels_validation, randomforest_predictions))
    print("Confusion Matrix:\n", confusion_matrix(labels_validation, randomforest_predictions))

    # Return best trained models
    return decisiontree_grid.best_estimator_, randomforest_grid.best_estimator_




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
def evaluate_on_test_set(decision_tree_model, random_forest_model,
                         features_testing, labels_testing):
    """
    Evaluates the trained models on the test set.
    Prints accuracy, precision, recall, F1-score and displays confusion matrices.
    """

    models = {
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    for name, model in models.items():
        print(f"\n=== {name} Evaluation on Test Set ===")

        # Make predictions
        predictions = model.predict(features_testing)

        # Compute metrics
        acc = accuracy_score(labels_testing, predictions)
        prec = precision_score(labels_testing, predictions, average='macro')
        rec = recall_score(labels_testing, predictions, average='macro')
        f1 = f1_score(labels_testing, predictions, average='macro')

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(labels_testing, predictions, labels=["POS", "NEG", "OBJ", "NEUTRAL"])
        print("Confusion Matrix:\n", cm)

        # Plot the confusion matrix as heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["POS", "NEG", "OBJ", "NEUTRAL"],
                    yticklabels=["POS", "NEG", "OBJ", "NEUTRAL"])
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()


if __name__ == "__main__":
    main()
