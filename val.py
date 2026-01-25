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
from sklearn.preprocessing import StandardScaler

# Name of the input file consisting of the tweets in arabic
file_name = "Arabic.txt"

# Define the set of valid sentiment LABELS
LABELS = {"POS", "NEG", "OBJ", "NEUTRAL"}


# Arabic stopwords from NLTK
#nltk.download('stopwords')
arabic_stopwords = set(stopwords.words('arabic'))
# Remove negation words from stopwords
negations = {"مش", "مو", "ما", "ليس", "لا", "لم", "لن", "بدون", "غير"}
arabic_stopwords = arabic_stopwords - negations
# Stemmer
stemmer = ISRIStemmer()
# Emoji dictionary
EMO_POS = " EMO_POS "
EMO_NEG = " EMO_NEG "
EMO_NEU = " EMO_NEU "

POS_EMOJIS = {
    "😂", "🤣", "😄", "😃", "😊", "😁", "😍", "🥰",
    "❤️", "💖", "💕", "👍", "👏", "🔥"
}

NEG_EMOJIS = {
    "😡", "🤬", "😠", "😢", "😭", "💔",
    "👎", "😞", "😤", "😰", "😨"
}

NEU_EMOJIS = {
    "😐", "😑", "😶", "🙄"
}

# Arabic and english punctuations
punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''

# --------------------------
# 1. Define hyperparameter grids for validation-based tuning
# --------------------------
decisiontree_params = {
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10]
}

randomforest_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

mlp_params = {
    "hidden_layer_sizes": [(128, 64), (64, 32)],
    "learning_rate_init": [0.001, 0.01],
    "max_iter": [100, 200]
}

nb_params = {
    "alpha": [0.1, 0.5, 1.0]
}

def main():
    # ==================================================
    # 1. Load dataset
    # ==================================================
    df = convert_to_csv_file()
    if df is None:
        return

    pd.set_option("display.max_colwidth", None)

    print("=== BEFORE CLEANING ===")
    print(df[["text", "label"]].head())
    data_analysis(df)

    # ==================================================
    # 2. Text preprocessing
    # ==================================================
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    print("\n=== AFTER CLEANING ===")
    print(df[["cleaned_text", "label"]].head())
    data_analysis(df)

    df.to_csv(f"{file_name}_preprocessed.csv", index=False, encoding="utf-8-sig")
    print(f"\nPreprocessed data saved to '{file_name}_preprocessed.csv'")

    # ==================================================
    # 3. Train / Val / Test split
    # ==================================================
    texts = df["cleaned_text"]
    labels = df["label"]

    texts_train, texts_val, texts_test, labels_train, labels_val, labels_test = \
        split_dataset(texts, labels)

    # ==================================================
    # 4. Class weights (for imbalance handling)
    # ==================================================
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(labels_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weight_dict)

    # ==================================================
    # 5. TF-IDF features (fit on TRAIN only)
    # ==================================================
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=3000,
        min_df=5,
        max_df=0.85
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(texts_train)
    X_val_tfidf   = tfidf_vectorizer.transform(texts_val)
    X_test_tfidf  = tfidf_vectorizer.transform(texts_test)

    # Dense TF-IDF for MLP
    X_train_tfidf_mlp = X_train_tfidf.toarray()
    X_val_tfidf_mlp   = X_val_tfidf.toarray()
    X_test_tfidf_mlp  = X_test_tfidf.toarray()

    # ==================================================
    # 6. Hand-engineered features
    # ==================================================
    from sklearn.preprocessing import StandardScaler

    X_train_hand = extract_handEngineered_features(df.loc[texts_train.index])
    X_val_hand   = extract_handEngineered_features(df.loc[texts_val.index])
    X_test_hand  = extract_handEngineered_features(df.loc[texts_test.index])

    scaler_hand = StandardScaler()
    X_train_hand = scaler_hand.fit_transform(X_train_hand)
    X_val_hand   = scaler_hand.transform(X_val_hand)
    X_test_hand  = scaler_hand.transform(X_test_hand)

    # ==================================================
    # 7. Word embeddings (FastText trained on TRAIN only)
    # ==================================================
    embedding_model = train_embedding_model(
        df.loc[texts_train.index],
        method="fasttext"
    )

    X_train_emb = compute_embedding_vectors(
        df.loc[texts_train.index], embedding_model, vector_size=100
    )
    X_val_emb = compute_embedding_vectors(
        df.loc[texts_val.index], embedding_model, vector_size=100
    )
    X_test_emb = compute_embedding_vectors(
        df.loc[texts_test.index], embedding_model, vector_size=100
    )

    # ==================================================
    # 8. MLP feature matrix (TF-IDF + handcrafted + embeddings)
    # ==================================================
    X_train_mlp = np.hstack([X_train_tfidf_mlp, X_train_hand, X_train_emb])
    X_val_mlp   = np.hstack([X_val_tfidf_mlp,   X_val_hand,   X_val_emb])
    X_test_mlp  = np.hstack([X_test_tfidf_mlp,  X_test_hand,  X_test_emb])

    # Scale ALL MLP inputs
    scaler_mlp = StandardScaler()
    X_train_mlp = scaler_mlp.fit_transform(X_train_mlp)
    X_val_mlp   = scaler_mlp.transform(X_val_mlp)
    X_test_mlp  = scaler_mlp.transform(X_test_mlp)

    # ==================================================
    # 9. Classical model feature matrices (TF-IDF + handcrafted)
    # ==================================================
    from scipy.sparse import hstack

    X_train = hstack([X_train_tfidf, X_train_hand])
    X_val   = hstack([X_val_tfidf,   X_val_hand])
    X_test  = hstack([X_test_tfidf,  X_test_hand])

    print("Training features shape:", X_train.shape)
    print("Validation features shape:", X_val.shape)
    print("Test features shape:", X_test.shape)

    # ==================================================
    # 10. Train models
    # ==================================================
    dt_model, rf_model, nb_model, mlp_model = train_and_evaluate_models(
        X_train,
        X_val,
        X_train_tfidf,
        X_val_tfidf,
        labels_train,
        labels_val,
        X_train_mlp,
        X_val_mlp,
        class_weight_dict
    )

    # ==================================================
    # 11. Final evaluation on test set
    # ==================================================
    evaluate_on_test_set(
        dt_model,
        rf_model,
        nb_model,
        mlp_model,
        X_test,
        X_test_tfidf,
        labels_test,
        X_test_mlp
    )




def train_embedding_model(df, method="fasttext", vector_size=100,
                          window=5, min_count=2, epochs=5):
    sentences = [text.split() for text in df["cleaned_text"]]

    if method == "word2vec":
        model = Word2Vec(sentences, vector_size=vector_size,
                         window=window, min_count=min_count,
                         workers=4, epochs=epochs)
    else:
        model = FastText(sentences, vector_size=vector_size,
                         window=window, min_count=min_count,
                         workers=4, epochs=epochs)
    return model



def compute_embedding_vectors(df, model, vector_size):
    embeddings = []

    for text in df["cleaned_text"]:
        words = text.split()
        vectors = [model.wv[w] for w in words if w in model.wv]

        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(vector_size))

    return np.array(embeddings)



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
                    print("Skipped malformed line:", line)

        # Convert the list of lists into a pandas DataFrame with columns "text" and "label"
        df = pd.DataFrame(data, columns=["text", "label"])

        df["label"] = df["label"].replace({"NEUTRAL": "OBJ"})
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
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s#]", '', text)  # remove other punctuation except hashtags
    # Remove punctuation and special characters (keep Arabic letters) except hashtags
    text = re.sub(r"[^\u0600-\u06FF\s#]", " ", text)
    return text




def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub(r"ي\b", "ى", text)  # only end of word
    text = re.sub(r"ه\b", "ة", text)  # common normalization direction
    return text



# Filters out common words that carry little sentiment meaning
def remove_stopwords(text):
    # Remove common Arabic stopwords.
    words = text.split()                # Split sentence into a list of words
    words = [w for w in words if w not in arabic_stopwords] # keep w in words only if it isnt a stopword
    return " ".join(words)



# Converts words to their root form to reduce lexical variation
def stem_text(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)




# Collapses repeated characters
def remove_elongation(text):
    return re.sub(r'(.)\1+', r'\1', text)




# Converts emojis into sentiment tokens (EMO_POS, EMO_NEG, EMO_NEU)
def replace_emojis(text):
    result = []
    for ch in text:
        if ch in POS_EMOJIS:
            result.append(EMO_POS)
        elif ch in NEG_EMOJIS:
            result.append(EMO_NEG)
        elif ch in NEU_EMOJIS:
            result.append(EMO_NEU)
        else:
            result.append(ch)
    return "".join(result)


def handle_negation(text):
    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        if tokens[i] in negations and i + 1 < len(tokens):
            # Keep negation word
            result.append(tokens[i])

            # Negate only the next non-stopword
            next_word = tokens[i + 1]
            if next_word not in arabic_stopwords:
                result.append("NOT_" + next_word)
            else:
                result.append(next_word)

            i += 2
        else:
            result.append(tokens[i])
            i += 1

    return " ".join(result)






def preprocess_text(text):
    # Normalize
    text = normalize_arabic(text)
    # Remove elongation
    text = remove_elongation(text)
    # Apply basic cleaning (remove URLs, HTML, numbers, punctuation)
    text = clean_text(text)
    # Keep hashtags as separate words
    text = re.sub(r"#(\w+)", r"\1", text)
    # Replace emojis with sentiment tokens
    text = replace_emojis(text)
    # Handle negation across multiple words (2 words after negation)
    text = handle_negation(text)
    # Stopword removal
    text = remove_stopwords(text)
    # Stemming
    text = stem_text(text)

    return text





from gensim.models import Word2Vec, FastText



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
def punctuation_count(text):
    count = 0
    for char in text:
        if char in string.punctuation or char in punctuations:
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





def negation_count(text):
    negations = ["مش", "مو", "ما", "ليس", "لا", "لم", "لن", "بدون", "غير"]
    return sum(text.count(neg) for neg in negations)



def emoji_features(text):
    return pd.Series({"has_positive_emoji": int("EMO_POS" in text),
                      "has_negative_emoji": int("EMO_NEG" in text)})




def dialect_feature(text):
    dialect_words = ["مش", "مو", "شو", "ليش", "هيك"]
    return int(any(word in text for word in dialect_words))




# Combine handcrafted features
def extract_handEngineered_features(df):
    df = df.copy()
    df["tweet_length"] = df["cleaned_text"].apply(tweet_length_feature)

    # Use RAW text for social signals
    df["punctuation_count"] = df["text"].apply(punctuation_count)
    df["char_elongation_count"] = df["text"].apply(char_elongation_count)
    df["word_repetition_count"] = df["text"].apply(word_repetition_count)
    df["hashtag_count"] = df["text"].apply(hashtag_count)
    df["emoji_count"] = df["text"].apply(emoji_count)
    df["emoticon_count"] = df["text"].apply(emoticon_count)
    df["repeated_punct_count"] = df["text"].apply(repeated_punctuation_count)

    # Fix: use 'cleaned_text' instead of 'clean_text'
    df["neg_count"] = df["cleaned_text"].apply(negation_count)
    df["dialect"] = df["cleaned_text"].apply(dialect_feature)
    emoji_df = df["cleaned_text"].apply(emoji_features)
    df = pd.concat([df, emoji_df], axis=1)

    return df[[
        "tweet_length",
        "punctuation_count",
        "char_elongation_count",
        "word_repetition_count",
        "hashtag_count",
        "emoji_count",
        "emoticon_count",
        "repeated_punct_count",
        "neg_count",
        "dialect",
        "has_positive_emoji",
        "has_negative_emoji"
    ]]





# Combine all features (TF-IDF + embeddings + handcrafted)
from scipy.sparse import hstack, csr_matrix



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





from scipy.sparse import vstack

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight


def train_and_evaluate_models(
    features_training,
    features_validation,
    features_training_tfidf,
    features_validation_tfidf,
    labels_training,
    labels_validation,
    X_train_mlp,
    X_val_mlp,
    class_weight_dict
):
    # ==================================================
    # 0. Prepare MLP inputs (dense + numeric)
    # ==================================================
    if hasattr(X_train_mlp, "toarray"):
        X_train_mlp = X_train_mlp.toarray()
    if hasattr(X_val_mlp, "toarray"):
        X_val_mlp = X_val_mlp.toarray()

    X_train_mlp = X_train_mlp.astype(np.float32)
    X_val_mlp   = X_val_mlp.astype(np.float32)


    assert X_train_mlp.shape[0] == len(labels_training)
    assert X_val_mlp.shape[0] == len(labels_validation)

    # ==================================================
    # 1. DECISION TREE — validation-based tuning
    # ==================================================
    best_f1 = 0
    best_dt_params = None

    for max_depth in [None, 10, 20, 30]:
        for min_samples_split in [2, 5, 10]:
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                class_weight=class_weight_dict
            )
            dt.fit(features_training, labels_training)
            preds = dt.predict(features_validation)
            f1 = f1_score(labels_validation, preds, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_dt_params = (max_depth, min_samples_split)

    print("=== Decision Tree Best Params (Validation) ===")
    print(f"max_depth={best_dt_params[0]}, min_samples_split={best_dt_params[1]}")

    X_dt_full = vstack([features_training, features_validation])
    y_dt_full = np.concatenate([labels_training, labels_validation])

    final_dt = DecisionTreeClassifier(
        max_depth=best_dt_params[0],
        min_samples_split=best_dt_params[1],
        random_state=42,
        class_weight=class_weight_dict
    )
    final_dt.fit(X_dt_full, y_dt_full)

    # ==================================================
    # 2. RANDOM FOREST — validation-based tuning
    # ==================================================
    best_f1 = 0
    best_rf_params = None

    for n_estimators in [100, 200]:
        for max_depth in [None, 10, 20]:
            for min_samples_split in [2, 5]:
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    class_weight=class_weight_dict,
                    n_jobs=-1
                )
                rf.fit(features_training, labels_training)
                preds = rf.predict(features_validation)
                f1 = f1_score(labels_validation, preds, average="macro")

                if f1 > best_f1:
                    best_f1 = f1
                    best_rf_params = (n_estimators, max_depth, min_samples_split)

    print("\n=== Random Forest Best Params (Validation) ===")
    print(
        f"n_estimators={best_rf_params[0]}, "
        f"max_depth={best_rf_params[1]}, "
        f"min_samples_split={best_rf_params[2]}"
    )

    X_rf_full = vstack([features_training, features_validation])
    y_rf_full = np.concatenate([labels_training, labels_validation])

    final_rf = RandomForestClassifier(
        n_estimators=best_rf_params[0],
        max_depth=best_rf_params[1],
        min_samples_split=best_rf_params[2],
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1
    )
    final_rf.fit(X_rf_full, y_rf_full)

    # ==================================================
    # 3. NAÏVE BAYES — validation-based alpha tuning
    # ==================================================
    best_f1 = 0
    best_alpha = None

    for alpha in [0.1, 0.3, 0.5, 1.0]:
        nb = MultinomialNB(alpha=alpha, fit_prior=False)
        nb.fit(features_training_tfidf, labels_training)
        preds = nb.predict(features_validation_tfidf)
        f1 = f1_score(labels_validation, preds, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha

    print("\n=== Naïve Bayes Best Alpha (Validation) ===")
    print(f"alpha={best_alpha}")

    X_nb_full = np.vstack([
        features_training_tfidf.toarray(),
        features_validation_tfidf.toarray()
    ])
    y_nb_full = np.concatenate([labels_training, labels_validation])

    final_nb = MultinomialNB(alpha=best_alpha,fit_prior=False)
    final_nb.fit(X_nb_full, y_nb_full)

    # ==================================================
    # 4. MLP — validation-based tuning (CLASS WEIGHTS ONLY)
    # ==================================================
    best_f1 = 0
    best_mlp_params = None

    classes = np.unique(labels_training)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_training
    )
    class_weight_mlp = dict(zip(classes, weights))

    sample_weight = np.array([class_weight_mlp[y] for y in labels_training])

    for hidden in [(128,), (128, 64)]:
        for lr in [0.001, 0.0005]:
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=lr,
                max_iter=500,
                random_state=42
            )

            mlp.fit(
                X_train_mlp,
                labels_training,
                sample_weight=sample_weight
            )

            preds = mlp.predict(X_val_mlp)
            f1 = f1_score(labels_validation, preds, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_mlp_params = (hidden, lr)

    print("\n=== MLP Best Params (Validation) ===")
    print(f"hidden_layers={best_mlp_params[0]}, learning_rate={best_mlp_params[1]}")

    # Retrain on TRAIN + VAL
    X_mlp_full = np.vstack([X_train_mlp, X_val_mlp])
    y_mlp_full = np.concatenate([labels_training, labels_validation])

    classes_full = np.unique(y_mlp_full)
    weights_full = compute_class_weight(
        class_weight="balanced",
        classes=classes_full,
        y=y_mlp_full
    )
    sample_weight_full = np.array(
        [dict(zip(classes_full, weights_full))[y] for y in y_mlp_full]
    )

    final_mlp = MLPClassifier(
        hidden_layer_sizes=best_mlp_params[0],
        learning_rate_init=best_mlp_params[1],
        max_iter=500,
        random_state=42
    )

    final_mlp.fit(X_mlp_full, y_mlp_full, sample_weight=sample_weight_full)

    return final_dt, final_rf, final_nb, final_mlp






from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def evaluate_on_test_set(decision_tree_model, random_forest_model, nb_model, mlp_model,
                         features_testing, features_testing_tfidf, labels_testing,X_test_mlp):
    """
    Evaluates the trained models on the test set.
    Prints accuracy, precision, recall, F1-score and displays confusion matrices.
    """

    models = {
        "Decision Tree": (decision_tree_model, features_testing),
        "Random Forest": (random_forest_model, features_testing),
        "Naïve Bayes": (nb_model, features_testing_tfidf),  # TF-IDF ONLY
        "MLP Neural Network": (mlp_model, X_test_mlp)
    }

    label_order = ["POS", "NEG", "OBJ"]

    for name, (model, features) in models.items():
        print(f"\n=== {name} Evaluation on Test Set ===")

        # Make predictions
        predictions = model.predict(features)

        # Compute metrics
        acc = accuracy_score(labels_testing, predictions)
        prec = precision_score(labels_testing, predictions, average='macro', zero_division=0)
        rec = recall_score(labels_testing, predictions, average='macro', zero_division=0)
        f1 = f1_score(labels_testing, predictions, average='macro', zero_division=0)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(labels_testing, predictions, labels=label_order)
        print("Confusion Matrix:\n", cm)

        # Plot confusion matrix as heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=label_order,
            yticklabels=label_order
        )
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()
        plt.close()  # Free resources

if __name__ == "__main__":
    main()
