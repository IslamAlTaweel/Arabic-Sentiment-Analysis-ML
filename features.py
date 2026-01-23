# arabic_features.py
from preprocessing import negations

dialect_words = ["مش", "مو", "شو", "ليش", "هيك"]

def negation_count(text):
    return sum(text.count(neg) for neg in negations)

def emoji_features(text):
    return {
        "has_positive_emoji": int("EMO_POS" in text),
        "has_negative_emoji": int("EMO_NEG" in text)
    }

def dialect_feature(text):
    return int(any(word in text for word in dialect_words))
