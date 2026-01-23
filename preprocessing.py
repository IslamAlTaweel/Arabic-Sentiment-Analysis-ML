# arabic_preprocessing.py
import re

emoji_dict = {
    "😂": " EMO_POS ",
    "❤️": " EMO_POS ",
    "😡": " EMO_NEG ",
    "😢": " EMO_NEG "
}

negations = ["مش", "مو", "ما", "ليس", "لا", "لم", "لن", "بدون", "غير"]

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text

def remove_elongation(text):
    return re.sub(r'(.)\1+', r'\1', text)

def replace_emojis(text):
    for emoji, token in emoji_dict.items():
        text = text.replace(emoji, token)
    return text

def handle_negation(text):
    for neg in negations:
        text = text.replace(neg, " NOT_")
    return text