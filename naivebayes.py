import nltk, string
import pandas as pd
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

CSV_PATH = r"C:\Users\Jonathan Philips\Coding\nltk-text-tone-processing\resources\responses.csv"

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_tokens(text):
    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]


def train_classifier():
    try:
        df = pd.read_csv(CSV_PATH, names=["text", "sentiment", "timestamp"], skiprows=1)
    except Exception as e:
        print("Error loading CSV:", e)
        return None, []

    documents = [
        (clean_tokens(row["text"]), row["sentiment"]) for _, row in df.iterrows()
    ]

    all_words = FreqDist(word for tokens, _ in documents for word in tokens)
    word_features = list(all_words)[:2000]

    def document_features(document):
        words = set(document)
        return {f"contains({w})": (w in words) for w in word_features}

    featuresets = [(document_features(d), c) for (d, c) in documents]
    if not featuresets:
        return None, []

    train_set = featuresets
    classifier = NaiveBayesClassifier.train(train_set)
    return classifier, word_features


classifier, word_features = train_classifier()


def classify_comment(text):
    if not classifier:
        return "unknown"
    tokens = clean_tokens(text)
    features = {f"contains({w})": (w in tokens) for w in word_features}
    return classifier.classify(features)
