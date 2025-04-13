import nltk, csv, os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from datetime import datetime

nltk.download("vader_lexicon")
nltk.download("punkt")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    scores = None
    entries = []
    csv_path = os.path.join("resources", "responses.csv")

    if request.method == "POST":
        text = request.form["text"].strip()

        if not text:
            return render_template("index.html", sentiment="Please enter some text.")

        sid = SentimentIntensityAnalyzer()
        sentence_results = []

        sentences = sent_tokenize(text)
        datestamp = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")

        for sentence in sentences:
            scores = sid.polarity_scores(sentence)
            if scores["compound"] >= 0.45:
                label = "Positive"
            elif scores["compound"] <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            sentence_results.append([sentence, label, scores])

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for sentence, label, _ in sentence_results:
                writer.writerow([sentence, label, datestamp, timestamp])

        pos_count = sum(1 for _, label, _ in sentence_results if label == "Positive")
        neg_count = sum(1 for _, label, _ in sentence_results if label == "Negative")
        neu_count = sum(1 for _, label, _ in sentence_results if label == "Neutral")

        if pos_count > max(neg_count, neu_count):
            overall_sentiment = "Overall Sentiment: Positive"
        elif neg_count > max(pos_count, neu_count):
            overall_sentiment = "Overall Sentiment: Negative"
        elif neu_count > max(pos_count, neg_count):
            overall_sentiment = "Overall Sentiment: Neutral"
        else:
            overall_sentiment = "Overall Sentiment: Mixed"

        return render_template(
            "index.html",
            sentence_results=sentence_results,
            overall_sentiment=overall_sentiment,
        )

    if os.path.exists(csv_path):
        df = pd.read_csv(
            csv_path, header=None, names=["Text", "Sentiment", "Date", "Time"]
        )

        if df.iloc[0]["Text"] == "Text" and df.iloc[0]["Sentiment"] == "Sentiment":
            df = df.iloc[1:]

        entries = df.tail(10).values.tolist()

    return render_template(
        "index.html", sentiment=sentiment, scores=scores, entries=entries
    )


if __name__ == "__main__":
    app.run(debug=True)
