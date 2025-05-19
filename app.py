from flask import Flask, render_template, request
from naivebayes import classify_comment, train_classifier
import csv, os
import pandas as pd
from collections import Counter

app = Flask(__name__)
CSV_PATH = r"C:\Users\Jonathan Philips\Coding\nltk-text-tone-processing\resources\responses.csv"


@app.route("/", methods=["GET", "POST"])
def index():
    label = ""

    if request.method == "POST":
        comment = request.form["comment"]
        label = classify_comment(comment)

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([comment, label])

        train_classifier()

    sentiment_counts = {}
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, names=["text", "sentiment", "timestamp"], skiprows=1)
            sentiment_counts = Counter(df["sentiment"])
        except Exception as e:
            print("Error reading CSV:", e)

    return render_template("index.html", label=label, sentiment_counts=sentiment_counts)


if __name__ == "__main__":
    app.run(debug=True)
