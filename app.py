from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    get_flashed_messages,
)
from datetime import datetime
from naivebayes import classify_comment, train_classifier
import csv, os
import pandas as pd
from collections import Counter

app = Flask(__name__)
app.secret_key = "new-refresh-clear-all-ting-22"

CSV_PATH = r"C:\Users\Jonathan Philips\Coding\nltk-text-tone-processing\resources\responses.csv"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        comment = request.form["comment"]
        label = classify_comment(comment)

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [comment, label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            )

        train_classifier()

        flash(label)
        return redirect(url_for("index"))

    messages = get_flashed_messages()
    label = messages[0] if messages else ""

    df = pd.read_csv(CSV_PATH, names=["text", "sentiment", "timestamp"], skiprows=1)
    valid_sentiments = {"positive", "negative", "neutral", "mixed"}
    df = df[df["sentiment"].str.lower().isin(valid_sentiments)]
    sentiment_counts = Counter(df["sentiment"])

    return render_template("index.html", label=label, sentiment_counts=sentiment_counts)


if __name__ == "__main__":
    app.run(debug=True)
