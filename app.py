from flask import Flask, render_template, request
import nltk, csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        if scores["compound"] >= 0.45:
            sentiment = "Positive"
        elif scores["compound"] <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        with open(
            "nltk-text-tone-processing\\resources\\responses.csv", "a", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([text, sentiment])
        return render_template("index.html", sentiment=sentiment)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
