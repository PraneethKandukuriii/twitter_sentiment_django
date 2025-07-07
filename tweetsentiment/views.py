import os
import joblib
import re
from django.shortcuts import render
from .forms import TweetForm

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'model', 'trained_model.sav')
vectorizer_path = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ✅ Text preprocessing to match training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)          # Remove mentions
    text = re.sub(r"#", "", text)             # Remove hashtag symbol
    text = re.sub(r"[^\w\s]", "", text)       # Remove punctuation
    text = re.sub(r"\d+", "", text)           # Remove numbers
    return text.strip()

# ✅ Predict function
def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]

    label_map = {
        0: "Negative 😠",
        1: "Positive 😊"
    }

    return label_map.get(prediction, "Unknown 🤔"), prediction

# ✅ View to handle form
def sentiment_view(request):
    sentiment = None
    is_positive = None

    if request.method == 'POST':
        form = TweetForm(request.POST)
        if form.is_valid():
            tweet = form.cleaned_data['tweet']
            sentiment, prediction = predict_sentiment(tweet)
            is_positive = "Yes ✅" if prediction == 1 else "No ❌"
    else:
        form = TweetForm()

    return render(request, 'tweetsentiment/index.html', {
        'form': form,
        'sentiment': sentiment,
        'is_positive': is_positive
    })
