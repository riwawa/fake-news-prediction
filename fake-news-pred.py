import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
CSV_PATH = "WELFake_Dataset.csv"
TARGET_COL = "label"


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Load dataset
data = pd.read_csv(CSV_PATH)

# Fill missing values
data["title"] = data["title"].fillna("")
data["text"] = data["text"].fillna("")

# Create one text column
data["content"] = (data["title"] + " " + data["text"]).apply(clean_text)

# Inputs and target
X = data["content"]
y = data[TARGET_COL]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

# Train
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_news(title, text):
    content = clean_text(title + " " + text)
    prediction = model.predict([content])[0]

    if prediction == 1:
        return "Real News 🟢"
    return "Fake News 🔴"


# Example
example_title = "Breaking: Government announces new economic policy"
example_text = "The finance minister introduced a new policy today after discussions in parliament."

result = predict_news(example_title, example_text)
print("Prediction:", result)