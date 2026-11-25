import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Review Sentiment Classifier", page_icon="💬", layout="centered")

# ---------- Built-in sample data ----------
POS = [
    "Absolutely love this product, works perfectly",
    "Great quality and fast delivery, highly recommend",
    "Excellent value for money, very happy with it",
    "Amazing performance, exceeded my expectations",
    "Super comfortable and looks fantastic",
    "Customer service was helpful and friendly",
    "Best purchase I have made this year",
    "Easy to use and very reliable",
    "Brilliant product, will buy again",
    "Packaging was neat and the item is superb",
    "Very satisfied, five stars from me",
    "Works like a charm, totally worth the price",
]
NEG = [
    "Terrible quality, broke after two days",
    "Worst product ever, total waste of money",
    "Very disappointed, does not work as described",
    "Delivery was late and the item was damaged",
    "Poor customer service, nobody replied to me",
    "Cheap material and awful finish, do not buy",
    "Stopped working within a week, very bad",
    "Not worth the price, I want a refund",
    "Horrible experience, the product is defective",
    "Battery drains quickly and the screen is faulty",
    "Misleading photos, the real item looks bad",
    "Extremely unhappy, I regret buying this",
]
NEU = [
    "It is okay, nothing special",
    "Average product, does the job",
    "Delivery was on time, product is as expected",
    "Decent for the price, not great not bad",
    "It works but could be better",
    "Mixed feelings, some features are good, some are not",
    "Neither impressed nor disappointed",
    "Fine for basic use, nothing more",
    "The item arrived and looks the same as the photo",
    "Acceptable quality, I might use it occasionally",
]


@st.cache_data
def sample_df() -> pd.DataFrame:
    rows = (
        [(t, "Positive") for t in POS]
        + [(t, "Negative") for t in NEG]
        + [(t, "Neutral") for t in NEU]
    )
    return pd.DataFrame(rows, columns=["review", "sentiment"])


@st.cache_resource
def train(df: pd.DataFrame):
    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()

    counts = df["sentiment"].value_counts()
    stratify = df["sentiment"] if counts.min() >= 2 and len(df) >= 20 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=stratify
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T
    model.fit(df["review"], df["sentiment"])  # refit on all data
    return model, acc, report


EMOJI = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}

# ---------- UI ----------
st.title("💬 Customer Review Sentiment Classifier")
st.caption("TF-IDF + Logistic Regression, trained live in your browser session.")

with st.sidebar:
    st.header("Training data")
    source = st.radio("Source", ["Built-in sample", "Upload my CSV"])
    if source == "Upload my CSV":
        up = st.file_uploader("CSV with columns: review, sentiment", type="csv")
        data = pd.read_csv(up) if up else None
        if data is not None and not {"review", "sentiment"}.issubset(data.columns):
            st.error("CSV must have 'review' and 'sentiment' columns.")
            data = None
    else:
        data = sample_df()
    if data is None:
        st.info("Upload a CSV to train, or switch to the built-in sample.")
        st.stop()
    st.write(f"{len(data)} labelled reviews")

model, acc, report = train(data)

tab1, tab2, tab3 = st.tabs(["Single review", "Batch CSV", "Model info"])

with tab1:
    text = st.text_area("Enter a customer review", height=120, placeholder="The product quality is amazing...")
    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter a review.")
        else:
            label = model.predict([text])[0]
            probs = model.predict_proba([text])[0]
            st.subheader(f"{EMOJI.get(label, '')} {label}")
            st.bar_chart(pd.Series(probs, index=model.classes_, name="Probability"))

with tab2:
    f = st.file_uploader("Upload reviews to classify (needs a 'review' column)", type="csv", key="batch")
    if f:
        batch = pd.read_csv(f)
        if "review" not in batch.columns:
            st.error("CSV needs a 'review' column.")
        else:
            batch["predicted_sentiment"] = model.predict(batch["review"].astype(str))
            st.dataframe(batch, use_container_width=True)
            st.bar_chart(batch["predicted_sentiment"].value_counts())
            st.download_button(
                "Download results",
                batch.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv",
            )

with tab3:
    st.metric("Hold-out accuracy", f"{acc:.0%}")
    st.dataframe(report.round(2), use_container_width=True)
    st.caption("The built-in sample is tiny, so use your own larger dataset for real results.")
