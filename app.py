import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# -------------------------------
# Load model and tokenizer
# -------------------------------
model = load_model("rnn_model.h5")
tokenizer = joblib.load("tokenizer.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AG News Classifier", page_icon="📰", layout="centered")

st.title("📰 AG News Text Classifier")
st.write("Classify news articles into **World, Sports, Business, or Sci/Tech** using an RNN model.")

# -------------------------------
# User Input
# -------------------------------
user_input = st.text_area("✍️ Enter a news headline or article text:")

if st.button("🔍 Predict"):
    if user_input.strip() != "":
        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=200, padding="post", truncating="post")

        # Predict
        prediction = model.predict(pad)
        pred_class = np.argmax(prediction)

        # Map class index to label
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        st.success(f"✅ Predicted Category: **{labels[pred_class]}**")

        # Show probabilities
        st.write("### Prediction Probabilities")
        probs = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
        st.bar_chart(probs)

    else:
        st.warning("⚠️ Please enter some text before predicting.")
