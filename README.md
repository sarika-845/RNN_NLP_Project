
# 📰 AG News Text Classification with RNN + Streamlit

## 📌 Problem Statement
Classify news articles into **World, Sports, Business, or Sci/Tech** using Natural Language Processing (NLP).  
This project demonstrates end-to-end ML workflow: **data preprocessing → model training → saving artifacts → deployment with Streamlit**.

---

## 🎯 Objectives
- Build an **RNN model** for text classification.
- Use **Tokenizer + Embedding + LSTM/RNN layers** for sequence modeling.
- Save trained model (`rnn_model.h5`) and tokenizer (`tokenizer.pkl`).
- Deploy with **Streamlit** for interactive predictions.
- Document workflow for interviews and GitHub.

---

## 🛠️ Tech Stack
- **Python** (ML + NLP)
- **TensorFlow/Keras** (RNN model)
- **Scikit-learn** (preprocessing utilities)
- **Joblib** (save/load tokenizer)
- **Streamlit** (deployment UI)
- **GitHub + Streamlit Cloud** (hosting)

---

## ⚙️ Workflow
1. **Data Preparation**  
   - Tokenized text → padded sequences.  
   - Labels encoded into 4 categories.

2. **Model Training**  
   - RNN layers for sequence learning.  
   - Trained on AG News dataset.  
   - Saved as `rnn_model.h5`.

3. **Artifacts**  
   - Tokenizer saved as `tokenizer.pkl`.  
   - Requirements pinned for reproducibility.

4. **Deployment**  
   - Streamlit app (`app.py`) loads model + tokenizer.  
   - User enters text → prediction displayed with probabilities.  
   - Hosted on Streamlit Cloud.



## 🚀 Live Demo

🌐 **Live Application:** [RNN NLP Project]  https://rnnnlpproject-h79zdsdeekugh3kw7daam2.streamlit.app/

🤖 The trained RNN-based NLP model is deployed using Streamlit, allowing users to enter text and generate predictions through an interactive web interface.

---

## 📂 Project Structure
RNN_NLP_Project/
│── venv/
│── app.py
│── requirements.txt
│── rnn_model.h5
│── tokenizer.pkl
│── README.md

📊 Results
Achieved ~89–91% accuracy on AG News dataset (validation set).

Streamlit UI provides real-time predictions.

Example:
Input: "Apple announces new AI-powered features in its latest iPhone release."
Output: Sci/Tech
