import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Smriti05wy/fake-review-distilbert"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Fake Review Detection (DistilBERT)")

text = st.text_area("Enter review text")

if st.button("Predict"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        if pred == 1:
            st.success("REAL REVIEW ✅")
        else:
            st.error("FAKE REVIEW ❌")
    else:
        st.warning("Enter text first")
