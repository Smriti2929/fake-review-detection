# 📌 Fake Review Detection using NLP

## 🧠 Overview

This project is an end-to-end Fake Review Detection system built using Natural Language Processing (NLP) and Machine Learning techniques.  
It classifies user reviews as Fake or Genuine using traditional ML models and transformer-based approaches.

The system includes a full ML pipeline — from text preprocessing and feature engineering to model training and cloud deployment.

---

## 🌐 Live Demo (Azure Deployment)

👉 https://fake-review-distilbert-app-123-axa4e7awadfsfqct.centralindia-01.azurewebsites.net/

The model is dynamically loaded in the Streamlit application during inference, ensuring lightweight deployment on Azure.

---

## 🤗 Hugging Face Model Repository

👉 https://huggingface.co/Smriti05wy/fake-review-distilbert/tree/main

The trained DistilBERT model (~257MB) is hosted on Hugging Face instead of GitHub due to repository size limitations.

GitHub is not suitable for storing large ML models, so the model is versioned and deployed using Hugging Face Model Hub, which provides:

- Efficient model storage and retrieval
- Public model access via API
- Better integration with Transformers library
- Scalable deployment for large NLP models

---

## 🚀 Features

- Text Cleaning (lowercasing, punctuation removal, stopword handling)
- TF-IDF Vectorization for feature extraction
- Sentiment-aware preprocessing
- Multiple ML models comparison:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost
- Deep Learning experiment using DistilBERT (Hugging Face Transformers)
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Deployed Streamlit-based web application on Azure Cloud

---

## 🧪 Models & Performance

| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|----------|--------|----------|
| Logistic Regression  | 88.3%    | 88.3%    | 88.3%  | 88.3%    |
| XGBoost              | 83.1%    | 83.1%    | 83.1%  | 83.1%    |
| SVM (Best Model)     | 88.26%   | 88.26%   | 88.26% | 88.26%   |
| DistilBERT (NLP DL)  | 88%      | 89%      | 88%    | 88%      |

👉 Final selected model: Support Vector Machine (SVM)

<img width="940" height="321" alt="image" src="https://github.com/user-attachments/assets/4f118b00-291e-4c19-bf1e-643fb508e561" />


---

## 🧠 Tech Stack

- Python
- Scikit-learn
- Pandas, NumPy
- NLTK / spaCy
- TF-IDF Vectorizer
- XGBoost
- Hugging Face Transformers (DistilBERT)
- Streamlit
- Azure App Service
- GitHub CI/CD

---

## 📊 Project Pipeline

1. Data Collection (Fake Review Dataset)
2. Text Preprocessing
3. Feature Engineering (TF-IDF)
4. Model Training (ML + DL experiments)
5. Model Evaluation
6. Model Selection (SVM)
7. Model Deployment
8. Cloud Deployment with CI/CD

---

## 📁 Sample Dataset

A 100-row sample dataset is included for testing and reproducibility:

- sample_data.csv

---

## 📈 Future Improvements

- Add confidence score visualization for predictions
- Improve model generalization using larger and more diverse datasets
- Add FastAPI backend for scalable inference
- Dockerize the application for cloud portability and reproducibility
- Add logging and monitoring for production-grade deployment
- Explore a hybrid model approach combining classical ML and transformer models
  - SVM (TF-IDF based classical model) for fast predictions
  - DistilBERT for contextual understanding
  - Weighted voting or ensemble strategy for final prediction

---

## 📌 Key Learnings

- End-to-end NLP pipeline development
- Model comparison and evaluation strategy
- Cloud deployment of ML applications
- Transformer-based model integration (DistilBERT)
- CI/CD using GitHub + Azure App Service

---
