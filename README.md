<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2E3192,100:1BFFFF&height=180&section=header&text=TicketSense&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Smart%20Support%20Ticket%20Intelligence%20System&descAlignY=58&descSize=18" width="100%"/>

<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&pause=1000&color=2E3192&center=true&vCenter=true&width=600&lines=Classifying+support+tickets+with+NLP+%2B+ML;Category+Prediction+%C2%B7+Priority+Prediction;TF-IDF+%C2%B7+Logistic+Regression+%C2%B7+Linear+SVM" alt="Typing SVG" />
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)

![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)

![License](https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge)

</div>

<br/>

## Project Overview

Support teams are flooded with tickets before anyone even reads them. **TicketSense** is an NLP and machine learning system that reads raw ticket text and predicts two things instantly:

1. **Category** — which support queue the ticket belongs to
2. **Priority** — how urgently it needs attention

These are treated as two independent classification tasks, trained on the same cleaned text input.


┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────────┐ ┌────────────┐
│ Raw Tickets│──▶│ Cleaning & │──▶│ TF-IDF │──▶│ Dual Classifier │──▶│ Prediction │
│(subj+body) │ │Deduplication │ │(1-2 grams) │ │ Category | Priority│ │ Output │
└────────────┘ └──────────────┘ └────────────┘ └──────────────────┘ └────────────┘
<br/>

## Dataset & Preprocessing

**Source:** [Tobi-Bueck Customer Support Tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face)

| Field | Description | Role |
|---|---|---|
| `subject` + `body` | Ticket text | Input |
| `queue` | Support category | Target 1 |
| `priority` | Urgency level | Target 2 |

**Preprocessing steps:**
- Merge `subject` and `body` into a single text field
- Drop rows with missing labels
- Remove duplicate tickets
- **Leakage prevention:** `queue`, `priority`, and `answer` are excluded from model input — only ticket text is used

<br/>

## Approach / Model Used

| Stage | Technique |
|---|---|
| Feature Engineering | TF-IDF, unigrams + bigrams, English stopwords removed, `min_df=2` |
| Models | Logistic Regression (baseline) vs. Linear SVM (comparison) — trained separately for category and priority |
| Train/Test Split | 80/20, stratified, `random_state=42` |
| Evaluation | Accuracy, Precision, Recall, F1-score, Macro F1, Confusion Matrix |

<br/>

## Model Results

**Category Classification**

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.76 | 0.74 | 0.71 | 0.52 |
| Linear SVM | 0.79 | 0.77 | 0.75 | 0.58 |

**Priority Classification**

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.68 | 0.66 | 0.64 | 0.61 |
| Linear SVM | 0.71 | 0.69 | 0.67 | 0.64 |

*Linear SVM outperforms Logistic Regression on both tasks. Macro F1 for category classification is lower than accuracy due to a long tail of underrepresented categories in the dataset (see Limitations).*

<br/>

## How to Run the Project

```bash
git clone &lt;YOUR_GITHUB_REPOSITORY_URL&gt;
cd TicketSense

<br>

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash
python src/data_preprocessing.py   # Step 1: clean & merge data
python src/train.py                # Step 2: train both models
python src/evaluate.py             # Step 3: evaluate performance
python src/predict.py              # Step 4: predict a new ticket


One Sample Prediction
Input
"My payment was deducted but my order was cancelled."

Output
Category : Billing and Payments
Priority : High

Code
python
def predict_ticket(ticket_text):
    return {
        "ticket": ticket_text,
        "category": category_model.predict([ticket_text])[0],
        "priority": priority_model.predict([ticket_text])[0]
    }


Limitations & Possible Improvements
Limitations
• TF-IDF lacks deep semantic understanding
• Rare classes (many categories have fewer than 400 samples) drag down macro F1 despite decent accuracy
• Performance depends heavily on dataset quality
Future Improvements
• Swap TF-IDF for transformer embeddings (BERT, DistilBERT)
• Add class_weight='balanced', hyperparameter tuning, and cross-validation
• Build a FastAPI REST API with a web dashboard
• Deploy via Docker with monitoring

Akshad Aloni — B.Tech CSE (Data Science)
Developed for the TARS Technologies AI/ML Internship Assessment
 ``` 
