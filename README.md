# Smart Support Ticket Intelligence System

AI/ML Internship Assessment Project — TARS Technologies

An NLP + ML system that predicts the **category** and **priority** of customer support tickets from their text.

## Example

**Input:** "My payment was deducted but my order was cancelled."

**Output:**
```

Category : Billing and Payments
Priority : High

```

## Problem Statement

Given ticket text (subject + body), predict:
1. **Category** (support queue)
2. **Priority** (urgency level)

These are two independent classification tasks.

## Dataset

[Tobi-Bueck Customer Support Tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face)

| Field | Description | Usage |
|---|---|---|
| `subject` + `body` | Ticket text | Input |
| `queue` | Category | Target |
| `priority` | Priority | Target |

**Note:** `queue`, `priority`, and `answer` are excluded from model input to prevent data leakage — only `subject + body` is used.

## Project Structure

```

smart-support-ticket-intelligence/
├── data/
│   ├── raw/support_tickets.csv
│   └── processed/customer_support_tickets_clean.csv
├── models/
│   ├── category_logistic_regression.joblib
│   ├── category_linear_svm.joblib
│   ├── priority_logistic_regression.joblib
│   └── priority_linear_svm.joblib
├── notebooks/exploratory_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── visualizations/
├── requirements.txt
├── app.py
└── README.md

```

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · TF-IDF · Logistic Regression · Linear SVM · Matplotlib · Seaborn · Joblib

## Installation

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd smart-support-ticket-intelligence

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Pipeline

```
Ticket Text → Cleaning → TF-IDF → [Category Model | Priority Model] → Prediction
```

**Data Cleaning:** handle missing values, remove duplicates, combine subject+body, drop rows without labels.

**Feature Engineering:** TF-IDF (unigrams + bigrams, English stopwords removed, `min_df=2`).

**Models:** Logistic Regression (baseline) and Linear SVM (comparison) — trained separately for category and priority.

**Split:** 80/20 train/test, stratified, `random_state=42`.

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, Macro F1, Confusion Matrix

| Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD |
| Linear SVM | TBD | TBD | TBD | TBD |

## Usage

```bash
python src/data_preprocessing.py   # 1. Clean data
python src/train.py                # 2. Train models
python src/evaluate.py             # 3. Evaluate models
python src/predict.py              # 4. Predict new ticket
```

**Prediction example:**

```python
def predict_ticket(ticket_text):
    return {
        "ticket": ticket_text,
        "category": category_model.predict([ticket_text])[0],
        "priority": priority_model.predict([ticket_text])[0]
    }
```

## Limitations

* TF-IDF lacks deep semantic understanding
* Rare classes may be underrepresented
* Performance depends on dataset quality

## Future Improvements

* Transformer-based embeddings (BERT, DistilBERT)
* Hyperparameter tuning, cross-validation, class balancing
* FastAPI REST API + web dashboard
* Docker + cloud deployment + monitoring

## Author

**Akshad Aloni** — B.Tech CSE (Data Science)

## License

Developed for educational and internship assessment purposes.
```
