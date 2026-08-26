<div align="center">

# 🎫 TicketSense
### Smart Support Ticket Intelligence System

*Reading between the lines of every customer complaint.*

`NLP` `TF-IDF` `Logistic Regression` `Linear SVM` `scikit-learn`

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)

![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)

![Status](https://img.shields.io/badge/status-in--progress-yellow)

![License](https://img.shields.io/badge/license-Educational-lightgrey)

</div>

---

## 💡 The Idea

Support teams drown in tickets before they even read them. **TicketSense** looks at raw ticket text and instantly tells you *where it should go* and *how fast it needs attention* — no manual triage required.


┌───────────────────────────────────────────┐
│ "My payment was deducted but my order │
│ was cancelled." │
└──────────────────┬──────────────────────────┘
▼
⚙️ clean → vectorize → classify
▼
┌──────────────────────────────┐
│ 🗂 Category : Billing & Payments │
│ 🔥 Priority : High │
└──────────────────────────────┘

---

## 🧠 System Design


┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────────┐ ┌────────────┐
│ Raw Tickets│──▶│ Cleaning & │──▶│ TF-IDF │──▶│ Dual Classifier │──▶│ Prediction │
│(subj+body) │ │Deduplication │ │(1-2 grams) │ │ Category | Priority│ │ Output │
└────────────┘ └──────────────┘ └────────────┘ └──────────────────┘ └────────────┘
Two independent classification heads share the same TF-IDF-vectorized input — **no leakage**, `queue`, `priority`, and `answer` are stripped before training.

---

## 📂 Project Structure


TicketSense/
├── 🗃️ data/
│ ├── raw/ → untouched source CSV
│ └── processed/ → cleaned, leak-free dataset
├── 📓 notebooks/ → EDA & experiments
├── 🧩 src/
│ ├── data_preprocessing.py → clean + merge subject/body
│ ├── train.py → fit LogReg & SVM models
│ ├── evaluate.py → metrics + confusion matrices
│ └── predict.py → single-ticket inference
├── 🧠 models/ → saved .joblib artifacts
├── 📊 visualizations/ → plots & charts
├── app.py → demo entry point
└── requirements.txt
---

## ⚙️ Quickstart

```bash
git clone &lt;YOUR_GITHUB_REPOSITORY_URL&gt;
cd TicketSense

<br>

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bash
python src/data_preprocessing.py   # 1️⃣ clean
python src/train.py                # 2️⃣ train
python src/evaluate.py             # 3️⃣ evaluate
python src/predict.py              # 4️⃣ predict


───

🔬 Pipeline Details

Stage
Approach

Cleaning
dedupe, drop unlabeled rows, merge subject + body

Features
TF-IDF · unigrams + bigrams · stopwords removed · min_df=2

Models
Logistic Regression (baseline) vs. Linear SVM (comparison)

Split
80/20 stratified · random_state=42


📈 Results

Model
Accuracy
Precision
Recall
Macro F1

Logistic Regression
TBD
TBD
TBD
TBD

Linear SVM
TBD
TBD
TBD
TBD



───

🚀 Roadmap
[  ]  Swap TF-IDF for BERT/DistilBERT embeddings
[  ]  Hyperparameter tuning + class balancing
[  ]  FastAPI service + live dashboard
[  ]  Docker image + cloud deployment

───

⚠️ Known Limitations
TF-IDF has no deep semantic understanding · rare classes may be underrepresented · quality is bounded by dataset quality.

───

Built by Akshad Aloni · B.Tech CSE (Data Science)
Developed for the TARS Technologies AI/ML Internship Assessment
 ``` 
