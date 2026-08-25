<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Smart Support Ticket Intelligence System</title>
  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.7;
      max-width: 1000px;
      margin: 0 auto;
      padding: 40px;
      color: #24292f;
      background: #ffffff;
    }

    h1 {
      font-size: 34px;
      border-bottom: 2px solid #24292f;
      padding-bottom: 12px;
    }

    h2 {
      margin-top: 38px;
      padding-bottom: 8px;
      border-bottom: 1px solid #d0d7de;
    }

    h3 {
      margin-top: 28px;
    }

    p {
      margin: 12px 0;
    }

    code {
      background: #f6f8fa;
      padding: 3px 6px;
      border-radius: 4px;
      font-family: Consolas, monospace;
    }

    pre {
      background: #f6f8fa;
      padding: 18px;
      border-radius: 8px;
      overflow-x: auto;
      border: 1px solid #d8dee4;
    }

    pre code {
      background: transparent;
      padding: 0;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
    }

    th,
    td {
      border: 1px solid #d0d7de;
      padding: 10px;
      text-align: left;
    }

    th {
      background: #f6f8fa;
    }

    blockquote {
      border-left: 4px solid #57606a;
      padding-left: 16px;
      color: #57606a;
      margin: 20px 0;
    }

    .architecture {
      background: #f6f8fa;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 20px;
      font-family: Consolas, monospace;
      white-space: pre;
      overflow-x: auto;
    }

    .note {
      background: #fff8c5;
      border: 1px solid #d4a72c;
      padding: 14px 18px;
      border-radius: 8px;
      margin: 20px 0;
    }

    .success {
      background: #dafbe1;
      border: 1px solid #1a7f37;
      padding: 14px 18px;
      border-radius: 8px;
      margin: 20px 0;
    }

    .muted {
      color: #57606a;
    }

    ul,
    ol {
      padding-left: 25px;
    }

    .badge {
      display: inline-block;
      background: #f6f8fa;
      border: 1px solid #d0d7de;
      padding: 4px 9px;
      border-radius: 20px;
      margin: 3px;
      font-size: 14px;
    }
  </style>
</head>

<body>

  <h1>Smart Support Ticket Intelligence System</h1>

  <p>
    An NLP-based machine learning system that analyzes customer support tickets
    and predicts the ticket category and priority.
  </p>

  <p>
    <span class="badge">Python</span>
    <span class="badge">Pandas</span>
    <span class="badge">Scikit-learn</span>
    <span class="badge">TF-IDF</span>
    <span class="badge">Logistic Regression</span>
    <span class="badge">Linear SVM</span>
    <span class="badge">Matplotlib</span>
    <span class="badge">Seaborn</span>
  </p>

  <hr>

  <h2>1. Project Overview</h2>

  <p>
    Customer support teams receive a large number of tickets every day.
    Manually identifying the category and urgency of every ticket can be
    time-consuming and inconsistent.
  </p>

  <p>
    This project builds a machine learning system that automatically analyzes
    customer support ticket text and predicts:
  </p>

  <ul>
    <li>Ticket Category</li>
    <li>Ticket Priority</li>
  </ul>

  <p>
    The system uses Natural Language Processing to convert ticket text into
    numerical features and classification models to generate predictions.
  </p>

  <h2>2. Problem Statement</h2>

  <p>
    Given a customer support ticket, the system should automatically determine
    the most appropriate category and priority.
  </p>

  <pre><code>Customer Support Ticket
          ↓
     Text Processing
          ↓
      TF-IDF Features
          ↓
   Machine Learning Models
       ↓          ↓
   Category     Priority
   Prediction   Prediction</code></pre>

  <p>
    Example:
  </p>

  <pre><code>Input:
"Payment was deducted but my order was cancelled."

Output:
Category: Billing and Payments
Priority: High</code></pre>

  <h2>3. Project Objectives</h2>

  <ul>
    <li>Prepare and clean a real-world customer support ticket dataset</li>
    <li>Understand the distribution of ticket categories and priorities</li>
    <li>Perform exploratory data analysis</li>
    <li>Convert text into numerical features using TF-IDF</li>
    <li>Train classification models</li>
    <li>Compare multiple machine learning approaches</li>
    <li>Evaluate models using standard classification metrics</li>
    <li>Build a simple interface for predicting new tickets</li>
    <li>Create reproducible and easy-to-understand ML code</li>
  </ul>

  <h2>4. Dataset</h2>

  <p>
    This project uses the publicly available
    <strong>Tobi-Bueck Customer Support Tickets</strong> dataset hosted on
    Hugging Face.
  </p>

  <p>
    Dataset:
    <a href="https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets">
      https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
    </a>
  </p>

  <p>
    The dataset contains customer support ticket information including ticket
    subject, ticket body, type, queue/category, priority, language and tags.
  </p>

  <h3>Important Dataset Fields</h3>

  <table>
    <thead>
      <tr>
        <th>Original Field</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>subject</td>
        <td>Customer ticket subject</td>
      </tr>
      <tr>
        <td>body</td>
        <td>Main customer support message</td>
      </tr>
      <tr>
        <td>queue</td>
        <td>Ticket category / target label</td>
      </tr>
      <tr>
        <td>priority</td>
        <td>Ticket priority / target label</td>
      </tr>
      <tr>
        <td>type</td>
        <td>Ticket type used for analysis</td>
      </tr>
      <tr>
        <td>language</td>
        <td>Ticket language</td>
      </tr>
      <tr>
        <td>tag_1 ... tag_8</td>
        <td>Additional ticket metadata</td>
      </tr>
    </tbody>
  </table>

  <h2>5. Feature and Target Definition</h2>

  <p>
    The customer-facing ticket text is used as the main input to avoid target
    leakage.
  </p>

  <pre><code>Input Features
--------------
subject + body

Target 1
--------
queue → Category

Target 2
--------
priority → Priority</code></pre>

  <p>
    The subject and body are combined into a single text feature called
    <code>ticket_text</code>.
  </p>

  <h2>6. Machine Learning Lifecycle</h2>

  <div class="architecture">1. Problem Definition
        ↓
2. Data Collection
        ↓
3. Data Understanding
        ↓
4. Data Cleaning
        ↓
5. Exploratory Data Analysis
        ↓
6. Text Preprocessing
        ↓
7. Feature Engineering
        ↓
8. Train/Test Split
        ↓
9. Model Training
        ↓
10. Model Evaluation
        ↓
11. Model Comparison
        ↓
12. Prediction System
        ↓
13. Documentation
        ↓
14. Optional API / Deployment</div>

  <h2>7. Data Collection</h2>

  <p>
    The dataset is loaded using the Hugging Face
    <code>datasets</code> library.
  </p>

  <pre><code>from datasets import load_dataset

dataset = load_dataset(
    "Tobi-Bueck/customer-support-tickets"
)

print(dataset)</code></pre>

  <p>
    The available dataset split is converted into a Pandas DataFrame for
    preprocessing and analysis.
  </p>

  <h2>8. Data Preparation</h2>

  <h3>8.1 Rename Important Columns</h3>

  <pre><code>df = df.rename(columns={
    "subject": "ticket_subject",
    "body": "ticket_body",
    "queue": "category",
    "priority": "priority"
})</code></pre>

  <h3>8.2 Handle Missing Values</h3>

  <pre><code>df["ticket_subject"] = df["ticket_subject"].fillna("")
df["ticket_body"] = df["ticket_body"].fillna("")</code></pre>

  <h3>8.3 Combine Ticket Text</h3>

  <pre><code>df["ticket_text"] = (
    df["ticket_subject"].astype(str)
    + " "
    + df["ticket_body"].astype(str)
).str.strip()</code></pre>

  <h3>8.4 Remove Duplicates</h3>

  <pre><code>df = df.drop_duplicates(
    subset=["ticket_text"]
)</code></pre>

  <h3>8.5 Remove Invalid Records</h3>

  <pre><code>df = df.dropna(
    subset=["category", "priority"]
)

df = df[
    df["ticket_text"].str.len() > 10
]</code></pre>

  <h2>9. Data Leakage Prevention</h2>

  <p>
    Data leakage occurs when information that would not be available at
    prediction time is accidentally provided to the model.
  </p>

  <p>
    The following fields are therefore not used as text input:
  </p>

  <ul>
    <li>category / queue</li>
    <li>priority</li>
    <li>answer</li>
    <li>Other target-derived information</li>
  </ul>

  <p>
    Only the customer's subject and message body are used to generate the
    prediction.
  </p>

  <h2>10. Exploratory Data Analysis</h2>

  <p>
    Before training the models, the dataset is analyzed to understand its
    structure and identify possible data quality issues.
  </p>

  <h3>Analysis Performed</h3>

  <ul>
    <li>Dataset dimensions</li>
    <li>Column names and data types</li>
    <li>Missing value analysis</li>
    <li>Duplicate analysis</li>
    <li>Category distribution</li>
    <li>Priority distribution</li>
    <li>Ticket text length</li>
  </ul>

  <h3>Visualizations</h3>

  <ol>
    <li>Ticket category distribution</li>
    <li>Ticket priority distribution</li>
    <li>Model confusion matrix</li>
  </ol>

  <p>
    The analysis helps identify class imbalance, rare categories and other
    potential issues before model training.
  </p>

  <h2>11. Text Preprocessing</h2>

  <p>
    Text must be converted into numerical features before it can be used by
    traditional machine learning algorithms.
  </p>

  <p>The main processing pipeline is:</p>

  <pre><code>Raw Ticket
    ↓
Lowercase
    ↓
Tokenization / Text Processing
    ↓
TF-IDF Vectorization
    ↓
Numerical Feature Matrix</code></pre>

  <h2>12. Feature Engineering — TF-IDF</h2>

  <p>
    Term Frequency-Inverse Document Frequency (TF-IDF) represents words based
    on their importance within the ticket and across the complete dataset.
  </p>

  <p>
    The project uses both unigram and bigram features so that individual words
    and short phrases can contribute to classification.
  </p>

  <pre><code>TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)</code></pre>

  <h2>13. Train/Test Split</h2>

  <p>
    The dataset is divided into training and testing subsets. The training
    data is used to learn patterns, while the testing data is kept separate
    for evaluating generalization.
  </p>

  <pre><code>Training Data → 80%
Testing Data  → 20%

random_state = 42</code></pre>

  <p>
    Stratified splitting is used so that the distribution of target classes is
    maintained as much as possible between training and testing datasets.
  </p>

  <h2>14. Model Selection</h2>

  <p>
    Two classical machine learning models are evaluated.
  </p>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Logistic Regression</td>
        <td>Strong and interpretable baseline for text classification</td>
      </tr>
      <tr>
        <td>Linear SVM</td>
        <td>Comparison model suited for high-dimensional sparse text features</td>
      </tr>
    </tbody>
  </table>

  <p>
    Both models use the same TF-IDF representation so that the comparison
    focuses on the classification algorithm.
  </p>

  <h2>15. Category Prediction</h2>

  <p>
    The category model predicts the appropriate support queue/category from
    the ticket text.
  </p>

  <pre><code>Ticket Text
    ↓
TF-IDF
    ↓
Classification Model
    ↓
Predicted Category</code></pre>

  <h2>16. Priority Prediction</h2>

  <p>
    A separate classification model predicts the priority of the support
    ticket.
  </p>

  <pre><code>Ticket Text
    ↓
TF-IDF
    ↓
Classification Model
    ↓
Predicted Priority</code></pre>

  <p>
    Separate models are used because category and priority represent two
    different prediction targets.
  </p>

  <h2>17. Evaluation</h2>

  <p>
    The models are evaluated using the metrics required by the technical
    assessment.
  </p>

  <h3>Accuracy</h3>

  <p>
    Measures the proportion of predictions that are correct.
  </p>

  <h3>Precision</h3>

  <p>
    Measures how many predicted instances of a class were actually correct.
  </p>

  <h3>Recall</h3>

  <p>
    Measures how many actual instances of a class were successfully identified.
  </p>

  <h3>F1-Score</h3>

  <p>
    Combines precision and recall into a single metric.
  </p>

  <h3>Confusion Matrix</h3>

  <p>
    Shows how predictions are distributed across the actual and predicted
    classes and helps identify which classes are being confused by the model.
  </p>

  <h2>18. Model Results</h2>

  <p class="note">
    The numerical results below should be filled using the actual output from
    the final training run. Results must not be manually invented.
  </p>

  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Accuracy</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Logistic Regression</td>
        <td>To be measured</td>
        <td>To be measured</td>
        <td>To be measured</td>
        <td>To be measured</td>
      </tr>
      <tr>
        <td>Linear SVM</td>
        <td>To be measured</td>
        <td>To be measured</td>
        <td>To be measured</td>
        <td>To be measured</td>
      </tr>
    </tbody>
  </table>

  <h2>19. Model Comparison</h2>

  <p>
    The final model is selected based on its evaluation performance and
    practical suitability for the support-ticket classification problem.
  </p>

  <p>
    The comparison will consider:
  </p>

  <ul>
    <li>Overall accuracy</li>
    <li>Macro precision</li>
    <li>Macro recall</li>
    <li>Macro F1-score</li>
    <li>Performance across individual classes</li>
    <li>Confusion matrix behaviour</li>
    <li>Training complexity</li>
  </ul>

  <p>
    The better-performing model will be selected for the final prediction
    system based on the actual evaluation results.
  </p>

  <h2>20. Prediction System</h2>

  <p>
    The project provides a simple way to enter a new support ticket and obtain
    predictions.
  </p>

  <pre><code>Enter support ticket:
Payment was deducted but my order was cancelled.

Predicted Category:
Billing and Payments

Predicted Priority:
High</code></pre>

  <p>
    The prediction system uses the trained category and priority models to
    independently generate both predictions.
  </p>

  <h2>21. System Design</h2>

  <div class="architecture">                    USER
                     │
                     ▼
             Support Ticket Text
                     │
                     ▼
             Input Validation
                     │
                     ▼
             Text Preprocessing
                     │
                     ▼
              TF-IDF Vectorizer
                     │
             ┌───────┴────────┐
             ▼                ▼
      Category Model      Priority Model
             │                │
             ▼                ▼
        Category           Priority
             │                │
             └───────┬────────┘
                     ▼
              Final Prediction</div>

  <h2>22. Detailed System Flow</h2>

  <ol>
    <li>User enters a customer support ticket</li>
    <li>The ticket subject and body are combined into text</li>
    <li>The text is cleaned and transformed</li>
    <li>The trained TF-IDF vectorizer converts text into numerical features</li>
    <li>The category model predicts the ticket category</li>
    <li>The priority model predicts the ticket priority</li>
    <li>The predictions are displayed to the user</li>
  </ol>

  <h2>23. Project Structure</h2>

  <pre><code>smart-support-ticket-intelligence/
│
├── data/
│   ├── raw/
│   │   └── support_tickets.csv
│   └── processed/
│       └── customer_support_tickets_clean.csv
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── category_vectorizer.pkl
│   └── priority_vectorizer.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_category.py
│   ├── train_priority.py
│   ├── evaluate.py
│   └── predict.py
│
├── visualizations/
│   ├── category_distribution.png
│   ├── priority_distribution.png
│   └── confusion_matrix.png
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE</code></pre>

  <h2>24. Technology Stack</h2>

  <table>
    <thead>
      <tr>
        <th>Technology</th>
        <th>Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Python</td>
        <td>Core programming language</td>
      </tr>
      <tr>
        <td>Pandas</td>
        <td>Data manipulation and analysis</td>
      </tr>
      <tr>
        <td>NumPy</td>
        <td>Numerical operations</td>
      </tr>
      <tr>
        <td>Scikit-learn</td>
        <td>Machine learning and evaluation</td>
      </tr>
      <tr>
        <td>TF-IDF</td>
        <td>Text feature extraction</td>
      </tr>
      <tr>
        <td>Matplotlib</td>
        <td>Visualization</td>
      </tr>
      <tr>
        <td>Seaborn</td>
        <td>Statistical visualization</td>
      </tr>
      <tr>
        <td>Joblib</td>
        <td>Model serialization</td>
      </tr>
      <tr>
        <td>Hugging Face Datasets</td>
        <td>Dataset loading</td>
      </tr>
    </tbody>
  </table>

  <h2>25. Installation</h2>

  <p>Clone the repository:</p>

  <pre><code>git clone &lt;YOUR_GITHUB_REPOSITORY_URL&gt;

cd smart-support-ticket-intelligence</code></pre>

  <p>Install dependencies:</p>

  <pre><code>pip install -r requirements.txt</code></pre>

  <h2>26. Requirements</h2>

  <pre><code>pandas
numpy
scikit-learn
matplotlib
seaborn
datasets
joblib</code></pre>

  <h2>27. Running the Project</h2>

  <h3>Step 1 — Prepare the Dataset</h3>

  <pre><code>python src/data_preprocessing.py</code></pre>

  <h3>Step 2 — Train the Models</h3>

  <pre><code>python src/train_category.py
python src/train_priority.py</code></pre>

  <h3>Step 3 — Evaluate the Models</h3>

  <pre><code>python src/evaluate.py</code></pre>

  <h3>Step 4 — Make a Prediction</h3>

  <pre><code>python src/predict.py</code></pre>

  <h2>28. Sample Prediction</h2>

  <pre><code>Input:
"My card was charged twice for the same transaction."

Prediction:
Category: Billing and Payments
Priority: High</code></pre>

  <p>
    The actual prediction displayed by the application should always be
    generated by the trained model.
  </p>

  <h2>29. Data Analysis Observations</h2>

  <p>
    The final observations should be written after performing the actual
    exploratory analysis.
  </p>

  <ul>
    <li>Identify the most common ticket categories.</li>
    <li>Identify the distribution of ticket priorities.</li>
    <li>Identify whether the dataset contains class imbalance.</li>
    <li>Identify any major missing-value or duplicate issues.</li>
  </ul>

  <p>
    Observations should be based on the actual dataset rather than assumptions.
  </p>

  <h2>30. Code Quality</h2>

  <ul>
    <li>Use clear variable and function names</li>
    <li>Keep data processing separate from model training</li>
    <li>Keep model evaluation separate from prediction</li>
    <li>Avoid unnecessary code duplication</li>
    <li>Use reusable functions where appropriate</li>
    <li>Keep configuration values easy to modify</li>
    <li>Document important decisions</li>
    <li>Use a requirements file for reproducibility</li>
  </ul>

  <h2>31. Limitations</h2>

  <ul>
    <li>Traditional TF-IDF models do not fully understand semantic meaning.</li>
    <li>Performance may decrease on tickets that are very different from the training data.</li>
    <li>Rare categories may have fewer training examples.</li>
    <li>The quality of predictions depends on the quality and distribution of the dataset.</li>
    <li>The system is intended as an ML assessment project rather than a production support platform.</li>
  </ul>

  <h2>32. Future Improvements</h2>

  <ul>
    <li>Use transformer-based embeddings for improved semantic understanding.</li>
    <li>Experiment with pretrained language models.</li>
    <li>Add confidence scores to predictions.</li>
    <li>Add explainable prediction reasons.</li>
    <li>Expose the model through a REST API.</li>
    <li>Create a lightweight web interface.</li>
    <li>Deploy the application.</li>
    <li>Monitor model performance after deployment.</li>
    <li>Retrain the model using newly labeled support tickets.</li>
  </ul>

  <h2>33. Optional REST API Architecture</h2>

  <p>
    A REST API can be added as an optional extension after the core ML system
    is complete.
  </p>

  <div class="architecture">Client
   │
   ▼
FastAPI REST API
   │
   ▼
Input Validation
   │
   ▼
ML Prediction Pipeline
   │
   ├── Category Model
   │
   └── Priority Model
   │
   ▼
JSON Response</div>

  <h2>34. Optional Web Interface</h2>

  <p>
    A simple web interface can allow users to enter a ticket and view the
    predicted category and priority.
  </p>

  <pre><code>┌───────────────────────────────────────┐
│     Smart Support Ticket System      │
├───────────────────────────────────────┤
│                                       │
│ Enter your support ticket             │
│                                       │
│ [ Payment was deducted but...       ] │
│                                       │
│           [ Analyze Ticket ]          │
│                                       │
│ Category: Billing and Payments        │
│ Priority: High                        │
│                                       │
└───────────────────────────────────────┘</code></pre>

  <h2>35. Why This Approach?</h2>

  <p>
    TF-IDF with classical linear classification models provides a strong,
    lightweight and explainable solution for this assessment.
  </p>

  <p>
    Deep learning is not necessary for the problem. A traditional NLP pipeline
    also makes the complete workflow easier to understand, evaluate and
    explain.
  </p>

  <p>
    The approach prioritizes clean preprocessing, appropriate feature
    engineering, model comparison and reliable evaluation rather than
    unnecessary system complexity.
  </p>

  <h2>36. Assessment Requirement Coverage</h2>

  <table>
    <thead>
      <tr>
        <th>Requirement</th>
        <th>Implementation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Public support-ticket dataset</td>
        <td>Tobi-Bueck Customer Support Tickets</td>
      </tr>
      <tr>
        <td>Dataset source and features</td>
        <td>Documented</td>
      </tr>
      <tr>
        <td>Missing values</td>
        <td>Checked and handled</td>
      </tr>
      <tr>
        <td>Duplicates</td>
        <td>Checked and removed where appropriate</td>
      </tr>
      <tr>
        <td>Category/priority distribution</td>
        <td>EDA</td>
      </tr>
      <tr>
        <td>2–3 visualizations</td>
        <td>Distribution charts and confusion matrices</td>
      </tr>
      <tr>
        <td>Classification model</td>
        <td>Logistic Regression</td>
      </tr>
      <tr>
        <td>Additional model</td>
        <td>Linear SVM</td>
      </tr>
      <tr>
        <td>Text feature extraction</td>
        <td>TF-IDF</td>
      </tr>
      <tr>
        <td>Accuracy</td>
        <td>Implemented</td>
      </tr>
      <tr>
        <td>Precision</td>
        <td>Implemented</td>
      </tr>
      <tr>
        <td>Recall</td>
        <td>Implemented</td>
      </tr>
      <tr>
        <td>F1-score</td>
        <td>Implemented</td>
      </tr>
      <tr>
        <td>Confusion matrix</td>
        <td>Implemented</td>
      </tr>
      <tr>
        <td>Simple prediction</td>
        <td>CLI prediction system</td>
      </tr>
      <tr>
        <td>README</td>
        <td>This document</td>
      </tr>
      <tr>
        <td>Requirements file</td>
        <td>Included</td>
      </tr>
      <tr>
        <td>Reproducible training</td>
        <td>Training scripts included</td>
      </tr>
    </tbody>
  </table>

  <h2>37. Security and Reliability Considerations</h2>

  <ul>
    <li>Do not expose model files or internal paths through user input.</li>
    <li>Validate ticket text before prediction.</li>
    <li>Keep training and test data separated.</li>
    <li>Avoid target leakage during feature engineering.</li>
    <li>Store trained models separately from raw datasets.</li>
  </ul>

  <h2>38. Final Outcome</h2>

  <div class="success">
    The completed system provides an end-to-end machine learning workflow for
    customer support ticket classification, starting from raw ticket data and
    ending with category and priority predictions.
  </div>

  <p>
    The project demonstrates data preparation, exploratory analysis, NLP
    feature engineering, supervised machine learning, model comparison,
    evaluation and inference in a single reproducible workflow.
  </p>

  <h2>39. Conclusion</h2>

  <p>
    The Smart Support Ticket Intelligence System demonstrates how traditional
    Natural Language Processing and machine learning can be used to automate
    the initial classification and prioritization of customer support tickets.
  </p>

  <p>
    The solution focuses on a simple, explainable and reproducible approach
    using TF-IDF and classification models while leaving room for future
    improvements such as transformer models, explainability, APIs and
    deployment.
  </p>

  <h2>40. Author</h2>

  <p>
    <strong>Akshad Aloni</strong><br>
    B.Tech — Computer Science and Engineering (Data Science)
  </p>

  <p class="muted">
    AI/ML Intern Technical Assessment — Smart Support Ticket Intelligence System
  </p>

</body>
</html>
