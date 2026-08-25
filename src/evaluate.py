from pathlib import Path
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_support_tickets_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"

VISUALIZATION_DIR = BASE_DIR / "visualizations"

REPORT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "evaluation_reports"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# CREATE DIRECTORIES
# ============================================================

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMART SUPPORT TICKET INTELLIGENCE SYSTEM")
print("MODEL EVALUATION PIPELINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/7] Loading processed dataset...")
print("-" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Columns : {df.columns.tolist()}")


# ============================================================
# 2. PREPARE DATA
# ============================================================

print("\n[2/7] Preparing evaluation data...")
print("-" * 70)

required_columns = [
    "ticket_text",
    "category",
    "priority"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


df["ticket_text"] = (
    df["ticket_text"]
    .fillna("")
    .astype(str)
)

df["category"] = (
    df["category"]
    .fillna("")
    .astype(str)
)

df["priority"] = (
    df["priority"]
    .fillna("")
    .astype(str)
)


# ============================================================
# 3. REPRODUCE SAME TRAIN / TEST SPLIT
# ============================================================

print("\n[3/7] Recreating test dataset...")
print("-" * 70)

from sklearn.model_selection import train_test_split


X = df["ticket_text"]

y_category = df["category"]

y_priority = df["priority"]


X_train, X_test, y_category_train, y_category_test = (
    train_test_split(
        X,
        y_category,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_category
    )
)


X_train_p, X_test_p, y_priority_train, y_priority_test = (
    train_test_split(
        X,
        y_priority,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_priority
    )
)


print(f"Training samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")


# ============================================================
# 4. LOAD MODELS
# ============================================================

print("\n[4/7] Loading trained models...")
print("-" * 70)

vectorizer_path = (
    MODEL_DIR
    / "tfidf_vectorizer.joblib"
)

category_logistic_path = (
    MODEL_DIR
    / "category_logistic_regression.joblib"
)

category_svm_path = (
    MODEL_DIR
    / "category_linear_svm.joblib"
)

priority_logistic_path = (
    MODEL_DIR
    / "priority_logistic_regression.joblib"
)

priority_svm_path = (
    MODEL_DIR
    / "priority_linear_svm.joblib"
)


required_models = [
    vectorizer_path,
    category_logistic_path,
    category_svm_path,
    priority_logistic_path,
    priority_svm_path
]


for model_path in required_models:

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )


vectorizer = joblib.load(
    vectorizer_path
)

category_logistic = joblib.load(
    category_logistic_path
)

category_svm = joblib.load(
    category_svm_path
)

priority_logistic = joblib.load(
    priority_logistic_path
)

priority_svm = joblib.load(
    priority_svm_path
)


print("All models loaded successfully")


# ============================================================
# 5. CREATE TF-IDF TEST FEATURES
# ============================================================

print("\n[5/7] Transforming test data using TF-IDF...")
print("-" * 70)

start_time = time.time()


X_test_tfidf = vectorizer.transform(
    X_test
)


X_test_p_tfidf = vectorizer.transform(
    X_test_p
)


elapsed = time.time() - start_time


print(
    f"Category test feature shape : "
    f"{X_test_tfidf.shape}"
)

print(
    f"Priority test feature shape : "
    f"{X_test_p_tfidf.shape}"
)

print(
    f"Transformation time          : "
    f"{elapsed:.2f} seconds"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_test_features,
    y_test,
    task_name,
    model_name
):

    print("\n" + "=" * 70)

    print(
        f"{task_name.upper()} - "
        f"{model_name.upper()}"
    )

    print("=" * 70)


    predictions = model.predict(
        X_test_features
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    print(
        f"\nAccuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Macro Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{f1_macro:.4f}"
    )

    print(
        f"Weighted F1    : "
        f"{f1_weighted:.4f}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report")
    print("-" * 70)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)


    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / f"{task_name.lower()}_"
          f"{model_name.lower().replace(' ', '_')}_"
          f"classification_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{task_name} - {model_name}\n"
        )

        file.write("=" * 70)

        file.write("\n\n")

        file.write(
            f"Accuracy       : {accuracy:.4f}\n"
        )

        file.write(
            f"Macro Precision: {precision:.4f}\n"
        )

        file.write(
            f"Macro Recall   : {recall:.4f}\n"
        )

        file.write(
            f"Macro F1       : {f1_macro:.4f}\n"
        )

        file.write(
            f"Weighted F1    : {f1_weighted:.4f}\n"
        )

        file.write("\n")

        file.write(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )


    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )


    # --------------------------------------------------------
    # SAVE RAW CONFUSION MATRIX
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )


    cm_csv_path = (
        REPORT_DIR
        / f"{task_name.lower()}_"
          f"{model_name.lower().replace(' ', '_')}_"
          f"confusion_matrix.csv"
    )


    cm_df.to_csv(
        cm_csv_path
    )


    # --------------------------------------------------------
    # VISUALIZE CONFUSION MATRIX
    # --------------------------------------------------------

    figure_size = (
        (18, 15)
        if task_name == "Category"
        else (9, 8)
    )


    fig, ax = plt.subplots(
        figsize=figure_size
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )


    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format="d",
        colorbar=True
    )


    ax.set_title(
        f"{task_name} Confusion Matrix - {model_name}",
        fontsize=16
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "True Label"
    )


    plt.tight_layout()


    image_path = (
        VISUALIZATION_DIR
        / f"{task_name.lower()}_"
          f"{model_name.lower().replace(' ', '_')}_"
          f"confusion_matrix.png"
    )


    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"\nConfusion matrix saved:"
    )

    print(
        image_path
    )


    return {
        "task": task_name,
        "model": model_name,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


# ============================================================
# 6. EVALUATE ALL MODELS
# ============================================================

print("\n[6/7] Evaluating all trained models...")
print("-" * 70)


results = []


# ------------------------------------------------------------
# CATEGORY - LOGISTIC REGRESSION
# ------------------------------------------------------------

results.append(
    evaluate_model(
        category_logistic,
        X_test_tfidf,
        y_category_test,
        "Category",
        "Logistic Regression"
    )
)


# ------------------------------------------------------------
# CATEGORY - LINEAR SVM
# ------------------------------------------------------------

results.append(
    evaluate_model(
        category_svm,
        X_test_tfidf,
        y_category_test,
        "Category",
        "Linear SVM"
    )
)


# ------------------------------------------------------------
# PRIORITY - LOGISTIC REGRESSION
# ------------------------------------------------------------

results.append(
    evaluate_model(
        priority_logistic,
        X_test_p_tfidf,
        y_priority_test,
        "Priority",
        "Logistic Regression"
    )
)


# ------------------------------------------------------------
# PRIORITY - LINEAR SVM
# ------------------------------------------------------------

results.append(
    evaluate_model(
        priority_svm,
        X_test_p_tfidf,
        y_priority_test,
        "Priority",
        "Linear SVM"
    )
)


# ============================================================
# 7. SAVE FINAL EVALUATION RESULTS
# ============================================================

print("\n[7/7] Saving evaluation summary...")
print("-" * 70)


results_df = pd.DataFrame(
    results
)


results_path = (
    REPORT_DIR
    / "evaluation_summary.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    f"\nEvaluation summary saved to:"
)

print(
    results_path
)


# ============================================================
# BEST MODELS
# ============================================================

category_results = results_df[
    results_df["task"] == "Category"
]


priority_results = results_df[
    results_df["task"] == "Priority"
]


best_category = category_results.loc[
    category_results["f1_macro"].idxmax()
]


best_priority = priority_results.loc[
    priority_results["f1_macro"].idxmax()
]


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL EVALUATION RESULTS")
print("=" * 70)


print("\nCATEGORY")

print(
    f"Best Model     : "
    f"{best_category['model']}"
)

print(
    f"Accuracy       : "
    f"{best_category['accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_category['precision_macro']:.4f}"
)

print(
    f"Macro Recall   : "
    f"{best_category['recall_macro']:.4f}"
)

print(
    f"Macro F1       : "
    f"{best_category['f1_macro']:.4f}"
)

print(
    f"Weighted F1    : "
    f"{best_category['f1_weighted']:.4f}"
)


print("\nPRIORITY")

print(
    f"Best Model     : "
    f"{best_priority['model']}"
)

print(
    f"Accuracy       : "
    f"{best_priority['accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_priority['precision_macro']:.4f}"
)

print(
    f"Macro Recall   : "
    f"{best_priority['recall_macro']:.4f}"
)

print(
    f"Macro F1       : "
    f"{best_priority['f1_macro']:.4f}"
)

print(
    f"Weighted F1    : "
    f"{best_priority['f1_weighted']:.4f}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n")
print("=" * 70)
print("GENERATED FILES")
print("=" * 70)


print("\nEvaluation reports:")

for file in sorted(
    REPORT_DIR.iterdir()
):

    if file.is_file():

        print(
            f"✓ {file}"
        )


print("\nVisualizations:")

for file in sorted(
    VISUALIZATION_DIR.iterdir()
):

    if file.is_file():

        print(
            f"✓ {file}"
        )


print("\n")
print("=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNext step:")

print(
    "Run: python src\\predict.py"
)

print(
    "This will test the trained models on new support tickets."
)