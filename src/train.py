from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "customer_support_tickets_clean.csv"
)

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
TEST_SIZE = 0.20


# Word-level TF-IDF
WORD_MAX_FEATURES = 150_000

# Character-level TF-IDF
CHAR_MAX_FEATURES = 100_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(series):
    """
    Basic text normalization.

    We intentionally do not remove stopwords because words such
    as 'not', 'cannot', 'unable', 'never', etc. can be extremely
    important for support-ticket classification.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1
    }


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    pipeline_start = time.time()

    print("=" * 70)
    print("SMART SUPPORT TICKET INTELLIGENCE SYSTEM")
    print("IMPROVED ML TRAINING PIPELINE")
    print("=" * 70)


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1/9] Loading processed dataset...")
    print("-" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")
    print(f"Columns : {df.columns.tolist()}")


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
            f"Missing required columns: {missing_columns}"
        )


    # ========================================================
    # 2. DATA CLEANING
    # ========================================================

    print("\n[2/9] Preparing training data...")
    print("-" * 70)

    df = df.copy()

    df["ticket_text"] = normalize_text(
        df["ticket_text"]
    )

    df["category"] = (
        df["category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["priority"] = (
        df["priority"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["ticket_text"].str.len() > 5) &
        (df["category"] != "") &
        (df["priority"] != "")
    ]

    df = df.drop_duplicates(
        subset=["ticket_text"]
    )

    df = df.reset_index(drop=True)


    print(
        f"Usable records : {len(df):,}"
    )

    print(
        f"Category classes: "
        f"{df['category'].nunique()}"
    )

    print(
        f"Priority classes: "
        f"{df['priority'].nunique()}"
    )


    # ========================================================
    # 3. TRAIN / TEST SPLIT
    # ========================================================

    print("\n[3/9] Creating stratified train/test split...")
    print("-" * 70)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"]
    )


    X_train = train_df["ticket_text"]
    X_test = test_df["ticket_text"]

    y_category_train = train_df["category"]
    y_category_test = test_df["category"]

    y_priority_train = train_df["priority"]
    y_priority_test = test_df["priority"]


    print(
        f"Training samples : {len(train_df):,}"
    )

    print(
        f"Testing samples  : {len(test_df):,}"
    )


    # ========================================================
    # 4. CREATE IMPROVED TF-IDF
    # ========================================================

    print("\n[4/9] Creating combined Word + Character TF-IDF...")
    print("-" * 70)

    vectorizer_start = time.time()


    # --------------------------------------------------------
    # WORD TF-IDF
    # --------------------------------------------------------

    word_vectorizer = TfidfVectorizer(

        analyzer="word",

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.98,

        sublinear_tf=True,

        max_features=WORD_MAX_FEATURES,

        strip_accents="unicode",

        lowercase=True
    )


    # --------------------------------------------------------
    # CHARACTER TF-IDF
    # --------------------------------------------------------

    char_vectorizer = TfidfVectorizer(

        analyzer="char",

        ngram_range=(3, 5),

        min_df=2,

        max_df=0.98,

        sublinear_tf=True,

        max_features=CHAR_MAX_FEATURES,

        lowercase=True
    )


    # --------------------------------------------------------
    # COMBINE BOTH
    # --------------------------------------------------------

    vectorizer = FeatureUnion([
        (
            "word",
            word_vectorizer
        ),
        (
            "char",
            char_vectorizer
        )
    ])


    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    print(
        f"Training feature shape : "
        f"{X_train_tfidf.shape}"
    )

    print(
        f"Testing feature shape  : "
        f"{X_test_tfidf.shape}"
    )

    print(
        f"TF-IDF time             : "
        f"{time.time() - vectorizer_start:.2f}s"
    )


    vectorizer_path = (
        MODEL_DIR
        / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
        compress=3
    )

    print(
        f"Vectorizer saved: {vectorizer_path}"
    )


    # ========================================================
    # 5. CATEGORY MODELS
    # ========================================================

    print("\n[5/9] Training CATEGORY models...")
    print("-" * 70)


    category_models = {

        "logistic_regression":
            LogisticRegression(
                C=3.0,
                max_iter=2000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "linear_svm":
            LinearSVC(
                C=1.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE
            )
    }


    category_results = []


    for model_name, model in category_models.items():

        print(
            f"\nTraining Category "
            f"{model_name}..."
        )

        start = time.time()

        model.fit(
            X_train_tfidf,
            y_category_train
        )

        metrics = evaluate_model(
            model,
            X_test_tfidf,
            y_category_test
        )

        training_time = (
            time.time() - start
        )

        model_path = (
            MODEL_DIR
            / f"category_{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
            compress=3
        )


        category_results.append({

            "task": "Category",

            "model": model_name,

            **metrics,

            "training_time_seconds":
                round(training_time, 2)
        })


        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1 : "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"Weighted F1 : "
            f"{metrics['f1_weighted']:.4f}"
        )

        print(
            f"Time : "
            f"{training_time:.2f}s"
        )

        print(
            f"Saved: {model_path}"
        )


    # ========================================================
    # 6. PRIORITY MODELS
    # ========================================================

    print("\n[6/9] Training PRIORITY models...")
    print("-" * 70)


    priority_models = {

        "logistic_regression":
            LogisticRegression(
                C=3.0,
                max_iter=2000,
                class_weight="balanced",
                solver="saga",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "linear_svm":
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE
            )
    }


    priority_results = []


    for model_name, model in priority_models.items():

        print(
            f"\nTraining Priority "
            f"{model_name}..."
        )

        start = time.time()

        model.fit(
            X_train_tfidf,
            y_priority_train
        )

        metrics = evaluate_model(
            model,
            X_test_tfidf,
            y_priority_test
        )

        training_time = (
            time.time() - start
        )


        model_path = (
            MODEL_DIR
            / f"priority_{model_name}.joblib"
        )


        joblib.dump(
            model,
            model_path,
            compress=3
        )


        priority_results.append({

            "task": "Priority",

            "model": model_name,

            **metrics,

            "training_time_seconds":
                round(training_time, 2)
        })


        print(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1 : "
            f"{metrics['f1_macro']:.4f}"
        )

        print(
            f"Weighted F1 : "
            f"{metrics['f1_weighted']:.4f}"
        )

        print(
            f"Time : "
            f"{training_time:.2f}s"
        )

        print(
            f"Saved: {model_path}"
        )


    # ========================================================
    # 7. MODEL COMPARISON
    # ========================================================

    print("\n[7/9] Comparing all models...")
    print("-" * 70)


    results = (
        category_results
        + priority_results
    )

    results_df = pd.DataFrame(
        results
    )


    results_df = results_df.sort_values(
        by=["task", "f1_macro"],
        ascending=[True, False]
    )


    results_path = (
        RESULT_DIR
        / "model_comparison.csv"
    )


    results_df.to_csv(
        results_path,
        index=False
    )


    print(
        results_df[
            [
                "task",
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "training_time_seconds"
            ]
        ].to_string(index=False)
    )


    print(
        f"\nResults saved to:\n"
        f"{results_path}"
    )


    # ========================================================
    # 8. BEST MODEL SELECTION
    # ========================================================

    print("\n[8/9] Selecting best models...")
    print("-" * 70)


    best_category = (
        results_df[
            results_df["task"] == "Category"
        ]
        .sort_values(
            "f1_macro",
            ascending=False
        )
        .iloc[0]
    )


    best_priority = (
        results_df[
            results_df["task"] == "Priority"
        ]
        .sort_values(
            "f1_macro",
            ascending=False
        )
        .iloc[0]
    )


    print("\nBEST CATEGORY MODEL")

    print(
        f"Model       : "
        f"{best_category['model']}"
    )

    print(
        f"Accuracy    : "
        f"{best_category['accuracy']:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{best_category['f1_macro']:.4f}"
    )


    print("\nBEST PRIORITY MODEL")

    print(
        f"Model       : "
        f"{best_priority['model']}"
    )

    print(
        f"Accuracy    : "
        f"{best_priority['accuracy']:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{best_priority['f1_macro']:.4f}"
    )


    # ========================================================
    # 9. FINAL SUMMARY
    # ========================================================

    print("\n[9/9] Training summary")
    print("-" * 70)

    total_time = (
        time.time() - pipeline_start
    )

    print(
        f"Total pipeline time : "
        f"{total_time:.2f}s"
    )

    print("\nGenerated files:")

    print(
        "✓ models/tfidf_vectorizer.joblib"
    )

    print(
        "✓ models/category_logistic_regression.joblib"
    )

    print(
        "✓ models/category_linear_svm.joblib"
    )

    print(
        "✓ models/priority_logistic_regression.joblib"
    )

    print(
        "✓ models/priority_linear_svm.joblib"
    )

    print(
        "✓ data/processed/model_comparison.csv"
    )


    print("\n" + "=" * 70)
    print(
        "IMPROVED MODEL TRAINING COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print("\nNext step:")

    print(
        r"Run: python src\evaluate.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()