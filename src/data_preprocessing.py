# ============================================================
# SMART SUPPORT TICKET INTELLIGENCE SYSTEM
# DATA PREPROCESSING PIPELINE
# ============================================================

import re
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_support_tickets_clean.csv"
)

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
)

PROCESSED_PATH.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Replace newline and tabs
    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text
    )

    # Keep alphanumeric characters
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def preprocess_data():

    print("=" * 70)
    print("SMART SUPPORT TICKET INTELLIGENCE SYSTEM")
    print("DATA PREPROCESSING")
    print("=" * 70)


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1/10] Loading cleaned dataset...")

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}\n\n"
            "Run data_collection.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    print(
        f"Dataset loaded successfully: {df.shape}"
    )


    # ========================================================
    # 2. CHECK REQUIRED COLUMNS
    # ========================================================

    print("\n[2/10] Checking required columns...")

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

    print("Required columns verified")


    # ========================================================
    # 3. SELECT MODELING COLUMNS
    # ========================================================

    df = df[
        [
            "ticket_text",
            "category",
            "priority"
        ]
    ].copy()

    print(
        "\nSelected columns:",
        df.columns.tolist()
    )


    # ========================================================
    # 4. HANDLE MISSING VALUES
    # ========================================================

    print("\n[3/10] Handling missing values...")

    print("\nMissing values before cleaning:")

    print(df.isnull().sum())

    df["ticket_text"] = (
        df["ticket_text"]
        .fillna("")
        .astype(str)
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


    # ========================================================
    # 5. CLEAN TEXT
    # ========================================================

    print("\n[4/10] Cleaning ticket text...")

    df["ticket_text"] = (
        df["ticket_text"]
        .apply(clean_text)
    )

    print("Text cleaning completed")


    # ========================================================
    # 6. REMOVE INVALID RECORDS
    # ========================================================

    print("\n[5/10] Removing invalid records...")

    original_size = len(df)

    # Remove empty text
    df = df[
        df["ticket_text"].str.len() > 0
    ]

    # Remove empty category
    df = df[
        df["category"].str.len() > 0
    ]

    # Remove empty priority
    df = df[
        df["priority"].str.len() > 0
    ]

    # Remove duplicate ticket texts
    df = df.drop_duplicates(
        subset=["ticket_text"],
        keep="first"
    )

    df = df.reset_index(drop=True)

    removed_records = (
        original_size - len(df)
    )

    print(
        "Records removed:",
        removed_records
    )

    print(
        "Records remaining:",
        len(df)
    )


    # ========================================================
    # 7. SAVE FINAL MODELING DATASET
    # ========================================================

    print("\n[6/10] Saving modeling dataset...")

    modeling_path = (
        PROCESSED_PATH
        / "modeling_dataset.csv"
    )

    df.to_csv(
        modeling_path,
        index=False
    )

    print(
        "Saved:",
        modeling_path
    )


    # ========================================================
    # 8. TRAIN TEST SPLIT
    # ========================================================

    print("\n[7/10] Creating train/test split...")

    X = df["ticket_text"]

    y_category = df["category"]

    y_priority = df["priority"]


    # --------------------------------------------------------
    # Category split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_category_train,
        y_category_test
    ) = train_test_split(

        X,
        y_category,

        test_size=0.20,

        random_state=42,

        stratify=y_category
    )


    # --------------------------------------------------------
    # Priority split
    # --------------------------------------------------------

    (
        X_train_priority,
        X_test_priority,
        y_priority_train,
        y_priority_test
    ) = train_test_split(

        X,
        y_priority,

        test_size=0.20,

        random_state=42,

        stratify=y_priority
    )


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # ========================================================
    # 9. SAVE TRAIN TEST TEXT DATA
    # ========================================================

    train_category_df = pd.DataFrame({
        "ticket_text": X_train,
        "category": y_category_train
    })

    test_category_df = pd.DataFrame({
        "ticket_text": X_test,
        "category": y_category_test
    })

    train_priority_df = pd.DataFrame({
        "ticket_text": X_train_priority,
        "priority": y_priority_train
    })

    test_priority_df = pd.DataFrame({
        "ticket_text": X_test_priority,
        "priority": y_priority_test
    })


    train_category_df.to_csv(
        PROCESSED_PATH / "train_category.csv",
        index=False
    )

    test_category_df.to_csv(
        PROCESSED_PATH / "test_category.csv",
        index=False
    )

    train_priority_df.to_csv(
        PROCESSED_PATH / "train_priority.csv",
        index=False
    )

    test_priority_df.to_csv(
        PROCESSED_PATH / "test_priority.csv",
        index=False
    )


    # ========================================================
    # 10. TF-IDF
    # ========================================================

    print("\n[8/10] Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(

        lowercase=True,

        stop_words="english",

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,

        max_features=100000
    )


    # IMPORTANT:
    # Fit TF-IDF ONLY on training data

    X_train_tfidf = (
        vectorizer.fit_transform(X_train)
    )

    X_test_tfidf = (
        vectorizer.transform(X_test)
    )


    print(
        "TF-IDF training shape:",
        X_train_tfidf.shape
    )

    print(
        "TF-IDF testing shape:",
        X_test_tfidf.shape
    )


    # ========================================================
    # SAVE TF-IDF
    # ========================================================

    save_npz(
        PROCESSED_PATH / "X_train_tfidf.npz",
        X_train_tfidf
    )

    save_npz(
        PROCESSED_PATH / "X_test_tfidf.npz",
        X_test_tfidf
    )


    # ========================================================
    # SAVE VECTORIZER
    # ========================================================

    joblib.dump(
        vectorizer,
        MODEL_PATH / "tfidf_vectorizer.joblib"
    )


    # ========================================================
    # LABEL ENCODING
    # ========================================================

    print("\n[9/10] Encoding target labels...")

    category_encoder = LabelEncoder()

    priority_encoder = LabelEncoder()


    # Category labels

    y_category_train_encoded = (
        category_encoder.fit_transform(
            y_category_train
        )
    )

    y_category_test_encoded = (
        category_encoder.transform(
            y_category_test
        )
    )


    # Priority labels

    y_priority_train_encoded = (
        priority_encoder.fit_transform(
            y_priority_train
        )
    )

    y_priority_test_encoded = (
        priority_encoder.transform(
            y_priority_test
        )
    )


    # ========================================================
    # SAVE LABEL ENCODERS
    # ========================================================

    joblib.dump(
        category_encoder,
        MODEL_PATH
        / "category_label_encoder.joblib"
    )

    joblib.dump(
        priority_encoder,
        MODEL_PATH
        / "priority_label_encoder.joblib"
    )


    # ========================================================
    # SAVE ENCODED TARGETS
    # ========================================================

    np.save(
        PROCESSED_PATH
        / "y_category_train.npy",
        y_category_train_encoded
    )

    np.save(
        PROCESSED_PATH
        / "y_category_test.npy",
        y_category_test_encoded
    )

    np.save(
        PROCESSED_PATH
        / "y_priority_train.npy",
        y_priority_train_encoded
    )

    np.save(
        PROCESSED_PATH
        / "y_priority_test.npy",
        y_priority_test_encoded
    )


    # ========================================================
    # LABEL MAPPINGS
    # ========================================================

    category_mapping = pd.DataFrame({
        "encoded_value": range(
            len(category_encoder.classes_)
        ),
        "category": category_encoder.classes_
    })

    priority_mapping = pd.DataFrame({
        "encoded_value": range(
            len(priority_encoder.classes_)
        ),
        "priority": priority_encoder.classes_
    })


    category_mapping.to_csv(
        PROCESSED_PATH
        / "category_label_mapping.csv",
        index=False
    )

    priority_mapping.to_csv(
        PROCESSED_PATH
        / "priority_label_mapping.csv",
        index=False
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n[10/10] PREPROCESSING SUMMARY")

    print("=" * 70)

    print(
        f"Final records           : {len(df):,}"
    )

    print(
        f"Training records        : {len(X_train):,}"
    )

    print(
        f"Testing records         : {len(X_test):,}"
    )

    print(
        f"TF-IDF features         : "
        f"{X_train_tfidf.shape[1]:,}"
    )

    print(
        f"Category classes        : "
        f"{len(category_encoder.classes_)}"
    )

    print(
        f"Priority classes        : "
        f"{len(priority_encoder.classes_)}"
    )

    print("\nCategory labels:")

    for index, label in enumerate(
        category_encoder.classes_
    ):

        print(
            f"{index} -> {label}"
        )


    print("\nPriority labels:")

    for index, label in enumerate(
        priority_encoder.classes_
    ):

        print(
            f"{index} -> {label}"
        )


    print("\nGenerated files:")

    files = [
        "modeling_dataset.csv",
        "train_category.csv",
        "test_category.csv",
        "train_priority.csv",
        "test_priority.csv",
        "X_train_tfidf.npz",
        "X_test_tfidf.npz",
        "tfidf_vectorizer.joblib",
        "category_label_encoder.joblib",
        "priority_label_encoder.joblib",
        "y_category_train.npy",
        "y_category_test.npy",
        "y_priority_train.npy",
        "y_priority_test.npy",
        "category_label_mapping.csv",
        "priority_label_mapping.csv"
    ]

    for file in files:
        print(f"✓ {file}")


    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nNEXT STEP:")
    print("Run src/train.py")
    print(
        "Train Logistic Regression and Linear SVM "
        "for category and priority prediction."
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    preprocess_data()