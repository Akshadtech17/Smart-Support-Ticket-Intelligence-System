from pathlib import Path
import re
import sys
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

CATEGORY_MODEL_PATH = (
    MODELS_DIR / "category_linear_svm.joblib"
)

PRIORITY_MODEL_PATH = (
    MODELS_DIR / "priority_linear_svm.joblib"
)


def clean_text(text):
    """
    Basic text normalization used before TF-IDF prediction.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def load_models():

    print("=" * 70)
    print("SMART SUPPORT TICKET INTELLIGENCE SYSTEM")
    print("PREDICTION PIPELINE")
    print("=" * 70)

    print("\n[1/3] Loading trained models...")
    print("-" * 70)

    required_files = [
        VECTORIZER_PATH,
        CATEGORY_MODEL_PATH,
        PRIORITY_MODEL_PATH
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required model file not found:\n{file_path}"
            )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    category_model = joblib.load(
        CATEGORY_MODEL_PATH
    )

    priority_model = joblib.load(
        PRIORITY_MODEL_PATH
    )

    print("TF-IDF vectorizer loaded successfully")
    print("Category model loaded successfully")
    print("Priority model loaded successfully")

    return (
        vectorizer,
        category_model,
        priority_model
    )


def predict_ticket(
    ticket_text,
    vectorizer,
    category_model,
    priority_model
):

    ticket_text = clean_text(
        ticket_text
    )

    if not ticket_text:

        raise ValueError(
            "Ticket text cannot be empty."
        )

    features = vectorizer.transform(
        [ticket_text]
    )

    category_prediction = category_model.predict(
        features
    )[0]

    priority_prediction = priority_model.predict(
        features
    )[0]

    result = {
        "ticket_text": ticket_text,
        "category": category_prediction,
        "priority": priority_prediction
    }

    return result


def display_prediction(result):

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print("\nTicket:")
    print("-" * 70)
    print(result["ticket_text"])

    print("\nPredicted Category:")
    print("-" * 70)
    print(result["category"])

    print("\nPredicted Priority:")
    print("-" * 70)
    print(result["priority"])

    print("\n" + "=" * 70)


def interactive_mode(
    vectorizer,
    category_model,
    priority_model
):

    print("\n")
    print("=" * 70)
    print("INTERACTIVE SUPPORT TICKET PREDICTION")
    print("=" * 70)

    print(
        "\nEnter a support ticket below."
    )

    print(
        "Type 'exit' to close the prediction system."
    )

    while True:

        print("\n" + "-" * 70)

        ticket_text = input(
            "Support Ticket: "
        ).strip()

        if ticket_text.lower() == "exit":

            print(
                "\nPrediction system closed."
            )

            break

        if not ticket_text:

            print(
                "Please enter a valid ticket."
            )

            continue

        try:

            result = predict_ticket(
                ticket_text,
                vectorizer,
                category_model,
                priority_model
            )

            display_prediction(
                result
            )

        except Exception as error:

            print(
                f"\nPrediction error: {error}"
            )


def run_demo(
    vectorizer,
    category_model,
    priority_model
):

    print("\n")
    print("=" * 70)
    print("DEMO PREDICTION")
    print("=" * 70)

    demo_tickets = [

        "My payment was deducted but my order was not completed.",

        "The application keeps crashing whenever I try to open it.",

        "I cannot access my account and need help resetting my password.",

        "The website is completely down and I cannot access any services.",

        "I want to return the product and receive a refund."
    ]

    for ticket in demo_tickets:

        result = predict_ticket(
            ticket,
            vectorizer,
            category_model,
            priority_model
        )

        display_prediction(
            result
        )


def main():

    try:

        (
            vectorizer,
            category_model,
            priority_model
        ) = load_models()

        run_demo(
            vectorizer,
            category_model,
            priority_model
        )

        interactive_mode(
            vectorizer,
            category_model,
            priority_model
        )

    except KeyboardInterrupt:

        print(
            "\n\nPrediction interrupted by user."
        )

        sys.exit(0)

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":

    main()