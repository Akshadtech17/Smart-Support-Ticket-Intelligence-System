"""
Smart Support Ticket Intelligence System
Production-ready FastAPI application

Run:
    uvicorn src.app:app --reload

Production:
    uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 2

Expected model artifacts:
    models/category_model.joblib
    models/priority_model.joblib

Optional:
    models/category_vectorizer.joblib
    models/priority_vectorizer.joblib

Recommended:
    Save complete sklearn Pipeline objects during training.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        BASE_DIR / "models",
    )
)

CATEGORY_MODEL_PATH = Path(
    os.getenv(
        "CATEGORY_MODEL_PATH",
        MODEL_DIR / "category_model.joblib",
    )
)

PRIORITY_MODEL_PATH = Path(
    os.getenv(
        "PRIORITY_MODEL_PATH",
        MODEL_DIR / "priority_model.joblib",
    )
)

CATEGORY_VECTORIZER_PATH = Path(
    os.getenv(
        "CATEGORY_VECTORIZER_PATH",
        MODEL_DIR / "category_vectorizer.joblib",
    )
)

PRIORITY_VECTORIZER_PATH = Path(
    os.getenv(
        "PRIORITY_VECTORIZER_PATH",
        MODEL_DIR / "priority_vectorizer.joblib",
    )
)

APP_NAME = "Smart Support Ticket Intelligence System"

APP_VERSION = os.getenv(
    "APP_VERSION",
    "1.0.0",
)

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development",
).lower()

MAX_TICKET_LENGTH = int(
    os.getenv(
        "MAX_TICKET_LENGTH",
        "5000",
    )
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(APP_NAME)


# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================

class ModelRegistry:
    """
    Stores loaded ML models and vectorizers.

    Models are loaded once when the application starts rather than
    loading them for every request.
    """

    def __init__(self) -> None:
        self.category_model: Any | None = None
        self.priority_model: Any | None = None

        self.category_vectorizer: Any | None = None
        self.priority_vectorizer: Any | None = None

        self.category_loaded = False
        self.priority_loaded = False

        self.loaded_at: float | None = None

    @property
    def ready(self) -> bool:
        return (
            self.category_loaded
            and self.priority_loaded
        )


registry = ModelRegistry()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_joblib_model(
    path: Path,
    model_name: str,
) -> Any:
    """
    Load a joblib model safely.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} not found at: {path}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"{model_name} path is not a file: {path}"
        )

    logger.info(
        "Loading %s from %s",
        model_name,
        path,
    )

    model = joblib.load(path)

    logger.info(
        "%s loaded successfully",
        model_name,
    )

    return model


def load_models() -> None:
    """
    Load all ML artifacts into memory.

    Preferred architecture:

        text
          ↓
        sklearn Pipeline
          ↓
        classifier

    If your training code saved the vectorizer and classifier
    separately, the API also supports that configuration.
    """

    logger.info(
        "Initializing ML models..."
    )

    registry.category_model = load_joblib_model(
        CATEGORY_MODEL_PATH,
        "Category model",
    )

    registry.priority_model = load_joblib_model(
        PRIORITY_MODEL_PATH,
        "Priority model",
    )

    # ------------------------------------------------------------------------
    # Optional vectorizers
    # ------------------------------------------------------------------------

    if CATEGORY_VECTORIZER_PATH.exists():

        logger.info(
            "Loading category vectorizer..."
        )

        registry.category_vectorizer = joblib.load(
            CATEGORY_VECTORIZER_PATH
        )

    if PRIORITY_VECTORIZER_PATH.exists():

        logger.info(
            "Loading priority vectorizer..."
        )

        registry.priority_vectorizer = joblib.load(
            PRIORITY_VECTORIZER_PATH
        )

    registry.category_loaded = (
        registry.category_model is not None
    )

    registry.priority_loaded = (
        registry.priority_model is not None
    )

    registry.loaded_at = time.time()

    logger.info(
        "Model initialization completed"
    )


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    """
    Application startup/shutdown lifecycle.
    """

    logger.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )

    logger.info(
        "Environment: %s",
        ENVIRONMENT,
    )

    try:
        load_models()

    except Exception:
        logger.exception(
            "Failed to load ML models"
        )

        # During development it is useful to see the real error.
        # In production, the API should not pretend it is ready.
        if ENVIRONMENT == "production":
            raise

    yield

    logger.info(
        "Shutting down application"
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "Production-ready REST API for automatically "
        "classifying customer support tickets by "
        "category and priority."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "*",
)

if allowed_origins_raw == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip()
        for origin in allowed_origins_raw.split(",")
        if origin.strip()
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST ID + REQUEST LOGGING
# ============================================================================

@app.middleware("http")
async def request_middleware(
    request: Request,
    call_next,
):
    """
    Adds request IDs and measures request latency.
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled exception | request_id=%s",
            request_id,
        )

        raise

    duration = (
        time.perf_counter()
        - start_time
    )

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time"
    ] = f"{duration:.6f}"

    logger.info(
        "%s %s -> %s | %.4fs | request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration,
        request_id,
    )

    return response


# ============================================================================
# SECURITY HEADERS
# ============================================================================

@app.middleware("http")
async def security_headers_middleware(
    request: Request,
    call_next,
):
    response = await call_next(request)

    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "no-referrer"

    response.headers[
        "Cache-Control"
    ] = "no-store"

    return response


# ============================================================================
# SCHEMAS
# ============================================================================

class TicketRequest(BaseModel):
    """
    Request body for ticket prediction.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    ticket: str = Field(
        ...,
        min_length=5,
        max_length=MAX_TICKET_LENGTH,
        description="Customer support ticket text",
        examples=[
            "Payment was deducted but my order was cancelled."
        ],
    )

    @field_validator("ticket")
    @classmethod
    def validate_ticket(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Ticket cannot be empty"
            )

        if len(value) < 5:
            raise ValueError(
                "Ticket must contain at least 5 characters"
            )

        return value


class PredictionResponse(BaseModel):
    """
    Prediction response returned by the API.
    """

    success: bool

    ticket: str

    category: str

    priority: str

    category_confidence: float | None = None

    priority_confidence: float | None = None

    model_version: str

    request_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    models_loaded: bool


# ============================================================================
# PREDICTION HELPERS
# ============================================================================

def prepare_text_for_model(
    text: str,
    vectorizer: Any | None,
) -> Any:
    """
    Transform raw text when a separate vectorizer is being used.

    If vectorizer is None, the original text is returned.
    """

    if vectorizer is None:
        return [text]

    return vectorizer.transform(
        [text]
    )


def predict_label(
    model: Any,
    text: str,
    vectorizer: Any | None = None,
) -> tuple[str, float | None]:
    """
    Perform prediction and optionally calculate confidence.

    Supports:

    1. sklearn Pipeline

    2. classifier + separate TF-IDF vectorizer

    3. classifiers without predict_proba
    """

    if model is None:
        raise RuntimeError(
            "ML model is not loaded"
        )

    try:

        # --------------------------------------------------------------
        # If a vectorizer is explicitly supplied, transform the text
        # before passing it to the classifier.
        # --------------------------------------------------------------

        if vectorizer is not None:

            features = vectorizer.transform(
                [text]
            )

        else:

            # Pipeline expects raw text.
            features = [text]

        prediction = model.predict(
            features
        )

        if prediction is None:
            raise RuntimeError(
                "Model returned no prediction"
            )

        label = prediction[0]

        confidence = None

        # --------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------

        if hasattr(
            model,
            "predict_proba",
        ):

            try:

                probabilities = (
                    model.predict_proba(
                        features
                    )
                )

                if (
                    probabilities is not None
                    and len(probabilities) > 0
                ):

                    confidence = float(
                        max(
                            probabilities[0]
                        )
                    )

            except Exception:
                logger.warning(
                    "Unable to calculate prediction probability",
                    exc_info=True,
                )

        return (
            str(label),
            confidence,
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        raise RuntimeError(
            f"Prediction failed: {exc}"
        ) from exc


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get(
    "/",
    tags=["System"],
)
async def root():
    """
    API information endpoint.
    """

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "environment": ENVIRONMENT,
        "docs": "/docs",
        "health": "/health",
        "prediction": "/api/v1/predict",
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health_check():
    """
    Health endpoint.

    Returns 200 when the application is running.
    """

    return HealthResponse(
        status="healthy",
        service=APP_NAME,
        version=APP_VERSION,
        environment=ENVIRONMENT,
        models_loaded=registry.ready,
    )


# ============================================================================
# READINESS CHECK
# ============================================================================

@app.get(
    "/ready",
    tags=["System"],
)
async def readiness_check():
    """
    Readiness endpoint.

    Useful for Docker/Kubernetes/load balancer probes.
    """

    if not registry.ready:

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "models_loaded": False,
            },
        )

    return {
        "status": "ready",
        "models_loaded": True,
    }


# ============================================================================
# MODEL INFORMATION
# ============================================================================

@app.get(
    "/api/v1/model",
    tags=["System"],
)
async def model_information():
    """
    Returns information about the loaded models.
    """

    if not registry.ready:

        raise HTTPException(
            status_code=503,
            detail="ML models are not ready",
        )

    return {
        "category_model": type(
            registry.category_model
        ).__name__,

        "priority_model": type(
            registry.priority_model
        ).__name__,

        "category_vectorizer": (
            type(
                registry.category_vectorizer
            ).__name__
            if registry.category_vectorizer
            else None
        ),

        "priority_vectorizer": (
            type(
                registry.priority_vectorizer
            ).__name__
            if registry.priority_vectorizer
            else None
        ),

        "model_version": APP_VERSION,
    }


# ============================================================================
# MAIN PREDICTION ENDPOINT
# ============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict_ticket(
    payload: TicketRequest,
    request: Request,
):
    """
    Predict ticket category and priority.

    Example:

        POST /api/v1/predict

        {
            "ticket": "Payment was deducted but my order was cancelled."
        }

    Response:

        {
            "success": true,
            "ticket": "...",
            "category": "Payment",
            "priority": "High",
            ...
        }
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    # ------------------------------------------------------------------------
    # Model readiness
    # ------------------------------------------------------------------------

    if not registry.ready:

        logger.error(
            "Prediction requested while models are unavailable | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction service is temporarily unavailable "
                "because ML models are not loaded."
            ),
        )

    ticket = payload.ticket.strip()

    logger.info(
        "Processing ticket prediction | request_id=%s",
        request_id,
    )

    try:

        # --------------------------------------------------------------------
        # Category
        # --------------------------------------------------------------------

        category, category_confidence = predict_label(
            model=registry.category_model,
            text=ticket,
            vectorizer=registry.category_vectorizer,
        )

        # --------------------------------------------------------------------
        # Priority
        # --------------------------------------------------------------------

        priority, priority_confidence = predict_label(
            model=registry.priority_model,
            text=ticket,
            vectorizer=registry.priority_vectorizer,
        )

        return PredictionResponse(
            success=True,
            ticket=ticket,
            category=category,
            priority=priority,
            category_confidence=(
                round(
                    category_confidence,
                    4,
                )
                if category_confidence is not None
                else None
            ),
            priority_confidence=(
                round(
                    priority_confidence,
                    4,
                )
                if priority_confidence is not None
                else None
            ),
            model_version=APP_VERSION,
            request_id=request_id,
        )

    except RuntimeError as exc:

        logger.exception(
            "Prediction runtime error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected prediction error | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process ticket prediction",
        ) from exc


# ============================================================================
# BATCH PREDICTION
# ============================================================================

class BatchTicketRequest(BaseModel):
    """
    Request body for batch predictions.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    tickets: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    @field_validator("tickets")
    @classmethod
    def validate_tickets(
        cls,
        values: list[str],
    ) -> list[str]:

        cleaned = []

        for ticket in values:

            ticket = ticket.strip()

            if len(ticket) < 5:
                raise ValueError(
                    "Every ticket must contain at least 5 characters"
                )

            if len(ticket) > MAX_TICKET_LENGTH:
                raise ValueError(
                    f"Ticket cannot exceed {MAX_TICKET_LENGTH} characters"
                )

            cleaned.append(ticket)

        return cleaned


@app.post(
    "/api/v1/predict/batch",
    tags=["Prediction"],
)
async def predict_batch(
    payload: BatchTicketRequest,
    request: Request,
):
    """
    Predict multiple tickets in a single request.

    Maximum:
        50 tickets/request
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    if not registry.ready:

        raise HTTPException(
            status_code=503,
            detail="ML models are not ready",
        )

    results = []

    for ticket in payload.tickets:

        try:

            category, category_confidence = predict_label(
                registry.category_model,
                ticket,
                registry.category_vectorizer,
            )

            priority, priority_confidence = predict_label(
                registry.priority_model,
                ticket,
                registry.priority_vectorizer,
            )

            results.append(
                {
                    "ticket": ticket,
                    "category": category,
                    "priority": priority,
                    "category_confidence": (
                        round(
                            category_confidence,
                            4,
                        )
                        if category_confidence is not None
                        else None
                    ),
                    "priority_confidence": (
                        round(
                            priority_confidence,
                            4,
                        )
                        if priority_confidence is not None
                        else None
                    ),
                }
            )

        except Exception:

            logger.exception(
                "Batch prediction failed | request_id=%s",
            )

            results.append(
                {
                    "ticket": ticket,
                    "category": None,
                    "priority": None,
                    "category_confidence": None,
                    "priority_confidence": None,
                    "error": "Prediction failed",
                }
            )

    return {
        "success": True,
        "count": len(results),
        "results": results,
        "model_version": APP_VERSION,
        "request_id": request_id,
    }


# ============================================================================
# GLOBAL EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
):
    """
    Consistent HTTP error format.
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
            },
            "request_id": request_id,
        },
    )


@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    """
    Prevents internal exception details from leaking
    to API consumers.
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    logger.exception(
        "Unhandled application exception | request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "message": "Internal server error",
                "status_code": 500,
            },
            "request_id": request_id,
        },
    )


# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=(
            ENVIRONMENT
            == "development"
        ),
  )
