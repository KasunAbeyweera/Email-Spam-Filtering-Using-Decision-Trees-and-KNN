# ============================================================
# FASTAPI MODEL SERVING
# ============================================================
#
# Supports either:
#   MODEL_SOURCE=joblib
#   MODEL_SOURCE=mlflow
#
# Joblib example:
#   export MODEL_SOURCE=joblib
#   export MODEL_PATH=binary_classification_pipeline.joblib
#
# MLflow example:
#   export MODEL_SOURCE=mlflow
#   export MODEL_URI="models:/credit-default-model@champion"
#
# Run:
#   uvicorn serve_model_fastapi:app --host 0.0.0.0 --port 8000
#
# Request:
#   POST /predict
#   {
#     "features": {
#       "age": 45,
#       "income": 120000,
#       "city": "Colombo",
#       "job_level": "senior"
#     }
#   }
# ============================================================

import os
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_SOURCE = os.getenv(
    "MODEL_SOURCE",
    "joblib"
).lower()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "binary_classification_pipeline.joblib"
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/model@champion"
)


app = FastAPI(
    title="Model Inference API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, Any]


def load_model():
    if MODEL_SOURCE == "joblib":
        return joblib.load(
            MODEL_PATH
        )

    if MODEL_SOURCE == "mlflow":
        return mlflow.pyfunc.load_model(
            MODEL_URI
        )

    raise ValueError(
        "MODEL_SOURCE must be 'joblib' or 'mlflow'"
    )


model = load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_source": MODEL_SOURCE,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_df = pd.DataFrame([
            request.features
        ])

        if MODEL_SOURCE == "joblib":
            prediction = model.predict(
                input_df
            )

            response = {
                "prediction": prediction[0].item()
                if hasattr(prediction[0], "item")
                else prediction[0]
            }

            if hasattr(
                model,
                "predict_proba"
            ):
                probabilities = (
                    model.predict_proba(
                        input_df
                    )
                )

                response[
                    "probabilities"
                ] = (
                    probabilities[0]
                    .tolist()
                )

            return response

        prediction = model.predict(
            input_df
        )

        if isinstance(
            prediction,
            pd.DataFrame
        ):
            prediction = (
                prediction
                .to_dict(
                    orient="records"
                )
            )

        elif isinstance(
            prediction,
            pd.Series
        ):
            prediction = (
                prediction
                .tolist()
            )

        elif hasattr(
            prediction,
            "tolist"
        ):
            prediction = (
                prediction
                .tolist()
            )

        return {
            "prediction": prediction
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
