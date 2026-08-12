"""
Instagram Post Performance Predictor — FastAPI
Run: uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import FullPostInput, PrePostInput, PredictionResponse
from api.predict import predict_full, predict_pre

app = FastAPI(
    title="Instagram Performance Predictor",
    description=(
        "Predicts whether an Instagram post will perform as low / medium / high / viral.\n\n"
        "Two modes:\n"
        "- **/predict/full** — post-hoc analysis using engagement data (91% accuracy)\n"
        "- **/predict/pre** — pre-posting prediction using metadata only (24% accuracy, honest)\n\n"
        "See README for full methodology."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "endpoints": {
            "POST /predict/full": "Post-hoc classification (all features, 91% acc)",
            "POST /predict/pre":  "Pre-posting prediction (metadata only, 24% acc)",
            "GET  /docs":         "Interactive Swagger UI",
        }
    }


@app.post("/predict/full", response_model=PredictionResponse, tags=["Prediction"])
def classify_post(payload: FullPostInput):
    """
    Classify an existing post using full engagement features.
    Best for: content audits, retrospective analysis, dataset labelling.
    Model: Random Forest (91.1% accuracy, 4-class, balanced dataset).
    """
    try:
        result = predict_full(payload.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/pre", response_model=PredictionResponse, tags=["Prediction"])
def classify_pre_posting(payload: PrePostInput):
    """
    Predict performance before publishing using only metadata.
    Best for: content strategy, A/B testing post plans.
    Model: Random Forest (24.6% accuracy — intentionally honest about metadata limits).
    """
    try:
        result = predict_pre(payload.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
