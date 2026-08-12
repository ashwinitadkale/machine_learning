# Instagram Post Performance Predictor

A full ML pipeline that classifies Instagram posts into **low / medium / high / viral** performance buckets — and uniquely includes a **pre-posting prediction mode** that forecasts virality using only information available *before* you publish.

---

## What makes this project different

Most engagement-prediction projects train on real-time engagement data (likes, saves, reach) — which is only available *after* a post goes live, making the model useless for content strategy. This project explicitly separates two scenarios:

| Mode | Features Used | Accuracy | Use Case |
|------|--------------|----------|----------|
| **Post-hoc analysis** | All 30 features incl. engagement | 91.1% | Audit / classify existing content |
| **Pre-posting prediction** | 15 metadata-only features | 24.6% | Strategy before you publish |

The pre-posting model's lower accuracy is *expected and honest* — it surfaces what metadata alone actually predicts, which is the real research question.

---

## Dataset

- **30,000** Instagram posts across 20 accounts (brand + creator)
- **10** content categories: Fitness, Food, Fashion, Travel, Technology, etc.
- **6** traffic sources: Home Feed, Hashtags, Reels Feed, Explore, External, Profile
- **4** media types: Reel, Image, Carousel
- Balanced target: ~7,500 posts per class

---

## Project Structure

```
instagram_performance_predictor/
├── data/
│   └── Instagram_Analytics.csv
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory analysis & visualizations
│   ├── 02_feature_engineering.ipynb  # Feature rationale + correlation heatmap
│   ├── 03_model_training.ipynb       # LR → RF → XGBoost + hyperparameter tuning
│   ├── 04_shap_explainability.ipynb  # SHAP values, waterfall + summary plots
│   └── 05_pre_posting_model.ipynb    # Pre-posting variant & honest comparison
├── api/
│   ├── main.py                       # FastAPI app (post-hoc + pre-posting endpoints)
│   ├── schemas.py                    # Pydantic input/output models
│   └── predict.py                    # Prediction logic with feature engineering
├── models/
│   ├── rf_full.pkl                   # Tuned Random Forest (full features)
│   ├── xgb_full.pkl                  # Tuned XGBoost (full features)
│   ├── rf_pre.pkl                    # Pre-posting RF model
│   └── scaler.pkl                    # StandardScaler for LR baseline
├── assets/
│   └── shap_importance.png
├── requirements.txt
└── README.md
```

---

## Models & Results

### Full Pipeline (post-hoc)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (baseline) | 87.1% | 0.871 |
| Random Forest (tuned) | **91.1%** | **0.911** |
| XGBoost (tuned) | 90.3% | 0.903 |

### Pre-Posting Pipeline

| Model | Accuracy | Macro F1 | Features |
|-------|----------|----------|----------|
| Random Forest | 24.6% | 0.245 | Metadata only (no engagement) |

**Key finding:** Metadata alone (posting time, caption length, hashtag count, account type, content category, media type) has minimal predictive power. Engagement signals dominate — meaning content quality as measured by audience response is what separates viral from low-performing posts, not scheduling strategy alone.

---

## Feature Engineering

8 new features engineered on top of the raw data:

| Feature | Formula | Intuition |
|---------|---------|-----------|
| `save_rate` | saves / reach | Bookmark signal — high intent |
| `comment_like_ratio` | comments / likes | Controversy / conversation depth |
| `share_rate` | shares / reach | Virality potential |
| `virality_score` | (shares + saves) / reach | Combined amplification metric |
| `engagement_depth` | comments / total_interactions | Quality over quantity |
| `impression_reach_ratio` | impressions / reach | Feed re-exposure rate |
| `hour_sin` / `hour_cos` | sin/cos of post_hour | Cyclical time encoding |

---

## SHAP Explainability

TreeSHAP is used to explain XGBoost predictions on a 500-post test sample. The viral class SHAP analysis reveals:

1. **`engagement_rate`** dominates with mean |SHAP| ~3.9 — far above all other features
2. **`save_rate`** (0.36) and **`likes`** (0.36) are the second-tier drivers
3. Metadata features (`media_type`, `post_hour`) have near-zero SHAP values for the viral class — confirming the pre-posting model's ceiling

---

## API Usage

```bash
# Install & run
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### POST `/predict/full` — post-hoc classification
```json
{
  "follower_count": 15000,
  "engagement_rate": 0.08,
  "likes": 1200,
  "comments": 45,
  "shares": 30,
  "saves": 200,
  "reach": 18000,
  "impressions": 22000,
  "followers_gained": 120,
  "has_call_to_action": 1,
  "post_hour": 19,
  "caption_length": 115,
  "hashtags_count": 8,
  "media_type": "reel",
  "content_category": "Fitness",
  "traffic_source": "Explore",
  "account_type": "creator",
  "day_of_week": "Saturday"
}
```

### POST `/predict/pre` — pre-posting prediction
```json
{
  "follower_count": 15000,
  "has_call_to_action": 1,
  "post_hour": 19,
  "caption_length": 115,
  "hashtags_count": 8,
  "media_type": "reel",
  "content_category": "Fitness",
  "traffic_source": "Explore",
  "account_type": "creator",
  "day_of_week": "Saturday"
}
```

Response:
```json
{
  "predicted_class": "high",
  "confidence": 0.72,
  "class_probabilities": {"low": 0.03, "medium": 0.11, "high": 0.72, "viral": 0.14},
  "model_used": "RandomForest (pre-posting)"
}
```

---

## Key Takeaways

- **Engagement signals > metadata** for post classification — a result worth communicating, not hiding
- **Cyclical time encoding** (`hour_sin`/`hour_cos`) is more principled than raw hour — the model sees that 23:00 and 00:00 are adjacent
- **SHAP over feature importance** — permutation-based importance is biased toward high-cardinality features; SHAP gives per-prediction attribution
- **Honest pre-posting comparison** is the differentiating design choice — most tutorials don't make this distinction

---

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · FastAPI · Pydantic · pandas · matplotlib · seaborn · joblib
