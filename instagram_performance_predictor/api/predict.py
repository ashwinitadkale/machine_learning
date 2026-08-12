import numpy as np
import pandas as pd
import joblib
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_rf_full = None
_rf_pre  = None

MEDIA_ENC    = {"carousel": 0, "image": 1, "reel": 2}
CAT_ENC      = {"Beauty":0,"Comedy":1,"Fashion":2,"Fitness":3,"Food":4,
                 "Lifestyle":5,"Music":6,"Photography":7,"Technology":8,"Travel":9}
SOURCE_ENC   = {"External":0,"Hashtags":1,"Home Feed":2,"Profile":3,"Reels Feed":4,"Explore":5}
ACCT_ENC     = {"brand":0,"creator":1}
DOW_ENC      = {"Friday":0,"Monday":1,"Saturday":2,"Sunday":3,"Thursday":4,"Tuesday":5,"Wednesday":6}

CLASSES = ["low", "medium", "high", "viral"]

FULL_FEATURES = [
    'follower_count','has_call_to_action','post_hour','caption_length','hashtags_count',
    'engagement_rate','reach','impressions','likes','comments','shares','saves','followers_gained',
    'total_interactions','save_rate','comment_like_ratio','share_rate',
    'impression_reach_ratio','virality_score','engagement_depth',
    'is_weekend','is_prime_time','is_morning','hour_sin','hour_cos',
    'media_type_enc','content_category_enc','traffic_source_enc','account_type_enc','day_of_week_enc'
]

PRE_FEATURES = [
    'follower_count','has_call_to_action','post_hour','caption_length','hashtags_count',
    'is_weekend','is_prime_time','is_morning','hour_sin','hour_cos',
    'media_type_enc','content_category_enc','traffic_source_enc','account_type_enc','day_of_week_enc'
]


def _load_models():
    global _rf_full, _rf_pre
    if _rf_full is None:
        _rf_full = joblib.load(os.path.join(BASE, "models", "rf_full.pkl"))
    if _rf_pre is None:
        _rf_pre = joblib.load(os.path.join(BASE, "models", "rf_pre.pkl"))


def _shared_features(d: dict) -> dict:
    total = d.get('likes',0) + d.get('comments',0) + d.get('shares',0) + d.get('saves',0)
    reach = d.get('reach', 1)
    hour  = d['post_hour']
    return {
        'total_interactions':     total,
        'save_rate':              d.get('saves',0)    / (reach + 1),
        'comment_like_ratio':     d.get('comments',0) / (d.get('likes',0) + 1),
        'share_rate':             d.get('shares',0)   / (reach + 1),
        'impression_reach_ratio': d.get('impressions',0) / (reach + 1),
        'virality_score':         (d.get('shares',0) + d.get('saves',0)) / (reach + 1),
        'engagement_depth':       d.get('comments',0) / (total + 1),
        'is_weekend':             int(d['day_of_week'] in ('Saturday','Sunday')),
        'is_prime_time':          int(18 <= hour <= 22),
        'is_morning':             int(6  <= hour <= 10),
        'hour_sin':               np.sin(2 * np.pi * hour / 24),
        'hour_cos':               np.cos(2 * np.pi * hour / 24),
        'media_type_enc':         MEDIA_ENC.get(d['media_type'], 0),
        'content_category_enc':   CAT_ENC.get(d['content_category'], 0),
        'traffic_source_enc':     SOURCE_ENC.get(d['traffic_source'], 0),
        'account_type_enc':       ACCT_ENC.get(d['account_type'], 0),
        'day_of_week_enc':        DOW_ENC.get(d['day_of_week'], 0),
    }


def predict_full(data: dict) -> dict:
    _load_models()
    row = {**data, **_shared_features(data)}
    X   = pd.DataFrame([row])[FULL_FEATURES]
    proba = _rf_full.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    return {
        "predicted_class":     CLASSES[idx],
        "confidence":          round(float(proba[idx]), 4),
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
        "model_used":          "RandomForest (full features, post-hoc)",
        "note":                "Uses real-time engagement data. Suitable for auditing existing posts."
    }


def predict_pre(data: dict) -> dict:
    _load_models()
    row = {**data, **_shared_features(data)}
    X   = pd.DataFrame([row])[PRE_FEATURES]
    proba = _rf_pre.predict_proba(X)[0]
    idx   = int(np.argmax(proba))
    return {
        "predicted_class":     CLASSES[idx],
        "confidence":          round(float(proba[idx]), 4),
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, proba)},
        "model_used":          "RandomForest (pre-posting, metadata only)",
        "note":                "Uses only pre-publish metadata. Accuracy is intentionally lower (~24%) — "
                               "metadata alone has limited predictive power. See README for full discussion."
    }
