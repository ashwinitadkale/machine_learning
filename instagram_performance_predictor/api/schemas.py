from pydantic import BaseModel, Field
from typing import Dict, Literal

MEDIA_TYPES      = Literal["reel", "image", "carousel"]
CONTENT_CATS     = Literal["Fitness","Food","Fashion","Travel","Technology","Beauty","Music","Photography","Lifestyle","Comedy"]
TRAFFIC_SOURCES  = Literal["Home Feed","Hashtags","Reels Feed","Explore","External","Profile"]
ACCOUNT_TYPES    = Literal["creator","brand"]
DAYS             = Literal["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


class FullPostInput(BaseModel):
    follower_count:    int   = Field(..., gt=0, example=15000)
    engagement_rate:   float = Field(..., ge=0, le=1, example=0.08)
    likes:             int   = Field(..., ge=0, example=1200)
    comments:          int   = Field(..., ge=0, example=45)
    shares:            int   = Field(..., ge=0, example=30)
    saves:             int   = Field(..., ge=0, example=200)
    reach:             int   = Field(..., gt=0, example=18000)
    impressions:       int   = Field(..., gt=0, example=22000)
    followers_gained:  int   = Field(..., ge=0, example=120)
    has_call_to_action:int   = Field(..., ge=0, le=1, example=1)
    post_hour:         int   = Field(..., ge=0, le=23, example=19)
    caption_length:    int   = Field(..., ge=0, example=115)
    hashtags_count:    int   = Field(..., ge=0, example=8)
    media_type:        MEDIA_TYPES     = "reel"
    content_category:  CONTENT_CATS   = "Fitness"
    traffic_source:    TRAFFIC_SOURCES = "Explore"
    account_type:      ACCOUNT_TYPES  = "creator"
    day_of_week:       DAYS           = "Saturday"


class PrePostInput(BaseModel):
    follower_count:    int   = Field(..., gt=0, example=15000)
    has_call_to_action:int   = Field(..., ge=0, le=1, example=1)
    post_hour:         int   = Field(..., ge=0, le=23, example=19)
    caption_length:    int   = Field(..., ge=0, example=115)
    hashtags_count:    int   = Field(..., ge=0, example=8)
    media_type:        MEDIA_TYPES     = "reel"
    content_category:  CONTENT_CATS   = "Fitness"
    traffic_source:    TRAFFIC_SOURCES = "Explore"
    account_type:      ACCOUNT_TYPES  = "creator"
    day_of_week:       DAYS           = "Saturday"


class PredictionResponse(BaseModel):
    predicted_class:      str
    confidence:           float
    class_probabilities:  Dict[str, float]
    model_used:           str
    note:                 str = ""
