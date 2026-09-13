from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class URLAnalysisRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="URL to analyze for cybersecurity threats.",
    )


class ModelPipeline(BaseModel):
    stage_1: str
    stage_2: str


class URLAnalysisResponse(BaseModel):
    url: str
    detector: str

    status: str

    threat_score: float
    decision_threshold: float

    risk_score: int
    risk_level: str

    threat_type: Optional[str] = None
    threat_type_confidence: Optional[float] = None

    threat_type_scores: Dict[str, float]

    security_signals: List[str]

    recommended_action: str

    models: ModelPipeline