from pydantic import BaseModel, Field
from typing import List, Literal

class CowSensorInput(BaseModel):
    cow_id: str = Field(..., example="C0001")
    milk_ec: float = Field(..., example=4.4)
    milk_temperature_c: float = Field(..., example=38.25)
    udder_temperature_c: float = Field(..., example=38.7)
    activity_index: float = Field(..., example=108.9)
    milk_colour_code: int = Field(
        ...,
        ge=0,
        le=4,
        example=1,
        description=(
            "Visual milk colour/appearance score observed at milking: "
            "0 = normal/white, 1 = slightly watery, 2 = yellowish, "
            "3 = blood-tinged, 4 = clots/flakes or other marked abnormality."
        ),
    )
    milk_ec_baseline: float = Field(..., example=4.307)
    milk_temp_baseline_c: float = Field(..., example=38.24)
    udder_temp_baseline_c: float = Field(..., example=38.588)
    activity_baseline: float = Field(..., example=107.525)
    forecast_horizon_days: Literal[7, 14] = Field(7, example=7, description="Forecast horizon in days")

class CowRecordInput(CowSensorInput):
    pass

class CowLatestRecordResponse(CowSensorInput):
    pass

class PredictionResponse(BaseModel):
    cow_id: str
    mastitis_risk_probability: float
    risk_category: str
    model_risk_category: str
    clinical_override: bool
    forecast_horizon_days: int
    confidence: str
    uncertainty_score: float
    top_factors: List[str]
    recommended_actions: List[str]
    timestamp: str

class HerdRiskInput(BaseModel):
    farm_id: str = Field(..., example="FARM_001")
    cows: List[CowSensorInput]

class HerdRiskResponse(BaseModel):
    farm_id: str
    total_cows_monitored: int
    high_risk_count: int
    high_risk_pct: float
    moderate_risk_count: int
    low_risk_count: int
    no_risk_count: int
    herd_health_index: float
    anomaly_alert: bool
