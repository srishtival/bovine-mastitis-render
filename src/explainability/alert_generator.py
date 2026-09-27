import json
import datetime
from src.models.calibration import map_risk_tier

def generate_clinician_alert(cow_id, probability, uncertainty=0.05, forecast_horizon=3, top_factors=None, timestamp=None, clinical_override=False):
    """
    Generates a clinician-safe, structured alert JSON object.
    Strictly avoids prescribing antimicrobial treatments.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
    model_risk_category, _ = map_risk_tier(probability)
    risk_category, action = map_risk_tier(1.0) if clinical_override else map_risk_tier(probability)
    
    if top_factors is None or len(top_factors) == 0:
        top_factors = [
            "Milk electrical conductivity elevated above personal baseline",
            "Somatic Cell Count (SCC) exceeding subclinical threshold",
            "Udder thermal asymmetry detected across quarters"
        ]
        
    confidence_level = "High" if uncertainty < 0.10 else ("Moderate" if uncertainty < 0.20 else "Low")
    
    recommended_actions = [
        action,
        "Perform California Mastitis Test (CMT) or lab SCC confirmation.",
        "Inspect milking machine cluster and claw vacuum alignment.",
        "Request veterinary clinical inspection if temperature or swelling develops."
    ]
    
    alert = {
        "cow_id": cow_id,
        "timestamp": timestamp,
        "mastitis_risk_probability": round(probability, 4),
        "risk_category": risk_category,
        "model_risk_category": model_risk_category,
        "clinical_override": clinical_override,
        "forecast_horizon_days": forecast_horizon,
        "confidence": confidence_level,
        "uncertainty_score": round(uncertainty, 4),
        "top_factors": top_factors,
        "recommended_actions": recommended_actions,
        "multilingual_dictionary": {
            "hi": {
                "risk_category": "उच्च जोखिम" if risk_category == "High Risk" else ("मध्यम जोखिम" if risk_category == "Moderate Risk" else "कम जोखिम"),
                "action": "कृपया सीएमटी परीक्षण करें और पशुचिकित्सक से परामर्श लें।"
            }
        }
    }
    return alert
