from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/api/analytics/summary")
def get_summary():
    return {
        "timestamp": datetime.utcnow(),
        "MRR": 20000,
        "active_users": 1340,
        "churn_rate": 0.03
    }
