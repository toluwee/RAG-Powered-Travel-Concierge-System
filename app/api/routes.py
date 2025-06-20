from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.travel import TravelRequest, TravelResponse
from app.services.travel_planner import TravelPlanner

router = APIRouter()
travel_planner = TravelPlanner()

@router.post("/plan", response_model=TravelResponse)
async def plan_trip(request: TravelRequest):
    """
    Plan a complex multi-city business trip based on the provided request.
    
    This endpoint processes travel requests with multiple segments, considering:
    - Traveler preferences and type
    - Corporate policies
    - Meeting schedules
    - Hotel and flight preferences
    - Budget constraints
    """
    try:
        response = await travel_planner.plan_trip(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error planning trip: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    } 