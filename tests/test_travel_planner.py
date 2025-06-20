"""
Tests for the Travel Planner service
"""

import pytest
from datetime import datetime, timedelta
from app.models.travel import (
    TravelRequest,
    TravelResponse,
    TripSegment,
    FlightSegment,
    HotelStay,
    Location,
    TravelerType,
    TripPurpose,
    TravelerPreferences
)

@pytest.fixture
def sample_travel_request():
    """Create a sample travel request for testing"""
    # Create locations
    seattle = Location(city="Seattle", country="USA")
    london = Location(city="London", country="UK")
    frankfurt = Location(city="Frankfurt", country="Germany")

    # Create flight segments
    flight1 = FlightSegment(
        origin=seattle,
        destination=london,
        departure_time=datetime.now() + timedelta(days=1),
        arrival_time=datetime.now() + timedelta(days=1, hours=8),
        airline="Delta",
        flight_number="DL123",
        price=1200.00
    )

    flight2 = FlightSegment(
        origin=london,
        destination=frankfurt,
        departure_time=datetime.now() + timedelta(days=3),
        arrival_time=datetime.now() + timedelta(days=3, hours=2),
        airline="Lufthansa",
        flight_number="LH456",
        price=400.00
    )

    # Create hotel stays
    hotel1 = HotelStay(
        location=london,
        check_in=datetime.now() + timedelta(days=1),
        check_out=datetime.now() + timedelta(days=3),
        hotel_name="Marriott London",
        chain="Marriott",
        room_type="Business",
        price_per_night=300.00,
        amenities=["24h Business Center", "Gym", "Restaurant"]
    )

    hotel2 = HotelStay(
        location=frankfurt,
        check_in=datetime.now() + timedelta(days=3),
        check_out=datetime.now() + timedelta(days=5),
        hotel_name="Hilton Frankfurt",
        chain="Hilton",
        room_type="Executive",
        price_per_night=250.00,
        amenities=["Business Center", "Spa", "Restaurant"]
    )

    # Create trip segments
    segment1 = TripSegment(
        flight=flight1,
        hotel=hotel1,
        purpose=TripPurpose.CLIENT_MEETING,
        meeting_times=[
            datetime.now() + timedelta(days=2, hours=10),
            datetime.now() + timedelta(days=2, hours=14)
        ]
    )

    segment2 = TripSegment(
        flight=flight2,
        hotel=hotel2,
        purpose=TripPurpose.CONFERENCE,
        meeting_times=[
            datetime.now() + timedelta(days=4, hours=9),
            datetime.now() + timedelta(days=4, hours=17)
        ]
    )

    # Create traveler preferences
    preferences = TravelerPreferences(
        preferred_airlines=["Delta", "Lufthansa"],
        preferred_hotel_chains=["Marriott", "Hilton"],
        seat_preference="aisle",
        meal_preference="vegetarian",
        requires_24h_business_center=True,
        preferred_flight_times=["morning", "afternoon"],
        loyalty_programs={
            "Delta": "Gold",
            "Marriott": "Platinum"
        }
    )

    # Create the complete travel request
    return TravelRequest(
        traveler_type=TravelerType.ROAD_WARRIOR,
        preferences=preferences,
        segments=[segment1, segment2],
        corporate_policy_id="CORP123",
        budget_constraints={
            "max_flight_price": 2000.00,
            "max_hotel_price_per_night": 400.00
        },
        special_requirements=[
            "Need visa for UK",
            "Requires airport transfer"
        ]
    )

@pytest.mark.asyncio
async def test_plan_trip(sample_travel_request):
    """Test the travel planning functionality"""
    from app.services.travel_planner import TravelPlanner
    
    planner = TravelPlanner()
    response = await planner.plan_trip(sample_travel_request)
    
    # Verify response structure
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.itinerary) == 2
    assert response.total_cost > 0
    assert isinstance(response.warnings, list)
    assert isinstance(response.recommendations, list)

@pytest.mark.asyncio
async def test_plan_trip_with_single_segment():
    """Test travel planning with a single segment"""
    from app.services.travel_planner import TravelPlanner
    
    # Create a simple single-segment request
    seattle = Location(city="Seattle", country="USA")
    london = Location(city="London", country="UK")
    
    flight = FlightSegment(
        origin=seattle,
        destination=london,
        departure_time=datetime.now() + timedelta(days=1),
        arrival_time=datetime.now() + timedelta(days=1, hours=8),
        airline="Delta",
        flight_number="DL123",
        price=1200.00
    )
    
    hotel = HotelStay(
        location=london,
        check_in=datetime.now() + timedelta(days=1),
        check_out=datetime.now() + timedelta(days=3),
        hotel_name="Marriott London",
        chain="Marriott",
        room_type="Business",
        price_per_night=300.00,
        amenities=["24h Business Center"]
    )
    
    segment = TripSegment(
        flight=flight,
        hotel=hotel,
        purpose=TripPurpose.CLIENT_MEETING,
        meeting_times=[datetime.now() + timedelta(days=2, hours=10)]
    )
    
    preferences = TravelerPreferences(
        preferred_airlines=["Delta"],
        preferred_hotel_chains=["Marriott"],
        seat_preference="aisle",
        requires_24h_business_center=True
    )
    
    request = TravelRequest(
        traveler_type=TravelerType.ROAD_WARRIOR,
        preferences=preferences,
        segments=[segment],
        budget_constraints={"max_flight_price": 2000.00}
    )
    
    planner = TravelPlanner()
    response = await planner.plan_trip(request)
    
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.itinerary) == 1
    assert response.total_cost > 0 