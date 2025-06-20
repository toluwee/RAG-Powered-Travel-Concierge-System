from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class TravelerType(str, Enum):
    ROAD_WARRIOR = "road_warrior"
    OCCASIONAL = "occasional"
    GROUP_BOOKER = "group_booker"

class TripPurpose(str, Enum):
    CLIENT_MEETING = "client_meeting"
    CONFERENCE = "conference"
    TRAINING = "training"
    OTHER = "other"

class TravelerPreferences(BaseModel):
    preferred_airlines: List[str] = Field(default_factory=list)
    preferred_hotel_chains: List[str] = Field(default_factory=list)
    seat_preference: Optional[str] = None
    meal_preference: Optional[str] = None
    requires_24h_business_center: bool = False
    preferred_flight_times: List[str] = Field(default_factory=list)
    loyalty_programs: Dict[str, str] = Field(default_factory=dict)

class Location(BaseModel):
    city: str
    country: str
    coordinates: Optional[Dict[str, float]] = None
    timezone: Optional[str] = None

class FlightSegment(BaseModel):
    origin: Location
    destination: Location
    departure_time: datetime
    arrival_time: datetime
    airline: str
    flight_number: str
    aircraft_type: Optional[str] = None
    duration: Optional[int] = None  # in minutes
    price: Optional[float] = None
    currency: str = "USD"

class HotelStay(BaseModel):
    location: Location
    check_in: datetime
    check_out: datetime
    hotel_name: str
    chain: Optional[str] = None
    room_type: str
    price_per_night: float
    currency: str = "USD"
    amenities: List[str] = Field(default_factory=list)
    distance_to_office: Optional[float] = None  # in kilometers

class TripSegment(BaseModel):
    flight: Optional[FlightSegment] = None
    hotel: Optional[HotelStay] = None
    purpose: TripPurpose
    meeting_times: Optional[List[datetime]] = None
    notes: Optional[str] = None

class TravelRequest(BaseModel):
    traveler_type: TravelerType
    preferences: TravelerPreferences
    segments: List[TripSegment]
    corporate_policy_id: Optional[str] = None
    budget_constraints: Optional[Dict[str, float]] = None
    special_requirements: Optional[List[str]] = None

class TravelResponse(BaseModel):
    request_id: str
    status: str
    itinerary: List[TripSegment]
    total_cost: float
    currency: str = "USD"
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow) 