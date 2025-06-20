from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from app.models.travel import (
    TravelRequest,
    TravelResponse,
    TripSegment,
    FlightSegment,
    HotelStay,
    Location
)
from app.core.config import settings
from app.core.model_training import TravelModelTrainer
from app.core.rag_system import TravelRAGSystem
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelPlanner:
    def __init__(self):
        # Initialize RAG system
        self.rag_system = TravelRAGSystem()
        
        # Initialize model trainer
        self.model_trainer = TravelModelTrainer()
        
        # Load fine-tuned model if available
        try:
            self.model, self.tokenizer = self.model_trainer.load_fine_tuned_model(
                "models/travel_concierge"
            )
        except Exception as e:
            logger.warning(f"Could not load fine-tuned model: {str(e)}")
            self.model = None
            self.tokenizer = None

    async def plan_trip(self, request: TravelRequest) -> TravelResponse:
        """Plan a complex multi-city business trip"""
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Process each segment
        processed_segments = []
        warnings = []
        recommendations = []
        total_cost = 0.0

        # Get relevant context from RAG system
        rag_context = self._get_rag_context(request)
        
        for segment in request.segments:
            # Process flight if needed
            if segment.flight:
                flight = await self._process_flight_segment(
                    segment.flight,
                    request.preferences,
                    rag_context
                )
                if flight:
                    segment.flight = flight
                    total_cost += flight.price or 0.0
                else:
                    warnings.append(f"Could not find suitable flight for {segment.flight.origin.city} to {segment.flight.destination.city}")

            # Process hotel if needed
            if segment.hotel:
                hotel = await self._process_hotel_segment(
                    segment.hotel,
                    request.preferences,
                    rag_context
                )
                if hotel:
                    segment.hotel = hotel
                    total_cost += hotel.price_per_night * (
                        (hotel.check_out - hotel.check_in).days
                    )
                else:
                    warnings.append(f"Could not find suitable hotel in {segment.hotel.location.city}")

            processed_segments.append(segment)

        # Generate recommendations using RAG
        recommendations.extend(self._generate_recommendations(rag_context, request))

        return TravelResponse(
            request_id=request_id,
            status="completed",
            itinerary=processed_segments,
            total_cost=total_cost,
            warnings=warnings,
            recommendations=recommendations
        )

    def _get_rag_context(self, request: TravelRequest) -> Dict[str, Any]:
        """Get relevant context from RAG system"""
        # Create a query based on the request
        query = f"""
        Business trip planning for {request.traveler_type} with {len(request.segments)} segments.
        Preferences: {request.preferences.dict()}
        Budget: {request.budget_constraints}
        Special requirements: {request.special_requirements}
        """
        
        # Query RAG system
        result = self.rag_system.query(query)
        return result

    async def _process_flight_segment(
        self,
        flight: FlightSegment,
        preferences: Dict,
        rag_context: Dict[str, Any]
    ) -> Optional[FlightSegment]:
        """Process and optimize a flight segment using fine-tuned model"""
        if self.model and self.tokenizer:
            # Use fine-tuned model for flight processing
            input_text = self._format_flight_input(flight, preferences, rag_context)
            # Process with model (simplified for example)
            # In practice, you'd want to implement proper model inference
            return flight
        else:
            # Fallback to basic processing
            return flight

    async def _process_hotel_segment(
        self,
        hotel: HotelStay,
        preferences: Dict,
        rag_context: Dict[str, Any]
    ) -> Optional[HotelStay]:
        """Process and optimize a hotel segment using fine-tuned model"""
        if self.model and self.tokenizer:
            # Use fine-tuned model for hotel processing
            input_text = self._format_hotel_input(hotel, preferences, rag_context)
            # Process with model (simplified for example)
            # In practice, you'd want to implement proper model inference
            return hotel
        else:
            # Fallback to basic processing
            return hotel

    def _format_flight_input(
        self,
        flight: FlightSegment,
        preferences: Dict,
        rag_context: Dict[str, Any]
    ) -> str:
        """Format input for flight processing"""
        return f"""
        <flight>
        From: {flight.origin.city}, {flight.origin.country}
        To: {flight.destination.city}, {flight.destination.country}
        Date: {flight.departure_time}
        Preferences: {preferences}
        Context: {rag_context}
        </flight>
        """

    def _format_hotel_input(
        self,
        hotel: HotelStay,
        preferences: Dict,
        rag_context: Dict[str, Any]
    ) -> str:
        """Format input for hotel processing"""
        return f"""
        <hotel>
        Location: {hotel.location.city}, {hotel.location.country}
        Check-in: {hotel.check_in}
        Check-out: {hotel.check_out}
        Preferences: {preferences}
        Context: {rag_context}
        </hotel>
        """

    def _generate_recommendations(
        self,
        rag_context: Dict[str, Any],
        request: TravelRequest
    ) -> List[str]:
        """Generate personalized recommendations based on RAG context"""
        recommendations = []
        
        # Add recommendations from RAG system
        if "answer" in rag_context:
            recommendations.append(rag_context["answer"])
        
        # Add source-based recommendations
        if "sources" in rag_context:
            for source in rag_context["sources"]:
                if source["metadata"]["type"] == "hotel":
                    recommendations.append(f"Hotel recommendation: {source['content']}")
                elif source["metadata"]["type"] == "flight":
                    recommendations.append(f"Flight recommendation: {source['content']}")
                elif source["metadata"]["type"] == "transportation":
                    recommendations.append(f"Transportation tip: {source['content']}")
        
        return recommendations 