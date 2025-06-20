#!/usr/bin/env python3
"""
Enhanced Automation script for Intelligent Travel Concierge System
Handles robust data generation, model training, RAG setup, and testing
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import random
from pathlib import Path
import itertools
from dataclasses import dataclass, asdict
import uuid

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.core.model_training import TravelModelTrainer
from app.core.rag_system import TravelRAGSystem
from app.services.travel_planner import TravelPlanner
from app.models.travel import (
    TravelRequest, TripSegment, FlightSegment, 
    HotelStay, Location, TravelerType, TripPurpose, TravelerPreferences
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TravelerProfile:
    """Comprehensive traveler profile for data generation"""
    segment: str
    seniority_level: str
    company_size: str
    travel_frequency: int  # trips per year
    budget_range: Tuple[int, int]
    preferred_airlines: List[str]
    preferred_hotel_chains: List[str]
    seat_preferences: List[str]
    meal_preferences: List[str]
    loyalty_programs: List[str]
    special_requirements: List[str]
    booking_lead_time: int  # days in advance
    flexibility_score: float  # 0-1, willingness to change plans

class TravelSystemAutomation:
    """Main automation class for the Travel Concierge System"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.synthetic_data_file = self.data_dir / "synthetic_training_data.json"
        self.rag_documents_file = self.data_dir / "rag_documents.json"
        self.travel_patterns_file = self.data_dir / "travel_patterns.json"
        
        # Create directories
        self._create_directories()
        
        # Initialize comprehensive data sets
        self._initialize_base_data()

    def _create_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        (self.models_dir / "travel_concierge").mkdir(exist_ok=True)

    def _initialize_base_data(self):
        """Initialize comprehensive base data for generation"""
        
        # Major business destinations with characteristics
        self.destinations = {
            "New York": {"timezone": "EST", "peak_season": ["Sep", "Oct", "Nov"], "business_districts": ["Manhattan Financial", "Midtown"], "avg_hotel_cost": 400},
            "London": {"timezone": "GMT", "peak_season": ["Jun", "Jul", "Aug"], "business_districts": ["City of London", "Canary Wharf"], "avg_hotel_cost": 350},
            "Tokyo": {"timezone": "JST", "peak_season": ["Mar", "Apr", "May"], "business_districts": ["Marunouchi", "Shibuya"], "avg_hotel_cost": 300},
            "Frankfurt": {"timezone": "CET", "peak_season": ["May", "Jun", "Sep"], "business_districts": ["Banking District", "Westend"], "avg_hotel_cost": 250},
            "Singapore": {"timezone": "SGT", "peak_season": ["Feb", "Mar", "Apr"], "business_districts": ["CBD", "Marina Bay"], "avg_hotel_cost": 280},
            "Sydney": {"timezone": "AEST", "peak_season": ["Dec", "Jan", "Feb"], "business_districts": ["CBD", "North Sydney"], "avg_hotel_cost": 320},
            "San Francisco": {"timezone": "PST", "peak_season": ["Sep", "Oct"], "business_districts": ["SOMA", "Financial District"], "avg_hotel_cost": 450},
            "Dubai": {"timezone": "GST", "peak_season": ["Nov", "Dec", "Jan"], "business_districts": ["DIFC", "Business Bay"], "avg_hotel_cost": 200},
            "Toronto": {"timezone": "EST", "peak_season": ["Jul", "Aug", "Sep"], "business_districts": ["Financial District", "King West"], "avg_hotel_cost": 180},
            "Hong Kong": {"timezone": "HKT", "peak_season": ["Oct", "Nov", "Dec"], "business_districts": ["Central", "Admiralty"], "avg_hotel_cost": 300},
            "Mumbai": {"timezone": "IST", "peak_season": ["Nov", "Dec", "Jan"], "business_districts": ["BKC", "Lower Parel"], "avg_hotel_cost": 120},
            "Paris": {"timezone": "CET", "peak_season": ["May", "Jun", "Sep"], "business_districts": ["La Défense", "8th Arrondissement"], "avg_hotel_cost": 380}
        }
        
        # Airlines with their characteristics
        self.airlines = {
            "Delta": {"regions": ["North America", "Europe"], "business_class": True, "loyalty": "SkyMiles", "avg_rating": 4.2},
            "United": {"regions": ["North America", "Asia", "Europe"], "business_class": True, "loyalty": "MileagePlus", "avg_rating": 4.0},
            "American": {"regions": ["North America", "Europe"], "business_class": True, "loyalty": "AAdvantage", "avg_rating": 3.9},
            "British Airways": {"regions": ["Europe", "Asia"], "business_class": True, "loyalty": "Executive Club", "avg_rating": 4.3},
            "Lufthansa": {"regions": ["Europe", "Asia"], "business_class": True, "loyalty": "Miles & More", "avg_rating": 4.4},
            "Emirates": {"regions": ["Middle East", "Asia", "Europe"], "business_class": True, "loyalty": "Skywards", "avg_rating": 4.6},
            "Singapore Airlines": {"regions": ["Asia", "Europe"], "business_class": True, "loyalty": "KrisFlyer", "avg_rating": 4.7},
            "ANA": {"regions": ["Asia", "North America"], "business_class": True, "loyalty": "Mileage Club", "avg_rating": 4.5},
            "Air France": {"regions": ["Europe", "Africa"], "business_class": True, "loyalty": "Flying Blue", "avg_rating": 4.1},
            "KLM": {"regions": ["Europe", "Asia"], "business_class": True, "loyalty": "Flying Blue", "avg_rating": 4.2}
        }
        
        # Hotel chains with characteristics
        self.hotel_chains = {
            "Marriott": {"tiers": ["Courtyard", "Marriott", "Renaissance", "Ritz-Carlton"], "loyalty": "Bonvoy", "business_amenities": True},
            "Hilton": {"tiers": ["Hampton", "Hilton", "Hampton Inn", "DoubleTree", "Conrad"], "loyalty": "Honors", "business_amenities": True},
            "Hyatt": {"tiers": ["Hyatt House", "Hyatt", "Grand Hyatt", "Park Hyatt"], "loyalty": "World of Hyatt", "business_amenities": True},
            "InterContinental": {"tiers": ["Holiday Inn", "InterContinental", "Regent"], "loyalty": "IHG Rewards", "business_amenities": True},
            "Accor": {"tiers": ["Novotel", "Pullman", "Sofitel"], "loyalty": "ALL", "business_amenities": True},
            "Four Seasons": {"tiers": ["Four Seasons"], "loyalty": None, "business_amenities": True},
            "W Hotels": {"tiers": ["W"], "loyalty": "Bonvoy", "business_amenities": True},
            "Le Meridien": {"tiers": ["Le Meridien"], "loyalty": "Bonvoy", "business_amenities": True}
        }
        
        # Traveler profiles
        self.traveler_profiles = [
            TravelerProfile(
                segment="C-Suite Executive",
                seniority_level="Executive",
                company_size="Enterprise",
                travel_frequency=50,
                budget_range=(8000, 15000),
                preferred_airlines=["Emirates", "Singapore Airlines", "British Airways"],
                preferred_hotel_chains=["Four Seasons", "Marriott", "Hyatt"],
                seat_preferences=["First Class", "Business Class"],
                meal_preferences=["Special Dietary", "Business Meal"],
                loyalty_programs=["Emirates Skywards Gold", "Marriott Bonvoy Platinum"],
                special_requirements=["Airport Lounge Access", "Concierge Service", "Late Checkout"],
                booking_lead_time=14,
                flexibility_score=0.3
            ),
            TravelerProfile(
                segment="Road Warrior",
                seniority_level="Senior",
                company_size="Large",
                travel_frequency=40,
                budget_range=(4000, 8000),
                preferred_airlines=["Delta", "United", "American"],
                preferred_hotel_chains=["Marriott", "Hilton", "Hyatt"],
                seat_preferences=["Business Class", "Premium Economy"],
                meal_preferences=["Standard", "Vegetarian"],
                loyalty_programs=["Delta SkyMiles Gold", "Marriott Bonvoy Gold"],
                special_requirements=["WiFi", "24h Business Center", "Express Checkout"],
                booking_lead_time=7,
                flexibility_score=0.6
            ),
            TravelerProfile(
                segment="Frequent Business Traveler",
                seniority_level="Mid-Level",
                company_size="Medium",
                travel_frequency=25,
                budget_range=(2500, 5000),
                preferred_airlines=["Delta", "United", "Lufthansa"],
                preferred_hotel_chains=["Marriott", "Hilton", "InterContinental"],
                seat_preferences=["Premium Economy", "Economy Plus"],
                meal_preferences=["Standard"],
                loyalty_programs=["United MileagePlus Silver", "Hilton Honors Gold"],
                special_requirements=["WiFi", "Fitness Center", "Business Center"],
                booking_lead_time=10,
                flexibility_score=0.5
            ),
            TravelerProfile(
                segment="Occasional Business Traveler",
                seniority_level="Junior",
                company_size="Small",
                travel_frequency=8,
                budget_range=(1500, 3000),
                preferred_airlines=["American", "United", "Delta"],
                preferred_hotel_chains=["Hilton", "Marriott", "InterContinental"],
                seat_preferences=["Economy", "Economy Plus"],
                meal_preferences=["Standard"],
                loyalty_programs=["Basic Memberships"],
                special_requirements=["WiFi", "Continental Breakfast"],
                booking_lead_time=21,
                flexibility_score=0.8
            )
        ]
        
        # Trip purposes with characteristics
        self.trip_purposes = {
            "client_meeting": {"urgency": "high", "flexibility": "low", "duration_range": (1, 3)},
            "conference": {"urgency": "medium", "flexibility": "medium", "duration_range": (2, 5)},
            "training": {"urgency": "low", "flexibility": "high", "duration_range": (3, 7)},
            "sales_presentation": {"urgency": "high", "flexibility": "low", "duration_range": (1, 2)},
            "board_meeting": {"urgency": "high", "flexibility": "very_low", "duration_range": (1, 2)},
            "team_building": {"urgency": "low", "flexibility": "high", "duration_range": (2, 4)},
            "product_launch": {"urgency": "high", "flexibility": "low", "duration_range": (2, 3)},
            "audit": {"urgency": "medium", "flexibility": "medium", "duration_range": (3, 5)},
            "negotiation": {"urgency": "high", "flexibility": "low", "duration_range": (1, 3)},
            "site_visit": {"urgency": "medium", "flexibility": "medium", "duration_range": (1, 4)}
        }

    def generate_synthetic_training_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate comprehensive synthetic training data"""
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        training_data = []
        
        for i in range(num_samples):
            # Select random traveler profile
            profile = random.choice(self.traveler_profiles)
            
            # Generate trip details
            origin = random.choice(list(self.destinations.keys()))
            destination = random.choice([d for d in self.destinations.keys() if d != origin])
            
            # Multi-city trip probability based on traveler segment
            multi_city_prob = 0.7 if profile.segment in ["C-Suite Executive", "Road Warrior"] else 0.3
            is_multi_city = random.random() < multi_city_prob
            
            destinations_list = [destination]
            if is_multi_city:
                additional_cities = random.sample([d for d in self.destinations.keys() if d not in [origin, destination]], 
                                                random.randint(1, 2))
                destinations_list.extend(additional_cities)
            
            # Trip purpose
            purpose = random.choice(list(self.trip_purposes.keys()))
            purpose_info = self.trip_purposes[purpose]
            
            # Trip duration
            duration = random.randint(*purpose_info["duration_range"])
            
            # Departure date
            lead_time_variation = random.randint(-profile.booking_lead_time//2, profile.booking_lead_time//2)
            departure_date = datetime.now() + timedelta(days=profile.booking_lead_time + lead_time_variation)
            
            # Budget calculation based on profile and trip characteristics
            base_budget = random.randint(*profile.budget_range)
            if is_multi_city:
                base_budget *= 1.5
            if purpose_info["urgency"] == "high":
                base_budget *= 1.2
            
            # Generate conversation scenario
            scenario = self._generate_conversation_scenario(
                profile, origin, destinations_list, departure_date, duration, purpose, base_budget
            )
            
            training_data.append(scenario)
            
            if i % 100 == 0:
                logger.info(f"Generated {i} scenarios...")
        
        # Add complex multi-turn conversations
        complex_scenarios = self._generate_complex_scenarios(num_samples // 10)
        training_data.extend(complex_scenarios)
        
        # Save to file
        with open(self.synthetic_data_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        logger.info(f"Generated {len(training_data)} total training examples")
        return training_data

    def _generate_conversation_scenario(self, profile: TravelerProfile, origin: str, 
                                      destinations: List[str], departure_date: datetime,
                                      duration: int, purpose: str, budget: float) -> Dict[str, Any]:
        """Generate a realistic conversation scenario"""
        
        # Natural language queries that users might ask
        query_templates = [
            f"I need to book a trip from {origin} to {', '.join(destinations)} for {purpose}",
            f"Can you help me plan a {duration}-day business trip to {destinations[0]}?",
            f"Looking for flights and hotels for a {purpose} in {destinations[0]} next month",
            f"I have a {purpose} in {destinations[0]} on {departure_date.strftime('%B %d')}. Need full travel arrangement",
            f"Book me a complete trip package: {origin} to {destinations[0]}, budget around ${int(budget)}"
        ]
        
        user_query = random.choice(query_templates)
        
        # Generate contextual constraints
        constraints = []
        if profile.flexibility_score < 0.4:
            constraints.append(f"Must depart after {(departure_date - timedelta(days=1)).strftime('%I %p')} on {departure_date.strftime('%A')}")
        if random.random() < 0.3:
            constraints.append(f"Prefer {random.choice(profile.preferred_airlines)} airlines")
        if random.random() < 0.4:
            constraints.append(f"Hotel must be within 15 minutes of {random.choice(self.destinations[destinations[0]]['business_districts'])}")
        
        # Generate expected response structure
        flight_options = self._generate_flight_options(origin, destinations[0], departure_date, profile)
        hotel_options = self._generate_hotel_options(destinations[0], departure_date, duration, profile)
        
        scenario = {
            "conversation_id": str(uuid.uuid4()),
            "user_profile": asdict(profile),
            "user_query": user_query,
            "constraints": constraints,
            "context": {
                "origin": origin,
                "destinations": destinations,
                "departure_date": departure_date.isoformat(),
                "duration": duration,
                "purpose": purpose,
                "budget": budget,
                "is_multi_city": len(destinations) > 1
            },
            "expected_response": {
                "flights": flight_options,
                "hotels": hotel_options,
                "total_estimate": sum([f["price"] for f in flight_options]) + sum([h["total_price"] for h in hotel_options]),
                "personalization_notes": self._generate_personalization_notes(profile, purpose),
                "alternatives": self._generate_alternatives(profile, budget)
            },
            "conversation_turns": self._generate_multi_turn_conversation(user_query, profile, constraints),
            "metadata": {
                "complexity_score": self._calculate_complexity_score(destinations, constraints, purpose),
                "personalization_factors": len([x for x in [profile.loyalty_programs, profile.special_requirements, constraints] if x]),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        return scenario

    def _generate_flight_options(self, origin: str, destination: str, departure_date: datetime, 
                               profile: TravelerProfile) -> List[Dict[str, Any]]:
        """Generate realistic flight options"""
        options = []
        
        for i in range(3):  # Generate 3 flight options
            airline = random.choice(profile.preferred_airlines[:2] + 
                                  [a for a in self.airlines.keys() if a not in profile.preferred_airlines][:2])
            
            # Price based on class and airline
            base_price = random.randint(800, 2500)
            if profile.seat_preferences[0] == "First Class":
                base_price *= 2.5
            elif profile.seat_preferences[0] == "Business Class":
                base_price *= 1.8
            elif profile.seat_preferences[0] == "Premium Economy":
                base_price *= 1.3
            
            # Add some randomness
            price = base_price + random.randint(-200, 300)
            
            departure_time = departure_date + timedelta(hours=random.randint(6, 22))
            flight_duration = random.randint(2, 12)  # hours
            
            options.append({
                "airline": airline,
                "flight_number": f"{airline[:2].upper()}{random.randint(100, 999)}",
                "departure": departure_time.isoformat(),
                "arrival": (departure_time + timedelta(hours=flight_duration)).isoformat(),
                "duration": f"{flight_duration}h {random.randint(0, 59)}m",
                "price": price,
                "class": profile.seat_preferences[0],
                "aircraft": random.choice(["Boeing 777", "Airbus A350", "Boeing 787", "Airbus A320"]),
                "stops": random.choice([0, 1]) if flight_duration > 8 else 0
            })
        
        return sorted(options, key=lambda x: x["price"])

    def _generate_hotel_options(self, destination: str, check_in: datetime, 
                              duration: int, profile: TravelerProfile) -> List[Dict[str, Any]]:
        """Generate realistic hotel options"""
        options = []
        
        dest_info = self.destinations[destination]
        
        for i in range(3):  # Generate 3 hotel options
            chain = random.choice(profile.preferred_hotel_chains[:2] + 
                                [c for c in self.hotel_chains.keys() if c not in profile.preferred_hotel_chains][:2])
            
            chain_info = self.hotel_chains[chain]
            tier = random.choice(chain_info["tiers"])
            
            # Price based on destination and tier
            base_price = dest_info["avg_hotel_cost"]
            if tier in ["Ritz-Carlton", "Four Seasons", "Park Hyatt", "Conrad"]:
                base_price *= 2
            elif tier in ["Renaissance", "DoubleTree", "Grand Hyatt"]:
                base_price *= 1.5
            
            nightly_rate = base_price + random.randint(-50, 100)
            total_price = nightly_rate * duration
            
            options.append({
                "name": f"{tier} {destination}",
                "chain": chain,
                "location": random.choice(dest_info["business_districts"]),
                "nightly_rate": nightly_rate,
                "total_price": total_price,
                "rating": round(random.uniform(3.8, 4.8), 1),
                "amenities": random.sample(["WiFi", "Business Center", "Gym", "Pool", "Spa", 
                                          "Concierge", "Room Service", "Laundry"], 
                                         random.randint(4, 7)),
                "distance_to_center": f"{random.uniform(0.5, 3.0):.1f} km",
                "check_in": check_in.isoformat(),
                "check_out": (check_in + timedelta(days=duration)).isoformat()
            })
        
        return sorted(options, key=lambda x: x["total_price"])

    def _generate_personalization_notes(self, profile: TravelerProfile, purpose: str) -> List[str]:
        """Generate personalization insights"""
        notes = []
        
        notes.append(f"Prioritized {profile.seat_preferences[0]} seats based on your travel profile")
        
        if profile.loyalty_programs:
            notes.append(f"Selected hotels/airlines to maximize {random.choice(profile.loyalty_programs)} benefits")
        
        if profile.special_requirements:
            notes.append(f"Ensured all accommodations include {random.choice(profile.special_requirements)}")
        
        if purpose in ["client_meeting", "board_meeting"]:
            notes.append("Recommended hotels within 15 minutes of business districts for easy meeting access")
        
        if profile.flexibility_score < 0.5:
            notes.append("Focused on direct flights to minimize travel disruption")
        
        return notes

    def _generate_alternatives(self, profile: TravelerProfile, budget: float) -> List[str]:
        """Generate alternative suggestions"""
        alternatives = []
        
        if budget > profile.budget_range[1] * 0.8:
            alternatives.append("Consider economy class to reduce costs by 40-60%")
        
        alternatives.append("Alternative dates could save up to $300 per flight")
        alternatives.append("Different hotel locations available with 20% savings")
        
        if profile.flexibility_score > 0.6:
            alternatives.append("Flexible booking options available for date changes")
        
        return alternatives

    def _generate_multi_turn_conversation(self, initial_query: str, profile: TravelerProfile, 
                                        constraints: List[str]) -> List[Dict[str, str]]:
        """Generate multi-turn conversation flow"""
        turns = [
            {"role": "user", "content": initial_query},
            {"role": "assistant", "content": "I'd be happy to help you plan this trip. Let me search for the best options based on your preferences."}
        ]
        
        # Add follow-up questions based on profile
        if not constraints:
            turns.extend([
                {"role": "assistant", "content": "Do you have any specific airline preferences or timing requirements?"},
                {"role": "user", "content": f"I prefer {random.choice(profile.preferred_airlines)} if possible, and need to arrive before noon."}
            ])
        
        # Add modification request
        turns.extend([
            {"role": "assistant", "content": "Here are the best options I found for your trip. Would you like me to adjust anything?"},
            {"role": "user", "content": random.choice([
                "Can you show me cheaper alternatives?",
                "I need a hotel closer to the financial district.",
                "Are there any direct flights available?",
                "Can you include airport transfer options?"
            ])}
        ])
        
        return turns

    def _generate_complex_scenarios(self, num_complex: int) -> List[Dict[str, Any]]:
        """Generate complex multi-city, multi-constraint scenarios"""
        logger.info(f"Generating {num_complex} complex scenarios...")
        
        complex_scenarios = []
        
        for i in range(num_complex):
            profile = random.choice([p for p in self.traveler_profiles if p.segment in ["C-Suite Executive", "Road Warrior"]])
            
            # Multi-city trip with 3-4 cities
            origin = random.choice(list(self.destinations.keys()))
            destinations = random.sample([d for d in self.destinations.keys() if d != origin], 3)
            
            # Complex constraints
            constraints = [
                f"Must arrive in {destinations[0]} before 2 PM for same-day meeting",
                f"Need to stay in {destinations[1]} over weekend",
                f"Prefer same hotel chain throughout trip",
                f"Budget not to exceed ${random.randint(8000, 12000)}",
                "Require business class for flights over 6 hours"
            ]
            
            scenario = {
                "conversation_id": str(uuid.uuid4()),
                "complexity_type": "multi_city_complex",
                "user_profile": asdict(profile),
                "user_query": f"I need to plan a complex business trip: {origin} → {' → '.join(destinations)} → {origin} over 10 days",
                "constraints": constraints,
                "context": {
                    "origin": origin,
                    "destinations": destinations,
                    "trip_type": "complex_multi_city",
                    "duration": 10,
                    "purposes": ["client_meeting", "conference", "negotiation"],
                    "budget": random.randint(8000, 12000)
                },
                "conversation_turns": self._generate_complex_conversation(profile, destinations, constraints),
                "metadata": {
                    "complexity_score": 10,
                    "requires_agentic_rag": True,
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            complex_scenarios.append(scenario)
        
        return complex_scenarios

    def _generate_complex_conversation(self, profile: TravelerProfile, destinations: List[str], 
                                     constraints: List[str]) -> List[Dict[str, str]]:
        """Generate complex multi-turn conversation"""
        return [
            {"role": "user", "content": f"I need help with a complex multi-city business trip to {', '.join(destinations)}. This involves multiple meetings and has several constraints."},
            {"role": "assistant", "content": "I'll help you plan this complex itinerary. Let me coordinate between different systems to find the optimal solution."},
            {"role": "user", "content": f"Key constraints: {'; '.join(constraints[:3])}"},
            {"role": "assistant", "content": "I understand the complexity. Let me use my agentic RAG system to coordinate flights, hotels, and meeting schedules across all cities."},
            {"role": "user", "content": "Also, can you check for any visa requirements or travel advisories?"},
            {"role": "assistant", "content": "Absolutely. I'm checking all requirements and will provide a comprehensive travel brief."}
        ]

    def _calculate_complexity_score(self, destinations: List[str], constraints: List[str], purpose: str) -> int:
        """Calculate complexity score for the scenario"""
        score = 1  # Base score
        
        score += len(destinations) - 1  # Multi-city adds complexity
        score += len(constraints)  # Each constraint adds complexity
        
        if purpose in ["board_meeting", "negotiation"]:
            score += 2  # High-stakes meetings
        
        return min(score, 10)  # Cap at 10

    def generate_rag_documents(self, num_documents: int = 500) -> List[Dict[str, Any]]:
        """Generate comprehensive RAG documents"""
        logger.info(f"Generating {num_documents} RAG documents...")
        
        rag_documents = []
        
        # Corporate travel policies
        policy_docs = self._generate_policy_documents()
        rag_documents.extend(policy_docs)
        
        # Destination guides
        destination_docs = self._generate_destination_documents()
        rag_documents.extend(destination_docs)
        
        # Airline and hotel information
        travel_provider_docs = self._generate_provider_documents()
        rag_documents.extend(travel_provider_docs)
        
        # Travel tips and best practices
        tips_docs = self._generate_travel_tips_documents()
        rag_documents.extend(tips_docs)
        
        # Seasonal and event information
        seasonal_docs = self._generate_seasonal_documents()
        rag_documents.extend(seasonal_docs)
        
        # Visa and travel requirements
        requirements_docs = self._generate_requirements_documents()
        rag_documents.extend(requirements_docs)
        
        # Save to file
        with open(self.rag_documents_file, 'w') as f:
            json.dump(rag_documents, f, indent=2, default=str)
        
        logger.info(f"Generated {len(rag_documents)} RAG documents")
        return rag_documents

    def _generate_policy_documents(self) -> List[Dict[str, Any]]:
            """Generate corporate travel policy documents"""
            policies = []
            
            policy_content = [
                {
                    "content": "Executive level travelers (VP and above) are authorized for business class on flights over 6 hours duration. Premium economy is approved for flights 3-6 hours.",
                    "metadata": {"source": "corporate_policy", "type": "flight_class", "applies_to": ["C-Suite Executive", "Road Warrior"]}
                },
                {
                    "content": "Preferred hotel chains include Marriott, Hilton, and Hyatt. Employees should prioritize these chains to maximize corporate rates and loyalty benefits.",
                    "metadata": {"source": "corporate_policy", "type": "accommodation", "cost_savings": "15-25%"}
                },
                {
                    "content": "All international business travel must be booked at least 14 days in advance unless classified as emergency travel. Emergency travel requires VP approval.",
                    "metadata": {"source": "corporate_policy", "type": "booking_requirements", "advance_notice": 14}
                },
                {
                    "content": "Travelers must select airlines with which the company has corporate agreements when available. These include Delta, United, American, British Airways, and Lufthansa.",
                    "metadata": {"source": "corporate_policy", "type": "airline_preference", "corporate_partners": True}
                },
                {
                    "content": "Maximum hotel rate allowances: Tier 1 cities $400/night, Tier 2 cities $300/night, Tier 3 cities $200/night. Exceeding these rates requires manager approval.",
                    "metadata": {"source": "corporate_policy", "type": "hotel_limits", "tier_based": True}
                },
                {
                    "content": "Ground transportation policy: Airport transfers via ride-share or taxi are reimbursable. Rental cars approved for trips over 3 days or when meetings are in multiple locations.",
                    "metadata": {"source": "corporate_policy", "type": "ground_transport", "conditions": ["duration", "meeting_locations"]}
                },
                {
                    "content": "Travel insurance is mandatory for all international trips and domestic trips over $3,000. Company provides coverage through Allianz Global.",
                    "metadata": {"source": "corporate_policy", "type": "insurance", "mandatory_conditions": ["international", "high_value_domestic"]}
                },
                {
                    "content": "Expense reporting must be completed within 30 days of trip completion. All receipts must be retained and uploaded to the expense system.",
                    "metadata": {"source": "corporate_policy", "type": "expense_reporting", "deadline_days": 30}
                }
            ]
            
            for i, policy in enumerate(policy_content):
                policies.append({
                    "id": f"policy_{i+1}",
                    "title": f"Corporate Travel Policy - {policy['metadata']['type'].replace('_', ' ').title()}",
                    "content": policy["content"],
                    "metadata": policy["metadata"],
                    "doc_type": "policy",
                    "created_at": datetime.now().isoformat()
                })
            
            return policies

    def _generate_destination_documents(self) -> List[Dict[str, Any]]:
        """Generate destination guide documents"""
        destination_docs = []
        
        for city, info in self.destinations.items():
            # Main destination guide
            destination_docs.append({
                "id": f"dest_{city.lower().replace(' ', '_')}",
                "title": f"Business Travel Guide - {city}",
                "content": f"{city} is a major business hub with primary business districts in {', '.join(info['business_districts'])}. "
                          f"Peak business season is {', '.join(info['peak_season'])}. Average hotel costs are ${info['avg_hotel_cost']} per night. "
                          f"Timezone: {info['timezone']}. Best areas for business travelers include accommodation near {info['business_districts'][0]} "
                          f"for easy access to corporate offices and meeting venues.",
                "metadata": {
                    "destination": city,
                    "timezone": info["timezone"],
                    "peak_season": info["peak_season"],
                    "avg_hotel_cost": info["avg_hotel_cost"],
                    "business_districts": info["business_districts"]
                },
                "doc_type": "destination_guide",
                "created_at": datetime.now().isoformat()
            })
            
            # Airport and transportation guide
            airport_codes = {
                "New York": "JFK/LGA/EWR", "London": "LHR/LGW/STN", "Tokyo": "NRT/HND",
                "Frankfurt": "FRA", "Singapore": "SIN", "Sydney": "SYD", "San Francisco": "SFO",
                "Dubai": "DXB", "Toronto": "YYZ", "Hong Kong": "HKG", "Mumbai": "BOM", "Paris": "CDG/ORY"
            }
            
            destination_docs.append({
                "id": f"transport_{city.lower().replace(' ', '_')}",
                "title": f"Transportation Guide - {city}",
                "content": f"Airport codes for {city}: {airport_codes.get(city, 'Multiple airports')}. "
                          f"Recommended airport transfer: Express train or taxi depending on traffic patterns. "
                          f"Local business transportation includes metro/subway, taxis, and ride-sharing services. "
                          f"Average airport to city center time: 30-60 minutes depending on traffic and method.",
                "metadata": {
                    "destination": city,
                    "airports": airport_codes.get(city, "Multiple"),
                    "transport_options": ["train", "taxi", "ride_share", "metro"]
                },
                "doc_type": "transportation",
                "created_at": datetime.now().isoformat()
            })
            
            # Business etiquette guide
            etiquette_guides = {
                "Tokyo": "Business cards exchanged with both hands and slight bow. Punctuality extremely important.",
                "London": "Formal business attire expected. Punctuality valued. Meetings often include tea service.",
                "Dubai": "Conservative dress code required. Business meetings may be scheduled around prayer times.",
                "Singapore": "Multicultural environment. English widely spoken. Respect for hierarchy important.",
                "Mumbai": "Relationship building crucial. Meetings may start with personal conversation.",
                "Frankfurt": "Direct communication style appreciated. Punctuality essential. Formal address expected."
            }
            
            if city in etiquette_guides:
                destination_docs.append({
                    "id": f"etiquette_{city.lower().replace(' ', '_')}",
                    "title": f"Business Etiquette - {city}",
                    "content": f"Business etiquette for {city}: {etiquette_guides[city]} Understanding local customs enhances business relationships.",
                    "metadata": {
                        "destination": city,
                        "category": "business_etiquette",
                        "cultural_considerations": True
                    },
                    "doc_type": "cultural_guide",
                    "created_at": datetime.now().isoformat()
                })
        
        return destination_docs

    def _generate_provider_documents(self) -> List[Dict[str, Any]]:
        """Generate airline and hotel provider documents"""
        provider_docs = []
        
        # Airline information
        for airline, info in self.airlines.items():
            provider_docs.append({
                "id": f"airline_{airline.lower().replace(' ', '_')}",
                "title": f"Airline Profile - {airline}",
                "content": f"{airline} operates primarily in {', '.join(info['regions'])} regions. "
                          f"Business class available: {info['business_class']}. "
                          f"Loyalty program: {info.get('loyalty', 'N/A')}. "
                          f"Average customer rating: {info['avg_rating']}/5. "
                          f"Known for reliable service and extensive route network in served regions.",
                "metadata": {
                    "provider": airline,
                    "type": "airline",
                    "regions": info["regions"],
                    "business_class": info["business_class"],
                    "loyalty_program": info.get("loyalty"),
                    "rating": info["avg_rating"]
                },
                "doc_type": "provider_info",
                "created_at": datetime.now().isoformat()
            })
        
        # Hotel chain information
        for chain, info in self.hotel_chains.items():
            provider_docs.append({
                "id": f"hotel_{chain.lower().replace(' ', '_')}",
                "title": f"Hotel Chain Profile - {chain}",
                "content": f"{chain} offers multiple tiers: {', '.join(info['tiers'])}. "
                          f"Loyalty program: {info.get('loyalty', 'None')}. "
                          f"Business amenities available: {info['business_amenities']}. "
                          f"Suitable for business travelers with consistent service standards across locations.",
                "metadata": {
                    "provider": chain,
                    "type": "hotel_chain",
                    "tiers": info["tiers"],
                    "loyalty_program": info.get("loyalty"),
                    "business_amenities": info["business_amenities"]
                },
                "doc_type": "provider_info",
                "created_at": datetime.now().isoformat()
            })
        
        return provider_docs

    def _generate_travel_tips_documents(self) -> List[Dict[str, Any]]:
        """Generate travel tips and best practices documents"""
        tips_docs = []
        
        travel_tips = [
            {
                "title": "Flight Booking Optimization",
                "content": "Book flights 2-8 weeks in advance for best prices. Tuesday and Wednesday departures are typically cheaper. "
                          "Consider nearby airports for potential savings. Use airline apps for real-time updates and mobile boarding passes.",
                "category": "flight_booking",
                "applies_to": "all_travelers"
            },
            {
                "title": "Hotel Selection Strategy",
                "content": "Choose hotels within 15 minutes of business destinations. Prioritize properties with business centers and reliable WiFi. "
                          "Book directly with hotels for room upgrades and flexible cancellation. Consider loyalty program benefits.",
                "category": "hotel_booking",
                "applies_to": "all_travelers"
            },
            {
                "title": "Packing Essentials for Business Travel",
                "content": "Pack one business outfit in carry-on to handle lost luggage. Bring universal adapter and portable chargers. "
                          "Keep important documents in easily accessible travel organizer. Pack wrinkle-resistant fabrics.",
                "category": "packing",
                "applies_to": "all_travelers"
            },
            {
                "title": "Time Zone Management",
                "content": "Adjust sleep schedule 1-2 days before travel. Stay hydrated during flight. Use natural light to reset circadian rhythm. "
                          "Consider melatonin for eastward travel. Schedule important meetings considering jet lag recovery.",
                "category": "jet_lag",
                "applies_to": "international_travelers"
            },
            {
                "title": "Airport Efficiency Tips",
                "content": "Enroll in TSA PreCheck or Global Entry for faster security. Use airline lounges for productive work time. "
                          "Download offline maps and save important phone numbers. Check in online and use mobile boarding passes.",
                "category": "airport_navigation",
                "applies_to": "frequent_travelers"
            },
            {
                "title": "Expense Management",
                "content": "Photograph receipts immediately. Use expense tracking apps. Separate personal and business expenses clearly. "
                          "Submit expense reports promptly. Keep digital and physical backups of important receipts.",
                "category": "expense_tracking",
                "applies_to": "all_travelers"
            },
            {
                "title": "Health and Safety Abroad",
                "content": "Research health requirements and vaccinations. Carry prescription medications in original containers. "
                          "Know emergency contact numbers for destinations. Consider travel health insurance for international trips.",
                "category": "health_safety",
                "applies_to": "international_travelers"
            },
            {
                "title": "Technology for Business Travel",
                "content": "Use VPN for secure internet access. Download offline maps and translation apps. Backup important files to cloud storage. "
                          "Carry portable WiFi hotspot for reliable connectivity. Keep devices charged with portable battery packs.",
                "category": "technology",
                "applies_to": "all_travelers"
            }
        ]
        
        for i, tip in enumerate(travel_tips):
            tips_docs.append({
                "id": f"tip_{i+1}",
                "title": tip["title"],
                "content": tip["content"],
                "metadata": {
                    "category": tip["category"],
                    "applies_to": tip["applies_to"],
                    "tip_type": "best_practice"
                },
                "doc_type": "travel_tip",
                "created_at": datetime.now().isoformat()
            })
        
        return tips_docs

    def _generate_seasonal_documents(self) -> List[Dict[str, Any]]:
        """Generate seasonal and event information documents"""
        seasonal_docs = []
        
        seasonal_info = [
            {
                "title": "Peak Business Travel Seasons",
                "content": "September-November and February-May are peak business travel seasons. Expect higher prices and limited availability. "
                          "Book accommodations 3-4 weeks in advance during peak seasons. Consider shoulder seasons for cost savings.",
                "season": "peak_business",
                "months": ["Sep", "Oct", "Nov", "Feb", "Mar", "Apr", "May"]
            },
            {
                "title": "Holiday Impact on Business Travel",
                "content": "Christmas/New Year (Dec 20-Jan 5), Easter week, and Thanksgiving week see reduced business activity but higher travel costs. "
                          "Many businesses close between Christmas and New Year. Plan accordingly for international destinations.",
                "season": "holidays",
                "impact": "high_cost_low_business"
            },
            {
                "title": "Conference Season Patterns",
                "content": "Major industry conferences typically occur in Spring (Mar-May) and Fall (Sep-Nov). Technology conferences peak in Spring. "
                          "Financial services conferences often in Fall. Book early as conference cities see hotel shortages.",
                "season": "conference",
                "industries": ["technology", "finance", "healthcare", "manufacturing"]
            },
            {
                "title": "Weather Considerations",
                "content": "Winter travel (Dec-Feb) to northern destinations may face weather delays. Summer travel (Jun-Aug) to hot climates requires "
                          "appropriate clothing. Monsoon season in Asia (Jun-Sep) can affect travel plans. Check weather forecasts weekly before travel.",
                "season": "weather_dependent",
                "risk_factors": ["delays", "cancellations", "comfort"]
            }
        ]
        
        for i, info in enumerate(seasonal_info):
            seasonal_docs.append({
                "id": f"seasonal_{i+1}",
                "title": info["title"],
                "content": info["content"],
                "metadata": {
                    "season_type": info["season"],
                    "planning_impact": "high",
                    "booking_advice": True
                },
                "doc_type": "seasonal_guide",
                "created_at": datetime.now().isoformat()
            })
        
        return seasonal_docs

    def _generate_requirements_documents(self) -> List[Dict[str, Any]]:
        """Generate visa and travel requirements documents"""
        requirements_docs = []
        
        visa_requirements = [
            {
                "title": "US Business Visa Requirements",
                "content": "US citizens traveling internationally for business may need visas depending on destination. "
                          "ESTA required for European travel. Business visas typically require invitation letters and proof of business purpose. "
                          "Process times vary from 3 days to 6 weeks. Some countries offer visa on arrival for business travelers.",
                "region": "outbound_us",
                "processing_time": "3_days_to_6_weeks"
            },
            {
                "title": "International Travel to US",
                "content": "Foreign nationals visiting US for business need B-1 visa or visa waiver (ESTA). "
                          "Requires demonstration of business purpose and intent to return home. Processing can take 2-12 weeks. "
                          "Some countries have expedited processing for business travelers.",
                "region": "inbound_us",
                "visa_types": ["B-1", "ESTA"]
            },
            {
                "title": "European Business Travel",
                "content": "Schengen area allows free movement between 26 countries for business. UK requires separate entry. "
                          "Business travelers can stay up to 90 days in 180-day period. Some countries require business registration for extended stays.",
                "region": "europe",
                "duration_limit": "90_days_per_180"
            },
            {
                "title": "Asia Pacific Business Requirements",
                "content": "Visa requirements vary significantly across Asia Pacific. Japan, Singapore, Hong Kong offer visa-free short stays for many nationalities. "
                          "China and India typically require business visas with invitation letters. Processing times 5-15 business days.",
                "region": "asia_pacific",
                "complexity": "high_variation"
            }
        ]
        
        for i, req in enumerate(visa_requirements):
            requirements_docs.append({
                "id": f"visa_req_{i+1}",
                "title": req["title"],
                "content": req["content"],
                "metadata": {
                    "requirement_type": "visa",
                    "region": req["region"],
                    "urgency": "plan_ahead"
                },
                "doc_type": "travel_requirements",
                "created_at": datetime.now().isoformat()
            })
        
        # Health and safety requirements
        health_requirements = [
            {
                "title": "COVID-19 Travel Considerations",
                "content": "Check current COVID-19 requirements for destinations including testing, vaccination proof, and quarantine rules. "
                          "Requirements change frequently. Verify with official government sources before travel. "
                          "Consider travel insurance covering COVID-related cancellations.",
                "category": "health",
                "dynamic": True
            },
            {
                "title": "Business Travel Health Precautions",
                "content": "Carry first aid kit with basic medications. Know location of nearest hospitals in destination cities. "
                          "Consider vaccination requirements for international destinations. Bring extra prescription medications.",
                "category": "health",
                "scope": "general"
            }
        ]
        
        for i, health in enumerate(health_requirements):
            requirements_docs.append({
                "id": f"health_req_{i+1}",
                "title": health["title"],
                "content": health["content"],
                "metadata": {
                    "requirement_type": "health",
                    "category": health["category"],
                    "update_frequency": "regular"
                },
                "doc_type": "health_requirements",
                "created_at": datetime.now().isoformat()
            })
        
        return requirements_docs

    def setup_rag_system(self):
        """Setup RAG system"""
        logger.info("Setting up RAG system...")
        
        with open(self.rag_documents_file, 'r') as f:
            documents = json.load(f)
        
        rag_system = TravelRAGSystem()
        
        for doc in documents:
            success = rag_system.add_document(doc["content"], doc["metadata"])
            if success:
                logger.info(f"Added document: {doc['metadata']['type']}")
        
        logger.info("RAG system setup complete")

    def train_model(self, training_data: List[Dict[str, Any]]):
        """Train the model"""
        logger.info("Starting model training...")
        
        try:
            trainer = TravelModelTrainer()
            trainer.fine_tune(
                training_data=training_data,
                num_epochs=1,
                batch_size=2,
                learning_rate=2e-5
            )
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")

    def create_test_request(self) -> TravelRequest:
        """Create test request"""
        seattle = Location(city="Seattle", country="USA")
        london = Location(city="London", country="UK")
        frankfurt = Location(city="Frankfurt", country="Germany")
        
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
        
        hotel1 = HotelStay(
            location=london,
            check_in=datetime.now() + timedelta(days=1),
            check_out=datetime.now() + timedelta(days=3),
            hotel_name="Marriott London",
            chain="Marriott",
            room_type="Business",
            price_per_night=300.00,
            amenities=["24h Business Center"]
        )
        
        hotel2 = HotelStay(
            location=frankfurt,
            check_in=datetime.now() + timedelta(days=3),
            check_out=datetime.now() + timedelta(days=5),
            hotel_name="Hilton Frankfurt",
            chain="Hilton",
            room_type="Executive",
            price_per_night=250.00,
            amenities=["Business Center"]
        )
        
        segment1 = TripSegment(
            flight=flight1,
            hotel=hotel1,
            purpose=TripPurpose.CLIENT_MEETING,
            meeting_times=[datetime.now() + timedelta(days=2, hours=10)]
        )
        
        segment2 = TripSegment(
            flight=flight2,
            hotel=hotel2,
            purpose=TripPurpose.CONFERENCE,
            meeting_times=[datetime.now() + timedelta(days=4, hours=9)]
        )
        
        preferences = TravelerPreferences(
            preferred_airlines=["Delta"],
            preferred_hotel_chains=["Marriott"],
            seat_preference="aisle",
            requires_24h_business_center=True
        )
        
        return TravelRequest(
            traveler_type=TravelerType.ROAD_WARRIOR,
            preferences=preferences,
            segments=[segment1, segment2],
            budget_constraints={"max_flight_price": 2000.00}
        )

    async def test_system(self):
        """Test the system"""
        logger.info("Testing system...")
        
        try:
            test_request = self.create_test_request()
            planner = TravelPlanner()
            response = await planner.plan_trip(test_request)
            
            logger.info("=== TEST RESULTS ===")
            logger.info(f"Request ID: {response.request_id}")
            logger.info(f"Status: {response.status}")
            logger.info(f"Total Cost: ${response.total_cost:.2f}")
            logger.info(f"Segments: {len(response.itinerary)}")
            
            if response.recommendations:
                logger.info("Recommendations:")
                for rec in response.recommendations:
                    logger.info(f"  - {rec}")
            
            logger.info("Test completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return False

    def run_full_setup(self):
        """Run full setup"""
        logger.info("Starting full setup...")
        
        try:
            training_data = self.generate_synthetic_training_data()
            self.generate_rag_documents()
            self.setup_rag_system()
            
            if os.getenv("TRAIN_MODEL", "false").lower() == "true":
                self.train_model(training_data)
            
            asyncio.run(self.test_system())
            logger.info("Full setup completed!")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            sys.exit(1)

def main():
    """Main entry point"""
    automation = TravelSystemAutomation()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "generate-data":
            automation.generate_synthetic_training_data()
            automation.generate_rag_documents()
        elif command == "setup-rag":
            automation.setup_rag_system()
        elif command == "train-model":
            with open(automation.synthetic_data_file, 'r') as f:
                training_data = json.load(f)
            automation.train_model(training_data)
        elif command == "test":
            asyncio.run(automation.test_system())
        elif command == "full":
            automation.run_full_setup()
        else:
            print("Available commands:")
            print("  generate-data  - Generate synthetic training data and RAG documents")
            print("  setup-rag      - Initialize and populate the RAG system")
            print("  train-model    - Fine-tune the model on training data")
            print("  test           - Test the complete system")
            print("  full           - Run complete setup (data + RAG + test)")
    else:
        # Default: run full setup
        automation.run_full_setup()

if __name__ == "__main__":
    main() 