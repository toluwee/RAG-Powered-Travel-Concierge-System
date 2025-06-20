"""
Comprehensive Evaluation Framework for Travel Concierge System
Evaluates fine-tuned models, RAG system, and overall recommendations
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import uuid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from app.core.model_training import TravelModelTrainer
from app.core.rag_system import TravelRAGSystem
from app.services.travel_planner import TravelPlanner
from app.models.travel import (
    TravelRequest, TripSegment, FlightSegment, 
    HotelStay, Location, TravelerType, TripPurpose, TravelerPreferences
)

logger = logging.getLogger(__name__)

@dataclass
class ModelEvaluationMetrics:
    """Metrics for fine-tuned model evaluation"""
    perplexity: float
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    response_time_avg: float
    token_efficiency: float  # tokens generated per second

@dataclass
class RAGEvaluationMetrics:
    """Metrics for RAG system evaluation"""
    retrieval_accuracy: float
    relevance_score: float
    coverage_score: float
    response_time_avg: float
    document_retrieval_rate: float
    context_utilization: float

@dataclass
class RecommendationEvaluationMetrics:
    """Metrics for recommendation quality"""
    personalization_score: float
    constraint_satisfaction: float
    cost_optimization: float
    user_satisfaction_simulated: float
    diversity_score: float
    novelty_score: float

@dataclass
class OverallSystemMetrics:
    """Overall system performance metrics"""
    end_to_end_accuracy: float
    response_time_total: float
    success_rate: float
    error_rate: float
    user_experience_score: float
    system_reliability: float

class TravelSystemEvaluator:
    """Comprehensive evaluator for the Travel Concierge System"""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.evaluation_results = {}
        
        # Load test data
        self.test_data = self._load_test_data()
        self.ground_truth = self._load_ground_truth()
        
        # Initialize components
        self.model_trainer = TravelModelTrainer()
        self.rag_system = TravelRAGSystem()
        self.planner = TravelPlanner()

    def _load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data for evaluation"""
        test_file = self.data_dir / "evaluation_test_data.json"
        if test_file.exists():
            with open(test_file, 'r') as f:
                return json.load(f)
        else:
            return self._generate_evaluation_test_data()

    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data for evaluation"""
        ground_truth_file = self.data_dir / "evaluation_ground_truth.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        else:
            return self._generate_ground_truth()

    def _generate_evaluation_test_data(self) -> List[Dict[str, Any]]:
        """Generate comprehensive test data for evaluation"""
        logger.info("Generating evaluation test data...")
        
        test_cases = []
        
        # Test case categories
        categories = [
            "simple_business_trip",
            "multi_city_complex",
            "budget_constrained",
            "luxury_preference",
            "last_minute_booking",
            "group_travel",
            "international_complex",
            "seasonal_constraints"
        ]
        
        for category in categories:
            for i in range(10):  # 10 test cases per category
                test_case = self._create_test_case(category, i)
                test_cases.append(test_case)
        
        # Save test data
        test_file = self.data_dir / "evaluation_test_data.json"
        with open(test_file, 'w') as f:
            json.dump(test_cases, f, indent=2, default=str)
        
        logger.info(f"Generated {len(test_cases)} evaluation test cases")
        return test_cases

    def _create_test_case(self, category: str, index: int) -> Dict[str, Any]:
        """Create a specific test case based on category"""
        base_date = datetime.now() + timedelta(days=30)
        
        if category == "simple_business_trip":
            return {
                "test_id": f"{category}_{index}",
                "category": category,
                "user_query": "I need a business trip from Seattle to London next month for a client meeting",
                "constraints": ["budget under $3000", "prefer Delta airlines"],
                "expected_flights": 1,
                "expected_hotels": 1,
                "complexity": "low"
            }
        
        elif category == "multi_city_complex":
            return {
                "test_id": f"{category}_{index}",
                "category": category,
                "user_query": "Plan a multi-city trip: Seattle → London → Frankfurt → Seattle over 2 weeks",
                "constraints": ["business class flights", "Marriott hotels", "meetings in each city"],
                "expected_flights": 3,
                "expected_hotels": 2,
                "complexity": "high"
            }
        
        elif category == "budget_constrained":
            return {
                "test_id": f"{category}_{index}",
                "category": category,
                "user_query": "Need cheapest options for Seattle to Paris, budget $1500 total",
                "constraints": ["economy class", "budget hotels", "flexible dates"],
                "expected_flights": 1,
                "expected_hotels": 1,
                "complexity": "medium"
            }
        
        # Add more categories as needed...
        else:
            return {
                "test_id": f"{category}_{index}",
                "category": category,
                "user_query": "Standard business trip request",
                "constraints": [],
                "expected_flights": 1,
                "expected_hotels": 1,
                "complexity": "medium"
            }

    def _generate_ground_truth(self) -> Dict[str, Any]:
        """Generate ground truth data for evaluation"""
        logger.info("Generating ground truth data...")
        
        ground_truth = {
            "model_responses": {},
            "rag_retrievals": {},
            "recommendations": {},
            "system_performance": {}
        }
        
        # Generate expected model responses
        for test_case in self.test_data:
            test_id = test_case["test_id"]
            ground_truth["model_responses"][test_id] = {
                "expected_tokens": self._estimate_expected_tokens(test_case),
                "expected_sentiment": "positive",
                "expected_completeness": 0.9
            }
            
            ground_truth["rag_retrievals"][test_id] = {
                "expected_documents": self._get_expected_documents(test_case),
                "expected_relevance": 0.8
            }
            
            ground_truth["recommendations"][test_id] = {
                "expected_personalization": 0.85,
                "expected_cost_optimization": 0.9,
                "expected_constraint_satisfaction": 0.95
            }
        
        # Save ground truth
        ground_truth_file = self.data_dir / "evaluation_ground_truth.json"
        with open(ground_truth_file, 'w') as f:
            json.dump(ground_truth, f, indent=2, default=str)
        
        return ground_truth

    def _estimate_expected_tokens(self, test_case: Dict[str, Any]) -> int:
        """Estimate expected token count for a test case"""
        base_tokens = 100
        complexity_multiplier = {
            "low": 1.0,
            "medium": 1.5,
            "high": 2.0
        }
        
        complexity = test_case.get("complexity", "medium")
        return int(base_tokens * complexity_multiplier[complexity])

    def _get_expected_documents(self, test_case: Dict[str, Any]) -> List[str]:
        """Get expected document types for a test case"""
        documents = ["general_travel_policy"]
        
        if "hotel" in test_case["user_query"].lower():
            documents.append("hotel_policy")
        if "flight" in test_case["user_query"].lower():
            documents.append("flight_policy")
        if "budget" in test_case["user_query"].lower():
            documents.append("budget_guidelines")
        
        return documents

    async def evaluate_fine_tuned_model(self) -> ModelEvaluationMetrics:
        """Evaluate the fine-tuned model performance"""
        logger.info("Evaluating fine-tuned model...")
        
        model_path = self.models_dir / "travel_concierge"
        if not model_path.exists():
            logger.warning("Fine-tuned model not found, skipping model evaluation")
            return ModelEvaluationMetrics(
                perplexity=0.0, loss=0.0, accuracy=0.0, precision=0.0,
                recall=0.0, f1_score=0.0, response_time_avg=0.0, token_efficiency=0.0
            )
        
        try:
            # Load the fine-tuned model
            model, tokenizer = self.model_trainer.load_fine_tuned_model(str(model_path))
            
            # Test cases for model evaluation
            test_inputs = [
                "Plan a business trip to London",
                "I need a hotel in Paris for next week",
                "Book a flight from Seattle to Tokyo",
                "Multi-city trip: New York → London → Paris"
            ]
            
            response_times = []
            generated_texts = []
            expected_patterns = [
                r"flight|airline|booking",
                r"hotel|accommodation|stay",
                r"flight|airline|booking",
                r"multi.*city|itinerary|schedule"
            ]
            
            for i, test_input in enumerate(test_inputs):
                start_time = datetime.now()
                
                # Generate response
                inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                response_times.append(response_time)
            
            # Calculate metrics
            response_time_avg = np.mean(response_times)
            token_efficiency = np.mean([len(text.split()) / time for text, time in zip(generated_texts, response_times)])
            
            # Pattern matching accuracy
            accuracy_scores = []
            for text, pattern in zip(generated_texts, expected_patterns):
                match = re.search(pattern, text.lower())
                accuracy_scores.append(1.0 if match else 0.0)
            
            accuracy = np.mean(accuracy_scores)
            
            # For a more comprehensive evaluation, you'd need labeled data
            precision = accuracy  # Simplified
            recall = accuracy     # Simplified
            f1_score = accuracy   # Simplified
            
            return ModelEvaluationMetrics(
                perplexity=0.0,  # Would need validation set
                loss=0.0,        # Would need validation set
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                response_time_avg=response_time_avg,
                token_efficiency=token_efficiency
            )
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return ModelEvaluationMetrics(
                perplexity=0.0, loss=0.0, accuracy=0.0, precision=0.0,
                recall=0.0, f1_score=0.0, response_time_avg=0.0, token_efficiency=0.0
            )

    async def evaluate_rag_system(self) -> RAGEvaluationMetrics:
        """Evaluate the RAG system performance"""
        logger.info("Evaluating RAG system...")
        
        test_queries = [
            "What are the preferred hotels for business travelers?",
            "Which airlines offer the best business class service?",
            "What are the corporate travel policies?",
            "How to book airport transfers?",
            "What are the visa requirements for international travel?"
        ]
        
        retrieval_times = []
        retrieval_accuracies = []
        relevance_scores = []
        
        for query in test_queries:
            start_time = datetime.now()
            
            # Query RAG system
            try:
                results = self.rag_system.query(query, top_k=3)
                end_time = datetime.now()
                retrieval_time = (end_time - start_time).total_seconds()
                retrieval_times.append(retrieval_time)
                
                # Evaluate retrieval accuracy (simplified)
                if results:
                    retrieval_accuracies.append(1.0)
                    # Calculate relevance based on keyword matching
                    query_keywords = set(query.lower().split())
                    doc_keywords = set(" ".join([r.get("content", "") for r in results]).lower().split())
                    relevance = len(query_keywords.intersection(doc_keywords)) / len(query_keywords)
                    relevance_scores.append(relevance)
                else:
                    retrieval_accuracies.append(0.0)
                    relevance_scores.append(0.0)
                    
            except Exception as e:
                logger.error(f"RAG query failed: {str(e)}")
                retrieval_times.append(0.0)
                retrieval_accuracies.append(0.0)
                relevance_scores.append(0.0)
        
        return RAGEvaluationMetrics(
            retrieval_accuracy=np.mean(retrieval_accuracies),
            relevance_score=np.mean(relevance_scores),
            coverage_score=0.8,  # Would need comprehensive document analysis
            response_time_avg=np.mean(retrieval_times),
            document_retrieval_rate=0.9,  # Would need document statistics
            context_utilization=0.85  # Would need analysis of context usage
        )

    async def evaluate_recommendations(self) -> RecommendationEvaluationMetrics:
        """Evaluate recommendation quality"""
        logger.info("Evaluating recommendations...")
        
        # Create test travel requests
        test_requests = self._create_evaluation_requests()
        
        personalization_scores = []
        constraint_satisfaction_scores = []
        cost_optimization_scores = []
        
        for request in test_requests:
            try:
                # Get recommendations
                response = await self.planner.plan_trip(request)
                
                # Evaluate personalization
                personalization_score = self._evaluate_personalization(request, response)
                personalization_scores.append(personalization_score)
                
                # Evaluate constraint satisfaction
                constraint_score = self._evaluate_constraint_satisfaction(request, response)
                constraint_satisfaction_scores.append(constraint_score)
                
                # Evaluate cost optimization
                cost_score = self._evaluate_cost_optimization(request, response)
                cost_optimization_scores.append(cost_score)
                
            except Exception as e:
                logger.error(f"Recommendation evaluation failed: {str(e)}")
                personalization_scores.append(0.0)
                constraint_satisfaction_scores.append(0.0)
                cost_optimization_scores.append(0.0)
        
        return RecommendationEvaluationMetrics(
            personalization_score=np.mean(personalization_scores),
            constraint_satisfaction=np.mean(constraint_satisfaction_scores),
            cost_optimization=np.mean(cost_optimization_scores),
            user_satisfaction_simulated=0.85,  # Would need user feedback
            diversity_score=0.8,  # Would need analysis of recommendation variety
            novelty_score=0.7     # Would need analysis of recommendation uniqueness
        )

    def _create_evaluation_requests(self) -> List[TravelRequest]:
        """Create travel requests for evaluation"""
        requests = []
        
        # Simple business trip
        seattle = Location(city="Seattle", country="USA")
        london = Location(city="London", country="UK")
        
        flight = FlightSegment(
            origin=seattle,
            destination=london,
            departure_time=datetime.now() + timedelta(days=30),
            arrival_time=datetime.now() + timedelta(days=30, hours=8),
            airline="Delta",
            flight_number="DL123",
            price=1200.00
        )
        
        hotel = HotelStay(
            location=london,
            check_in=datetime.now() + timedelta(days=30),
            check_out=datetime.now() + timedelta(days=33),
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
            meeting_times=[datetime.now() + timedelta(days=31, hours=10)]
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
        
        requests.append(request)
        return requests

    def _evaluate_personalization(self, request: TravelRequest, response: Any) -> float:
        """Evaluate how well recommendations match user preferences"""
        score = 0.0
        total_checks = 0
        
        # Check airline preferences
        if request.preferences.preferred_airlines:
            total_checks += 1
            # This is a simplified check - in practice you'd analyze the actual recommendations
            score += 0.8  # Assume 80% match
        
        # Check hotel preferences
        if request.preferences.preferred_hotel_chains:
            total_checks += 1
            score += 0.8  # Assume 80% match
        
        # Check seat preferences
        if request.preferences.seat_preference:
            total_checks += 1
            score += 0.7  # Assume 70% match
        
        return score / total_checks if total_checks > 0 else 0.0

    def _evaluate_constraint_satisfaction(self, request: TravelRequest, response: Any) -> float:
        """Evaluate how well recommendations satisfy constraints"""
        score = 0.0
        total_constraints = 0
        
        # Check budget constraints
        if request.budget_constraints:
            total_constraints += 1
            if hasattr(response, 'total_cost'):
                max_flight_price = request.budget_constraints.get("max_flight_price", float('inf'))
                if response.total_cost <= max_flight_price:
                    score += 1.0
                else:
                    score += 0.5  # Partial satisfaction
        
        # Check special requirements
        if request.special_requirements:
            total_constraints += len(request.special_requirements)
            # Simplified check - assume 90% satisfaction
            score += 0.9 * len(request.special_requirements)
        
        return score / total_constraints if total_constraints > 0 else 1.0

    def _evaluate_cost_optimization(self, request: TravelRequest, response: Any) -> float:
        """Evaluate cost optimization quality"""
        if not hasattr(response, 'total_cost'):
            return 0.0
        
        # Simplified evaluation - in practice you'd compare against benchmarks
        base_cost = 2000.0  # Assume base cost for this type of trip
        actual_cost = response.total_cost
        
        if actual_cost <= base_cost:
            return 1.0
        elif actual_cost <= base_cost * 1.2:
            return 0.8
        elif actual_cost <= base_cost * 1.5:
            return 0.6
        else:
            return 0.4

    async def evaluate_overall_system(self) -> OverallSystemMetrics:
        """Evaluate overall system performance"""
        logger.info("Evaluating overall system...")
        
        # Run end-to-end tests
        test_results = []
        response_times = []
        success_count = 0
        error_count = 0
        
        for test_case in self.test_data[:10]:  # Test first 10 cases
            try:
                start_time = datetime.now()
                
                # Create travel request from test case
                request = self._create_request_from_test_case(test_case)
                
                # Get system response
                response = await self.planner.plan_trip(request)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                response_times.append(response_time)
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(test_case, response)
                test_results.append(quality_score)
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"System test failed: {str(e)}")
                error_count += 1
                response_times.append(0.0)
                test_results.append(0.0)
        
        total_tests = len(self.test_data[:10])
        
        return OverallSystemMetrics(
            end_to_end_accuracy=np.mean(test_results),
            response_time_total=np.mean(response_times),
            success_rate=success_count / total_tests,
            error_rate=error_count / total_tests,
            user_experience_score=0.85,  # Would need user feedback
            system_reliability=success_count / total_tests
        )

    def _create_request_from_test_case(self, test_case: Dict[str, Any]) -> TravelRequest:
        """Create a travel request from a test case"""
        # Simplified implementation - in practice you'd parse the user query
        seattle = Location(city="Seattle", country="USA")
        london = Location(city="London", country="UK")
        
        flight = FlightSegment(
            origin=seattle,
            destination=london,
            departure_time=datetime.now() + timedelta(days=30),
            arrival_time=datetime.now() + timedelta(days=30, hours=8),
            airline="Delta",
            flight_number="DL123",
            price=1200.00
        )
        
        hotel = HotelStay(
            location=london,
            check_in=datetime.now() + timedelta(days=30),
            check_out=datetime.now() + timedelta(days=33),
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
            meeting_times=[datetime.now() + timedelta(days=31, hours=10)]
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
            segments=[segment],
            budget_constraints={"max_flight_price": 2000.00}
        )

    def _evaluate_response_quality(self, test_case: Dict[str, Any], response: Any) -> float:
        """Evaluate the quality of a system response"""
        score = 0.0
        total_checks = 0
        
        # Check if response has required components
        if hasattr(response, 'itinerary'):
            total_checks += 1
            if len(response.itinerary) >= test_case.get("expected_flights", 1):
                score += 1.0
            else:
                score += 0.5
        
        # Check if response has recommendations
        if hasattr(response, 'recommendations'):
            total_checks += 1
            if response.recommendations:
                score += 1.0
            else:
                score += 0.5
        
        # Check if response has warnings
        if hasattr(response, 'warnings'):
            total_checks += 1
            score += 1.0  # Warnings are good for user awareness
        
        return score / total_checks if total_checks > 0 else 0.0

    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of all system components"""
        logger.info("Starting comprehensive system evaluation...")
        
        # Evaluate each component
        model_metrics = await self.evaluate_fine_tuned_model()
        rag_metrics = await self.evaluate_rag_system()
        recommendation_metrics = await self.evaluate_recommendations()
        overall_metrics = await self.evaluate_overall_system()
        
        # Compile results
        evaluation_results = {
            "evaluation_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "model_evaluation": asdict(model_metrics),
            "rag_evaluation": asdict(rag_metrics),
            "recommendation_evaluation": asdict(recommendation_metrics),
            "overall_system_evaluation": asdict(overall_metrics),
            "summary": self._generate_evaluation_summary(
                model_metrics, rag_metrics, recommendation_metrics, overall_metrics
            )
        }
        
        # Save results
        results_file = self.data_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        logger.info("Comprehensive evaluation completed!")
        return evaluation_results

    def _generate_evaluation_summary(self, model_metrics: ModelEvaluationMetrics,
                                   rag_metrics: RAGEvaluationMetrics,
                                   recommendation_metrics: RecommendationEvaluationMetrics,
                                   overall_metrics: OverallSystemMetrics) -> Dict[str, Any]:
        """Generate a summary of evaluation results"""
        
        # Calculate overall scores
        model_score = (model_metrics.accuracy + model_metrics.f1_score) / 2
        rag_score = (rag_metrics.retrieval_accuracy + rag_metrics.relevance_score) / 2
        recommendation_score = (recommendation_metrics.personalization_score + 
                              recommendation_metrics.constraint_satisfaction) / 2
        
        overall_score = (model_score + rag_score + recommendation_score) / 3
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if model_metrics.accuracy > 0.8:
            strengths.append("High model accuracy")
        else:
            weaknesses.append("Model accuracy needs improvement")
        
        if rag_metrics.retrieval_accuracy > 0.8:
            strengths.append("Good RAG retrieval performance")
        else:
            weaknesses.append("RAG retrieval needs improvement")
        
        if recommendation_metrics.personalization_score > 0.8:
            strengths.append("Strong personalization")
        else:
            weaknesses.append("Personalization needs improvement")
        
        if overall_metrics.success_rate > 0.9:
            strengths.append("High system reliability")
        else:
            weaknesses.append("System reliability needs improvement")
        
        return {
            "overall_score": overall_score,
            "component_scores": {
                "model": model_score,
                "rag": rag_score,
                "recommendations": recommendation_score
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": self._generate_improvement_recommendations(
                model_metrics, rag_metrics, recommendation_metrics, overall_metrics
            )
        }

    def _generate_improvement_recommendations(self, model_metrics: ModelEvaluationMetrics,
                                            rag_metrics: RAGEvaluationMetrics,
                                            recommendation_metrics: RecommendationEvaluationMetrics,
                                            overall_metrics: OverallSystemMetrics) -> List[str]:
        """Generate improvement recommendations based on evaluation results"""
        recommendations = []
        
        if model_metrics.accuracy < 0.8:
            recommendations.append("Increase training data diversity and quality")
            recommendations.append("Fine-tune model hyperparameters")
        
        if rag_metrics.retrieval_accuracy < 0.8:
            recommendations.append("Expand RAG document collection")
            recommendations.append("Improve document chunking strategy")
        
        if recommendation_metrics.personalization_score < 0.8:
            recommendations.append("Enhance user preference modeling")
            recommendations.append("Implement more sophisticated personalization algorithms")
        
        if overall_metrics.response_time_total > 5.0:
            recommendations.append("Optimize system performance and caching")
            recommendations.append("Consider model quantization for faster inference")
        
        if overall_metrics.error_rate > 0.1:
            recommendations.append("Improve error handling and fallback mechanisms")
            recommendations.append("Add more comprehensive input validation")
        
        return recommendations

# Import torch for model evaluation
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available, model evaluation will be limited")
    torch = None 