#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Travel Concierge System
Demonstrates how to evaluate fine-tuned models, RAG system, and overall recommendations
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.core.evaluation import TravelSystemEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationRunner:
    """Runner for comprehensive system evaluation"""
    
    def __init__(self):
        self.evaluator = TravelSystemEvaluator()
        self.results = {}
    
    async def run_model_evaluation(self) -> Dict[str, Any]:
        """Evaluate the fine-tuned model"""
        logger.info("=== FINE-TUNED MODEL EVALUATION ===")
        
        try:
            metrics = await self.evaluator.evaluate_fine_tuned_model()
            
            results = {
                "model_accuracy": metrics.accuracy,
                "model_precision": metrics.precision,
                "model_recall": metrics.recall,
                "model_f1_score": metrics.f1_score,
                "response_time_avg": metrics.response_time_avg,
                "token_efficiency": metrics.token_efficiency,
                "perplexity": metrics.perplexity,
                "loss": metrics.loss
            }
            
            logger.info(f"Model Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"Model F1 Score: {metrics.f1_score:.3f}")
            logger.info(f"Average Response Time: {metrics.response_time_avg:.3f}s")
            logger.info(f"Token Efficiency: {metrics.token_efficiency:.2f} tokens/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    async def run_rag_evaluation(self) -> Dict[str, Any]:
        """Evaluate the RAG system"""
        logger.info("=== RAG SYSTEM EVALUATION ===")
        
        try:
            metrics = await self.evaluator.evaluate_rag_system()
            
            results = {
                "retrieval_accuracy": metrics.retrieval_accuracy,
                "relevance_score": metrics.relevance_score,
                "coverage_score": metrics.coverage_score,
                "response_time_avg": metrics.response_time_avg,
                "document_retrieval_rate": metrics.document_retrieval_rate,
                "context_utilization": metrics.context_utilization
            }
            
            logger.info(f"Retrieval Accuracy: {metrics.retrieval_accuracy:.3f}")
            logger.info(f"Relevance Score: {metrics.relevance_score:.3f}")
            logger.info(f"Coverage Score: {metrics.coverage_score:.3f}")
            logger.info(f"Average Response Time: {metrics.response_time_avg:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"RAG evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    async def run_recommendation_evaluation(self) -> Dict[str, Any]:
        """Evaluate recommendation quality"""
        logger.info("=== RECOMMENDATION EVALUATION ===")
        
        try:
            metrics = await self.evaluator.evaluate_recommendations()
            
            results = {
                "personalization_score": metrics.personalization_score,
                "constraint_satisfaction": metrics.constraint_satisfaction,
                "cost_optimization": metrics.cost_optimization,
                "user_satisfaction_simulated": metrics.user_satisfaction_simulated,
                "diversity_score": metrics.diversity_score,
                "novelty_score": metrics.novelty_score
            }
            
            logger.info(f"Personalization Score: {metrics.personalization_score:.3f}")
            logger.info(f"Constraint Satisfaction: {metrics.constraint_satisfaction:.3f}")
            logger.info(f"Cost Optimization: {metrics.cost_optimization:.3f}")
            logger.info(f"User Satisfaction: {metrics.user_satisfaction_simulated:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Recommendation evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    async def run_overall_system_evaluation(self) -> Dict[str, Any]:
        """Evaluate overall system performance"""
        logger.info("=== OVERALL SYSTEM EVALUATION ===")
        
        try:
            metrics = await self.evaluator.evaluate_overall_system()
            
            results = {
                "end_to_end_accuracy": metrics.end_to_end_accuracy,
                "response_time_total": metrics.response_time_total,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate,
                "user_experience_score": metrics.user_experience_score,
                "system_reliability": metrics.system_reliability
            }
            
            logger.info(f"End-to-End Accuracy: {metrics.end_to_end_accuracy:.3f}")
            logger.info(f"Success Rate: {metrics.success_rate:.3f}")
            logger.info(f"Error Rate: {metrics.error_rate:.3f}")
            logger.info(f"System Reliability: {metrics.system_reliability:.3f}")
            logger.info(f"Total Response Time: {metrics.response_time_total:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Overall system evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation of all components"""
        logger.info("=== COMPREHENSIVE SYSTEM EVALUATION ===")
        
        # Run individual evaluations
        model_results = await self.run_model_evaluation()
        rag_results = await self.run_rag_evaluation()
        recommendation_results = await self.run_recommendation_evaluation()
        overall_results = await self.run_overall_system_evaluation()
        
        # Compile comprehensive results
        comprehensive_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_evaluation": model_results,
            "rag_evaluation": rag_results,
            "recommendation_evaluation": recommendation_results,
            "overall_system_evaluation": overall_results,
            "summary": self._generate_comprehensive_summary(
                model_results, rag_results, recommendation_results, overall_results
            )
        }
        
        # Save results
        results_file = Path("data") / "comprehensive_evaluation_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive evaluation results saved to {results_file}")
        return comprehensive_results
    
    def _generate_comprehensive_summary(self, model_results: Dict[str, Any],
                                      rag_results: Dict[str, Any],
                                      recommendation_results: Dict[str, Any],
                                      overall_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all evaluation results"""
        
        # Calculate component scores
        model_score = model_results.get("model_accuracy", 0.0)
        rag_score = rag_results.get("retrieval_accuracy", 0.0)
        recommendation_score = recommendation_results.get("personalization_score", 0.0)
        overall_score = overall_results.get("end_to_end_accuracy", 0.0)
        
        # Calculate overall system score
        system_score = (model_score + rag_score + recommendation_score + overall_score) / 4
        
        # Identify strengths and areas for improvement
        strengths = []
        areas_for_improvement = []
        
        if model_score > 0.8:
            strengths.append("High model accuracy and performance")
        else:
            areas_for_improvement.append("Model accuracy needs improvement")
        
        if rag_score > 0.8:
            strengths.append("Excellent RAG retrieval performance")
        else:
            areas_for_improvement.append("RAG system needs optimization")
        
        if recommendation_score > 0.8:
            strengths.append("Strong personalization capabilities")
        else:
            areas_for_improvement.append("Personalization needs enhancement")
        
        if overall_score > 0.8:
            strengths.append("High end-to-end system reliability")
        else:
            areas_for_improvement.append("System reliability needs improvement")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            model_results, rag_results, recommendation_results, overall_results
        )
        
        return {
            "overall_system_score": system_score,
            "component_scores": {
                "model": model_score,
                "rag": rag_score,
                "recommendations": recommendation_score,
                "overall_system": overall_score
            },
            "strengths": strengths,
            "areas_for_improvement": areas_for_improvement,
            "recommendations": recommendations,
            "performance_grade": self._calculate_performance_grade(system_score)
        }
    
    def _generate_recommendations(self, model_results: Dict[str, Any],
                                rag_results: Dict[str, Any],
                                recommendation_results: Dict[str, Any],
                                overall_results: Dict[str, Any]) -> list:
        """Generate specific recommendations based on evaluation results"""
        recommendations = []
        
        # Model recommendations
        if model_results.get("model_accuracy", 0.0) < 0.8:
            recommendations.append("Increase training data diversity and quality")
            recommendations.append("Fine-tune model hyperparameters")
            recommendations.append("Consider using a larger base model")
        
        if model_results.get("response_time_avg", 0.0) > 2.0:
            recommendations.append("Optimize model inference performance")
            recommendations.append("Consider model quantization")
        
        # RAG recommendations
        if rag_results.get("retrieval_accuracy", 0.0) < 0.8:
            recommendations.append("Expand RAG document collection")
            recommendations.append("Improve document chunking strategy")
            recommendations.append("Enhance vector embedding quality")
        
        if rag_results.get("relevance_score", 0.0) < 0.8:
            recommendations.append("Improve document relevance scoring")
            recommendations.append("Implement better query preprocessing")
        
        # Recommendation recommendations
        if recommendation_results.get("personalization_score", 0.0) < 0.8:
            recommendations.append("Enhance user preference modeling")
            recommendations.append("Implement more sophisticated personalization algorithms")
            recommendations.append("Add more user preference data")
        
        if recommendation_results.get("constraint_satisfaction", 0.0) < 0.8:
            recommendations.append("Improve constraint handling logic")
            recommendations.append("Add more constraint validation")
        
        # Overall system recommendations
        if overall_results.get("success_rate", 0.0) < 0.9:
            recommendations.append("Improve error handling and fallback mechanisms")
            recommendations.append("Add more comprehensive input validation")
            recommendations.append("Implement better exception handling")
        
        if overall_results.get("response_time_total", 0.0) > 5.0:
            recommendations.append("Optimize system performance and caching")
            recommendations.append("Consider parallel processing")
            recommendations.append("Implement request queuing and prioritization")
        
        return recommendations
    
    def _calculate_performance_grade(self, score: float) -> str:
        """Calculate performance grade based on overall score"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B+ (Good)"
        elif score >= 0.6:
            return "B (Satisfactory)"
        elif score >= 0.5:
            return "C (Needs Improvement)"
        else:
            return "D (Poor)"
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report"""
        print("\n" + "="*80)
        print("TRAVEL CONCIERGE SYSTEM EVALUATION REPORT")
        print("="*80)
        
        summary = results.get("summary", {})
        
        print(f"\nOverall System Score: {summary.get('overall_system_score', 0.0):.3f}")
        print(f"Performance Grade: {summary.get('performance_grade', 'N/A')}")
        
        print("\nComponent Scores:")
        component_scores = summary.get("component_scores", {})
        for component, score in component_scores.items():
            print(f"  {component.title()}: {score:.3f}")
        
        print("\nStrengths:")
        for strength in summary.get("strengths", []):
            print(f"  ✓ {strength}")
        
        print("\nAreas for Improvement:")
        for area in summary.get("areas_for_improvement", []):
            print(f"  ⚠ {area}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(summary.get("recommendations", []), 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)

async def main():
    """Main evaluation function"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        runner = EvaluationRunner()
        
        if command == "model":
            await runner.run_model_evaluation()
        elif command == "rag":
            await runner.run_rag_evaluation()
        elif command == "recommendations":
            await runner.run_recommendation_evaluation()
        elif command == "overall":
            await runner.run_overall_system_evaluation()
        elif command == "comprehensive":
            results = await runner.run_comprehensive_evaluation()
            runner.print_evaluation_report(results)
        else:
            print("Available commands:")
            print("  model           - Evaluate fine-tuned model only")
            print("  rag             - Evaluate RAG system only")
            print("  recommendations - Evaluate recommendations only")
            print("  overall         - Evaluate overall system only")
            print("  comprehensive   - Run comprehensive evaluation (default)")
    else:
        # Default: run comprehensive evaluation
        runner = EvaluationRunner()
        results = await runner.run_comprehensive_evaluation()
        runner.print_evaluation_report(results)

if __name__ == "__main__":
    asyncio.run(main()) 