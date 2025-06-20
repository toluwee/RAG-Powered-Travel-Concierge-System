#!/usr/bin/env python3
"""
Simple test script to demonstrate the evaluation process
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.core.evaluation import TravelSystemEvaluator

async def test_evaluation():
    """Test the evaluation system"""
    print("Testing Travel Concierge System Evaluation...")
    
    try:
        # Initialize evaluator
        evaluator = TravelSystemEvaluator()
        
        print("✓ Evaluator initialized successfully")
        
        # Test RAG evaluation (this should work even without models)
        print("\nTesting RAG System Evaluation...")
        rag_metrics = await evaluator.evaluate_rag_system()
        print(f"✓ RAG Retrieval Accuracy: {rag_metrics.retrieval_accuracy:.3f}")
        print(f"✓ RAG Relevance Score: {rag_metrics.relevance_score:.3f}")
        
        # Test recommendation evaluation
        print("\nTesting Recommendation Evaluation...")
        rec_metrics = await evaluator.evaluate_recommendations()
        print(f"✓ Personalization Score: {rec_metrics.personalization_score:.3f}")
        print(f"✓ Constraint Satisfaction: {rec_metrics.constraint_satisfaction:.3f}")
        
        # Test overall system evaluation
        print("\nTesting Overall System Evaluation...")
        overall_metrics = await evaluator.evaluate_overall_system()
        print(f"✓ End-to-End Accuracy: {overall_metrics.end_to_end_accuracy:.3f}")
        print(f"✓ Success Rate: {overall_metrics.success_rate:.3f}")
        
        # Test model evaluation (may fail if no model is available)
        print("\nTesting Model Evaluation...")
        try:
            model_metrics = await evaluator.evaluate_fine_tuned_model()
            print(f"✓ Model Accuracy: {model_metrics.accuracy:.3f}")
            print(f"✓ Model F1 Score: {model_metrics.f1_score:.3f}")
        except Exception as e:
            print(f"⚠ Model evaluation skipped: {str(e)}")
            print("   (This is normal if no fine-tuned model is available)")
        
        print("\n=== EVALUATION SUMMARY ===")
        print(f"RAG Performance: {'Good' if rag_metrics.retrieval_accuracy > 0.7 else 'Needs Improvement'}")
        print(f"Recommendation Quality: {'Good' if rec_metrics.personalization_score > 0.7 else 'Needs Improvement'}")
        print(f"System Reliability: {'Good' if overall_metrics.success_rate > 0.8 else 'Needs Improvement'}")
        
        print("\n✓ Evaluation test completed successfully!")
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_evaluation())
    sys.exit(0 if success else 1) 