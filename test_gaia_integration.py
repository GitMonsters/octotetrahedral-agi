#!/usr/bin/env python3
"""
Test Suite for Integrated GAIA Solver with Semantic Matching

Tests the complete Phase 2 integration:
1. SemanticAnswerMatcher integration
2. Enhanced question answering with confidence scores
3. Performance metrics and accuracy tracking
4. Learning system recording
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any

# Must be run from the project directory
try:
    from ngvt_gaia_solver import (
        NGVTGAIAOrchestrator,
        GAIAQuestion,
        GAIASolveResult,
    )
    from ngvt_semantic_matcher import SemanticAnswerMatcher
except ImportError as e:
    print(f"Import error (expected if Inspect AI not installed): {e}")
    # Continue anyway for testing semantic matcher


class GAIAIntegrationTest:
    """Test suite for GAIA solver integration"""
    
    def __init__(self):
        self.test_questions: List[GAIAQuestion] = []
        self.results: List[GAIASolveResult] = []
        self.matcher_tests: List[Dict[str, Any]] = []
        
    def setup_test_questions(self):
        """Setup comprehensive test questions covering all levels"""
        self.test_questions = [
            # Level 1: Simple retrieval
            GAIAQuestion(
                question_id="L1_Q1",
                question="What is the capital of France?",
                answer="Paris",
                level=1
            ),
            GAIAQuestion(
                question_id="L1_Q2",
                question="Who wrote Romeo and Juliet?",
                answer="William Shakespeare",
                level=1
            ),
            
            # Level 2: Multi-step reasoning
            GAIAQuestion(
                question_id="L2_Q1",
                question="If a train travels at 60 mph for 2 hours, how far does it travel?",
                answer="120 miles",
                level=2
            ),
            GAIAQuestion(
                question_id="L2_Q2",
                question="What is the largest country in the world by area?",
                answer="Russia",
                level=2
            ),
            
            # Level 3: Complex investigation
            GAIAQuestion(
                question_id="L3_Q1",
                question="Find a research paper about machine learning published in 2023",
                answer="Machine Learning Research Paper 2023",
                level=3
            ),
            GAIAQuestion(
                question_id="L3_Q2",
                question="What are the main components of a neural network?",
                answer="neurons, layers, weights, biases, activation functions",
                level=3
            ),
        ]
    
    def setup_semantic_matcher_tests(self):
        """Setup tests for semantic answer matching"""
        self.matcher_tests = [
            # Exact matches
            {
                "predicted": "Paris",
                "correct": "Paris",
                "expected_match": True,
                "description": "Exact match - case sensitive"
            },
            {
                "predicted": "paris",
                "correct": "PARIS",
                "expected_match": True,
                "description": "Exact match - case insensitive"
            },
            {
                "predicted": "Paris ",
                "correct": " Paris",
                "expected_match": True,
                "description": "Exact match - with whitespace"
            },
            
            # Substring matches
            {
                "predicted": "Based on analysis: Paris",
                "correct": "Paris",
                "expected_match": True,
                "description": "Substring match with prefix"
            },
            {
                "predicted": "The capital of France is Paris",
                "correct": "Paris",
                "expected_match": True,
                "description": "Substring match - predicted contains correct"
            },
            
            # Multi-word exact
            {
                "predicted": "William Shakespeare",
                "correct": "william shakespeare",
                "expected_match": True,
                "description": "Multi-word exact match"
            },
            
            # Partial/fuzzy
            {
                "predicted": "William Shakespare",
                "correct": "William Shakespeare",
                "expected_match": False,  # With threshold 0.75, this should fail
                "description": "Typo in name"
            },
            
            # Wrong answers
            {
                "predicted": "London",
                "correct": "Paris",
                "expected_match": False,
                "description": "Completely different answers"
            },
            {
                "predicted": "",
                "correct": "Paris",
                "expected_match": False,
                "description": "Empty predicted answer"
            },
        ]
    
    async def test_solver_integration(self):
        """Test the integrated solver with semantic matching"""
        print("\n" + "="*80)
        print("GAIA SOLVER INTEGRATION TEST - Semantic Matching")
        print("="*80 + "\n")
        
        try:
            orchestrator = NGVTGAIAOrchestrator(use_semantic_matching=True)
            
            print("Testing solver with semantic matching enabled\n")
            
            for question in self.test_questions:
                result = await orchestrator.solve_question(question)
                self.results.append(result)
                
                print(f"Question ID: {result.question_id}")
                print(f"Level: {question.level}")
                print(f"Question: {result.question[:70]}...")
                print(f"Predicted: {result.predicted_answer[:50]}...")
                print(f"Correct: {result.correct_answer[:50]}...")
                print(f"Result: {'✓ CORRECT' if result.is_correct else '✗ INCORRECT'}")
                print(f"Confidence: {result.confidence:.1%}")
                print(f"Time: {result.solve_time_ms:.1f}ms")
                print()
            
            # Print overall stats
            self._print_solver_stats()
            
        except Exception as e:
            print(f"Error during solver test: {e}")
            import traceback
            traceback.print_exc()
    
    def test_semantic_matcher(self):
        """Test semantic answer matching independently"""
        print("\n" + "="*80)
        print("SEMANTIC ANSWER MATCHER TEST")
        print("="*80 + "\n")
        
        # Test without embeddings first (faster)
        matcher = SemanticAnswerMatcher(use_embeddings=False)
        
        correct_count = 0
        total_count = len(self.matcher_tests)
        
        print("Testing string-based matching (no embeddings):\n")
        
        for test in self.matcher_tests:
            pred = test["predicted"]
            correct = test["correct"]
            expected = test["expected_match"]
            
            is_match, confidence = matcher.match_answers(pred, correct, threshold=0.75)
            
            # Check if result matches expectation
            success = is_match == expected
            if success:
                correct_count += 1
            
            status = "✓" if success else "✗"
            print(f"{status} {test['description']}")
            print(f"   Predicted: '{pred}' vs Correct: '{correct}'")
            print(f"   Match: {is_match}, Confidence: {confidence:.2%}")
            print()
        
        print(f"Matcher Test Results: {correct_count}/{total_count} passed ({correct_count/total_count*100:.1f}%)\n")
        
        # Try with embeddings if available
        print("Testing with embeddings (if available):\n")
        try:
            matcher_semantic = SemanticAnswerMatcher(use_embeddings=True)
            
            if matcher_semantic.use_embeddings:
                print("✓ Embeddings available, testing semantic matching\n")
                
                # Test a few key cases
                semantic_tests = [
                    ("paris is the capital", "Paris"),
                    ("william shakespeare wrote plays", "William Shakespeare"),
                ]
                
                for pred, correct in semantic_tests:
                    is_match, confidence = matcher_semantic.match_answers(
                        pred, correct, threshold=0.7
                    )
                    print(f"Pred: '{pred}' vs Correct: '{correct}'")
                    print(f"   Match: {is_match}, Confidence: {confidence:.2%}\n")
            else:
                print("✗ Embeddings not available, skipping semantic test\n")
                
        except Exception as e:
            print(f"Note: Semantic matching test skipped ({e})\n")
    
    def _print_solver_stats(self):
        """Print solver performance statistics"""
        if not self.results:
            print("No results to analyze")
            return
        
        print("\nPerformance Statistics:")
        print("-" * 60)
        
        correct = sum(1 for r in self.results if r.is_correct)
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0.0
        
        print(f"Total Questions: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # By level
        by_level = {}
        for result in self.results:
            # Extract level from question ID
            level_str = result.question_id.split('_')[0]  # e.g., "L1"
            level = int(level_str[1]) if level_str.startswith('L') else 1
            
            if level not in by_level:
                by_level[level] = {'correct': 0, 'total': 0, 'times': []}
            
            by_level[level]['total'] += 1
            by_level[level]['times'].append(result.solve_time_ms)
            if result.is_correct:
                by_level[level]['correct'] += 1
        
        print("\nBy Difficulty Level:")
        for level in sorted(by_level.keys()):
            stats = by_level[level]
            level_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0.0
            print(f"  Level {level}: {stats['correct']}/{stats['total']} ({level_acc:.1%}) - {avg_time:.1f}ms avg")
        
        avg_confidence = sum(r.confidence for r in self.results) / total if total > 0 else 0.0
        avg_time = sum(r.solve_time_ms for r in self.results) / total if total > 0 else 0.0
        
        print(f"\nAverage Confidence: {avg_confidence:.1%}")
        print(f"Average Solve Time: {avg_time:.1f}ms")
        print(f"Total Execution Time: {sum(r.solve_time_ms for r in self.results):.1f}ms")
        
        print("\n" + "="*80 + "\n")
    
    def generate_report(self):
        """Generate JSON report of test results"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "GAIA Solver Integration with Semantic Matching",
            "matcher_tests": {
                "total": len(self.matcher_tests),
                "tests": self.matcher_tests
            },
            "solver_results": {
                "total_questions": len(self.results),
                "results": [
                    {
                        "question_id": r.question_id,
                        "is_correct": r.is_correct,
                        "confidence": r.confidence,
                        "solve_time_ms": r.solve_time_ms,
                        "models_used": r.models_used,
                        "tools_used": r.tools_used,
                    }
                    for r in self.results
                ]
            }
        }
        
        if self.results:
            correct = sum(1 for r in self.results if r.is_correct)
            report["solver_results"]["accuracy"] = correct / len(self.results)
            report["solver_results"]["correct"] = correct
        
        # Save report
        report_file = f"gaia_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {report_file}")
        return report_file


async def main():
    """Run all integration tests"""
    test_suite = GAIAIntegrationTest()
    
    # Test semantic matcher first
    test_suite.setup_semantic_matcher_tests()
    test_suite.test_semantic_matcher()
    
    # Test solver integration
    test_suite.setup_test_questions()
    
    try:
        await test_suite.test_solver_integration()
    except Exception as e:
        print(f"\nNote: Solver integration test requires Inspect AI: {e}")
        print("This is expected if Inspect AI is not installed.\n")
    
    # Generate report
    test_suite.generate_report()
    
    print("="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
