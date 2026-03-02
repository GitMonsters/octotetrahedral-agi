#!/usr/bin/env python3
"""
Phase 3: Official GAIA Benchmark Evaluation Script

Runs NGVT GAIA Solver against official GAIA benchmark from UK Government
Inspect AI framework with semantic answer matching.

Usage:
    export HF_TOKEN='your_token'
    python phase3_evaluation.py --limit 10  # Test with 10 questions
    python phase3_evaluation.py --full      # Run full 450 questions
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import NGVT components
from ngvt_gaia_solver import (
    NGVTGAIAOrchestrator,
    GAIAQuestion,
    GAIASolveResult,
)
from ngvt_semantic_matcher import SemanticAnswerMatcher

# Check if HF token is available for official GAIA benchmark
import os
HF_TOKEN = os.environ.get('HF_TOKEN', '')
OFFICIAL_GAIA_AVAILABLE = bool(HF_TOKEN)
if not OFFICIAL_GAIA_AVAILABLE:
    print("⚠ HF_TOKEN not set - using mock data only. Set HF_TOKEN to access official GAIA benchmark.")


class Phase3Evaluator:
    """Main evaluator for Phase 3 official benchmark evaluation"""
    
    def __init__(self, use_official_benchmark: bool = True):
        self.use_official = use_official_benchmark and OFFICIAL_GAIA_AVAILABLE
        self.orchestrator = NGVTGAIAOrchestrator(use_semantic_matching=True)
        self.results: List[GAIASolveResult] = []
        self.question_levels: Dict[str, int] = {}  # question_id -> level
        self.start_time = None
        self.end_time = None
        
    async def run_evaluation(
        self,
        limit: Optional[int] = None,
        level: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run evaluation against official GAIA benchmark or mock data
        
        Args:
            limit: Max number of questions to evaluate
            level: Specific level (1, 2, or 3) or None for all
            verbose: Print progress updates
        
        Returns:
            Dict with evaluation results and metrics
        """
        self.start_time = datetime.now()
        
        if verbose:
            print("\n" + "="*80)
            print("PHASE 3: OFFICIAL GAIA BENCHMARK EVALUATION")
            print("="*80 + "\n")
            
            if self.use_official:
                print("Mode: OFFICIAL Benchmark (HuggingFace GAIA dataset)")
            else:
                print("Mode: MOCK Data (fallback for testing)")
            
            if level:
                print(f"Level: {level} only")
            if limit:
                print(f"Limit: {limit} questions")
            print()
        
        # Get questions
        questions = await self._get_questions(limit=limit, level=level)
        
        if verbose:
            print(f"Loaded {len(questions)} questions\n")
        
        # Run evaluation
        for i, question in enumerate(questions, 1):
            if verbose and i % 10 == 1:
                print(f"Progress: {i}/{len(questions)}...", end="\r", flush=True)
            
            self.question_levels[question.question_id] = question.level
            result = await self.orchestrator.solve_question(question)
            self.results.append(result)
        
        if verbose:
            print(f"Progress: {len(questions)}/{len(questions)} - Complete!   \n")
        
        self.end_time = datetime.now()
        
        # Generate report
        report = self._generate_report()
        
        if verbose:
            self._print_report(report)
        
        return report
    
    async def _get_questions(
        self,
        limit: Optional[int] = None,
        level: Optional[int] = None
    ) -> List[GAIAQuestion]:
        """Get questions from official benchmark or mock data"""
        
        if self.use_official:
            return await self._get_official_questions(limit=limit, level=level)
        else:
            return self._get_mock_questions(limit=limit, level=level)
    
    async def _get_official_questions(
        self,
        limit: Optional[int] = None,
        level: Optional[int] = None
    ) -> List[GAIAQuestion]:
        """Load questions from official HuggingFace GAIA dataset"""
        
        try:
            from datasets import load_dataset
            
            print("Loading official GAIA dataset from HuggingFace...")
            
            # Load from HuggingFace with token and correct config
            dataset = load_dataset(
                'gaia-benchmark/GAIA',
                '2023_all',
                split='validation',
                token=HF_TOKEN
            )
            
            # Download file attachments from HuggingFace
            from huggingface_hub import hf_hub_download
            
            questions = []
            for item in dataset:
                # Use correct field names from GAIA dataset
                question_text = item.get('Question', '')
                file_name = item.get('file_name', '')
                file_path_field = item.get('file_path', '')
                answer = item.get('Final answer', '')
                level_num = item.get('Level', 1)
                question_id = item.get('task_id', f"q_{len(questions)}")
                
                # Download file attachment if present
                local_file_path = None
                if file_name and file_path_field:
                    try:
                        local_file_path = hf_hub_download(
                            repo_id='gaia-benchmark/GAIA',
                            filename=file_path_field,
                            repo_type='dataset',
                            token=HF_TOKEN
                        )
                    except Exception as e:
                        print(f"  Warning: Could not download file {file_name}: {e}")
                        local_file_path = file_name  # Fallback to bare filename
                
                q = GAIAQuestion(
                    question_id=str(question_id),
                    question=question_text,
                    answer=answer,
                    file=local_file_path,
                    level=int(level_num)  # Level comes as string from dataset
                )
                questions.append(q)
            
            print(f"Loaded {len(questions)} questions from official GAIA dataset")
            
            # Filter by level if specified
            if level:
                questions = [q for q in questions if q.level == level]
            
            # Apply limit
            if limit:
                questions = questions[:limit]
            
            return questions
            
        except Exception as e:
            print(f"⚠ Error loading official dataset: {e}")
            print("Falling back to mock data...\n")
            self.use_official = False
            return self._get_mock_questions(limit=limit, level=level)
    
    def _get_mock_questions(
        self,
        limit: Optional[int] = None,
        level: Optional[int] = None
    ) -> List[GAIAQuestion]:
        """Get mock questions for testing"""
        
        all_questions = [
            # Level 1 - Simple Retrieval
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
            GAIAQuestion(
                question_id="L1_Q3",
                question="What is the largest planet in our solar system?",
                answer="Jupiter",
                level=1
            ),
            GAIAQuestion(
                question_id="L1_Q4",
                question="In what year did the Titanic sink?",
                answer="1912",
                level=1
            ),
            GAIAQuestion(
                question_id="L1_Q5",
                question="What is the chemical symbol for gold?",
                answer="Au",
                level=1
            ),
            
            # Level 2 - Multi-step Reasoning
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
            GAIAQuestion(
                question_id="L2_Q3",
                question="How many sides does a hexagon have?",
                answer="6",
                level=2
            ),
            GAIAQuestion(
                question_id="L2_Q4",
                question="What is the square root of 144?",
                answer="12",
                level=2
            ),
            GAIAQuestion(
                question_id="L2_Q5",
                question="Which planet is known as the Red Planet?",
                answer="Mars",
                level=2
            ),
            
            # Level 3 - Complex Investigation
            GAIAQuestion(
                question_id="L3_Q1",
                question="What are the main components of a neural network?",
                answer="neurons, layers, weights, biases, activation functions",
                level=3
            ),
            GAIAQuestion(
                question_id="L3_Q2",
                question="Explain the concept of entropy in thermodynamics",
                answer="measure of disorder or randomness in a system",
                level=3
            ),
        ]
        
        # Filter by level if specified
        if level:
            all_questions = [q for q in all_questions if q.level == level]
        
        # Apply limit
        if limit:
            all_questions = all_questions[:limit]
        
        return all_questions
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        if not self.results:
            return {'error': 'No results to report'}
        
        # Calculate metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0
        
        # By level
        by_level = {}
        for result in self.results:
            level = self.question_levels.get(result.question_id, 1)
            if level not in by_level:
                by_level[level] = {'correct': 0, 'total': 0, 'times': []}
            by_level[level]['total'] += 1
            by_level[level]['times'].append(result.solve_time_ms)
            if result.is_correct:
                by_level[level]['correct'] += 1
        
        # Time metrics
        elapsed = (self.end_time - self.start_time).total_seconds()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'official' if self.use_official else 'mock',
            'total_questions': total,
            'correct': correct,
            'accuracy': accuracy,
            'accuracy_percentage': f"{accuracy*100:.1f}%",
            'total_time_seconds': elapsed,
            'avg_time_per_question_ms': (elapsed / total * 1000) if total > 0 else 0,
            'throughput_qps': total / elapsed if elapsed > 0 else 0,
            'by_level': {
                k: {
                    'correct': v['correct'],
                    'total': v['total'],
                    'accuracy': f"{v['correct']/v['total']*100:.1f}%" if v['total'] > 0 else "0%",
                    'avg_time_ms': sum(v['times']) / len(v['times']) if v['times'] else 0
                }
                for k, v in sorted(by_level.items())
            },
            'avg_confidence': sum(r.confidence for r in self.results) / total if total > 0 else 0,
            'results': [
                {
                    'question_id': r.question_id,
                    'question': r.question[:150],
                    'predicted_answer': r.predicted_answer,
                    'expected_answer': r.correct_answer,
                    'correct': r.is_correct,
                    'confidence': r.confidence,
                    'time_ms': r.solve_time_ms,
                    'reasoning_trace': r.reasoning_trace[:500] if r.reasoning_trace else '',
                }
                for r in self.results
            ]
        }
        
        return report
    
    def _print_report(self, report: Dict[str, Any]):
        """Print formatted report to console"""
        
        print("="*80)
        print("EVALUATION RESULTS")
        print("="*80 + "\n")
        
        print(f"Mode: {'OFFICIAL Benchmark' if report['mode'] == 'official' else 'MOCK Data'}")
        print(f"Total Questions: {report['total_questions']}")
        print(f"Correct: {report['correct']}")
        print(f"Accuracy: {report['accuracy_percentage']}")
        print(f"\nPerformance:")
        print(f"  Total Time: {report['total_time_seconds']:.2f}s")
        print(f"  Avg per Question: {report['avg_time_per_question_ms']:.2f}ms")
        print(f"  Throughput: {report['throughput_qps']:.0f} questions/second")
        print(f"  Avg Confidence: {report['avg_confidence']:.1%}")
        
        if report['by_level']:
            print(f"\nBy Difficulty Level:")
            for level, stats in sorted(report['by_level'].items()):
                print(f"  Level {level}: {stats['correct']}/{stats['total']} ({stats['accuracy']}) - {stats['avg_time_ms']:.2f}ms avg")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save report to JSON file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"phase3_evaluation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {filename}")
        return filename


async def main():
    """Main evaluation runner"""
    
    # Check for command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3 GAIA Benchmark Evaluation')
    parser.add_argument('--limit', type=int, help='Limit to N questions')
    parser.add_argument('--level', type=int, choices=[1, 2, 3], help='Evaluate only specific level')
    parser.add_argument('--mock', action='store_true', help='Force mock data (no HuggingFace)')
    parser.add_argument('--full', action='store_true', help='Run full 450 questions')
    parser.add_argument('--quick', action='store_true', help='Quick test (10 questions)')
    
    args = parser.parse_args()
    
    # Determine mode
    use_official = not args.mock and OFFICIAL_GAIA_AVAILABLE
    
    # Set limit
    limit = args.limit
    if args.full:
        limit = None  # All questions
    elif args.quick:
        limit = 10
    elif not limit:
        limit = 10  # Default to 10 for safety
    
    # Create evaluator
    evaluator = Phase3Evaluator(use_official_benchmark=use_official)
    
    # Run evaluation
    report = await evaluator.run_evaluation(limit=limit, level=args.level, verbose=True)
    
    # Save report
    evaluator.save_report(report)
    
    # Print summary
    print("\nPhase 3 Evaluation Complete!")
    print(f"Accuracy: {report['accuracy_percentage']}")
    print(f"Questions: {report['total_questions']}")
    print(f"Time: {report['total_time_seconds']:.2f}s")


if __name__ == '__main__':
    asyncio.run(main())
