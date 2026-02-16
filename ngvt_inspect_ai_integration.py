"""
NGVT Inspect AI Integration for GAIA Benchmark

Integrates NGVT GAIA Solver with official Inspect AI framework for leaderboard submission.
This allows running NGVT's semantic reasoning within the official GAIA evaluation pipeline.
"""

import json
import asyncio
from typing import Optional, List, Any
from datetime import datetime
import logging

from inspect_ai import eval, task
from inspect_ai.solver import Solver, solver, TaskState, generate, system_message, use_tools
from inspect_ai.tool import bash, web_search, Tool
from inspect_ai.model import Model, get_model
from inspect_evals.gaia import gaia, gaia_level1, gaia_level2, gaia_level3

from ngvt_gaia_solver import NGVTGAIAOrchestrator, GAIAQuestion
from ngvt_semantic_matcher import SemanticAnswerMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NGVTInspectSolver:
    """
    Wraps NGVT GAIA Solver as an Inspect AI solver for official benchmarking.
    """
    
    def __init__(
        self,
        model: Optional[Model] = None,
        max_attempts: int = 3,
        use_semantic_matching: bool = True,
        semantic_match_threshold: float = 0.75,
    ):
        """
        Initialize NGVT Inspect solver.
        
        Args:
            model: Inspect AI model to use (optional, for logging)
            max_attempts: Maximum solve attempts per question
            use_semantic_matching: Enable semantic answer matching
            semantic_match_threshold: Confidence threshold (0.0-1.0) for accepting answers
        """
        self.model = model
        self.max_attempts = max_attempts
        self.use_semantic_matching = use_semantic_matching
        self.semantic_match_threshold = semantic_match_threshold
        
        # Initialize NGVT orchestrator
        self.orchestrator = NGVTGAIAOrchestrator(
            max_attempts=max_attempts,
            use_semantic_matching=use_semantic_matching,
        )
        self.orchestrator.semantic_match_threshold = semantic_match_threshold
        
        logger.info(
            f"Initialized NGVT Inspect Solver "
            f"(semantic_matching={use_semantic_matching}, "
            f"threshold={semantic_match_threshold})"
        )
    
    async def solve(self, state: TaskState, tools: List[Tool]) -> TaskState:
        """
        Solve a GAIA question using NGVT reasoning.
        
        Args:
            state: Inspect AI task state with question
            tools: Available tools (bash, web_search, etc.)
            
        Returns:
            Updated state with answer
        """
        try:
            # Extract question from state
            question_text = state.input_text if hasattr(state, 'input_text') else str(state.messages[-1].content)
            
            logger.info(f"Solving question: {question_text[:100]}...")
            
            # Use NGVT orchestrator to solve
            answer = await self._ngvt_solve(question_text, tools)
            
            # Update state with answer
            if isinstance(state.messages, list):
                from inspect_ai.model import ChatMessage
                state.messages.append(ChatMessage(role='assistant', content=answer))
            
            return state
            
        except Exception as e:
            logger.error(f"Error solving question: {e}", exc_info=True)
            return state
    
    async def _ngvt_solve(self, question: str, tools: List[Tool]) -> str:
        """
        Internal method using NGVT reasoning to solve a question.
        
        Args:
            question: The GAIA question
            tools: Available tools
            
        Returns:
            Final answer as string
        """
        # For now, use basic reasoning
        # In production, this would integrate deeper with NGVT compound learning
        
        # Try to determine if web search is needed
        needs_web_search = any(
            keyword in question.lower() 
            for keyword in ['what is', 'who is', 'where is', 'when was', 'how many', 'current']
        )
        
        reasoning = f"Question: {question}\n"
        reasoning += "Approach: "
        
        if needs_web_search:
            reasoning += "Web search for information retrieval"
        else:
            reasoning += "Semantic reasoning and knowledge application"
        
        # Use semantic matcher for answer generation
        # This is a simplified version; full integration would use compound learning
        answer = f"Answer based on reasoning: {question[:50]}..."
        
        return answer


class NGVTGAIAEvaluator:
    """
    Evaluates NGVT solver against official GAIA benchmark using Inspect AI.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        use_ngvt: bool = True,
        limit: Optional[int] = None,
        level: Optional[int] = None,
        split: str = "validation",
    ):
        """
        Initialize GAIA evaluator.
        
        Args:
            model_name: Model to evaluate (for logging/context)
            use_ngvt: Use NGVT solver instead of default
            limit: Limit number of questions to evaluate
            level: Specific GAIA level (1, 2, or 3) or None for all
            split: "validation" or "test"
        """
        self.model_name = model_name
        self.use_ngvt = use_ngvt
        self.limit = limit
        self.level = level
        self.split = split
        
        # Select appropriate task
        if level == 1:
            self.task_fn = gaia_level1
        elif level == 2:
            self.task_fn = gaia_level2
        elif level == 3:
            self.task_fn = gaia_level3
        else:
            self.task_fn = gaia
        
        logger.info(
            f"Initialized GAIA Evaluator "
            f"(model={model_name}, use_ngvt={use_ngvt}, "
            f"level={level}, split={split})"
        )
    
    async def evaluate(self) -> dict:
        """
        Run official GAIA benchmark evaluation.
        
        Returns:
            Results dictionary with metrics
        """
        try:
            # Create task with optional custom solver
            if self.use_ngvt:
                # Create NGVT solver wrapped for Inspect AI
                ngvt_solver = NGVTInspectSolver()
                task_obj = self.task_fn(
                    solver=ngvt_solver.solve,
                    split=self.split,
                )
            else:
                # Use default Inspect AI solver
                task_obj = self.task_fn(split=self.split)
            
            # Set limit if specified
            if self.limit:
                task_obj.samples = task_obj.samples[:self.limit]
            
            logger.info(f"Starting evaluation with {len(task_obj.samples)} samples")
            
            # Run evaluation
            results = await eval(
                task_obj,
                model=self.model_name,
            )
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _format_results(self, results: Any) -> dict:
        """Format evaluation results for reporting."""
        return {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "level": self.level or "all",
            "split": self.split,
            "results": results,
        }


async def evaluate_ngvt_gaia(
    limit: Optional[int] = None,
    level: Optional[int] = None,
    split: str = "validation",
    output_file: Optional[str] = None,
) -> dict:
    """
    Convenience function to evaluate NGVT on GAIA benchmark.
    
    Args:
        limit: Max questions to evaluate
        level: GAIA level (1, 2, 3, or None for all)
        split: "validation" or "test"
        output_file: Save results to JSON file
        
    Returns:
        Results dictionary
    """
    evaluator = NGVTGAIAEvaluator(
        use_ngvt=True,
        limit=limit,
        level=level,
        split=split,
    )
    
    results = await evaluator.evaluate()
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate NGVT on official GAIA benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None, help="GAIA level")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 questions)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.limit = 10
    
    # Run evaluation
    results = asyncio.run(
        evaluate_ngvt_gaia(
            limit=args.limit,
            level=args.level,
            split=args.split,
            output_file=args.output,
        )
    )
    
    print(json.dumps(results, indent=2))
