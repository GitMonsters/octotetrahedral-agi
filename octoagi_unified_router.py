#!/usr/bin/env python3
"""
OctoAGI Unified Router - Intelligent Mode Selection
====================================================

Automatically routes queries to the optimal processing mode:
- Popperian: System commands, file operations, app control
- Perplexity: Knowledge questions, explanations, research
- CodeGen: Code generation, debugging, refactoring
- Hybrid: Complex tasks requiring multiple modes

Uses pattern matching + confidence scoring to select the best approach.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(Enum):
    POPPERIAN = "popperian"  # Command execution
    PERPLEXITY = "perplexity"  # Knowledge/research
    CODEGEN = "codegen"  # Code generation
    HYBRID = "hybrid"  # Multiple modes needed


@dataclass
class RouteDecision:
    mode: ProcessingMode
    confidence: float
    reasoning: str
    secondary_mode: Optional[ProcessingMode] = None
    extraction: Optional[Dict] = None


class OctoAGIRouter:
    """
    Intelligent router that selects optimal processing mode based on query analysis.
    """
    
    def __init__(self):
        # Command patterns for Popperian mode
        self.command_patterns = [
            r"^(run|execute|cmd|command)\s+",
            r"^(open|launch|start)\s+\w+",
            r"^(find|search|locate)\s+file",
            r"^(create|make|write)\s+file",
            r"^(list|show|display)\s+.*\s+(files|directories)",
            r"^\w+\s+-[a-zA-Z]+",  # Unix-style flags
        ]
        
        # Knowledge patterns for Perplexity mode
        self.knowledge_patterns = [
            r"^(what|why|how|when|where|who)\s+",
            r"(explain|describe|tell me about)",
            r"(meaning of|definition of)",
            r"(history of|background on)",
            r"^(is|are|does|do|can|could|should)\s+",
        ]
        
        # Code patterns for CodeGen mode
        self.code_patterns = [
            r"(write|create|generate)\s+(code|function|class|script)",
            r"(implement|code)\s+.*\s+(algorithm|function)",
            r"(debug|fix|refactor|optimize)\s+.*code",
            r"(convert|translate)\s+.*\s+to\s+\w+",
            r"```[\w]*\n",  # Code blocks
        ]
        
    def route(self, query: str) -> RouteDecision:
        """
        Analyze query and determine optimal processing mode.
        """
        query_lower = query.lower().strip()
        
        # Score each mode
        popperian_score = self._score_popperian(query_lower)
        perplexity_score = self._score_perplexity(query_lower)
        codegen_score = self._score_codegen(query_lower)
        
        scores = {
            ProcessingMode.POPPERIAN: popperian_score,
            ProcessingMode.PERPLEXITY: perplexity_score,
            ProcessingMode.CODEGEN: codegen_score,
        }
        
        # Select primary mode
        primary = max(scores, key=scores.get)
        confidence = scores[primary]
        
        # Check if hybrid approach needed
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] > 0.5 and sorted_scores[1][1] > 0.4:
            # Close scores suggest hybrid approach
            return RouteDecision(
                mode=ProcessingMode.HYBRID,
                confidence=0.7,
                reasoning=f"Hybrid: {sorted_scores[0][0].value} (primary) + {sorted_scores[1][0].value}",
                secondary_mode=sorted_scores[1][0],
                extraction=self._extract_components(query, sorted_scores[0][0])
            )
        
        return RouteDecision(
            mode=primary,
            confidence=confidence,
            reasoning=self._get_reasoning(primary, query_lower),
            extraction=self._extract_components(query, primary)
        )
    
    def _score_popperian(self, query: str) -> float:
        """Score likelihood this is a system command."""
        score = 0.0
        
        for pattern in self.command_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.3
        
        # Boost for imperative verbs
        if any(query.startswith(v) for v in ["run", "open", "create", "find", "list", "execute"]):
            score += 0.2
        
        # Boost for file/system terms
        if any(term in query for term in ["file", "directory", "folder", "app", "terminal", "command"]):
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_perplexity(self, query: str) -> float:
        """Score likelihood this is a knowledge question."""
        score = 0.0
        
        for pattern in self.knowledge_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.3
        
        # Boost for question marks
        if "?" in query:
            score += 0.2
        
        # Boost for knowledge-seeking terms
        if any(term in query for term in ["explain", "meaning", "definition", "history", "why", "how"]):
            score += 0.15
        
        # Penalty if looks like command
        if re.search(r"^\w+\s+-[a-zA-Z]+", query):
            score -= 0.3
        
        return max(min(score, 1.0), 0.0)
    
    def _score_codegen(self, query: str) -> float:
        """Score likelihood this is a code generation request."""
        score = 0.0
        
        for pattern in self.code_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.3
        
        # Boost for code-related terms
        if any(term in query for term in ["code", "function", "class", "algorithm", "debug", "refactor"]):
            score += 0.15
        
        # Boost for programming languages
        if any(lang in query for lang in ["python", "javascript", "java", "c++", "rust", "go"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_components(self, query: str, mode: ProcessingMode) -> Dict:
        """Extract relevant components from query based on mode."""
        if mode == ProcessingMode.POPPERIAN:
            # Extract command
            match = re.search(r"^(?:run|execute|cmd|command)\s+(.+)", query, re.IGNORECASE)
            if match:
                return {"command": match.group(1).strip()}
        
        elif mode == ProcessingMode.CODEGEN:
            # Extract language and task
            lang_match = re.search(r"\b(python|javascript|java|c\+\+|rust|go)\b", query, re.IGNORECASE)
            task_match = re.search(r"(write|create|implement|debug|refactor|optimize)\s+(.+)", query, re.IGNORECASE)
            
            result = {}
            if lang_match:
                result["language"] = lang_match.group(1).lower()
            if task_match:
                result["action"] = task_match.group(1).lower()
                result["task"] = task_match.group(2).strip()
            return result
        
        return {}
    
    def _get_reasoning(self, mode: ProcessingMode, query: str) -> str:
        """Generate human-readable reasoning for mode selection."""
        if mode == ProcessingMode.POPPERIAN:
            return "Detected system command/action request → Popperian execution mode"
        elif mode == ProcessingMode.PERPLEXITY:
            return "Detected knowledge/research question → Perplexity search mode"
        elif mode == ProcessingMode.CODEGEN:
            return "Detected code generation request → VortexDisCode CodeGen mode"
        else:
            return f"Selected {mode.value} mode"


def test_router():
    """Test the router with various queries."""
    router = OctoAGIRouter()
    
    test_queries = [
        "run ls -la",
        "What is Popperian philosophy?",
        "Create a binary search function in Python",
        "Find file named test.py",
        "Explain how quantum computing works",
        "Debug this code: def foo(): print('hello'",
        "Open Chrome browser",
        "Why is the sky blue?",
        "Implement quicksort algorithm",
        "List all Python files in current directory",
    ]
    
    print("=" * 70)
    print("OctoAGI Unified Router - Test Results")
    print("=" * 70)
    
    for query in test_queries:
        decision = router.route(query)
        print(f"\nQuery: {query}")
        print(f"  Mode: {decision.mode.value}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Reasoning: {decision.reasoning}")
        if decision.extraction:
            print(f"  Extracted: {decision.extraction}")


if __name__ == "__main__":
    test_router()
