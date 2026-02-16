"""
Semantic Answer Matching for GAIA Solver

Implements embeddings-based answer matching to handle:
- Paraphrasing and synonyms
- Partial answers
- Different phrasings of the same concept
"""

import os
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SemanticAnswerMatcher:
    """
    Uses embeddings to match answers semantically instead of exact string matching.
    Falls back to simple string matching if embeddings unavailable.
    """
    
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.embeddings_model = None
        
        if use_embeddings:
            self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        try:
            # Try to use sentence-transformers for semantic similarity
            from sentence_transformers import SentenceTransformer, util
            
            logger.info("Loading semantic embedding model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.similarity_util = util
            logger.info("Semantic embeddings ready")
            
        except ImportError:
            logger.warning("sentence-transformers not installed")
            logger.warning("Install with: pip install sentence-transformers")
            logger.warning("Falling back to string-based matching")
            self.use_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            logger.warning("Falling back to string-based matching")
            self.use_embeddings = False
    
    def match_answers(
        self,
        predicted: str,
        correct: str,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        Match predicted answer against correct answer.
        
        Returns:
            (is_match: bool, confidence: float)
        """
        
        if not predicted or not correct:
            return False, 0.0
        
        # Try exact match first (fastest)
        if self._exact_match(predicted, correct):
            return True, 1.0
        
        # Try normalized number match (for numeric answers)
        if self._number_match(predicted, correct):
            return True, 0.98
        
        # Try substring match (only if correct answer is contained in predicted)
        if self._substring_match(predicted, correct):
            return True, 0.95
        
        # Try semantic match if available
        if self.use_embeddings and self.embeddings_model:
            return self._semantic_match(predicted, correct, threshold)
        
        # Fall back to fuzzy string matching
        return self._fuzzy_match(predicted, correct, threshold)
    
    def _exact_match(self, pred: str, correct: str) -> bool:
        """Exact string match (case-insensitive)"""
        pred_clean = pred.lower().strip()
        correct_clean = correct.lower().strip()
        
        # Remove "Based on analysis: " prefix if present
        if pred_clean.startswith("based on analysis: "):
            pred_clean = pred_clean.replace("based on analysis: ", "")
        
        # Remove trailing punctuation
        pred_clean = pred_clean.rstrip('.,;:!?')
        correct_clean = correct_clean.rstrip('.,;:!?')
        
        return pred_clean == correct_clean
    
    def _number_match(self, pred: str, correct: str) -> bool:
        """Check if both answers are the same number (handles formatting differences)"""
        import re
        
        # Try to extract numbers from both
        try:
            # Remove commas and spaces in numbers
            pred_clean = pred.strip().replace(',', '').replace(' ', '')
            correct_clean = correct.strip().replace(',', '').replace(' ', '')
            
            # Try direct float comparison
            pred_num = float(pred_clean)
            correct_num = float(correct_clean)
            
            # Exact match or very close (for floating point issues)
            if pred_num == correct_num:
                return True
            if correct_num != 0 and abs(pred_num - correct_num) / abs(correct_num) < 0.001:
                return True
        except (ValueError, ZeroDivisionError):
            pass
        
        return False
    
    def _substring_match(self, pred: str, correct: str) -> bool:
        """Check if correct answer is contained in predicted (not vice versa for short answers)"""
        pred_clean = pred.lower().strip()
        correct_clean = correct.lower().strip()
        
        # Remove prefix
        if pred_clean.startswith("based on analysis: "):
            pred_clean = pred_clean.replace("based on analysis: ", "")
        
        # Strip trailing punctuation
        pred_clean = pred_clean.rstrip('.,;:!?')
        correct_clean = correct_clean.rstrip('.,;:!?')
        
        # Only allow substring match if the correct answer is meaningful length
        if len(correct_clean) <= 2:
            return False
        
        # Check if correct answer is contained in predicted
        # But NOT if predicted is much longer (likely a hallucinated sentence containing the word)
        if correct_clean in pred_clean:
            # For short correct answers, predicted must also be short
            if len(correct_clean) < 10 and len(pred_clean) > len(correct_clean) * 5:
                return False
            return True
        
        # Check if predicted is contained in correct (e.g., pred="Paris", correct="Paris, France")
        if len(pred_clean) > 3 and pred_clean in correct_clean:
            return True
        
        return False
    
    def _semantic_match(
        self,
        predicted: str,
        correct: str,
        threshold: float
    ) -> Tuple[bool, float]:
        """Match using semantic similarity with embeddings"""
        try:
            # Get embeddings
            pred_embedding = self.embeddings_model.encode(predicted, convert_to_tensor=True)
            correct_embedding = self.embeddings_model.encode(correct, convert_to_tensor=True)
            
            # Calculate similarity
            similarity = self.similarity_util.pytorch_cos_sim(
                pred_embedding,
                correct_embedding
            ).item()
            
            # Clamp to [0, 1]
            similarity = max(0.0, min(1.0, similarity))
            
            is_match = similarity >= threshold
            return is_match, similarity
            
        except Exception as e:
            logger.debug(f"Semantic matching failed: {e}")
            return False, 0.0
    
    def _fuzzy_match(
        self,
        predicted: str,
        correct: str,
        threshold: float
    ) -> Tuple[bool, float]:
        """Fuzzy string matching using difflib"""
        try:
            from difflib import SequenceMatcher
            
            ratio = SequenceMatcher(
                None,
                predicted.lower(),
                correct.lower()
            ).ratio()
            
            is_match = ratio >= threshold
            return is_match, ratio
            
        except Exception as e:
            logger.debug(f"Fuzzy matching failed: {e}")
            return False, 0.0


def install_semantic_dependencies():
    """Install required packages for semantic matching"""
    import subprocess
    import sys
    
    print("Installing semantic matching dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "sentence-transformers", "torch"
    ])
    print("Installation complete!")


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    matcher = SemanticAnswerMatcher(use_embeddings=False)  # Disable for testing
    
    test_cases = [
        # Exact matches
        ("Paris", "Paris", True),
        ("paris", "PARIS", True),
        
        # Substring matches
        ("The capital of France is Paris", "Paris", True),
        ("Based on analysis: Paris", "Paris", True),
        
        # Different answers
        ("London", "Paris", False),
        ("France", "Paris", False),
    ]
    
    print("Testing Answer Matching")
    print("=" * 60)
    
    for pred, correct, expected in test_cases:
        is_match, confidence = matcher.match_answers(pred, correct)
        status = "✓" if is_match == expected else "✗"
        print(f"{status} Pred: '{pred}' vs Correct: '{correct}'")
        print(f"   Match: {is_match}, Confidence: {confidence:.2%}")
    
    print("=" * 60)
