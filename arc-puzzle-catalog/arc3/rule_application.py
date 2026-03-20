"""
Rule Application Engine — Apply inferred transformation rules to new grids.

Manages rule execution, confidence scoring, fallback chains, and voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from arc3.reasoning import TransformationRule


@dataclass
class Prediction:
    """A grid prediction with confidence information."""
    grid: np.ndarray
    rule: TransformationRule
    confidence: float
    rule_index: int
    method: str  # 'primary', 'fallback', 'ensemble'


class RuleApplicator:
    """Applies transformation rules to grids with confidence tracking."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.rules: list[TransformationRule] = []
        self.prediction_history: list[Prediction] = []

    def load_rules(self, rules: list[TransformationRule]):
        """Load inferred rules."""
        self.rules = sorted(rules, key=lambda r: r.confidence, reverse=True)

    def apply_primary_rule(self, grid: np.ndarray) -> Optional[Prediction]:
        """Apply highest-confidence rule."""
        if not self.rules:
            return None
        
        rule = self.rules[0]
        if rule.confidence < self.confidence_threshold:
            return None
        
        result = rule.apply(grid)
        if result is not None:
            pred = Prediction(
                grid=result,
                rule=rule,
                confidence=rule.confidence,
                rule_index=0,
                method='primary',
            )
            self.prediction_history.append(pred)
            return pred
        
        return None

    def apply_with_fallback(self, grid: np.ndarray, max_attempts: int = 3) -> Optional[Prediction]:
        """Try rules in order until one succeeds."""
        for i, rule in enumerate(self.rules[:max_attempts]):
            if rule.confidence < self.confidence_threshold:
                continue
            
            result = rule.apply(grid)
            if result is not None:
                pred = Prediction(
                    grid=result,
                    rule=rule,
                    confidence=rule.confidence,
                    rule_index=i,
                    method='fallback' if i > 0 else 'primary',
                )
                self.prediction_history.append(pred)
                return pred
        
        return None

    def apply_ensemble(self, grid: np.ndarray, voting: bool = True) -> Optional[Prediction]:
        """Apply multiple rules and vote on result."""
        predictions = []
        
        for i, rule in enumerate(self.rules):
            if rule.confidence < 0.3:  # Lower threshold for ensemble
                break
            
            result = rule.apply(grid)
            if result is not None:
                predictions.append((result, rule, i))
        
        if not predictions:
            return None
        
        if voting and len(predictions) > 1:
            # Vote by comparing grid hashes
            vote_map = {}
            for grid_pred, rule, idx in predictions:
                grid_hash = hash(grid_pred.tobytes())
                if grid_hash not in vote_map:
                    vote_map[grid_hash] = (grid_pred, [], 0.0)
                vote_map[grid_hash][1].append(rule)
                vote_map[grid_hash] = (grid_pred, vote_map[grid_hash][1],
                                       np.mean([r.confidence for r in vote_map[grid_hash][1]]))
            
            # Get majority vote
            best_grid, rules_voted, avg_conf = max(vote_map.values(), key=lambda x: x[2])
            rule = rules_voted[0]
        else:
            # Use highest confidence
            best_grid, rule, _ = predictions[0]
            avg_conf = rule.confidence
        
        pred = Prediction(
            grid=best_grid,
            rule=rule,
            confidence=avg_conf,
            rule_index=0,
            method='ensemble',
        )
        self.prediction_history.append(pred)
        return pred

    def score_prediction(self, prediction: Prediction, reference: Optional[np.ndarray] = None) -> dict:
        """Score a prediction against reference or return confidence metrics."""
        score_info = {
            'confidence': prediction.confidence,
            'rule_type': prediction.rule.rule_type,
            'method': prediction.method,
            'rule_index': prediction.rule_index,
        }
        
        if reference is not None:
            match = np.array_equal(prediction.grid, reference)
            score_info['exact_match'] = match
            score_info['similarity'] = float(np.sum(prediction.grid == reference) / reference.size) if reference.size > 0 else 0.0
        
        return score_info

    def get_top_predictions(self, grid: np.ndarray, count: int = 3) -> list[Prediction]:
        """Get top N predictions from different rules."""
        predictions = []
        
        for i, rule in enumerate(self.rules[:count]):
            result = rule.apply(grid)
            if result is not None:
                pred = Prediction(
                    grid=result,
                    rule=rule,
                    confidence=rule.confidence,
                    rule_index=i,
                    method='ranked',
                )
                predictions.append(pred)
        
        return predictions

    def get_decision_path(self, grid: np.ndarray) -> dict:
        """Get detailed explanation of prediction decision."""
        primary = self.apply_primary_rule(grid)
        
        if not primary:
            primary = self.apply_with_fallback(grid, max_attempts=1)
        
        if not primary:
            # Return best effort
            if self.rules:
                result = self.rules[0].apply(grid)
                if result is not None:
                    primary = Prediction(
                        grid=result,
                        rule=self.rules[0],
                        confidence=self.rules[0].confidence,
                        rule_index=0,
                        method='best_effort',
                    )
        
        return {
            'predicted_output': primary.grid if primary else None,
            'rule': primary.rule.description if primary else None,
            'confidence': primary.confidence if primary else 0.0,
            'method': primary.method if primary else 'none',
            'rule_count': len(self.rules),
            'threshold': self.confidence_threshold,
        }

    def reset(self):
        """Clear prediction history."""
        self.prediction_history.clear()


class EnsemblePredictor:
    """Ensemble of rule applicators with weighted voting."""

    def __init__(self):
        self.applicators: list[RuleApplicator] = []
        self.weights: list[float] = []

    def add_applicator(self, applicator: RuleApplicator, weight: float = 1.0):
        """Add a rule applicator to the ensemble."""
        self.applicators.append(applicator)
        self.weights.append(weight)

    def predict(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Get ensemble prediction."""
        predictions = []
        weighted_scores = []
        
        for applicator, weight in zip(self.applicators, self.weights):
            pred = applicator.apply_primary_rule(grid)
            if pred:
                predictions.append(pred.grid)
                weighted_scores.append(pred.confidence * weight)
        
        if not predictions:
            return None
        
        # Vote by grid hash
        vote_map = {}
        for pred, score in zip(predictions, weighted_scores):
            grid_hash = hash(pred.tobytes())
            if grid_hash not in vote_map:
                vote_map[grid_hash] = (pred, 0.0)
            vote_map[grid_hash] = (pred, vote_map[grid_hash][1] + score)
        
        # Get highest scoring prediction
        best_pred, _ = max(vote_map.values(), key=lambda x: x[1])
        return best_pred

    def get_predictions(self, grid: np.ndarray, count: int = 3) -> list[np.ndarray]:
        """Get top N ensemble predictions."""
        all_preds = []
        
        for applicator in self.applicators:
            preds = applicator.get_top_predictions(grid, count)
            for pred in preds:
                all_preds.append((pred.grid, pred.confidence))
        
        # Deduplicate and sort
        unique = {}
        for grid_pred, conf in all_preds:
            grid_hash = hash(grid_pred.tobytes())
            if grid_hash not in unique or conf > unique[grid_hash][1]:
                unique[grid_hash] = (grid_pred, conf)
        
        sorted_preds = sorted(unique.values(), key=lambda x: x[1], reverse=True)
        return [pred for pred, _ in sorted_preds[:count]]
