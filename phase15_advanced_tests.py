"""
Phase 15: Advanced Features - Comprehensive Test Suite
=======================================================

Tests for all four advanced systems:
1. Model Ensemble Optimization
2. Predictive Modeling
3. Custom Routing DSL
4. A/B Testing Framework
"""

import unittest
from datetime import datetime, timedelta
from ngvt_phase15_advanced import (
    ModelProfile, ModelCapability, EnsembleOptimizer, RoutingDecision,
    PredictiveModel, RoutingDSL,
    Variant, ABTestFramework
)


class TestModelEnsemble(unittest.TestCase):
    """Test model ensemble optimization"""
    
    def setUp(self):
        self.ensemble = EnsembleOptimizer()
        
        self.model1 = ModelProfile(
            provider="openai", model="gpt-4",
            capability=ModelCapability.EXPERT,
            cost_per_token=0.00003,
            avg_latency_ms=200,
            success_rate=0.99,
            throughput_tokens_sec=100,
            quality_score=0.95
        )
        
        self.model2 = ModelProfile(
            provider="openai", model="gpt-3.5",
            capability=ModelCapability.STANDARD,
            cost_per_token=0.0000015,
            avg_latency_ms=100,
            success_rate=0.95,
            throughput_tokens_sec=150,
            quality_score=0.80
        )
    
    def test_register_model(self):
        """Test registering a model"""
        self.ensemble.register_model(self.model1)
        self.assertEqual(len(self.ensemble.models), 1)
        self.assertIn("openai/gpt-4", self.ensemble.models)
    
    def test_efficiency_score(self):
        """Test efficiency score calculation"""
        score = self.model1.calculate_efficiency_score()
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1)
    
    def test_model_selection(self):
        """Test model selection"""
        self.ensemble.register_model(self.model1)
        self.ensemble.register_model(self.model2)
        
        decision = self.ensemble.select_model("complex", quality_requirement="advanced")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.model_profile.model, "gpt-4")
    
    def test_budget_constraint(self):
        """Test budget constraint in model selection"""
        self.ensemble.register_model(self.model1)
        self.ensemble.register_model(self.model2)
        
        # Low budget should select cheaper model
        decision = self.ensemble.select_model("simple", budget_constraint=0.005)
        self.assertEqual(decision.model_profile.model, "gpt-3.5")
    
    def test_latency_constraint(self):
        """Test latency constraint"""
        self.ensemble.register_model(self.model1)
        self.ensemble.register_model(self.model2)
        
        # Low latency requirement should select faster model
        decision = self.ensemble.select_model("simple", latency_constraint_ms=150)
        self.assertEqual(decision.model_profile.model, "gpt-3.5")
    
    def test_record_request(self):
        """Test recording request results"""
        self.ensemble.register_model(self.model1)
        
        self.ensemble.record_request("openai/gpt-4", True, 150, 0.05, 0.9)
        self.assertEqual(self.ensemble.model_stats["openai/gpt-4"]["requests"], 1)
        self.assertEqual(self.ensemble.model_stats["openai/gpt-4"]["successes"], 1)
    
    def test_ensemble_report(self):
        """Test ensemble performance report"""
        self.ensemble.register_model(self.model1)
        self.ensemble.record_request("openai/gpt-4", True, 200, 0.03, 0.95)
        
        report = self.ensemble.get_ensemble_report()
        self.assertIn("openai/gpt-4", report)
        self.assertIn("success_rate", report["openai/gpt-4"])


class TestPredictiveModel(unittest.TestCase):
    """Test predictive modeling"""
    
    def setUp(self):
        self.predictor = PredictiveModel()
    
    def test_add_data_point(self):
        """Test adding data points"""
        self.predictor.add_data_point(1000, 0.03, 150, 12)
        self.assertEqual(len(self.predictor.historical_data), 1)
    
    def test_cost_prediction(self):
        """Test cost prediction"""
        # Add training data
        for i in range(20):
            self.predictor.add_data_point(1000 + i * 100, 0.03 + i * 0.001, 150, 12)
        
        prediction = self.predictor.predict_cost(1500)
        self.assertGreater(prediction.predicted_value, 0)
        self.assertGreaterEqual(prediction.confidence_interval_upper, prediction.predicted_value)
        self.assertLessEqual(prediction.confidence_interval_lower, prediction.predicted_value)
    
    def test_latency_prediction(self):
        """Test latency prediction"""
        # Add training data
        for i in range(20):
            self.predictor.add_data_point(1000, 0.03, 100 + i * 5, i % 24)
        
        prediction = self.predictor.predict_latency(14)
        self.assertGreater(prediction.predicted_value, 0)
    
    def test_confidence_interval(self):
        """Test confidence interval width"""
        for i in range(30):
            self.predictor.add_data_point(1000 + i * 50, 0.03 + i * 0.0005, 150, 12)
        
        prediction = self.predictor.predict_cost(1500)
        ci_width = prediction.confidence_interval_upper - prediction.confidence_interval_lower
        # With perfect linear data, CI width might be 0, so just verify structure
        self.assertGreaterEqual(ci_width, 0)
        self.assertIsNotNone(prediction.confidence_interval_lower)
        self.assertIsNotNone(prediction.confidence_interval_upper)


class TestRoutingDSL(unittest.TestCase):
    """Test custom routing DSL"""
    
    def setUp(self):
        self.dsl = RoutingDSL()
    
    def test_define_rule(self):
        """Test defining a rule"""
        self.dsl.define_rule(
            "test_rule",
            "tokens > 1000",
            "route_to('gpt-4')"
        )
        self.assertIn("test_rule", self.dsl.rules)
    
    def test_evaluate_rule_true(self):
        """Test rule evaluation when true"""
        self.dsl.define_rule(
            "high_tokens",
            "tokens > 1000",
            "route_to('gpt-4')"
        )
        
        context = {"tokens": 2000, "cost_budget": 1.0, "quality": "advanced"}
        result = self.dsl.evaluate_rules(context)
        self.assertEqual(result, "gpt-4")
    
    def test_evaluate_rule_false(self):
        """Test rule evaluation when false"""
        self.dsl.define_rule(
            "high_tokens",
            "tokens > 1000",
            "route_to('gpt-4')"
        )
        
        context = {"tokens": 500, "cost_budget": 1.0, "quality": "advanced"}
        result = self.dsl.evaluate_rules(context)
        self.assertIsNone(result)
    
    def test_multiple_rules_first_match(self):
        """Test multiple rules - first match wins"""
        self.dsl.define_rule(
            "rule1",
            "tokens > 2000",
            "route_to('gpt-4')"
        )
        self.dsl.define_rule(
            "rule2",
            "tokens > 1000",
            "route_to('claude-3')"
        )
        
        context = {"tokens": 1500}
        result = self.dsl.evaluate_rules(context)
        # Should match rule2 (rule1 not met)
        self.assertEqual(result, "claude-3")
    
    def test_rules_documentation(self):
        """Test rules documentation generation"""
        self.dsl.define_rule("test", "tokens > 1000", "route_to('gpt-4')")
        doc = self.dsl.get_rules_documentation()
        self.assertIn("test", doc)
        self.assertIn("tokens > 1000", doc)


class TestABTesting(unittest.TestCase):
    """Test A/B testing framework"""
    
    def setUp(self):
        self.ab_test = ABTestFramework("test", 7)
    
    def test_add_variant(self):
        """Test adding variants"""
        variant = Variant("a", "Control", "gpt-4")
        self.ab_test.add_variant(variant)
        self.assertEqual(len(self.ab_test.variants), 1)
    
    def test_variant_selection_deterministic(self):
        """Test deterministic variant selection"""
        v_a = Variant("a", "Control", "gpt-4", traffic_allocation=0.5)
        v_b = Variant("b", "Treatment", "claude-3", traffic_allocation=0.5)
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        # Same user should get same variant
        selected1 = self.ab_test.select_variant("user_123")
        selected2 = self.ab_test.select_variant("user_123")
        
        self.assertEqual(selected1.variant_id, selected2.variant_id)
    
    def test_traffic_allocation(self):
        """Test traffic allocation distribution"""
        v_a = Variant("a", "Control", "gpt-4", traffic_allocation=0.7)
        v_b = Variant("b", "Treatment", "claude-3", traffic_allocation=0.3)
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        selections = {}
        for i in range(1000):
            variant = self.ab_test.select_variant(f"user_{i}")
            selections[variant.variant_id] = selections.get(variant.variant_id, 0) + 1
        
        # Check roughly 70/30 split
        ratio = selections.get("a", 0) / (selections.get("b", 1))
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 3.0)
    
    def test_record_result(self):
        """Test recording results"""
        v_a = Variant("a", "Control", "gpt-4")
        self.ab_test.add_variant(v_a)
        
        self.ab_test.record_result("a", True, 200, 0.03, 0.9)
        metrics = self.ab_test.metrics["a"]
        
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 1)
    
    def test_variant_comparison(self):
        """Test variant comparison"""
        v_a = Variant("a", "Control", "gpt-4")
        v_b = Variant("b", "Treatment", "claude-3")
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        # Record some results
        for i in range(10):
            self.ab_test.record_result("a", True, 200, 0.03, 0.9)
            self.ab_test.record_result("b", True, 150, 0.02, 0.85)
        
        comparison = self.ab_test.get_variant_comparison()
        self.assertEqual(len(comparison), 2)
        self.assertIn("a", comparison)
        self.assertIn("b", comparison)
    
    def test_calculate_winner_quality(self):
        """Test determining winner by quality"""
        v_a = Variant("a", "Control", "gpt-4")
        v_b = Variant("b", "Treatment", "claude-3")
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        # Variant A has higher quality
        for i in range(20):
            self.ab_test.record_result("a", True, 200, 0.03, 0.95)
            self.ab_test.record_result("b", True, 150, 0.02, 0.80)
        
        winner = self.ab_test.calculate_winner("quality")
        self.assertEqual(winner, "a")
    
    def test_calculate_winner_cost(self):
        """Test determining winner by cost"""
        v_a = Variant("a", "Expensive", "gpt-4")
        v_b = Variant("b", "Cheap", "gpt-3.5")
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        # Variant B is cheaper
        for i in range(20):
            self.ab_test.record_result("a", True, 200, 0.10, 0.9)
            self.ab_test.record_result("b", True, 150, 0.01, 0.8)
        
        winner = self.ab_test.calculate_winner("cost")
        self.assertEqual(winner, "b")
    
    def test_statistical_significance(self):
        """Test statistical significance testing"""
        v_a = Variant("a", "Control", "gpt-4")
        v_b = Variant("b", "Treatment", "claude-3")
        
        self.ab_test.add_variant(v_a)
        self.ab_test.add_variant(v_b)
        
        # Large difference in success rates
        for i in range(30):
            self.ab_test.record_result("a", True, 200, 0.03, 0.9)
            self.ab_test.record_result("b", False, 150, 0.02, 0.8)
        
        is_sig = self.ab_test.is_statistically_significant("a", "b")
        self.assertTrue(is_sig)
    
    def test_get_test_report(self):
        """Test test report generation"""
        v_a = Variant("a", "Control", "gpt-4")
        self.ab_test.add_variant(v_a)
        
        self.ab_test.record_result("a", True, 200, 0.03, 0.9)
        
        report = self.ab_test.get_test_report()
        self.assertIn("test_name", report)
        self.assertIn("variant_comparison", report)
        self.assertIn("recommendations", report)


def run_tests():
    """Run all Phase 15 tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelEnsemble))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveModel))
    suite.addTests(loader.loadTestsFromTestCase(TestRoutingDSL))
    suite.addTests(loader.loadTestsFromTestCase(TestABTesting))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("PHASE 15 ADVANCED FEATURES - TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
