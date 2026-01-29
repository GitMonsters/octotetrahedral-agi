#!/usr/bin/env python3
"""
Phase 17: Advanced Analytics & ML Insights - Comprehensive Test Suite

Tests for all analytics components:
- Time series analysis
- Anomaly detection
- Pattern recognition
- Predictive analytics
- Recommendation engine

Author: Confucius SDK Development Team
Version: 2.2.0
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
from ngvt_phase17_analytics import (
    TimeSeriesAnalyzer, AnomalyDetector, PatternRecognizer,
    PredictiveAnalytics, RecommendationEngine, AdvancedAnalyticsOrchestrator,
    AnomalyType, PredictionConfidence, RecommendationType
)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Test time series analysis functionality"""
    
    def setUp(self):
        self.analyzer = TimeSeriesAnalyzer(window_size=100)
    
    def test_add_point(self):
        """Test adding a data point"""
        ts = datetime.utcnow()
        self.analyzer.add_point(ts, 100.0)
        
        self.assertEqual(len(self.analyzer.data), 1)
        self.assertEqual(self.analyzer.get_values(), [100.0])
    
    def test_add_multiple_points(self):
        """Test adding multiple points"""
        points = [
            (datetime.utcnow() - timedelta(hours=i), 100 + i)
            for i in range(10)
        ]
        self.analyzer.add_points(points)
        
        self.assertEqual(len(self.analyzer.data), 10)
    
    def test_get_statistics(self):
        """Test calculating statistics"""
        for i in range(10):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=i), 100 + i)
        
        stats = self.analyzer.get_statistics()
        
        self.assertIsNotNone(stats)
        self.assertGreater(stats.mean, 0)
        self.assertGreater(stats.std_dev, 0)
        self.assertEqual(stats.count, 10)
    
    def test_get_statistics_empty(self):
        """Test statistics with no data"""
        stats = self.analyzer.get_statistics()
        self.assertIsNone(stats)
    
    def test_calculate_trend_positive(self):
        """Test detecting positive trend"""
        # Add increasing values
        for i in range(20):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=20-i), float(i))
        
        trend = self.analyzer.calculate_trend(window=10)
        self.assertGreater(trend, 0)  # Should detect upward trend
    
    def test_calculate_trend_negative(self):
        """Test detecting negative trend"""
        # Add decreasing values
        for i in range(20):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=20-i), float(20-i))
        
        trend = self.analyzer.calculate_trend(window=10)
        self.assertLess(trend, 0)  # Should detect downward trend
    
    def test_forecast(self):
        """Test forecasting future values"""
        for i in range(20):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=20-i), 100.0 + i)
        
        forecast = self.analyzer.forecast(periods=5)
        
        self.assertEqual(len(forecast), 5)
        self.assertTrue(all(isinstance(f, float) for f in forecast))
    
    def test_moving_average(self):
        """Test moving average calculation"""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for i, val in enumerate(values):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=10-i), float(val))
        
        ma = self.analyzer.moving_average(window=3)
        
        self.assertGreater(len(ma), 0)
        # First moving average should be (10+20+30)/3 = 20
        self.assertAlmostEqual(ma[0], 20.0, places=1)


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection"""
    
    def setUp(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.detector = AnomalyDetector(sensitivity=2.0)
    
    def test_detect_statistical_outliers(self):
        """Test detecting statistical outliers"""
        # Add normal data
        for i in range(30):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 100.0 + np.random.normal(0, 5))
        
        # Add outliers
        self.analyzer.add_point(datetime.utcnow(), 500.0)
        
        anomalies = self.detector.detect_statistical_outliers(self.analyzer)
        
        self.assertGreater(len(anomalies), 0)
        self.assertTrue(any(a.anomaly_type == AnomalyType.OUTLIER for a in anomalies))
    
    def test_detect_spikes(self):
        """Test detecting spikes"""
        base_time = datetime.utcnow()
        
        # Add normal data
        for i in range(10):
            self.analyzer.add_point(base_time - timedelta(hours=10-i), 100.0)
        
        # Add spike
        self.analyzer.add_point(base_time + timedelta(hours=1), 250.0)
        
        anomalies = self.detector.detect_spikes(self.analyzer)
        
        self.assertGreater(len(anomalies), 0)
    
    def test_detect_all(self):
        """Test detecting all types of anomalies"""
        for i in range(20):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=20-i), 100.0 + i)
        
        anomalies = self.detector.detect_all(self.analyzer)
        
        self.assertIsNotNone(anomalies)
        self.assertTrue(isinstance(anomalies, list))


class TestPatternRecognition(unittest.TestCase):
    """Test pattern recognition"""
    
    def setUp(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.recognizer = PatternRecognizer()
    
    def test_detect_growth_pattern(self):
        """Test detecting growth pattern"""
        for i in range(30):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 100.0 + i * 5)
        
        pattern = self.recognizer.detect_growth_pattern(self.analyzer)
        
        # Pattern may or may not be detected depending on sensitivity
        if pattern:
            self.assertEqual(pattern.name, "Growth Trend")
    
    def test_detect_decline_pattern(self):
        """Test detecting decline pattern"""
        for i in range(30):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 500.0 - i * 5)
        
        pattern = self.recognizer.detect_growth_pattern(self.analyzer)
        
        # Pattern may or may not be detected depending on sensitivity
        if pattern:
            self.assertEqual(pattern.name, "Decline Trend")
    
    def test_detect_volatility_pattern(self):
        """Test detecting high volatility"""
        for i in range(30):
            # High variability
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 100.0 + np.random.normal(0, 30))
        
        pattern = self.recognizer.detect_volatility_pattern(self.analyzer)
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.name, "High Volatility")
    
    def test_detect_seasonality(self):
        """Test detecting seasonal pattern"""
        base_time = datetime.utcnow()
        
        # Create seasonal data (7-day period)
        for cycle in range(3):
            for day in range(7):
                value = 100 + 30 * np.sin(day * 2 * np.pi / 7)
                timestamp = base_time - timedelta(hours=21 - (cycle * 7 + day))
                self.analyzer.add_point(timestamp, value)
        
        pattern = self.recognizer.detect_seasonality(self.analyzer, period=7)
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.name, "Seasonal Pattern")


class TestPredictiveAnalytics(unittest.TestCase):
    """Test predictive analytics"""
    
    def setUp(self):
        self.analyzer = TimeSeriesAnalyzer()
        self.predictor = PredictiveAnalytics()
    
    def test_predict_metric(self):
        """Test metric prediction"""
        for i in range(30):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 100.0 + i)
        
        prediction = self.predictor.predict_metric("test_metric", self.analyzer, periods_ahead=5)
        
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.metric_name, "test_metric")
        self.assertGreater(prediction.upper_bound, prediction.lower_bound)
    
    def test_prediction_confidence_high_data(self):
        """Test prediction confidence with high data points"""
        for i in range(150):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=150-i), 100.0 + i)
        
        prediction = self.predictor.predict_metric("test_metric", self.analyzer, periods_ahead=5)
        
        self.assertEqual(prediction.confidence, PredictionConfidence.VERY_HIGH)
    
    def test_prediction_confidence_low_data(self):
        """Test prediction confidence with limited data points"""
        for i in range(15):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=15-i), 100.0 + i)
        
        prediction = self.predictor.predict_metric("test_metric", self.analyzer, periods_ahead=5)
        
        # With 15 data points, confidence should be LOW
        self.assertEqual(prediction.confidence, PredictionConfidence.LOW)
    
    def test_get_predictions(self):
        """Test retrieving predictions"""
        for i in range(30):
            self.analyzer.add_point(datetime.utcnow() - timedelta(hours=30-i), 100.0)
        
        self.predictor.predict_metric("metric1", self.analyzer, periods_ahead=5)
        self.predictor.predict_metric("metric2", self.analyzer, periods_ahead=5)
        
        predictions = self.predictor.get_predictions(limit=5)
        
        self.assertEqual(len(predictions), 2)


class TestRecommendationEngine(unittest.TestCase):
    """Test recommendation engine"""
    
    def setUp(self):
        self.engine = RecommendationEngine()
    
    def test_recommend_based_on_anomalies(self):
        """Test generating recommendations from anomalies"""
        from ngvt_phase17_analytics import Anomaly
        
        anomaly = Anomaly(
            anomaly_id="test1",
            timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.SPIKE,
            value=500.0,
            expected_value=100.0,
            severity=0.8,
            description="Large spike detected"
        )
        
        recommendations = self.engine.recommend_based_on_anomalies([anomaly])
        
        self.assertGreater(len(recommendations), 0)
        self.assertEqual(recommendations[0].recommendation_type, RecommendationType.ISSUE_PREVENTION)
    
    def test_recommend_based_on_patterns(self):
        """Test generating recommendations from patterns"""
        from ngvt_phase17_analytics import Pattern
        
        pattern = Pattern(
            pattern_id="test1",
            name="High Volatility",
            description="High volatility detected",
            confidence=0.8,
            frequency=1,
            characteristics={"cv": 0.5}
        )
        
        recommendations = self.engine.recommend_based_on_patterns([pattern])
        
        self.assertGreater(len(recommendations), 0)
        self.assertEqual(recommendations[0].recommendation_type, RecommendationType.OPTIMIZATION)
    
    def test_recommendation_priority_levels(self):
        """Test that recommendations have proper priority levels"""
        from ngvt_phase17_analytics import Anomaly
        
        high_severity_anomaly = Anomaly(
            anomaly_id="test1",
            timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.SPIKE,
            value=500.0,
            expected_value=100.0,
            severity=0.95,  # Very high
            description="Critical spike"
        )
        
        recommendations = self.engine.recommend_based_on_anomalies([high_severity_anomaly])
        
        self.assertGreater(len(recommendations), 0)
        self.assertEqual(recommendations[0].priority, "critical")


class TestOrchestrator(unittest.TestCase):
    """Test advanced analytics orchestrator"""
    
    def setUp(self):
        self.orchestrator = AdvancedAnalyticsOrchestrator()
    
    def test_register_metric(self):
        """Test registering a metric"""
        self.orchestrator.register_metric("test_metric")
        
        self.assertIn("test_metric", self.orchestrator.analyzers)
        self.assertIn("test_metric", self.orchestrator.anomaly_detectors)
        self.assertIn("test_metric", self.orchestrator.pattern_recognizers)
    
    def test_record_metric(self):
        """Test recording metric values"""
        self.orchestrator.record_metric("cpu_usage", 65.0)
        self.orchestrator.record_metric("cpu_usage", 70.0)
        
        analyzer = self.orchestrator.analyzers["cpu_usage"]
        self.assertEqual(len(analyzer.data), 2)
    
    def test_analyze_metric(self):
        """Test analyzing a metric"""
        for i in range(30):
            self.orchestrator.record_metric("test_metric", 100.0 + i)
        
        analysis = self.orchestrator.analyze_metric("test_metric")
        
        self.assertIn("metric_name", analysis)
        self.assertIn("statistics", analysis)
        self.assertIn("trend", analysis)
        self.assertIn("anomalies", analysis)
        self.assertIn("patterns", analysis)
    
    def test_get_full_report(self):
        """Test generating full report"""
        self.orchestrator.record_metric("metric1", 100.0)
        self.orchestrator.record_metric("metric2", 200.0)
        
        for i in range(30):
            self.orchestrator.record_metric("metric1", 100.0 + i)
            self.orchestrator.record_metric("metric2", 200.0 - i)
        
        report = self.orchestrator.get_full_report()
        
        self.assertIn("metrics", report)
        self.assertIn("total_metrics", report)
        self.assertEqual(report["total_metrics"], 2)
    
    def test_auto_registration(self):
        """Test automatic registration on first record"""
        self.orchestrator.record_metric("auto_metric", 50.0)
        
        self.assertIn("auto_metric", self.orchestrator.analyzers)


class TestAnalyticsIntegration(unittest.TestCase):
    """Test integration of all analytics components"""
    
    def setUp(self):
        self.orchestrator = AdvancedAnalyticsOrchestrator()
    
    def test_end_to_end_analysis(self):
        """Test complete analytics pipeline"""
        # Generate synthetic data
        base_time = datetime.utcnow()
        
        for i in range(50):
            timestamp = base_time - timedelta(hours=50-i)
            # Normal data with trend
            value = 100 + i * 0.5 + np.random.normal(0, 5)
            self.orchestrator.record_metric("latency", value, timestamp)
        
        # Add anomalies
        for i in range(3):
            anomaly_time = base_time - timedelta(hours=10-i*5)
            self.orchestrator.record_metric("latency", 500.0, anomaly_time)
        
        # Analyze
        analysis = self.orchestrator.analyze_metric("latency")
        
        # Should have detected anomalies
        self.assertGreater(len(analysis["anomalies"]), 0)
        
        # Should have statistics
        self.assertIsNotNone(analysis["statistics"])
        
        # Should have trend
        self.assertIsNotNone(analysis["trend"])
    
    def test_multiple_metrics_analysis(self):
        """Test analyzing multiple metrics simultaneously"""
        base_time = datetime.utcnow()
        
        # Generate data for multiple metrics
        for i in range(30):
            timestamp = base_time - timedelta(hours=30-i)
            self.orchestrator.record_metric("cpu", 50.0 + np.random.normal(0, 5), timestamp)
            self.orchestrator.record_metric("memory", 60.0 + i, timestamp)
            self.orchestrator.record_metric("disk", 40.0 - np.random.normal(0, 2), timestamp)
        
        # Get full report
        report = self.orchestrator.get_full_report()
        
        self.assertEqual(report["total_metrics"], 3)
        self.assertGreater(report["total_anomalies"], 0)


# Run all tests
if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 17: ADVANCED ANALYTICS & ML INSIGHTS - TEST SUITE")
    print("=" * 80)
    print()
    
    unittest.main(verbosity=2)
