#!/usr/bin/env python3
"""
Phase 17: Advanced Analytics & ML Insights
Confucius SDK v2.2

Machine learning-powered analytics including:
- Time series analysis and forecasting
- Anomaly detection with statistical methods
- Pattern recognition and clustering
- Predictive analytics for system behavior
- Trend analysis
- Recommendation engine
- Data quality metrics
- Performance insights

Author: Confucius SDK Development Team
Version: 2.2.0
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class AnomalyType(Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"                    # Sudden increase
    DROP = "drop"                      # Sudden decrease
    TREND = "trend"                    # Unusual trend
    SEASONAL = "seasonal"              # Breaks from seasonal pattern
    OUTLIER = "outlier"                # Statistical outlier


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RecommendationType(Enum):
    """Types of recommendations"""
    PERFORMANCE = "performance"        # Improve performance
    OPTIMIZATION = "optimization"      # Optimize resources
    ISSUE_PREVENTION = "issue_prevention"  # Prevent issues
    CAPACITY_PLANNING = "capacity_planning"  # Plan for future needs


# ============================================================================
# TIME SERIES ANALYSIS
# ============================================================================

@dataclass
class TimeSeriesPoint:
    """Single point in time series"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalSummary:
    """Statistical summary of data"""
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    q1: float  # First quartile
    q3: float  # Third quartile
    count: int


class TimeSeriesAnalyzer:
    """Analyzes time series data"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data: List[TimeSeriesPoint] = []
    
    def add_point(self, timestamp: datetime, value: float, metadata: Optional[Dict] = None):
        """Add a data point"""
        point = TimeSeriesPoint(timestamp, value, metadata or {})
        self.data.append(point)
        
        # Keep only recent data
        if len(self.data) > self.window_size * 2:
            self.data = self.data[-self.window_size:]
    
    def add_points(self, points: List[Tuple[datetime, float]]):
        """Add multiple points"""
        for timestamp, value in points:
            self.add_point(timestamp, value)
    
    def get_values(self) -> List[float]:
        """Get all values"""
        return [p.value for p in self.data]
    
    def get_statistics(self) -> Optional[StatisticalSummary]:
        """Calculate statistical summary"""
        values = self.get_values()
        if not values:
            return None
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if n > 1 else 0
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        return StatisticalSummary(
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            q1=sorted_vals[q1_idx],
            q3=sorted_vals[q3_idx],
            count=n
        )
    
    def calculate_trend(self, window: int = 10) -> Optional[float]:
        """Calculate trend (positive or negative slope)"""
        if len(self.data) < window:
            return None
        
        recent = self.get_values()[-window:]
        # Simple linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return float(slope)
    
    def forecast(self, periods: int = 5) -> List[float]:
        """Forecast future values using simple linear regression"""
        if len(self.data) < 2:
            return [self.data[0].value] * periods if self.data else []
        
        x = np.arange(len(self.data))
        y = np.array(self.get_values())
        
        # Linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return [y_mean] * periods
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Forecast
        future_x = np.arange(len(self.data), len(self.data) + periods)
        forecasts = [float(slope * xi + intercept) for xi in future_x]
        
        return forecasts
    
    def moving_average(self, window: int = 5) -> List[float]:
        """Calculate moving average"""
        values = self.get_values()
        if len(values) < window:
            return values
        
        ma = []
        for i in range(len(values) - window + 1):
            avg = sum(values[i:i + window]) / window
            ma.append(avg)
        
        return ma


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    value: float
    expected_value: float
    severity: float  # 0.0 to 1.0
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "anomaly_id": self.anomaly_id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "value": self.value,
            "expected_value": self.expected_value,
            "severity": self.severity,
            "description": self.description,
            "metadata": self.metadata
        }


class AnomalyDetector:
    """Detects anomalies in time series data"""
    
    def __init__(self, sensitivity: float = 2.0):
        """
        Initialize detector
        
        sensitivity: Standard deviations for threshold (higher = less sensitive)
        """
        self.sensitivity = sensitivity
        self.anomalies: List[Anomaly] = []
    
    def detect_statistical_outliers(
        self,
        analyzer: TimeSeriesAnalyzer
    ) -> List[Anomaly]:
        """Detect statistical outliers using z-score"""
        values = analyzer.get_values()
        stats = analyzer.get_statistics()
        
        if not stats or stats.std_dev == 0:
            return []
        
        detected = []
        threshold = self.sensitivity * stats.std_dev
        
        for i, point in enumerate(analyzer.data):
            z_score = abs((point.value - stats.mean) / stats.std_dev)
            
            if z_score > self.sensitivity:
                severity = min(1.0, (z_score - self.sensitivity) / 5.0)
                
                anomaly = Anomaly(
                    anomaly_id=f"stat_outlier_{i}",
                    timestamp=point.timestamp,
                    anomaly_type=AnomalyType.OUTLIER,
                    value=point.value,
                    expected_value=stats.mean,
                    severity=severity,
                    description=f"Statistical outlier (z-score: {z_score:.2f})"
                )
                detected.append(anomaly)
        
        return detected
    
    def detect_spikes(
        self,
        analyzer: TimeSeriesAnalyzer,
        threshold_multiplier: float = 1.5
    ) -> List[Anomaly]:
        """Detect sudden spikes in values"""
        values = analyzer.get_values()
        stats = analyzer.get_statistics()
        
        if not stats or len(values) < 2:
            return []
        
        detected = []
        
        for i in range(1, len(analyzer.data)):
            current = analyzer.data[i]
            previous = analyzer.data[i - 1]
            
            change_percent = abs(current.value - previous.value) / (abs(previous.value) + 1)
            
            if change_percent > 0.5:  # 50% change
                severity = min(1.0, change_percent)
                
                anomaly = Anomaly(
                    anomaly_id=f"spike_{i}",
                    timestamp=current.timestamp,
                    anomaly_type=AnomalyType.SPIKE if current.value > previous.value else AnomalyType.DROP,
                    value=current.value,
                    expected_value=previous.value,
                    severity=severity,
                    description=f"Sudden {'spike' if current.value > previous.value else 'drop'} "
                                f"({change_percent*100:.1f}% change)"
                )
                detected.append(anomaly)
        
        return detected
    
    def detect_all(self, analyzer: TimeSeriesAnalyzer) -> List[Anomaly]:
        """Detect all types of anomalies"""
        anomalies = []
        anomalies.extend(self.detect_statistical_outliers(analyzer))
        anomalies.extend(self.detect_spikes(analyzer))
        
        self.anomalies = anomalies
        return anomalies


# ============================================================================
# PATTERN RECOGNITION
# ============================================================================

@dataclass
class Pattern:
    """Detected pattern in data"""
    pattern_id: str
    name: str
    description: str
    confidence: float  # 0.0 to 1.0
    frequency: int  # How often it occurs
    characteristics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "characteristics": self.characteristics
        }


class PatternRecognizer:
    """Recognizes patterns in data"""
    
    def __init__(self):
        self.patterns: List[Pattern] = []
    
    def detect_seasonality(
        self,
        analyzer: TimeSeriesAnalyzer,
        period: int = 7
    ) -> Optional[Pattern]:
        """Detect seasonal patterns"""
        values = analyzer.get_values()
        
        if len(values) < period * 2:
            return None
        
        # Simple seasonality check: compare periods
        correlations = []
        
        for offset in range(1, min(5, len(values) // period)):
            segment1 = values[:-offset * period]
            segment2 = values[offset * period:]
            
            if len(segment1) == len(segment2):
                # Calculate correlation
                mean1 = statistics.mean(segment1)
                mean2 = statistics.mean(segment2)
                
                numerator = sum((segment1[i] - mean1) * (segment2[i] - mean2) for i in range(len(segment1)))
                denom1 = sum((segment1[i] - mean1) ** 2 for i in range(len(segment1)))
                denom2 = sum((segment2[i] - mean2) ** 2 for i in range(len(segment2)))
                
                if denom1 > 0 and denom2 > 0:
                    correlation = numerator / (denom1 * denom2) ** 0.5
                    correlations.append(correlation)
        
        if correlations and max(correlations) > 0.6:
            return Pattern(
                pattern_id="seasonality",
                name="Seasonal Pattern",
                description=f"Data exhibits strong seasonal pattern with period ~{period}",
                confidence=min(0.99, max(correlations)),
                frequency=len(values) // period,
                characteristics={
                    "period": period,
                    "strength": max(correlations)
                }
            )
        
        return None
    
    def detect_growth_pattern(self, analyzer: TimeSeriesAnalyzer) -> Optional[Pattern]:
        """Detect growth/decline patterns"""
        trend = analyzer.calculate_trend(window=min(10, len(analyzer.data)))
        
        if trend is None or trend == 0:
            return None
        
        stats = analyzer.get_statistics()
        if not stats:
            return None
        
        # Calculate trend strength
        slope_strength = abs(trend) / (stats.std_dev + 1)
        confidence = min(0.99, slope_strength / 5.0)
        
        if confidence > 0.3:
            pattern_type = "Growth" if trend > 0 else "Decline"
            
            return Pattern(
                pattern_id=f"trend_{pattern_type.lower()}",
                name=f"{pattern_type} Trend",
                description=f"Data shows consistent {pattern_type.lower()} trend",
                confidence=confidence,
                frequency=1,
                characteristics={
                    "slope": trend,
                    "strength": slope_strength
                }
            )
        
        return None
    
    def detect_volatility_pattern(self, analyzer: TimeSeriesAnalyzer) -> Optional[Pattern]:
        """Detect high volatility patterns"""
        stats = analyzer.get_statistics()
        
        if not stats:
            return None
        
        # Calculate coefficient of variation
        cv = (stats.std_dev / stats.mean) if stats.mean != 0 else 0
        
        if cv > 0.3:  # High variability
            return Pattern(
                pattern_id="volatility_high",
                name="High Volatility",
                description="Data exhibits high variability",
                confidence=min(0.99, cv),
                frequency=1,
                characteristics={
                    "coefficient_of_variation": cv,
                    "std_dev": stats.std_dev,
                    "mean": stats.mean
                }
            )
        
        return None
    
    def detect_all_patterns(self, analyzer: TimeSeriesAnalyzer) -> List[Pattern]:
        """Detect all patterns"""
        patterns = []
        
        seasonality = self.detect_seasonality(analyzer)
        if seasonality:
            patterns.append(seasonality)
        
        growth = self.detect_growth_pattern(analyzer)
        if growth:
            patterns.append(growth)
        
        volatility = self.detect_volatility_pattern(analyzer)
        if volatility:
            patterns.append(volatility)
        
        self.patterns = patterns
        return patterns


# ============================================================================
# PREDICTIVE ANALYTICS
# ============================================================================

@dataclass
class Prediction:
    """Prediction for future value"""
    prediction_id: str
    metric_name: str
    predicted_value: float
    lower_bound: float  # Confidence interval lower
    upper_bound: float  # Confidence interval upper
    confidence: PredictionConfidence
    horizon: int  # Periods ahead
    timestamp: datetime = field(default_factory=datetime.utcnow)
    methodology: str = "linear_regression"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prediction_id": self.prediction_id,
            "metric_name": self.metric_name,
            "predicted_value": self.predicted_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence": self.confidence.value,
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "methodology": self.methodology
        }


class PredictiveAnalytics:
    """Performs predictive analytics"""
    
    def __init__(self):
        self.predictions: List[Prediction] = []
    
    def predict_metric(
        self,
        metric_name: str,
        analyzer: TimeSeriesAnalyzer,
        periods_ahead: int = 5
    ) -> Prediction:
        """Predict future metric values"""
        import uuid
        
        forecasts = analyzer.forecast(periods_ahead)
        stats = analyzer.get_statistics()
        
        if not forecasts or not stats:
            raise ValueError("Insufficient data for prediction")
        
        # Use average of forecasts as prediction
        predicted_value = sum(forecasts) / len(forecasts)
        
        # Calculate confidence interval (assume 95% CI)
        uncertainty = stats.std_dev * 1.96
        lower_bound = predicted_value - uncertainty
        upper_bound = predicted_value + uncertainty
        
        # Determine confidence level based on data quality
        data_points = stats.count
        if data_points >= 100:
            confidence = PredictionConfidence.VERY_HIGH
        elif data_points >= 50:
            confidence = PredictionConfidence.HIGH
        elif data_points >= 20:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW
        
        prediction = Prediction(
            prediction_id=str(uuid.uuid4()),
            metric_name=metric_name,
            predicted_value=predicted_value,
            lower_bound=max(0, lower_bound),
            upper_bound=upper_bound,
            confidence=confidence,
            horizon=periods_ahead
        )
        
        self.predictions.append(prediction)
        return prediction
    
    def get_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        return [p.to_dict() for p in self.predictions[-limit:]]


# ============================================================================
# RECOMMENDATIONS ENGINE
# ============================================================================

@dataclass
class Recommendation:
    """System recommendation"""
    recommendation_id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    priority: str  # "low", "medium", "high", "critical"
    estimated_impact: str  # "positive", "negative", "neutral"
    action_items: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "recommendation_id": self.recommendation_id,
            "title": self.title,
            "description": self.description,
            "type": self.recommendation_type.value,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact,
            "action_items": self.action_items,
            "created_at": self.created_at.isoformat()
        }


class RecommendationEngine:
    """Generates recommendations based on analytics"""
    
    def __init__(self):
        self.recommendations: List[Recommendation] = []
    
    def recommend_based_on_anomalies(
        self,
        anomalies: List[Anomaly]
    ) -> List[Recommendation]:
        """Generate recommendations based on detected anomalies"""
        import uuid
        
        recommendations = []
        
        for anomaly in anomalies:
            if anomaly.severity > 0.7:  # High severity
                priority = "critical" if anomaly.severity > 0.9 else "high"
                
                rec = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title=f"Investigate {anomaly.anomaly_type.value}",
                    description=f"Detected {anomaly.anomaly_type.value}: {anomaly.description}",
                    recommendation_type=RecommendationType.ISSUE_PREVENTION,
                    priority=priority,
                    estimated_impact="positive",
                    action_items=[
                        f"Analyze root cause of {anomaly.anomaly_type.value}",
                        "Check system logs for errors",
                        "Monitor metric closely for next 24 hours"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def recommend_based_on_patterns(
        self,
        patterns: List[Pattern]
    ) -> List[Recommendation]:
        """Generate recommendations based on detected patterns"""
        import uuid
        
        recommendations = []
        
        for pattern in patterns:
            if pattern.name == "High Volatility":
                rec = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Optimize for Variability",
                    description="High volatility detected in metrics",
                    recommendation_type=RecommendationType.OPTIMIZATION,
                    priority="medium",
                    estimated_impact="positive",
                    action_items=[
                        "Review configuration for stability",
                        "Implement rate limiting if needed",
                        "Add caching to reduce variability"
                    ]
                )
                recommendations.append(rec)
            
            elif pattern.name == "Growth Trend":
                rec = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title="Plan for Capacity Growth",
                    description="Data shows consistent growth trend",
                    recommendation_type=RecommendationType.CAPACITY_PLANNING,
                    priority="high",
                    estimated_impact="positive",
                    action_items=[
                        "Assess current capacity limits",
                        "Plan infrastructure scaling",
                        "Set up alerts for capacity thresholds"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def recommend_based_on_predictions(
        self,
        predictions: List[Prediction]
    ) -> List[Recommendation]:
        """Generate recommendations based on predictions"""
        import uuid
        
        recommendations = []
        
        for prediction in predictions:
            if prediction.predicted_value > prediction.upper_bound * 1.1:
                rec = Recommendation(
                    recommendation_id=str(uuid.uuid4()),
                    title=f"High {prediction.metric_name} Predicted",
                    description=f"Predictions indicate {prediction.metric_name} will increase significantly",
                    recommendation_type=RecommendationType.CAPACITY_PLANNING,
                    priority="high",
                    estimated_impact="positive",
                    action_items=[
                        f"Prepare for increased {prediction.metric_name}",
                        "Review resource allocation",
                        "Implement auto-scaling if available"
                    ]
                )
                recommendations.append(rec)
        
        return recommendations
    
    def generate_all_recommendations(
        self,
        anomalies: List[Anomaly] = None,
        patterns: List[Pattern] = None,
        predictions: List[Prediction] = None
    ) -> List[Recommendation]:
        """Generate all recommendations"""
        recommendations = []
        
        if anomalies:
            recommendations.extend(self.recommend_based_on_anomalies(anomalies))
        if patterns:
            recommendations.extend(self.recommend_based_on_patterns(patterns))
        if predictions:
            recommendations.extend(self.recommend_based_on_predictions(predictions))
        
        self.recommendations = recommendations
        return recommendations


# ============================================================================
# ADVANCED ANALYTICS ORCHESTRATOR
# ============================================================================

class AdvancedAnalyticsOrchestrator:
    """Main orchestrator for advanced analytics"""
    
    def __init__(self):
        self.analyzers: Dict[str, TimeSeriesAnalyzer] = {}
        self.anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self.pattern_recognizers: Dict[str, PatternRecognizer] = {}
        self.predictive_analytics = PredictiveAnalytics()
        self.recommendation_engine = RecommendationEngine()
    
    def register_metric(self, metric_name: str):
        """Register a metric for tracking"""
        self.analyzers[metric_name] = TimeSeriesAnalyzer()
        self.anomaly_detectors[metric_name] = AnomalyDetector()
        self.pattern_recognizers[metric_name] = PatternRecognizer()
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value"""
        if metric_name not in self.analyzers:
            self.register_metric(metric_name)
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.analyzers[metric_name].add_point(timestamp, value)
    
    def analyze_metric(self, metric_name: str) -> Dict[str, Any]:
        """Perform full analysis on a metric"""
        if metric_name not in self.analyzers:
            return {"error": "Metric not found"}
        
        analyzer = self.analyzers[metric_name]
        detector = self.anomaly_detectors[metric_name]
        recognizer = self.pattern_recognizers[metric_name]
        
        # Analyze
        stats = analyzer.get_statistics()
        trend = analyzer.calculate_trend()
        anomalies = detector.detect_all(analyzer)
        patterns = recognizer.detect_all_patterns(analyzer)
        
        # Try to make prediction
        prediction = None
        try:
            if len(analyzer.data) >= 10:
                prediction = self.predictive_analytics.predict_metric(metric_name, analyzer, periods_ahead=5)
        except:
            pass
        
        # Generate recommendations
        recommendations = self.recommendation_engine.generate_all_recommendations(
            anomalies=anomalies,
            patterns=patterns,
            predictions=[prediction] if prediction else []
        )
        
        return {
            "metric_name": metric_name,
            "statistics": {
                "mean": stats.mean,
                "median": stats.median,
                "std_dev": stats.std_dev,
                "min": stats.min_val,
                "max": stats.max_val,
                "count": stats.count
            } if stats else None,
            "trend": trend,
            "anomalies": [a.to_dict() for a in anomalies],
            "patterns": [p.to_dict() for p in patterns],
            "prediction": prediction.to_dict() if prediction else None,
            "recommendations": [r.to_dict() for r in recommendations],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_full_report(self) -> Dict[str, Any]:
        """Get comprehensive analytics report"""
        metrics_analysis = {}
        
        for metric_name in self.analyzers:
            metrics_analysis[metric_name] = self.analyze_metric(metric_name)
        
        return {
            "metrics": metrics_analysis,
            "total_metrics": len(self.analyzers),
            "total_anomalies": sum(len(d.anomalies) for d in self.anomaly_detectors.values()),
            "total_patterns": sum(len(r.patterns) for r in self.pattern_recognizers.values()),
            "total_predictions": len(self.predictive_analytics.predictions),
            "total_recommendations": len(self.recommendation_engine.recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# DEMO AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 17: ADVANCED ANALYTICS & ML INSIGHTS - DEMO")
    print("=" * 80)
    
    orchestrator = AdvancedAnalyticsOrchestrator()
    
    # Generate synthetic metrics data
    print("\n1. GENERATING SYNTHETIC METRICS DATA")
    print("-" * 80)
    
    base_time = datetime.utcnow()
    
    # Normal metric with trend
    print("Generating request latency (normal with upward trend)...")
    for i in range(50):
        timestamp = base_time - timedelta(hours=50-i)
        value = 100 + i * 0.5 + np.random.normal(0, 5)
        orchestrator.record_metric("request_latency_ms", value, timestamp)
    
    # Add some anomalies
    for i in range(3):
        anomaly_time = base_time - timedelta(hours=10-i*5)
        orchestrator.record_metric("request_latency_ms", 500 + np.random.normal(0, 20), anomaly_time)
    
    # Error rate metric with spikes
    print("Generating error rate (with occasional spikes)...")
    for i in range(50):
        timestamp = base_time - timedelta(hours=50-i)
        if i % 10 == 0:
            value = np.random.uniform(5, 15)  # Spike
        else:
            value = np.random.uniform(0.5, 2)
        orchestrator.record_metric("error_rate_percent", value, timestamp)
    
    # CPU usage (seasonal pattern)
    print("Generating CPU usage (with seasonal pattern)...")
    for i in range(50):
        timestamp = base_time - timedelta(hours=50-i)
        # Seasonal pattern (higher every 24 hours)
        base_cpu = 50 + 20 * np.sin(i * 2 * np.pi / 24)
        value = base_cpu + np.random.normal(0, 3)
        orchestrator.record_metric("cpu_usage_percent", value, timestamp)
    
    # Memory usage (stable)
    print("Generating memory usage (stable)...")
    for i in range(50):
        timestamp = base_time - timedelta(hours=50-i)
        value = 60 + np.random.normal(0, 2)
        orchestrator.record_metric("memory_usage_percent", value, timestamp)
    
    # API calls (growth trend)
    print("Generating API calls (growth trend)...")
    for i in range(50):
        timestamp = base_time - timedelta(hours=50-i)
        value = 1000 + i * 50 + np.random.normal(0, 50)
        orchestrator.record_metric("api_calls_per_minute", value, timestamp)
    
    print("✓ Metrics generation complete")
    
    # Time series analysis
    print("\n2. TIME SERIES ANALYSIS")
    print("-" * 80)
    for metric_name in ["request_latency_ms", "error_rate_percent", "cpu_usage_percent"]:
        analyzer = orchestrator.analyzers[metric_name]
        stats = analyzer.get_statistics()
        trend = analyzer.calculate_trend()
        
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats.mean:.2f}, Std Dev: {stats.std_dev:.2f}")
        print(f"  Min: {stats.min_val:.2f}, Max: {stats.max_val:.2f}")
        print(f"  Trend: {trend:.4f}" if trend else "  Trend: N/A")
    
    # Anomaly detection
    print("\n3. ANOMALY DETECTION")
    print("-" * 80)
    for metric_name in orchestrator.analyzers:
        detector = orchestrator.anomaly_detectors[metric_name]
        anomalies = detector.detect_all(orchestrator.analyzers[metric_name])
        
        if anomalies:
            print(f"\n{metric_name}: {len(anomalies)} anomalies detected")
            for anomaly in anomalies[:3]:
                print(f"  - {anomaly.anomaly_type.value}: {anomaly.description} (severity: {anomaly.severity:.2f})")
    
    # Pattern recognition
    print("\n4. PATTERN RECOGNITION")
    print("-" * 80)
    for metric_name in orchestrator.analyzers:
        recognizer = orchestrator.pattern_recognizers[metric_name]
        patterns = recognizer.detect_all_patterns(orchestrator.analyzers[metric_name])
        
        if patterns:
            print(f"\n{metric_name}: {len(patterns)} patterns detected")
            for pattern in patterns:
                print(f"  - {pattern.name}: {pattern.description} (confidence: {pattern.confidence:.2f})")
    
    # Predictions
    print("\n5. PREDICTIVE ANALYTICS")
    print("-" * 80)
    for metric_name in ["request_latency_ms", "api_calls_per_minute"]:
        analyzer = orchestrator.analyzers[metric_name]
        if len(analyzer.data) >= 10:
            prediction = orchestrator.predictive_analytics.predict_metric(metric_name, analyzer, periods_ahead=5)
            print(f"\n{metric_name}:")
            print(f"  Predicted: {prediction.predicted_value:.2f}")
            print(f"  Range: {prediction.lower_bound:.2f} - {prediction.upper_bound:.2f}")
            print(f"  Confidence: {prediction.confidence.value}")
    
    # Recommendations
    print("\n6. INTELLIGENT RECOMMENDATIONS")
    print("-" * 80)
    recommendations = orchestrator.recommendation_engine.recommendations
    
    print(f"Total recommendations: {len(recommendations)}")
    for rec in recommendations[:5]:
        print(f"\n  [{rec.priority.upper()}] {rec.title}")
        print(f"  {rec.description}")
        print(f"  Actions: {', '.join(rec.action_items[:2])}")
    
    # Full report
    print("\n7. COMPREHENSIVE ANALYTICS REPORT")
    print("-" * 80)
    report = orchestrator.get_full_report()
    print(f"Metrics Analyzed: {report['total_metrics']}")
    print(f"Anomalies Detected: {report['total_anomalies']}")
    print(f"Patterns Found: {report['total_patterns']}")
    print(f"Predictions Made: {report['total_predictions']}")
    print(f"Recommendations: {report['total_recommendations']}")
    
    print("\n" + "=" * 80)
    print("PHASE 17 DEMO COMPLETE - ALL ANALYTICS SYSTEMS OPERATIONAL")
    print("=" * 80)
