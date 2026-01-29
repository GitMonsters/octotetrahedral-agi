"""
Phase 14: Analytics & Reporting Engine
======================================

Provides comprehensive analytics, cost analysis, performance tracking, 
usage pattern analysis, and automated reporting for the Confucius SDK.

Features:
- Cost analysis dashboard with spending trends
- Performance metrics tracking and trend analysis
- Usage pattern detection and anomaly detection
- Optimization recommendations
- Comprehensive HTML and JSON reporting
"""

import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum


# ============================================================================
# DATA MODELS
# ============================================================================

class TrendDirection(Enum):
    """Trend direction indicator"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class CostCategory(Enum):
    """Cost category for analysis"""
    COMPLETION = "completion"
    PROMPT = "prompt"
    CACHE = "cache"
    EMBEDDING = "embedding"


@dataclass
class CostRecord:
    """Individual cost record"""
    timestamp: datetime
    provider: str
    model: str
    tokens_used: int
    cost: float
    category: CostCategory = CostCategory.COMPLETION
    tokens_cached: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "cost": cost,
            "category": self.category.value,
            "tokens_cached": self.tokens_cached
        }


@dataclass
class PerformanceMetric:
    """Performance metric record"""
    timestamp: datetime
    service: str
    response_time_ms: float
    tokens_processed: int
    throughput_req_s: float
    error: bool = False
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "response_time_ms": self.response_time_ms,
            "tokens_processed": self.tokens_processed,
            "throughput_req_s": self.throughput_req_s,
            "error": self.error,
            "error_type": self.error_type
        }


@dataclass
class TrendMetric:
    """Trend metric with direction and percentage change"""
    metric_name: str
    current_value: float
    previous_value: float
    direction: TrendDirection
    percent_change: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "current": self.current_value,
            "previous": self.previous_value,
            "direction": self.direction.value,
            "percent_change": f"{self.percent_change:.2f}%"
        }


@dataclass
class UsagePattern:
    """Detected usage pattern"""
    pattern_type: str  # "peak_hour", "model_preference", "error_spike", etc.
    description: str
    frequency: str  # "hourly", "daily", "weekly"
    confidence: float  # 0.0-1.0
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": f"{self.confidence:.2f}",
            "data": self.data
        }


@dataclass
class OptimizationRecommendation:
    """Recommended optimization"""
    recommendation_id: str
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    estimated_savings: float  # Estimated cost savings in dollars
    estimated_latency_improvement_ms: float  # Estimated latency improvement
    implementation_effort: str  # "easy", "medium", "hard"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.recommendation_id,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "estimated_savings": f"${self.estimated_savings:.4f}",
            "latency_improvement_ms": f"{self.estimated_latency_improvement_ms:.2f}",
            "implementation_effort": self.implementation_effort
        }


# ============================================================================
# COST ANALYSIS ENGINE
# ============================================================================

class CostAnalyzer:
    """Analyzes costs across providers and models"""
    
    def __init__(self):
        self.cost_records: List[CostRecord] = []
        self.daily_breakdown: Dict[str, float] = {}
        self.provider_breakdown: Dict[str, float] = {}
        self.model_breakdown: Dict[str, float] = {}
        
    def add_cost_record(self, record: CostRecord) -> None:
        """Add a cost record"""
        self.cost_records.append(record)
        self._update_breakdowns(record)
    
    def _update_breakdowns(self, record: CostRecord) -> None:
        """Update cost breakdowns"""
        # Daily breakdown
        date_key = record.timestamp.strftime("%Y-%m-%d")
        self.daily_breakdown[date_key] = self.daily_breakdown.get(date_key, 0) + record.cost
        
        # Provider breakdown
        self.provider_breakdown[record.provider] = self.provider_breakdown.get(record.provider, 0) + record.cost
        
        # Model breakdown
        model_key = f"{record.provider}/{record.model}"
        self.model_breakdown[model_key] = self.model_breakdown.get(model_key, 0) + record.cost
    
    def get_total_cost(self) -> float:
        """Get total cost"""
        return sum(r.cost for r in self.cost_records)
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """Get daily costs for last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return {
            k: v for k, v in self.daily_breakdown.items()
            if datetime.fromisoformat(k) >= cutoff
        }
    
    def get_provider_costs(self) -> Dict[str, Dict[str, Any]]:
        """Get costs by provider with percentages"""
        total = self.get_total_cost()
        if total == 0:
            return {}
        
        return {
            provider: {
                "cost": cost,
                "percent": (cost / total * 100),
                "record_count": len([r for r in self.cost_records if r.provider == provider])
            }
            for provider, cost in self.provider_breakdown.items()
        }
    
    def get_model_costs(self) -> Dict[str, Dict[str, Any]]:
        """Get costs by model"""
        total = self.get_total_cost()
        if total == 0:
            return {}
        
        return {
            model: {
                "cost": cost,
                "percent": (cost / total * 100),
                "record_count": len([r for r in self.cost_records 
                                   if f"{r.provider}/{r.model}" == model])
            }
            for model, cost in self.model_breakdown.items()
        }
    
    def get_daily_average(self, days: int = 30) -> float:
        """Get average daily cost"""
        daily = self.get_daily_costs(days)
        if not daily:
            return 0.0
        return sum(daily.values()) / len(daily)
    
    def get_cost_trend(self, days: int = 30) -> TrendMetric:
        """Get cost trend (increasing/decreasing)"""
        daily = self.get_daily_costs(days)
        if len(daily) < 2:
            return TrendMetric("daily_cost", 0, 0, TrendDirection.STABLE, 0)
        
        sorted_days = sorted(daily.items())
        first_half_avg = statistics.mean([v for k, v in sorted_days[:len(sorted_days)//2]])
        second_half_avg = statistics.mean([v for k, v in sorted_days[len(sorted_days)//2:]])
        
        change = second_half_avg - first_half_avg
        percent_change = (change / first_half_avg * 100) if first_half_avg > 0 else 0
        
        direction = TrendDirection.UP if change > 0 else (TrendDirection.DOWN if change < 0 else TrendDirection.STABLE)
        
        return TrendMetric("daily_cost", second_half_avg, first_half_avg, direction, percent_change)
    
    def get_most_expensive_models(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get most expensive models"""
        sorted_models = sorted(self.model_breakdown.items(), key=lambda x: x[1], reverse=True)
        return sorted_models[:limit]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        total_cost = self.get_total_cost()
        daily_avg = self.get_daily_average()
        trend = self.get_cost_trend()
        
        # Project monthly cost
        projected_monthly = daily_avg * 30
        
        return {
            "total_cost": f"${total_cost:.4f}",
            "total_records": len(self.cost_records),
            "daily_average": f"${daily_avg:.4f}",
            "projected_monthly": f"${projected_monthly:.4f}",
            "trend": trend.to_dict(),
            "by_provider": self.get_provider_costs(),
            "by_model": self.get_model_costs(),
            "top_models": [
                {"model": m, "cost": f"${c:.4f}"}
                for m, c in self.get_most_expensive_models(5)
            ]
        }


# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """Analyzes performance metrics"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.service_stats: Dict[str, Dict[str, Any]] = {}
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add performance metric"""
        self.metrics.append(metric)
        self._update_stats(metric)
    
    def _update_stats(self, metric: PerformanceMetric) -> None:
        """Update service statistics"""
        if metric.service not in self.service_stats:
            self.service_stats[metric.service] = {
                "response_times": [],
                "error_count": 0,
                "total_count": 0
            }
        
        stats = self.service_stats[metric.service]
        stats["response_times"].append(metric.response_time_ms)
        stats["total_count"] += 1
        if metric.error:
            stats["error_count"] += 1
    
    def get_service_stats(self, service: str) -> Dict[str, Any]:
        """Get stats for a service"""
        if service not in self.service_stats:
            return {}
        
        stats = self.service_stats[service]
        times = stats["response_times"]
        
        if not times:
            return {
                "service": service,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "min_response_time_ms": 0,
                "max_response_time_ms": 0,
                "error_rate": 0,
                "total_requests": 0
            }
        
        sorted_times = sorted(times)
        total = stats["total_count"]
        error_rate = (stats["error_count"] / total * 100) if total > 0 else 0
        
        return {
            "service": service,
            "avg_response_time_ms": statistics.mean(times),
            "p95_response_time_ms": statistics.quantiles(times, n=20)[18] if len(times) > 1 else times[0],
            "p99_response_time_ms": statistics.quantiles(times, n=100)[98] if len(times) > 1 else times[0],
            "min_response_time_ms": min(times),
            "max_response_time_ms": max(times),
            "error_rate": f"{error_rate:.2f}%",
            "total_requests": total
        }
    
    def get_all_services_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all services"""
        return {
            service: self.get_service_stats(service)
            for service in self.service_stats.keys()
        }
    
    def get_performance_trend(self, service: str, hours: int = 24) -> TrendMetric:
        """Get performance trend for a service"""
        cutoff = datetime.now() - timedelta(hours=hours)
        service_metrics = [m for m in self.metrics if m.service == service and m.timestamp >= cutoff]
        
        if len(service_metrics) < 2:
            return TrendMetric(f"{service}_latency", 0, 0, TrendDirection.STABLE, 0)
        
        mid = len(service_metrics) // 2
        first_half = statistics.mean([m.response_time_ms for m in service_metrics[:mid]])
        second_half = statistics.mean([m.response_time_ms for m in service_metrics[mid:]])
        
        change = second_half - first_half
        percent_change = (change / first_half * 100) if first_half > 0 else 0
        
        # Lower latency is better, so trend is opposite
        direction = TrendDirection.DOWN if change < 0 else (TrendDirection.UP if change > 0 else TrendDirection.STABLE)
        
        return TrendMetric(f"{service}_latency", second_half, first_half, direction, percent_change)
    
    def get_throughput_stats(self, service: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get throughput statistics"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        service_metrics = [m for m in self.metrics if m.service == service and m.timestamp >= cutoff]
        
        if not service_metrics:
            return {"service": service, "requests_per_second": 0, "total_tokens": 0}
        
        request_count = len(service_metrics)
        total_tokens = sum(m.tokens_processed for m in service_metrics)
        time_span_seconds = (service_metrics[-1].timestamp - service_metrics[0].timestamp).total_seconds() or 1
        
        return {
            "service": service,
            "requests_per_second": request_count / time_span_seconds,
            "total_tokens": total_tokens,
            "tokens_per_second": total_tokens / time_span_seconds,
            "window_minutes": window_minutes
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "services": self.get_all_services_stats(),
            "trends": {
                service: self.get_performance_trend(service).to_dict()
                for service in self.service_stats.keys()
            },
            "throughput": {
                service: self.get_throughput_stats(service)
                for service in self.service_stats.keys()
            }
        }


# ============================================================================
# USAGE PATTERN ANALYZER
# ============================================================================

class PatternAnalyzer:
    """Detects usage patterns and anomalies"""
    
    def __init__(self):
        self.patterns: List[UsagePattern] = []
        self.hourly_usage: Dict[int, int] = {}  # hour -> count
        self.model_usage: Dict[str, int] = {}   # model -> count
    
    def add_usage(self, hour: int, model: str) -> None:
        """Record usage"""
        self.hourly_usage[hour] = self.hourly_usage.get(hour, 0) + 1
        self.model_usage[model] = self.model_usage.get(model, 0) + 1
    
    def analyze_peak_hours(self) -> Optional[UsagePattern]:
        """Analyze peak usage hours"""
        if not self.hourly_usage:
            return None
        
        avg_usage = statistics.mean(self.hourly_usage.values())
        peak_hours = [h for h, count in self.hourly_usage.items() if count > avg_usage * 1.5]
        
        if peak_hours:
            return UsagePattern(
                pattern_type="peak_hours",
                description=f"Peak usage detected at hours: {sorted(peak_hours)}",
                frequency="daily",
                confidence=0.85,
                data={"peak_hours": sorted(peak_hours), "avg_usage": avg_usage}
            )
        return None
    
    def analyze_model_preference(self) -> Optional[UsagePattern]:
        """Analyze model usage preferences"""
        if not self.model_usage:
            return None
        
        sorted_models = sorted(self.model_usage.items(), key=lambda x: x[1], reverse=True)
        top_model = sorted_models[0]
        total_usage = sum(self.model_usage.values())
        
        if total_usage == 0:
            return None
        
        top_model_percent = (top_model[1] / total_usage) * 100
        
        if top_model_percent > 50:
            return UsagePattern(
                pattern_type="model_preference",
                description=f"Strong preference for {top_model[0]} ({top_model_percent:.1f}% of usage)",
                frequency="ongoing",
                confidence=0.90,
                data={
                    "preferred_model": top_model[0],
                    "usage_percent": top_model_percent,
                    "top_5_models": [{"model": m, "count": c} for m, c in sorted_models[:5]]
                }
            )
        return None
    
    def get_all_patterns(self) -> List[UsagePattern]:
        """Get all detected patterns"""
        patterns = []
        
        peak = self.analyze_peak_hours()
        if peak:
            patterns.append(peak)
        
        preference = self.analyze_model_preference()
        if preference:
            patterns.append(preference)
        
        return patterns
    
    def get_patterns_summary(self) -> Dict[str, Any]:
        """Get patterns summary"""
        patterns = self.get_all_patterns()
        return {
            "pattern_count": len(patterns),
            "patterns": [p.to_dict() for p in patterns],
            "hourly_usage": self.hourly_usage,
            "model_usage": self.model_usage
        }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """Generates optimization recommendations"""
    
    def __init__(self, cost_analyzer: CostAnalyzer, perf_analyzer: PerformanceAnalyzer, 
                 pattern_analyzer: PatternAnalyzer):
        self.cost = cost_analyzer
        self.perf = perf_analyzer
        self.patterns = pattern_analyzer
        self.recommendations: List[OptimizationRecommendation] = []
    
    def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate all recommendations"""
        self.recommendations = []
        
        self._recommend_model_optimization()
        self._recommend_rate_limiting()
        self._recommend_caching()
        self._recommend_batching()
        self._recommend_error_reduction()
        self._recommend_load_balancing()
        
        return self.recommendations
    
    def _recommend_model_optimization(self) -> None:
        """Recommend cheaper model usage"""
        top_models = self.cost.get_most_expensive_models(3)
        total_cost = self.cost.get_total_cost()
        
        if total_cost == 0 or not top_models:
            return
        
        top_cost = top_models[0][1]
        percent = (top_cost / total_cost) * 100
        
        if percent > 40:
            self.recommendations.append(OptimizationRecommendation(
                recommendation_id="opt_001",
                priority="high",
                title="Switch to cheaper models",
                description=f"Model {top_models[0][0]} accounts for {percent:.1f}% of costs. Consider switching to cheaper alternatives like GPT-3.5 or Claude Haiku.",
                estimated_savings=top_cost * 0.3,
                estimated_latency_improvement_ms=-5,  # Might be slightly slower
                implementation_effort="medium"
            ))
    
    def _recommend_rate_limiting(self) -> None:
        """Recommend rate limiting based on patterns"""
        patterns_summary = self.patterns.get_patterns_summary()
        
        if patterns_summary.get("pattern_count", 0) > 0:
            self.recommendations.append(OptimizationRecommendation(
                recommendation_id="opt_002",
                priority="medium",
                title="Implement intelligent rate limiting",
                description="Usage patterns show predictable peak hours. Implement rate limiting that adapts to peak times.",
                estimated_savings=100 * 0.05,  # Rough estimate
                estimated_latency_improvement_ms=10,
                implementation_effort="easy"
            ))
    
    def _recommend_caching(self) -> None:
        """Recommend response caching"""
        self.recommendations.append(OptimizationRecommendation(
            recommendation_id="opt_003",
            priority="high",
            title="Implement response caching",
            description="Enable caching for frequently requested prompts. This can reduce API calls by 20-40%.",
            estimated_savings=self.cost.get_total_cost() * 0.25,
            estimated_latency_improvement_ms=50,
            implementation_effort="medium"
        ))
    
    def _recommend_batching(self) -> None:
        """Recommend request batching"""
        self.recommendations.append(OptimizationRecommendation(
            recommendation_id="opt_004",
            priority="medium",
            title="Implement batch processing",
            description="Group requests into batches for better throughput and cost efficiency.",
            estimated_savings=self.cost.get_total_cost() * 0.10,
            estimated_latency_improvement_ms=-20,  # Slightly higher latency for batching
            implementation_effort="medium"
        ))
    
    def _recommend_error_reduction(self) -> None:
        """Recommend error reduction"""
        perf_summary = self.perf.get_performance_summary()
        
        for service_data in perf_summary.get("services", {}).values():
            error_rate = float(service_data.get("error_rate", "0%").rstrip("%"))
            if error_rate > 5:
                self.recommendations.append(OptimizationRecommendation(
                    recommendation_id="opt_005",
                    priority="high",
                    title=f"Reduce error rate for {service_data['service']}",
                    description=f"Error rate of {error_rate:.1f}% is above acceptable threshold. Implement better error handling and retry logic.",
                    estimated_savings=0,  # Reduce wasted API calls
                    estimated_latency_improvement_ms=0,
                    implementation_effort="hard"
                ))
                break
    
    def _recommend_load_balancing(self) -> None:
        """Recommend load balancing"""
        self.recommendations.append(OptimizationRecommendation(
            recommendation_id="opt_006",
            priority="medium",
            title="Implement provider load balancing",
            description="Distribute requests across multiple providers for better reliability and potentially lower costs.",
            estimated_savings=self.cost.get_total_cost() * 0.15,
            estimated_latency_improvement_ms=0,
            implementation_effort="hard"
        ))
    
    def get_recommendations_summary(self) -> Dict[str, Any]:
        """Get recommendations summary"""
        recs = self.generate_recommendations()
        
        high_priority = [r for r in recs if r.priority == "high"]
        total_savings = sum(r.estimated_savings for r in recs)
        
        return {
            "total_recommendations": len(recs),
            "high_priority_count": len(high_priority),
            "estimated_total_savings": f"${total_savings:.4f}",
            "recommendations": [r.to_dict() for r in recs[:5]],  # Top 5
            "implementation_priority": [r.to_dict() for r in high_priority]
        }


# ============================================================================
# COMPREHENSIVE REPORTING ENGINE
# ============================================================================

class AnalyticsReporter:
    """Generates comprehensive reports"""
    
    def __init__(self, cost_analyzer: CostAnalyzer, perf_analyzer: PerformanceAnalyzer,
                 pattern_analyzer: PatternAnalyzer, recommendation_engine: RecommendationEngine):
        self.cost = cost_analyzer
        self.perf = perf_analyzer
        self.patterns = pattern_analyzer
        self.recommendations = recommendation_engine
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate comprehensive JSON report"""
        return {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_cost": self.cost.get_cost_summary()["total_cost"],
                "services_monitored": len(self.perf.service_stats),
                "patterns_detected": self.patterns.get_patterns_summary()["pattern_count"],
                "recommendations": len(self.recommendations.generate_recommendations())
            },
            "cost_analysis": self.cost.get_cost_summary(),
            "performance_analysis": self.perf.get_performance_summary(),
            "usage_patterns": self.patterns.get_patterns_summary(),
            "recommendations": self.recommendations.get_recommendations_summary()
        }
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report"""
        report = self.generate_json_report()
        
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confucius SDK Analytics Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            border-bottom: 2px solid #eee;
            padding-bottom: 30px;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section h2 {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        
        .section h2::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 30px;
            background: #667eea;
            margin-right: 15px;
            border-radius: 2px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border: 2px solid #667eea30;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .summary-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .table th {
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        .table td {
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        
        .table tr:hover {
            background: #f5f5f5;
        }
        
        .recommendation {
            background: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        .recommendation.high {
            border-left-color: #e74c3c;
        }
        
        .recommendation.medium {
            border-left-color: #f39c12;
        }
        
        .recommendation.low {
            border-left-color: #27ae60;
        }
        
        .recommendation h4 {
            color: #333;
            margin-bottom: 5px;
        }
        
        .priority-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .priority-badge.high {
            background: #e74c3c;
            color: white;
        }
        
        .priority-badge.medium {
            background: #f39c12;
            color: white;
        }
        
        .priority-badge.low {
            background: #27ae60;
            color: white;
        }
        
        .trend {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .trend.up {
            background: #e74c3c;
            color: white;
        }
        
        .trend.down {
            background: #27ae60;
            color: white;
        }
        
        .trend.stable {
            background: #95a5a6;
            color: white;
        }
        
        .pattern {
            background: #ecf0f1;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #667eea;
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Confucius SDK Analytics Report</h1>
            <p>Generated: """ + report["report_timestamp"] + """</p>
        </div>
        
        <div class="content">
            <!-- SUMMARY SECTION -->
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Total Cost</h3>
                        <div class="value">""" + str(report["summary"]["total_cost"]) + """</div>
                    </div>
                    <div class="summary-card">
                        <h3>Services</h3>
                        <div class="value">""" + str(report["summary"]["services_monitored"]) + """</div>
                    </div>
                    <div class="summary-card">
                        <h3>Patterns Detected</h3>
                        <div class="value">""" + str(report["summary"]["patterns_detected"]) + """</div>
                    </div>
                    <div class="summary-card">
                        <h3>Recommendations</h3>
                        <div class="value">""" + str(report["summary"]["recommendations"]) + """</div>
                    </div>
                </div>
            </div>
            
            <!-- COST ANALYSIS SECTION -->
            <div class="section">
                <h2>Cost Analysis</h2>
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Daily Average</h3>
                        <div class="value">""" + str(report["cost_analysis"]["daily_average"]) + """</div>
                    </div>
                    <div class="summary-card">
                        <h3>Projected Monthly</h3>
                        <div class="value">""" + str(report["cost_analysis"]["projected_monthly"]) + """</div>
                    </div>
                    <div class="summary-card">
                        <h3>Total Records</h3>
                        <div class="value">""" + str(report["cost_analysis"]["total_records"]) + """</div>
                    </div>
                </div>
                
                <h3>Cost Trend</h3>
                <div class="trend """ + report["cost_analysis"]["trend"]["direction"] + """">
                    """ + report["cost_analysis"]["trend"]["direction"].upper() + """ """ + report["cost_analysis"]["trend"]["percent_change"] + """
                </div>
            </div>
            
            <!-- PERFORMANCE SECTION -->
            <div class="section">
                <h2>Performance Metrics</h2>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Service</th>
                            <th>Avg Response Time</th>
                            <th>P95 Response Time</th>
                            <th>Error Rate</th>
                            <th>Total Requests</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for service, stats in report["performance_analysis"].get("services", {}).items():
            html += f"""
                        <tr>
                            <td>{stats.get('service', 'N/A')}</td>
                            <td>{stats.get('avg_response_time_ms', 0):.2f}ms</td>
                            <td>{stats.get('p95_response_time_ms', 0):.2f}ms</td>
                            <td>{stats.get('error_rate', '0%')}</td>
                            <td>{stats.get('total_requests', 0)}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <!-- USAGE PATTERNS SECTION -->
            <div class="section">
                <h2>Usage Patterns</h2>
        """
        
        for pattern in report["usage_patterns"].get("patterns", []):
            html += f"""
                <div class="pattern">
                    <h4>{pattern['type'].upper()}</h4>
                    <p>{pattern['description']}</p>
                    <p><small>Confidence: {pattern['confidence']}</small></p>
                </div>
            """
        
        html += """
            </div>
            
            <!-- RECOMMENDATIONS SECTION -->
            <div class="section">
                <h2>Optimization Recommendations</h2>
                <p><strong>Estimated Total Savings:</strong> """ + str(report["recommendations"]["estimated_total_savings"]) + """</p>
        """
        
        for rec in report["recommendations"].get("recommendations", []):
            html += f"""
                <div class="recommendation {rec['priority'].lower()}">
                    <h4>{rec['title']}<span class="priority-badge {rec['priority'].lower()}">{rec['priority'].upper()}</span></h4>
                    <p>{rec['description']}</p>
                    <p><small>
                        Estimated Savings: {rec['estimated_savings']} |
                        Implementation: {rec['implementation_effort']}
                    </small></p>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the Confucius SDK Analytics Engine (Phase 14)</p>
            <p>For more information, visit: <a href="https://github.com/GitMonsters/octotetrahedral-agi">Confucius SDK Repository</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def save_json_report(self, filename: str) -> None:
        """Save JSON report to file"""
        with open(filename, 'w') as f:
            json.dump(self.generate_json_report(), f, indent=2)
    
    def save_html_report(self, filename: str) -> None:
        """Save HTML report to file"""
        with open(filename, 'w') as f:
            f.write(self.generate_html_report())


# ============================================================================
# ANALYTICS ENGINE FACADE
# ============================================================================

class AnalyticsEngine:
    """Main analytics engine combining all components"""
    
    def __init__(self):
        self.cost_analyzer = CostAnalyzer()
        self.perf_analyzer = PerformanceAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.recommendation_engine = RecommendationEngine(
            self.cost_analyzer,
            self.perf_analyzer,
            self.pattern_analyzer
        )
        self.reporter = AnalyticsReporter(
            self.cost_analyzer,
            self.perf_analyzer,
            self.pattern_analyzer,
            self.recommendation_engine
        )
    
    def add_cost_record(self, timestamp: datetime, provider: str, model: str, 
                       tokens: int, cost: float, category: CostCategory = CostCategory.COMPLETION) -> None:
        """Add a cost record"""
        record = CostRecord(timestamp, provider, model, tokens, cost, category)
        self.cost_analyzer.add_cost_record(record)
    
    def add_performance_metric(self, timestamp: datetime, service: str, 
                             response_time_ms: float, tokens_processed: int,
                             throughput_req_s: float, error: bool = False,
                             error_type: Optional[str] = None) -> None:
        """Add a performance metric"""
        metric = PerformanceMetric(timestamp, service, response_time_ms, 
                                  tokens_processed, throughput_req_s, error, error_type)
        self.perf_analyzer.add_metric(metric)
    
    def record_usage(self, hour: int, model: str) -> None:
        """Record usage pattern"""
        self.pattern_analyzer.add_usage(hour, model)
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive report"""
        return self.reporter.generate_json_report()
    
    def get_html_report(self) -> str:
        """Get HTML report"""
        return self.reporter.generate_html_report()
    
    def save_reports(self, json_file: str, html_file: str) -> None:
        """Save both reports"""
        self.reporter.save_json_report(json_file)
        self.reporter.save_html_report(html_file)


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo_analytics_engine():
    """Demonstrate the analytics engine"""
    engine = AnalyticsEngine()
    
    print("Confucius SDK - Phase 14: Analytics & Reporting Engine")
    print("=" * 60)
    
    # Add sample cost records
    print("\nAdding sample cost records...")
    for i in range(10):
        engine.add_cost_record(
            datetime.now() - timedelta(hours=i),
            provider="openai" if i % 3 == 0 else "anthropic",
            model="gpt-4" if i % 3 == 0 else "claude-3-sonnet",
            tokens=1000 + (i * 100),
            cost=0.05 + (i * 0.01)
        )
    
    # Add performance metrics
    print("Adding sample performance metrics...")
    for i in range(20):
        engine.add_performance_metric(
            datetime.now() - timedelta(minutes=i),
            service="llm-api",
            response_time_ms=100 + (i * 2),
            tokens_processed=500,
            throughput_req_s=10.5,
            error=(i % 20 == 0)
        )
    
    # Record usage patterns
    print("Recording usage patterns...")
    for hour in range(24):
        for _ in range(5 if hour in [9, 14, 18] else 2):
            engine.record_usage(hour, "gpt-4" if hour < 12 else "claude-3-sonnet")
    
    # Get report
    print("\nGenerating analytics report...")
    report = engine.get_report()
    
    print("\n" + "=" * 60)
    print("ANALYTICS REPORT")
    print("=" * 60)
    
    print("\nExecution Summary:")
    for key, value in report["summary"].items():
        print(f"  • {key}: {value}")
    
    print("\nCost Summary:")
    print(f"  • Total Cost: {report['cost_analysis']['total_cost']}")
    print(f"  • Daily Average: {report['cost_analysis']['daily_average']}")
    print(f"  • Projected Monthly: {report['cost_analysis']['projected_monthly']}")
    print(f"  • Trend: {report['cost_analysis']['trend']['direction'].upper()} {report['cost_analysis']['trend']['percent_change']}")
    
    print("\nPerformance Summary:")
    for service, stats in report["performance_analysis"]["services"].items():
        print(f"  • {service}:")
        print(f"      - Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        print(f"      - P95 Response Time: {stats['p95_response_time_ms']:.2f}ms")
        print(f"      - Error Rate: {stats['error_rate']}")
    
    print("\nUsage Patterns Detected:")
    print(f"  • Total Patterns: {report['usage_patterns']['pattern_count']}")
    for pattern in report["usage_patterns"]["patterns"]:
        print(f"  • {pattern['type']}: {pattern['description']}")
    
    print("\nTop Recommendations:")
    print(f"  • Estimated Total Savings: {report['recommendations']['estimated_total_savings']}")
    print(f"  • High Priority Items: {report['recommendations']['high_priority_count']}")
    for rec in report["recommendations"]["recommendations"][:3]:
        print(f"  • {rec['title']} (Priority: {rec['priority']})")
    
    # Save reports
    print("\nSaving reports...")
    engine.save_reports("analytics_report.json", "analytics_report.html")
    print("  ✓ Saved: analytics_report.json")
    print("  ✓ Saved: analytics_report.html")
    
    print("\n" + "=" * 60)
    print("Phase 14 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_analytics_engine()
