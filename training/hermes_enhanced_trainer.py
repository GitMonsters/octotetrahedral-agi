"""
HERMES Integration for ARC Training
====================================

Bridges background task orchestration with ARC training pipeline.
Enables:
- Background puzzle solving agents during training
- Asynchronous result collection
- Learning experience recording
- Task queue management

Usage:
    from training.hermes_enhanced_trainer import HermesTrainingIntegration
    
    hermes = HermesTrainingIntegration(enabled=True, learning_engine=trainer)
    
    # Start agents
    hermes.initialize_agents()
    
    # During training loop
    hermes.queue_solve_task(task_id="1a2e2828")
    
    # Collect results
    results = hermes.collect_results()
    
    # Generate report
    hermes.generate_summary()
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Result from a puzzle solve attempt."""
    task_id: str
    agent_id: str
    attempted_at: float
    duration_ms: float
    success: bool
    exact_match: bool = False
    grid_accuracy: float = 0.0
    cell_accuracy: float = 0.0
    confidence: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentStats:
    """Statistics for a background agent."""
    agent_id: str
    role: str
    tasks_queued: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_ms: float = 0.0
    avg_success_rate: float = 0.0
    avg_exact_match: float = 0.0
    avg_grid_accuracy: float = 0.0
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HermesTaskQueue:
    """Manages task queue for background agents."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queue: List[Dict[str, Any]] = []
        self.executing: Dict[str, Dict[str, Any]] = {}
        self.completed: List[SolveResult] = []
        self.failed: List[Tuple[str, str]] = []
    
    def enqueue(self, task_id: str, params: Dict[str, Any]) -> bool:
        """Add task to queue."""
        if len(self.queue) >= self.max_queue_size:
            logger.warning(f"Task queue full ({self.max_queue_size}), skipping {task_id}")
            return False
        
        self.queue.append({
            'task_id': task_id,
            'params': params,
            'queued_at': time.time(),
        })
        return True
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """Get next task from queue."""
        if not self.queue:
            return None
        return self.queue.pop(0)
    
    def mark_executing(self, task_id: str, agent_id: str):
        """Mark task as executing."""
        self.executing[task_id] = {
            'agent_id': agent_id,
            'started_at': time.time(),
        }
    
    def mark_completed(self, result: SolveResult):
        """Record completed task."""
        self.completed.append(result)
        if result.task_id in self.executing:
            del self.executing[result.task_id]
    
    def mark_failed(self, task_id: str, error: str):
        """Record failed task."""
        self.failed.append((task_id, error))
        if task_id in self.executing:
            del self.executing[task_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            'queued': len(self.queue),
            'executing': len(self.executing),
            'completed': len(self.completed),
            'failed': len(self.failed),
            'total_tasks': len(self.queue) + len(self.executing) + len(self.completed) + len(self.failed),
        }


class HermesTrainingIntegration:
    """
    Integrates HERMES background task orchestration with ARC training.
    
    Features:
    - Agent creation and lifecycle management
    - Task queueing during training
    - Asynchronous result collection
    - Learning signal recording
    - HTML report generation
    """
    
    def __init__(
        self,
        enabled: bool = False,
        learning_engine=None,
        log_frequency: int = 100,
        output_dir: str = "logs/hermes",
        max_parallel_agents: int = 4,
        max_queue_size: int = 1000
    ):
        """
        Initialize HERMES training integration.
        
        Args:
            enabled: Whether to enable HERMES integration
            learning_engine: Reference to training engine for feedback
            log_frequency: Report results every N steps
            output_dir: Directory for HTML reports
            max_parallel_agents: Maximum concurrent agents
            max_queue_size: Maximum tasks in queue
        """
        self.enabled = enabled
        self.learning_engine = learning_engine
        self.log_frequency = log_frequency
        self.output_dir = Path(output_dir)
        self.max_parallel_agents = max_parallel_agents
        
        # Create output directory
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task queue
        self.task_queue = HermesTaskQueue(max_queue_size=max_queue_size)
        
        # Agents
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_stats: Dict[str, AgentStats] = {}
        
        # Training integration
        self.session_start_time = time.time()
        self.last_report_time = time.time()
        self.total_tasks_queued = 0
        self.total_results_collected = 0
        
        # Learning signals
        self.learning_events: List[Dict[str, Any]] = []
    
    def initialize_agents(self, num_agents: int = 3):
        """Create and initialize background solving agents."""
        if not self.enabled:
            return
        
        logger.info(f"Initializing {num_agents} HERMES agents...")
        
        roles = ["solver", "validator", "learner"]
        
        for i in range(min(num_agents, self.max_parallel_agents)):
            agent_id = f"hermes-agent-{i:02d}"
            role = roles[i % len(roles)]
            
            self.agents[agent_id] = {
                'id': agent_id,
                'role': role,
                'created_at': datetime.now().isoformat(),
                'tasks_queued': 0,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_duration_ms': 0.0,
            }
            
            self.agent_stats[agent_id] = AgentStats(
                agent_id=agent_id,
                role=role,
                created_at=datetime.now().isoformat(),
            )
            
            logger.info(f"  Created {role} agent: {agent_id}")
    
    def queue_solve_task(
        self,
        task_id: str,
        params: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Queue a puzzle solving task.
        
        Args:
            task_id: Puzzle ID to solve
            params: Additional parameters
            agent_id: Specific agent (if None, use round-robin)
        
        Returns:
            Whether task was queued
        """
        if not self.enabled:
            return False
        
        # Default params
        if params is None:
            params = {}
        
        # Select agent (round-robin if not specified)
        if not agent_id:
            available = [
                a for a in self.agents.values()
                if a.get('tasks_queued', 0) <= self.max_parallel_agents
            ]
            if not available:
                logger.warning("No available agents for task queueing")
                return False
            agent = min(available, key=lambda a: a.get('tasks_queued', 0))
            agent_id = agent['id']
        
        # Enqueue
        task_params = {
            'task_id': task_id,
            'agent_id': agent_id,
            'timestamp': time.time(),
            **params
        }
        
        success = self.task_queue.enqueue(task_id, task_params)
        
        if success:
            self.agents[agent_id]['tasks_queued'] += 1
            self.total_tasks_queued += 1
            self.agent_stats[agent_id].tasks_queued += 1
        
        return success
    
    def record_solve_result(
        self,
        task_id: str,
        agent_id: str,
        duration_ms: float,
        success: bool,
        exact_match: bool = False,
        grid_accuracy: float = 0.0,
        cell_accuracy: float = 0.0,
        confidence: float = 0.0,
        error_message: Optional[str] = None
    ) -> SolveResult:
        """Record result from completed solve task."""
        if not self.enabled:
            return None
        
        result = SolveResult(
            task_id=task_id,
            agent_id=agent_id,
            attempted_at=time.time(),
            duration_ms=duration_ms,
            success=success,
            exact_match=exact_match,
            grid_accuracy=grid_accuracy,
            cell_accuracy=cell_accuracy,
            confidence=confidence,
            error_message=error_message,
        )
        
        # Update queue
        if success:
            self.task_queue.mark_completed(result)
            self.agents[agent_id]['tasks_completed'] += 1
            self.agent_stats[agent_id].tasks_completed += 1
        else:
            self.task_queue.mark_failed(task_id, error_message or "Unknown error")
            self.agents[agent_id]['tasks_failed'] += 1
            self.agent_stats[agent_id].tasks_failed += 1
        
        # Update statistics
        self.agent_stats[agent_id].total_duration_ms += duration_ms
        if self.agent_stats[agent_id].tasks_completed > 0:
            completed = self.agent_stats[agent_id].tasks_completed
            self.agent_stats[agent_id].avg_exact_match = (
                (self.agent_stats[agent_id].avg_exact_match * (completed - 1) + (1 if exact_match else 0)) / completed
            )
            self.agent_stats[agent_id].avg_grid_accuracy = (
                (self.agent_stats[agent_id].avg_grid_accuracy * (completed - 1) + grid_accuracy) / completed
            )
        
        # Record learning event
        self._record_learning_event({
            'type': 'solve_result',
            'task_id': task_id,
            'agent_id': agent_id,
            'success': success,
            'exact_match': exact_match,
            'grid_accuracy': grid_accuracy,
            'confidence': confidence,
        })
        
        self.total_results_collected += 1
        return result
    
    def _record_learning_event(self, event: Dict[str, Any]):
        """Record a learning event for later analysis."""
        event['timestamp'] = time.time()
        self.learning_events.append(event)
    
    def collect_results(self) -> Dict[str, Any]:
        """Get current task results."""
        if not self.enabled:
            return {}
        
        stats = self.task_queue.get_stats()
        completed_results = self.task_queue.completed
        
        return {
            'queue_stats': stats,
            'completed_count': len(completed_results),
            'failed_count': len(self.task_queue.failed),
            'results': [r.to_dict() for r in completed_results[-10:]],  # Last 10
        }
    
    def should_report(self, step: int) -> bool:
        """Check if it's time to report results."""
        return self.enabled and (step % self.log_frequency == 0)
    
    def get_summary(self, training_step: int = 0) -> Dict[str, Any]:
        """Get summary of HERMES activity."""
        if not self.enabled:
            return {}
        
        session_duration = time.time() - self.session_start_time
        queue_stats = self.task_queue.get_stats()
        
        agent_summaries = {}
        total_exact_match = 0
        total_grid_accuracy = 0
        total_success = 0
        
        for agent_id, stats in self.agent_stats.items():
            agent_summaries[agent_id] = {
                'role': stats.role,
                'tasks_queued': stats.tasks_queued,
                'tasks_completed': stats.tasks_completed,
                'tasks_failed': stats.tasks_failed,
                'success_rate': stats.tasks_completed / max(stats.tasks_queued, 1),
                'avg_exact_match': stats.avg_exact_match,
                'avg_grid_accuracy': stats.avg_grid_accuracy,
                'total_duration_ms': stats.total_duration_ms,
            }
            
            total_exact_match += stats.avg_exact_match * stats.tasks_completed
            total_grid_accuracy += stats.avg_grid_accuracy * stats.tasks_completed
            total_success += stats.tasks_completed
        
        overall_exact_match = total_exact_match / max(total_success, 1)
        overall_grid_accuracy = total_grid_accuracy / max(total_success, 1)
        
        return {
            'enabled': self.enabled,
            'training_step': training_step,
            'session_duration_s': session_duration,
            'queue_stats': queue_stats,
            'total_tasks_queued': self.total_tasks_queued,
            'total_results_collected': self.total_results_collected,
            'agent_summaries': agent_summaries,
            'overall_exact_match': overall_exact_match,
            'overall_grid_accuracy': overall_grid_accuracy,
            'learning_events_count': len(self.learning_events),
        }
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """Generate HTML report of HERMES activity."""
        if not self.enabled:
            return ""
        
        if output_path is None:
            output_path = self.output_dir / "hermes_training_report.html"
        else:
            output_path = Path(output_path)
        
        summary = self.get_summary()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HERMES Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
        h2 {{ color: #0066cc; margin-top: 20px; }}
        .summary {{ background: white; padding: 15px; border-radius: 5px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat {{ display: inline-block; margin-right: 30px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #0066cc; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        .agent-section {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #0066cc; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🤖 HERMES Training Integration Report</h1>
    <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
    
    <div class="summary">
        <h2>Session Overview</h2>
        <div class="stat">
            <div class="stat-value">{summary.get('total_tasks_queued', 0)}</div>
            <div class="stat-label">Tasks Queued</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary.get('total_results_collected', 0)}</div>
            <div class="stat-label">Results Collected</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary.get('overall_exact_match', 0):.1%}</div>
            <div class="stat-label">Exact Match Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary.get('overall_grid_accuracy', 0):.1%}</div>
            <div class="stat-label">Grid Accuracy</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary.get('session_duration_s', 0):.1f}s</div>
            <div class="stat-label">Session Duration</div>
        </div>
    </div>
    
    <div class="summary">
        <h2>Queue Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Tasks Queued</td>
                <td>{summary.get('queue_stats', {}).get('queued', 0)}</td>
            </tr>
            <tr>
                <td>Tasks Executing</td>
                <td>{summary.get('queue_stats', {}).get('executing', 0)}</td>
            </tr>
            <tr>
                <td>Tasks Completed</td>
                <td class="success">{summary.get('queue_stats', {}).get('completed', 0)}</td>
            </tr>
            <tr>
                <td>Tasks Failed</td>
                <td class="error">{summary.get('queue_stats', {}).get('failed', 0)}</td>
            </tr>
        </table>
    </div>
    
    <h2>Agent Performance</h2>
"""
        
        for agent_id, agent_stats in summary.get('agent_summaries', {}).items():
            html += f"""
    <div class="agent-section">
        <h3>{agent_id} ({agent_stats['role'].upper()})</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Tasks Queued</td>
                <td>{agent_stats['tasks_queued']}</td>
            </tr>
            <tr>
                <td>Completed</td>
                <td class="success">{agent_stats['tasks_completed']}</td>
            </tr>
            <tr>
                <td>Failed</td>
                <td class="error">{agent_stats['tasks_failed']}</td>
            </tr>
            <tr>
                <td>Success Rate</td>
                <td>{agent_stats['success_rate']:.1%}</td>
            </tr>
            <tr>
                <td>Exact Match Rate</td>
                <td>{agent_stats['avg_exact_match']:.1%}</td>
            </tr>
            <tr>
                <td>Grid Accuracy</td>
                <td>{agent_stats['avg_grid_accuracy']:.1%}</td>
            </tr>
            <tr>
                <td>Total Duration</td>
                <td>{agent_stats['total_duration_ms']:.1f}ms</td>
            </tr>
        </table>
    </div>
"""
        
        html += """
    <div class="summary">
        <h2>Learning Insights</h2>
        <p>Total learning events recorded: """ + str(summary.get('learning_events_count', 0)) + """</p>
        <p>These events capture solve attempts, successes, and failures for continuous learning.</p>
    </div>
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px;">
        <p>HERMES Training Integration Report - OctoTetrahedral AGI</p>
    </footer>
</body>
</html>
"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HERMES report saved to {output_path}")
        return str(output_path)
    
    def save_metrics_json(self, output_path: Optional[str] = None) -> str:
        """Save metrics as JSON for analysis."""
        if not self.enabled:
            return ""
        
        if output_path is None:
            output_path = self.output_dir / "hermes_training_metrics.json"
        else:
            output_path = Path(output_path)
        
        summary = self.get_summary()
        
        # Convert all agent stats to dict
        metrics = {
            'summary': summary,
            'learning_events': self.learning_events,
            'task_results': [r.to_dict() for r in self.task_queue.completed],
            'timestamp': datetime.now().isoformat(),
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"HERMES metrics saved to {output_path}")
        return str(output_path)
