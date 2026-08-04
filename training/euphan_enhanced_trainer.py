"""
EUPHAN Integration for ARC Training
Adds observability hooks to track 8-limb parallel processing during inference
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import time
import torch
from pathlib import Path
from dataclasses import dataclass, asdict

# We'll import EuphanCompoundBridge when available, but make it optional
try:
    from core.euphan_compound_bridge import EuphanCompoundBridge, LimbName
    HAS_EUPHAN = True
except ImportError:
    HAS_EUPHAN = False
    LimbName = None

logger = logging.getLogger(__name__)


@dataclass
class LimbEventRecord:
    """Single limb event during inference"""
    limb_name: str
    action: str
    timestamp: float
    duration: float
    confidence: float = 0.5
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    parent_event_id: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LimbEventLogger:
    """
    Tracks all limb events during a forward pass.
    
    Features:
    - Record timing for each limb
    - Extract confidence scores
    - Track parent/child relationships
    - Generate HTML visualization
    """
    
    def __init__(self, task_id: str = "inference", enabled: bool = True):
        """
        Initialize limb event logger.
        
        Args:
            task_id: ID for this inference task
            enabled: Whether to log events
        """
        self.enabled = enabled
        self.task_id = task_id
        self.events: List[LimbEventRecord] = []
        self.session_start_time = time.time()
        self.event_counter = 0
        
        logger.info(f"LimbEventLogger initialized for {task_id}")
    
    def log_limb_event(
        self,
        limb_name: str,
        action: str,
        duration: float,
        confidence: float = 0.5,
        input_shape: Optional[Tuple] = None,
        output_shape: Optional[Tuple] = None,
        parent_event_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Log a single limb event.
        
        Args:
            limb_name: Name of limb (e.g., 'reasoning', 'memory')
            action: What the limb did (e.g., 'forward', 'update')
            duration: How long it took (seconds)
            confidence: Confidence score (0-1)
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
            parent_event_id: ID of parent event (for hierarchy)
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        if not self.enabled:
            return -1
        
        event = LimbEventRecord(
            limb_name=limb_name,
            action=action,
            timestamp=time.time() - self.session_start_time,
            duration=duration,
            confidence=confidence,
            input_shape=input_shape,
            output_shape=output_shape,
            parent_event_id=parent_event_id,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.event_counter += 1
        
        return self.event_counter - 1
    
    def get_events(self) -> List[Dict]:
        """Get all events as dicts"""
        return [
            {
                **asdict(event),
                'event_id': i
            }
            for i, event in enumerate(self.events)
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.events:
            return {}
        
        limb_stats = {}
        for event in self.events:
            if event.limb_name not in limb_stats:
                limb_stats[event.limb_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            
            stats = limb_stats[event.limb_name]
            stats['count'] += 1
            stats['total_time'] += event.duration
            stats['confidences'].append(event.confidence)
        
        # Compute averages
        for limb_name, stats in limb_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
            del stats['confidences']
        
        return {
            'task_id': self.task_id,
            'num_events': len(self.events),
            'total_time': sum(e.duration for e in self.events),
            'limb_stats': limb_stats,
            'session_duration': time.time() - self.session_start_time
        }
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate HTML visualization of limb timeline.
        
        Args:
            output_path: Where to save HTML (optional)
        
        Returns:
            HTML string
        """
        events = self.get_events()
        stats = self.get_statistics()
        
        # Compute timeline positions (cumulative time)
        current_time = 0.0
        positions = []
        max_time = max((e['timestamp'] + e['duration'] for e in events), default=1.0)
        
        for event in events:
            positions.append({
                'start': event['timestamp'],
                'end': event['timestamp'] + event['duration'],
                'duration': event['duration'],
                'limb': event['limb_name'],
                'action': event['action'],
                'confidence': event['confidence']
            })
        
        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>EUPHAN Limb Timeline - {self.task_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 20px 0; }}
                .stat-box {{ background: #e8f4f8; padding: 10px; border-left: 4px solid #007acc; }}
                .timeline {{ margin: 20px 0; }}
                .limb-track {{ margin: 10px 0; padding: 5px; background: #f9f9f9; border: 1px solid #ddd; }}
                .limb-name {{ font-weight: bold; width: 120px; display: inline-block; }}
                .event-bar {{ 
                    display: inline-block; 
                    height: 30px;
                    background: #4CAF50;
                    border: 1px solid #333;
                    margin: 2px;
                    padding: 2px;
                    vertical-align: top;
                    border-radius: 3px;
                    color: white;
                    font-size: 11px;
                    line-height: 26px;
                    text-align: center;
                }}
                .legend {{ margin: 20px 0; }}
                .legend-item {{ display: inline-block; margin-right: 20px; }}
                .confidence-high {{ background: #4CAF50; }}
                .confidence-medium {{ background: #FFC107; }}
                .confidence-low {{ background: #f44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>EUPHAN Limb Event Timeline</h1>
                <p>Task: {self.task_id}</p>
                <p>Session Duration: {stats.get('session_duration', 0):.3f}s</p>
            </div>
            
            <div class="stats">
        """
        
        # Add statistics
        if 'limb_stats' in stats:
            for limb_name, limb_stat in stats['limb_stats'].items():
                html += f"""
                <div class="stat-box">
                    <strong>{limb_name}</strong><br>
                    Events: {limb_stat['count']}<br>
                    Time: {limb_stat['total_time']:.3f}s<br>
                    Conf: {limb_stat['avg_confidence']:.2f}
                </div>
                """
        
        html += """
            </div>
            
            <div class="timeline">
                <h2>Event Timeline</h2>
        """
        
        # Group by limb
        limbs_seen = {}
        for event in events:
            limb = event['limb_name']
            if limb not in limbs_seen:
                limbs_seen[limb] = []
            limbs_seen[limb].append(event)
        
        # Render timeline per limb
        for limb_name, limb_events in sorted(limbs_seen.items()):
            html += f'<div class="limb-track"><span class="limb-name">{limb_name}</span>'
            
            for event in limb_events:
                # Confidence color
                conf = event['confidence']
                if conf >= 0.7:
                    color_class = 'confidence-high'
                elif conf >= 0.4:
                    color_class = 'confidence-medium'
                else:
                    color_class = 'confidence-low'
                
                # Event bar (width proportional to duration)
                width_pct = (event['duration'] / max_time * 80) if max_time > 0 else 5
                html += f"""
                <span class="event-bar {color_class}" 
                      title="{event['action']} ({event['duration']:.3f}s, conf={event['confidence']:.2f})"
                      style="width: {width_pct}%">
                    {event['action']}
                </span>
                """
            
            html += '</div>'
        
        html += """
            </div>
            
            <div class="legend">
                <h3>Confidence Levels</h3>
                <div class="legend-item"><span class="event-bar confidence-high"></span> High (≥0.7)</div>
                <div class="legend-item"><span class="event-bar confidence-medium"></span> Medium (0.4-0.7)</div>
                <div class="legend-item"><span class="event-bar confidence-low"></span> Low (<0.4)</div>
            </div>
        </body>
        </html>
        """
        
        # Save if output path provided
        if output_path:
            Path(output_path).write_text(html)
            logger.info(f"HTML report saved to {output_path}")
        
        return html


class EuphanTrainingIntegration:
    """
    Integrates EUPHAN observability into training.
    
    Manages limb event logging across training steps/validation passes.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        use_euphan_bridge: bool = False,
        log_frequency: int = 100
    ):
        """
        Initialize EUPHAN integration.
        
        Args:
            enabled: Whether to log limb events
            use_euphan_bridge: Whether to use EuphanCompoundBridge (requires implementation)
            log_frequency: How often to generate reports (every N steps)
        """
        self.enabled = enabled
        self.use_euphan_bridge = use_euphan_bridge and HAS_EUPHAN
        self.log_frequency = log_frequency
        
        # Initialize bridge if available
        self.euphan_bridge = None
        if self.use_euphan_bridge:
            try:
                self.euphan_bridge = EuphanCompoundBridge()
                logger.info("EuphanCompoundBridge initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EuphanCompoundBridge: {e}")
                self.use_euphan_bridge = False
        
        # Current session logger
        self.current_logger: Optional[LimbEventLogger] = None
        self.step_count = 0
        self.all_reports = []
        
        if self.enabled:
            logger.info(f"EuphanTrainingIntegration initialized (log_freq={log_frequency})")
    
    def start_session(self, task_id: str = "inference") -> LimbEventLogger:
        """Start a new logging session"""
        if not self.enabled:
            return None
        
        self.current_logger = LimbEventLogger(task_id=task_id, enabled=True)
        return self.current_logger
    
    def end_session(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """End current session and generate report"""
        if not self.enabled or self.current_logger is None:
            return {}
        
        stats = self.current_logger.get_statistics()
        
        # Generate HTML if output dir specified
        if output_dir:
            output_path = Path(output_dir) / f"euphan_{self.current_logger.task_id}.html"
            self.current_logger.generate_html_report(str(output_path))
        
        self.step_count += 1
        self.all_reports.append(stats)
        
        return stats
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get statistics across all sessions"""
        if not self.all_reports:
            return {}
        
        return {
            'total_sessions': len(self.all_reports),
            'sessions': self.all_reports
        }
