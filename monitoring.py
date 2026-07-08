"""Real-time coherence monitoring for the unified cognitive stack.

.. deprecated::
    This top-level module is retained for backward compatibility only.
    The canonical implementation now lives in the ``monitoring`` package
    (``monitoring/inference_monitor.py``).  Import directly from the package::

        from monitoring import InferenceMonitor, CoherenceAlert, MonitoringStats
"""

from monitoring.inference_monitor import (  # noqa: F401  (re-export)
    CoherenceAlert,
    InferenceMonitor,
    MonitoringStats,
)

__all__ = ["CoherenceAlert", "InferenceMonitor", "MonitoringStats"]
