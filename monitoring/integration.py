"""Integration module — wires MetricsRecorder, CLI monitor, and web dashboard together."""

from __future__ import annotations

import threading
from typing import Any

from monitoring.cli_monitor import CLIMonitor
from monitoring.config import MonitoringConfig
from monitoring.metrics_recorder import MetricsRecorder


class MonitoringSystem:
    """Coordinator that initialises and manages all monitoring layers.

    Usage (context manager)::

        with MonitoringSystem(model, enable_cli=True, enable_web=True) as monitor:
            model.forward(...)
            stats = monitor.get_stats()

    Or manually::

        monitor = MonitoringSystem(model)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(
        self,
        model: Any,
        config: MonitoringConfig | None = None,
        enable_cli: bool = False,
        enable_web: bool = False,
    ) -> None:
        self._model = model
        self._config = config or MonitoringConfig()
        self._enable_cli = enable_cli
        self._enable_web = enable_web

        self.recorder = MetricsRecorder(config=self._config)
        self._cli_monitor: CLIMonitor | None = None
        self._web_thread: threading.Thread | None = None
        self._web_app: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "MonitoringSystem":
        """Start all enabled monitoring components."""
        self.recorder.start_recording(self._model)

        if self._enable_cli:
            self._cli_monitor = CLIMonitor(
                recorder=self.recorder, config=self._config
            )
            self._cli_monitor.start()

        if self._enable_web:
            self._start_web()

        return self

    def stop(self) -> None:
        """Gracefully stop all monitoring components."""
        self.recorder.stop_recording()

        if self._cli_monitor is not None:
            self._cli_monitor.stop()
            self._cli_monitor = None

        if self._web_thread is not None:
            # Signal the web server to stop (best-effort; uvicorn needs SIGINT)
            self._web_thread = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return current rolling statistics from the recorder."""
        return self.recorder.get_rolling_stats()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MonitoringSystem":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_web(self) -> None:
        try:
            from monitoring.web_dashboard import create_app
        except ImportError:
            return

        try:
            import uvicorn
        except ImportError:
            return

        self._web_app = create_app(
            recorder=self.recorder, config=self._config
        )

        def _run() -> None:
            uvicorn.run(
                self._web_app,
                host="0.0.0.0",
                port=self._config.web_port,
                log_level="error",
            )

        self._web_thread = threading.Thread(target=_run, daemon=True)
        self._web_thread.start()
