"""
Agent Communication Protocol  (ACP)
=====================================
Multi-agent coordination layer for the Recursive Engine.

Inspired by OpenClaw's ACP design: multiple agents collaborate on a single
task across different software environments. Each agent has a role, a
communication channel, and can broadcast or consume messages from a shared
message bus.

Design
------
    AgentMessage        — typed message between agents
    MessageBus          — thread-safe publish/subscribe bus
    AgentRole           — enum of cognitive roles (maps to 8 limbs)
    AgentNode           — single agent participant on the bus
    ACPOrchestrator     — coordinates a team of AgentNodes on a shared task
    HeartbeatScheduler  — proactive 30-min check-in (OpenClaw Heartbeat)

The ACP lets the Recursive Engine's 8 specialized limbs
(Memory, Planning, Language, Spatial, Reasoning,
MetaCognition, Perception, Action) act as separate communicating agents
rather than monolithic forward passes.

Usage
-----
    bus = MessageBus()

    planning_agent = AgentNode(AgentRole.PLANNING, bus)
    action_agent   = AgentNode(AgentRole.ACTION,   bus)

    orchestrator = ACPOrchestrator(bus, agents=[planning_agent, action_agent])
    orchestrator.broadcast(task="Solve ARC task abc82100")

    # Agents pick up the message and respond asynchronously
    response = action_agent.receive(timeout=5.0)
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Agent roles  (mirror the 8 cognitive limbs)
# ─────────────────────────────────────────────────────────────────────────────

class AgentRole(Enum):
    MEMORY          = "memory"
    PLANNING        = "planning"
    LANGUAGE        = "language"
    SPATIAL         = "spatial"
    REASONING       = "reasoning"
    META_COGNITION  = "metacognition"
    PERCEPTION      = "perception"
    ACTION          = "action"
    ORCHESTRATOR    = "orchestrator"   # coordination role


# ─────────────────────────────────────────────────────────────────────────────
# Message types
# ─────────────────────────────────────────────────────────────────────────────

class MessageType(Enum):
    TASK_ASSIGN    = "task_assign"    # Orchestrator → agent: here's your sub-task
    TASK_RESULT    = "task_result"    # Agent → orchestrator: here's my output
    BROADCAST      = "broadcast"      # One → all: share context
    QUERY          = "query"          # Agent → agent: ask a question
    RESPONSE       = "response"       # Agent → agent: answer a query
    HEARTBEAT      = "heartbeat"      # Scheduler → user: proactive check-in
    STATUS_UPDATE  = "status_update"  # Agent → orchestrator: progress ping
    ERROR          = "error"          # Any → orchestrator: signal failure


@dataclass
class AgentMessage:
    msg_type:  MessageType
    sender:    AgentRole
    recipient: Optional[AgentRole]          # None = broadcast
    payload:   Dict[str, Any]
    msg_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    reply_to:  Optional[str] = None         # msg_id of message being answered

    def reply(self, sender: AgentRole, payload: Dict[str, Any]) -> "AgentMessage":
        return AgentMessage(
            msg_type=MessageType.RESPONSE,
            sender=sender,
            recipient=self.sender,
            payload=payload,
            reply_to=self.msg_id,
        )

    def to_json(self) -> str:
        return json.dumps({
            "msg_id":    self.msg_id,
            "msg_type":  self.msg_type.value,
            "sender":    self.sender.value,
            "recipient": self.recipient.value if self.recipient else None,
            "payload":   self.payload,
            "timestamp": self.timestamp,
            "reply_to":  self.reply_to,
        })

    @classmethod
    def from_json(cls, raw: str) -> "AgentMessage":
        d = json.loads(raw)
        return cls(
            msg_type=MessageType(d["msg_type"]),
            sender=AgentRole(d["sender"]),
            recipient=AgentRole(d["recipient"]) if d.get("recipient") else None,
            payload=d["payload"],
            msg_id=d["msg_id"],
            timestamp=d["timestamp"],
            reply_to=d.get("reply_to"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Message bus
# ─────────────────────────────────────────────────────────────────────────────

class MessageBus:
    """
    Thread-safe publish/subscribe message bus.

    Agents subscribe to their own role channel.  Broadcast messages are
    delivered to all subscribers.  The bus also keeps a bounded history
    for replay / audit.
    """

    MAX_HISTORY = 1000

    def __init__(self):
        self._queues:  Dict[AgentRole, queue.Queue] = {}
        self._lock     = threading.Lock()
        self._history: List[AgentMessage] = []

    def subscribe(self, role: AgentRole) -> None:
        with self._lock:
            if role not in self._queues:
                self._queues[role] = queue.Queue()

    def publish(self, msg: AgentMessage) -> None:
        """Route message to recipient queue (or all queues if broadcast)."""
        with self._lock:
            self._history.append(msg)
            if len(self._history) > self.MAX_HISTORY:
                self._history.pop(0)

            if msg.recipient is None:
                # Broadcast to all subscribers
                for q in self._queues.values():
                    q.put(msg)
            else:
                if msg.recipient in self._queues:
                    self._queues[msg.recipient].put(msg)

    def receive(
        self,
        role: AgentRole,
        timeout: float = 1.0,
    ) -> Optional[AgentMessage]:
        """Non-blocking receive with timeout."""
        if role not in self._queues:
            self.subscribe(role)
        try:
            return self._queues[role].get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, role: AgentRole) -> List[AgentMessage]:
        """Drain all pending messages for a role without blocking."""
        msgs = []
        if role not in self._queues:
            return msgs
        while True:
            try:
                msgs.append(self._queues[role].get_nowait())
            except queue.Empty:
                break
        return msgs

    @property
    def history(self) -> List[AgentMessage]:
        with self._lock:
            return list(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# Agent node
# ─────────────────────────────────────────────────────────────────────────────

class AgentNode:
    """
    A single participant on the ACP message bus.

    Subclass this and override `on_message()` to implement agent behaviour.
    Alternatively, register a `handler` callback.
    """

    def __init__(
        self,
        role: AgentRole,
        bus: MessageBus,
        handler: Optional[Callable[[AgentMessage], Optional[AgentMessage]]] = None,
    ):
        self.role    = role
        self.bus     = bus
        self.handler = handler
        bus.subscribe(role)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def send(
        self,
        recipient: Optional[AgentRole],
        msg_type: MessageType,
        payload: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> str:
        """Send a message; returns its msg_id."""
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self.role,
            recipient=recipient,
            payload=payload,
            reply_to=reply_to,
        )
        self.bus.publish(msg)
        return msg.msg_id

    def receive(self, timeout: float = 1.0) -> Optional[AgentMessage]:
        return self.bus.receive(self.role, timeout=timeout)

    def reply(self, original: AgentMessage, payload: Dict[str, Any]) -> None:
        self.bus.publish(original.reply(self.role, payload))

    def on_message(self, msg: AgentMessage) -> Optional[AgentMessage]:
        """Override in subclasses. Return a reply message or None."""
        if self.handler:
            return self.handler(msg)
        return None

    def start_listening(self, poll_interval: float = 0.1) -> None:
        """Start background thread that processes incoming messages."""
        self._running = True

        def loop():
            while self._running:
                msg = self.receive(timeout=poll_interval)
                if msg:
                    response = self.on_message(msg)
                    if response:
                        self.bus.publish(response)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ACPOrchestrator:
    """
    Coordinates a team of AgentNodes on a shared task.

    Implements the ACP pattern:
        broadcast(task) → decompose → assign sub-tasks → collect results → merge

    Maps to the Recursive Engine's meta-learner layer (M_ψ) which decides
    which modules to activate and how to combine their outputs.
    """

    def __init__(self, bus: MessageBus, agents: List[AgentNode]):
        self.bus    = bus
        self.agents = {a.role: a for a in agents}
        self.node   = AgentNode(AgentRole.ORCHESTRATOR, bus)
        self._results: Dict[str, Any] = {}

    def broadcast(self, task: str, context: Optional[Dict] = None) -> str:
        """Broadcast a task to all registered agents. Returns broadcast msg_id."""
        return self.node.send(
            recipient=None,
            msg_type=MessageType.BROADCAST,
            payload={"task": task, "context": context or {}},
        )

    def assign(
        self, role: AgentRole, sub_task: str, context: Optional[Dict] = None
    ) -> str:
        """Assign a sub-task to a specific agent role. Returns msg_id."""
        return self.node.send(
            recipient=role,
            msg_type=MessageType.TASK_ASSIGN,
            payload={"sub_task": sub_task, "context": context or {}},
        )

    def collect_results(
        self, timeout_per_agent: float = 5.0
    ) -> Dict[AgentRole, Any]:
        """
        Wait for TASK_RESULT messages from all registered agents.
        Returns dict of role → result payload.
        """
        results: Dict[AgentRole, Any] = {}
        deadline = time.time() + timeout_per_agent * len(self.agents)

        while len(results) < len(self.agents) and time.time() < deadline:
            msg = self.node.receive(timeout=0.5)
            if msg and msg.msg_type == MessageType.TASK_RESULT:
                results[msg.sender] = msg.payload
        return results

    def run_task(
        self,
        task: str,
        decompose: Optional[Dict[AgentRole, str]] = None,
        timeout: float = 30.0,
    ) -> Dict[AgentRole, Any]:
        """
        Full orchestration cycle:
        1. Broadcast task (or assign decomposed sub-tasks)
        2. Collect results from all agents
        3. Return merged result dict

        Args
        ----
        task        : high-level task description
        decompose   : optional {role: sub_task} mapping for fine-grained control
        timeout     : total seconds to wait for all results
        """
        if decompose:
            for role, sub_task in decompose.items():
                self.assign(role, sub_task)
        else:
            self.broadcast(task)

        return self.collect_results(timeout_per_agent=timeout / max(len(self.agents), 1))

    def start_all(self) -> None:
        for agent in self.agents.values():
            agent.start_listening()

    def stop_all(self) -> None:
        for agent in self.agents.values():
            agent.stop_listening()


# ─────────────────────────────────────────────────────────────────────────────
# Heartbeat Scheduler  (OpenClaw Heartbeat — proactive 30-min check-in)
# ─────────────────────────────────────────────────────────────────────────────

HeartbeatCallback = Callable[["HeartbeatScheduler"], None]


class HeartbeatScheduler:
    """
    Proactive heartbeat system — the Recursive Engine checks in on its own
    schedule rather than waiting to be queried.

    Inspired by OpenClaw's 30-minute Heartbeat feature.  At each tick:
        - Runs the user-supplied callback (e.g., check task queue, post update)
        - Publishes a HEARTBEAT message on the ACP bus if one is provided
        - Records tick history for meta-learning introspection

    Usage
    -----
        def on_heartbeat(sched):
            print(f"Tick #{sched.tick_count}: agent is alive")

        scheduler = HeartbeatScheduler(
            interval_sec=1800,      # 30 minutes
            callback=on_heartbeat,
        )
        scheduler.start()
        # ... application runs ...
        scheduler.stop()
    """

    DEFAULT_INTERVAL = 1800  # 30 minutes in seconds

    def __init__(
        self,
        interval_sec: float = DEFAULT_INTERVAL,
        callback: Optional[HeartbeatCallback] = None,
        bus: Optional[MessageBus] = None,
        jitter_sec: float = 0.0,   # random jitter to avoid thundering herd
    ):
        self.interval_sec = interval_sec
        self.callback     = callback
        self.bus          = bus
        self.jitter_sec   = jitter_sec
        self.tick_count   = 0
        self.last_tick:   Optional[float] = None
        self._thread:     Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._tick_history: List[Dict[str, Any]] = []

    def _tick(self) -> None:
        import random
        self.tick_count += 1
        self.last_tick = time.time()
        record = {
            "tick": self.tick_count,
            "timestamp": self.last_tick,
        }

        # Run user callback
        if self.callback:
            try:
                self.callback(self)
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = f"error: {exc}"

        # Publish heartbeat on ACP bus
        if self.bus:
            msg = AgentMessage(
                msg_type=MessageType.HEARTBEAT,
                sender=AgentRole.ORCHESTRATOR,
                recipient=None,
                payload={"tick": self.tick_count, "timestamp": self.last_tick},
            )
            self.bus.publish(msg)

        self._tick_history.append(record)
        if len(self._tick_history) > 500:
            self._tick_history.pop(0)

    def _run(self) -> None:
        import random
        while not self._stop_event.is_set():
            jitter = random.uniform(0, self.jitter_sec) if self.jitter_sec else 0
            wait = self.interval_sec + jitter
            self._stop_event.wait(timeout=wait)
            if not self._stop_event.is_set():
                self._tick()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def force_tick(self) -> None:
        """Trigger an immediate heartbeat tick (useful for testing)."""
        self._tick()

    @property
    def tick_history(self) -> List[Dict[str, Any]]:
        return list(self._tick_history)

    def status(self) -> Dict[str, Any]:
        return {
            "interval_sec": self.interval_sec,
            "tick_count":   self.tick_count,
            "last_tick":    self.last_tick,
            "running":      bool(self._thread and self._thread.is_alive()),
        }
