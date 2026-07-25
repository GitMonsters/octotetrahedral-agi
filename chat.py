#!/usr/bin/env python3
"""
OctoTetrahedral AGI - Interactive Terminal Chat Interface

Usage:
    python3 chat.py [--url URL] [--key API_KEY]

Commands in chat:
    /ask <question>                  - Ask a question
    /prompt <text> [--mode MODE]     - Send a prompt (modes: answer, code, creative, technical)
    /command <cmd> <text>            - Execute a command (summarize, translate, analyze, expand, simplify)
    /chat <message>                  - Regular conversational chat
    /history                         - Show conversation history
    /clear                           - Clear conversation
    /help                            - Show help menu
    /exit or /quit                   - Exit the chat
"""

import sys
import argparse
from typing import Optional
from datetime import datetime

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Install with: pip install requests")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.prompt import Prompt
    from rich import box
except ImportError:
    print("Missing dependency: rich. Install with: pip install rich")
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
except ImportError:
    print("Missing dependency: prompt_toolkit. Install with: pip install prompt_toolkit")
    sys.exit(1)


console = Console()


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------


class OctoAGIClient:
    """HTTP client for the OctoTetrahedral AGI API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def ask(self, question: str) -> dict:
        return self._post("/ask", {"question": question})

    def prompt(self, text: str, mode: str = "answer", max_length: int = 200,
               temperature: float = 0.7) -> dict:
        return self._post("/prompt", {
            "prompt": text,
            "mode": mode,
            "max_length": max_length,
            "temperature": temperature,
        })

    def chat(self, messages: list, system_prompt: Optional[str] = None) -> dict:
        payload: dict = {"messages": messages}
        if system_prompt:
            payload["system_prompt"] = system_prompt
        return self._post("/chat", payload)

    def command(self, cmd: str, input_text: str, options: Optional[dict] = None) -> dict:
        payload: dict = {"command": cmd, "input_text": input_text}
        if options:
            payload["options"] = options
        return self._post("/command", payload)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _device_badge(device: str) -> str:
    badges = {"mps": "🍎 MPS", "cuda": "🟢 CUDA", "cpu": "⚙️ CPU"}
    for key, badge in badges.items():
        if key in device:
            return badge
    return f"📟 {device}"


def show_welcome(base_url: str, health_info: Optional[dict]) -> None:
    title = Text("OctoTetrahedral AGI Chat", style="bold bright_cyan")
    if health_info:
        device = health_info.get("device", "unknown")
        status = health_info.get("status", "?")
        subtitle = (
            f"[green]{status.upper()}[/green]  {_device_badge(device)}  "
            f"[dim]{base_url}[/dim]"
        )
    else:
        subtitle = f"[dim]{base_url}[/dim]"

    panel = Panel(
        f"[bold bright_cyan]Welcome to OctoTetrahedral AGI Interactive Chat[/bold bright_cyan]\n"
        f"{subtitle}\n\n"
        "[dim]Type [bold]/help[/bold] for available commands or just start chatting![/dim]",
        title="[bold cyan]🐙 OctoAGI[/bold cyan]",
        border_style="bright_cyan",
        box=box.ROUNDED,
    )
    console.print(panel)


def show_help() -> None:
    table = Table(
        title="Available Commands",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
        show_header=True,
        header_style="bold bright_cyan",
    )
    table.add_column("Command", style="bold yellow", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Example", style="dim")

    rows = [
        ("/ask <question>",
         "Ask the AGI a question",
         "/ask What is quantum computing?"),
        ("/prompt <text> [--mode MODE]",
         "Send a prompt (modes: answer, code, creative, technical)",
         "/prompt Write a hello-world function --mode code"),
        ("/command <cmd> <text>",
         "Run a command: summarize | translate | analyze | expand | simplify",
         "/command summarize Your long text here"),
        ("/chat <message>",
         "Conversational chat (maintains history)",
         "/chat Tell me more"),
        ("(plain text)",
         "Same as /chat – send anything without a prefix",
         "Hello! How are you?"),
        ("/history",
         "Show conversation history",
         "/history"),
        ("/clear",
         "Clear conversation history",
         "/clear"),
        ("/help",
         "Show this help menu",
         "/help"),
        ("/exit  /quit",
         "Exit the chat",
         "/exit"),
    ]
    for cmd, desc, example in rows:
        table.add_row(cmd, desc, example)

    console.print(table)


def _render_response(data: dict, cmd_type: str) -> None:
    """Render an API response with rich formatting."""
    device = data.get("device", "")
    latency = data.get("latency_ms", 0.0)
    footer = f"[dim]{_device_badge(device)}  ⏱ {latency:.1f} ms[/dim]"

    if cmd_type == "ask":
        content = data.get("answer", "")
        title = "💡 Answer"
        border = "bright_green"
    elif cmd_type == "prompt":
        content = data.get("response", "")
        mode = data.get("mode", "")
        title = f"📝 Response  [dim]({mode})[/dim]"
        border = "bright_blue"
    elif cmd_type == "chat":
        content = data.get("response", "")
        title = "💬 Chat"
        border = "bright_magenta"
    elif cmd_type == "command":
        content = data.get("output", "")
        cmd_name = data.get("command", "")
        title = f"⚙️ {cmd_name.capitalize()}"
        border = "bright_yellow"
    else:
        content = str(data)
        title = "Response"
        border = "white"

    console.print(
        Panel(
            content,
            title=title,
            subtitle=footer,
            border_style=border,
            box=box.ROUNDED,
        )
    )


def _render_error(message: str) -> None:
    console.print(
        Panel(
            f"[bold red]{message}[/bold red]",
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        )
    )


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class ConversationHistory:
    def __init__(self):
        self.turns: list = []  # list of {"role": ..., "content": ..., "ts": ...}

    def add(self, role: str, content: str) -> None:
        self.turns.append({
            "role": role,
            "content": content,
            "ts": datetime.now().strftime("%H:%M:%S"),
        })

    def as_messages(self) -> list:
        return [{"role": t["role"], "content": t["content"]} for t in self.turns]

    def clear(self) -> None:
        self.turns.clear()

    def display(self) -> None:
        if not self.turns:
            console.print("[dim]No conversation history yet.[/dim]")
            return

        table = Table(
            title="Conversation History",
            box=box.SIMPLE_HEAVY,
            border_style="cyan",
            show_header=True,
            header_style="bold bright_cyan",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Role", style="bold", width=10)
        table.add_column("Content", style="white")

        role_styles = {"user": "bright_blue", "assistant": "bright_green", "system": "yellow"}
        for i, turn in enumerate(self.turns, 1):
            role = turn["role"]
            style = role_styles.get(role, "white")
            table.add_row(
                str(i),
                turn["ts"],
                f"[{style}]{role}[/{style}]",
                turn["content"][:120] + ("…" if len(turn["content"]) > 120 else ""),
            )
        console.print(table)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

class ChatApp:
    def __init__(self, client: OctoAGIClient):
        self.client = client
        self.history = ConversationHistory()

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    def handle_ask(self, args: str) -> None:
        if not args.strip():
            _render_error("Usage: /ask <your question>")
            return
        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            data = self.client.ask(args.strip())
        self.history.add("user", f"/ask {args.strip()}")
        self.history.add("assistant", data.get("answer", ""))
        _render_response(data, "ask")

    def handle_prompt(self, args: str) -> None:
        # Support optional --mode flag at the end
        mode = "answer"
        text = args.strip()
        if "--mode" in text:
            parts = text.rsplit("--mode", 1)
            text = parts[0].strip()
            mode = parts[1].strip().split()[0] if parts[1].strip() else "answer"

        valid_modes = {"answer", "code", "creative", "technical"}
        if mode not in valid_modes:
            _render_error(f"Invalid mode '{mode}'. Choose from: {', '.join(sorted(valid_modes))}")
            return

        if not text:
            _render_error("Usage: /prompt <text> [--mode answer|code|creative|technical]")
            return

        with console.status("[cyan]Generating…[/cyan]", spinner="dots"):
            data = self.client.prompt(text, mode=mode)
        self.history.add("user", f"/prompt [{mode}] {text}")
        self.history.add("assistant", data.get("response", ""))
        _render_response(data, "prompt")

    def handle_command(self, args: str) -> None:
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            _render_error(
                "Usage: /command <cmd> <text>\n"
                "Commands: summarize, translate, analyze, expand, simplify"
            )
            return
        cmd, input_text = parts[0].lower(), parts[1]
        valid_cmds = {"summarize", "translate", "analyze", "expand", "simplify"}
        if cmd not in valid_cmds:
            _render_error(f"Unknown command '{cmd}'. Choose from: {', '.join(sorted(valid_cmds))}")
            return
        with console.status(f"[cyan]{cmd.capitalize()}ing…[/cyan]", spinner="dots"):
            data = self.client.command(cmd, input_text)
        self.history.add("user", f"/command {cmd}: {input_text}")
        self.history.add("assistant", data.get("output", ""))
        _render_response(data, "command")

    def handle_chat(self, message: str) -> None:
        if not message.strip():
            _render_error("Usage: /chat <your message>  (or just type without a prefix)")
            return
        self.history.add("user", message.strip())
        with console.status("[cyan]Thinking…[/cyan]", spinner="dots"):
            data = self.client.chat(self.history.as_messages())
        response = data.get("response", "")
        self.history.add("assistant", response)
        _render_response(data, "chat")

    # ------------------------------------------------------------------
    # Input dispatch
    # ------------------------------------------------------------------

    def dispatch(self, line: str) -> bool:
        """Process a single input line. Returns False when the user wants to exit."""
        stripped = line.strip()
        if not stripped:
            return True

        if stripped.lower() in {"/exit", "/quit"}:
            console.print("\n[bold bright_cyan]Goodbye! 👋[/bold bright_cyan]\n")
            return False

        if stripped.lower() == "/help":
            show_help()
            return True

        if stripped.lower() == "/history":
            self.history.display()
            return True

        if stripped.lower() == "/clear":
            self.history.clear()
            console.print("[dim]Conversation history cleared.[/dim]")
            return True

        if stripped.lower().startswith("/ask ") or stripped.lower() == "/ask":
            args = stripped[5:].strip()
            self._safe_call(self.handle_ask, args)
            return True

        if stripped.lower().startswith("/prompt ") or stripped.lower() == "/prompt":
            args = stripped[8:].strip()
            self._safe_call(self.handle_prompt, args)
            return True

        if stripped.lower().startswith("/command ") or stripped.lower() == "/command":
            args = stripped[9:].strip()
            self._safe_call(self.handle_command, args)
            return True

        if stripped.lower().startswith("/chat ") or stripped.lower() == "/chat":
            args = stripped[6:].strip()
            self._safe_call(self.handle_chat, args)
            return True

        if stripped.startswith("/"):
            _render_error(f"Unknown command '{stripped.split()[0]}'. Type /help for available commands.")
            return True

        # Plain text → chat
        self._safe_call(self.handle_chat, stripped)
        return True

    def _safe_call(self, fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            _render_error(
                "Cannot connect to the API server.\n"
                "Make sure the server is running: python3 -m uvicorn api:app --host 0.0.0.0 --port 8000"
            )
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            _render_error(f"HTTP {status}: {detail}")
        except requests.exceptions.Timeout:
            _render_error("Request timed out. The server may be overloaded.")
        except Exception as exc:
            _render_error(f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OctoTetrahedral AGI Interactive Terminal Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        metavar="URL",
        help="Base URL of the AGI API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--key",
        default=None,
        metavar="API_KEY",
        help="API key (bearer token). If not provided you will be prompted.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_url: str = args.url
    api_key: Optional[str] = args.key

    # Determine API key
    if not api_key:
        console.print(
            "\n[bold yellow]No API key provided via --key.[/bold yellow]\n"
            "You can generate one with: [bold]python3 auth.py generate[/bold]\n"
        )
        try:
            api_key = Prompt.ask("[bold cyan]Enter your API key[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Cancelled.[/dim]")
            sys.exit(0)

    if not api_key:
        console.print("[bold red]No API key provided. Exiting.[/bold red]")
        sys.exit(1)

    client = OctoAGIClient(base_url, api_key)

    # Try to fetch health info for the welcome banner
    health_info: Optional[dict] = None
    try:
        health_info = client.health()
    except Exception:
        pass  # Server may not be running yet; we still show the UI

    show_welcome(base_url, health_info)

    # Set up prompt_toolkit session
    pt_style = Style.from_dict({
        "prompt": "bold ansicyan",
    })
    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        style=pt_style,
        multiline=True,
    )

    app = ChatApp(client)

    console.print("[dim]Tip: Use [bold]Alt+Enter[/bold] or [bold]Escape Enter[/bold] to submit multi-line input.[/dim]\n")

    while True:
        try:
            line = session.prompt("🐙 You: ")
        except KeyboardInterrupt:
            console.print("\n[dim](Use /exit to quit)[/dim]")
            continue
        except EOFError:
            console.print("\n[bold bright_cyan]Goodbye! 👋[/bold bright_cyan]\n")
            break

        if not app.dispatch(line):
            break


if __name__ == "__main__":
    main()
