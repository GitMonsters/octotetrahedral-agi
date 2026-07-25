"""
Tests for the interactive terminal chat interface (chat.py).

These tests cover the pure-logic portions of the chat module that do NOT require
a running API server: argument parsing, conversation history management, and
command dispatch routing.  Network calls are mocked.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import chat as chat_mod

OctoAGIClient = chat_mod.OctoAGIClient
ConversationHistory = chat_mod.ConversationHistory
ChatApp = chat_mod.ChatApp
build_arg_parser = chat_mod.build_arg_parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(base_url="http://localhost:8000", api_key="test-key"):
    return OctoAGIClient(base_url, api_key)


def _make_app():
    return ChatApp(_make_client())


# ---------------------------------------------------------------------------
# OctoAGIClient tests
# ---------------------------------------------------------------------------

class TestOctoAGIClient(unittest.TestCase):
    def test_base_url_trailing_slash_stripped(self):
        c = OctoAGIClient("http://localhost:8000/", "k")
        self.assertEqual(c.base_url, "http://localhost:8000")

    def test_authorization_header_contains_bearer(self):
        c = _make_client(api_key="my-secret-key")
        self.assertIn("Bearer", c.headers["Authorization"])
        self.assertIn("my-secret-key", c.headers["Authorization"])

    def test_content_type_header(self):
        c = _make_client()
        self.assertEqual(c.headers["Content-Type"], "application/json")


# ---------------------------------------------------------------------------
# ConversationHistory tests
# ---------------------------------------------------------------------------

class TestConversationHistory(unittest.TestCase):
    def test_empty_history(self):
        h = ConversationHistory()
        self.assertEqual(len(h.turns), 0)
        self.assertEqual(h.as_messages(), [])

    def test_add_and_as_messages(self):
        h = ConversationHistory()
        h.add("user", "Hello")
        h.add("assistant", "Hi there!")
        msgs = h.as_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"role": "user", "content": "Hello"})
        self.assertEqual(msgs[1], {"role": "assistant", "content": "Hi there!"})

    def test_clear(self):
        h = ConversationHistory()
        h.add("user", "Test")
        h.clear()
        self.assertEqual(len(h.turns), 0)

    def test_turns_have_timestamp(self):
        h = ConversationHistory()
        h.add("user", "msg")
        self.assertIn("ts", h.turns[0])

    def test_display_empty_does_not_raise(self):
        h = ConversationHistory()
        # Should not raise even with an empty history
        h.display()

    def test_display_with_entries_does_not_raise(self):
        h = ConversationHistory()
        h.add("user", "Question?")
        h.add("assistant", "Answer!")
        h.display()  # Should not raise


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------

class TestArgParser(unittest.TestCase):
    def test_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.url, "http://localhost:8000")
        self.assertIsNone(args.key)

    def test_custom_url_and_key(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--url", "http://example.com:9000", "--key", "abc123"])
        self.assertEqual(args.url, "http://example.com:9000")
        self.assertEqual(args.key, "abc123")


# ---------------------------------------------------------------------------
# ChatApp dispatch tests (mocked client)
# ---------------------------------------------------------------------------

class TestChatAppDispatch(unittest.TestCase):
    def _app_with_mock(self):
        app = _make_app()
        app.client = MagicMock()
        return app

    def test_exit_returns_false(self):
        app = _make_app()
        self.assertFalse(app.dispatch("/exit"))

    def test_quit_returns_false(self):
        app = _make_app()
        self.assertFalse(app.dispatch("/quit"))

    def test_exit_case_insensitive(self):
        app = _make_app()
        self.assertFalse(app.dispatch("/EXIT"))

    def test_help_returns_true(self):
        app = _make_app()
        self.assertTrue(app.dispatch("/help"))

    def test_clear_empties_history(self):
        app = _make_app()
        app.history.add("user", "something")
        app.dispatch("/clear")
        self.assertEqual(len(app.history.turns), 0)

    def test_history_returns_true(self):
        app = _make_app()
        self.assertTrue(app.dispatch("/history"))

    def test_unknown_slash_command_returns_true(self):
        app = _make_app()
        result = app.dispatch("/unknowncommand")
        self.assertTrue(result)

    def test_empty_input_returns_true(self):
        app = _make_app()
        self.assertTrue(app.dispatch(""))
        self.assertTrue(app.dispatch("   "))

    def test_ask_calls_client_ask(self):
        app = self._app_with_mock()
        app.client.ask.return_value = {
            "answer": "42", "device": "cpu", "latency_ms": 1.0
        }
        result = app.dispatch("/ask What is 6x7?")
        self.assertTrue(result)
        app.client.ask.assert_called_once_with("What is 6x7?")

    def test_ask_adds_to_history(self):
        app = self._app_with_mock()
        app.client.ask.return_value = {
            "answer": "A fine answer", "device": "cpu", "latency_ms": 0.5
        }
        app.dispatch("/ask Some question")
        roles = [t["role"] for t in app.history.turns]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_prompt_calls_client_prompt(self):
        app = self._app_with_mock()
        app.client.prompt.return_value = {
            "response": "some code", "mode": "code", "device": "cpu", "latency_ms": 2.0
        }
        result = app.dispatch("/prompt Write hello world --mode code")
        self.assertTrue(result)
        app.client.prompt.assert_called_once_with("Write hello world", mode="code")

    def test_prompt_default_mode_is_answer(self):
        app = self._app_with_mock()
        app.client.prompt.return_value = {
            "response": "r", "mode": "answer", "device": "cpu", "latency_ms": 1.0
        }
        app.dispatch("/prompt Explain something")
        _, kwargs = app.client.prompt.call_args
        self.assertEqual(kwargs.get("mode"), "answer")

    def test_command_calls_client_command(self):
        app = self._app_with_mock()
        app.client.command.return_value = {
            "output": "Summary text", "command": "summarize",
            "device": "cpu", "latency_ms": 1.5
        }
        result = app.dispatch("/command summarize This is a long text")
        self.assertTrue(result)
        app.client.command.assert_called_once_with("summarize", "This is a long text")

    def test_chat_calls_client_chat(self):
        app = self._app_with_mock()
        app.client.chat.return_value = {
            "response": "Nice to meet you!", "device": "cpu", "latency_ms": 0.8
        }
        result = app.dispatch("/chat Hello there")
        self.assertTrue(result)
        app.client.chat.assert_called_once()

    def test_plain_text_routes_to_chat(self):
        app = self._app_with_mock()
        app.client.chat.return_value = {
            "response": "reply", "device": "cpu", "latency_ms": 1.0
        }
        result = app.dispatch("Hello, how are you?")
        self.assertTrue(result)
        app.client.chat.assert_called_once()

    def test_chat_builds_message_history(self):
        app = self._app_with_mock()
        app.client.chat.return_value = {
            "response": "reply", "device": "cpu", "latency_ms": 0.5
        }
        app.dispatch("First message")
        app.dispatch("Second message")
        # history should contain 4 turns: 2 user + 2 assistant
        self.assertEqual(len(app.history.turns), 4)

    def test_connection_error_handled_gracefully(self):
        import requests as req
        app = _make_app()
        app.client = MagicMock()
        app.client.ask.side_effect = req.exceptions.ConnectionError("refused")
        # Should not raise – error is displayed to console
        result = app.dispatch("/ask Will this crash?")
        self.assertTrue(result)

    def test_http_error_handled_gracefully(self):
        import requests as req
        app = _make_app()
        app.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.json.return_value = {"detail": "Invalid API key"}
        app.client.ask.side_effect = req.exceptions.HTTPError(response=mock_resp)
        result = app.dispatch("/ask Anything?")
        self.assertTrue(result)

    def test_ask_empty_args_shows_error(self):
        app = self._app_with_mock()
        result = app.dispatch("/ask")
        self.assertTrue(result)
        app.client.ask.assert_not_called()

    def test_prompt_invalid_mode_shows_error(self):
        app = self._app_with_mock()
        result = app.dispatch("/prompt Something --mode invalidmode")
        self.assertTrue(result)
        app.client.prompt.assert_not_called()

    def test_command_invalid_cmd_shows_error(self):
        app = self._app_with_mock()
        result = app.dispatch("/command unknown_cmd some text")
        self.assertTrue(result)
        app.client.command.assert_not_called()

    def test_command_missing_text_shows_error(self):
        app = self._app_with_mock()
        result = app.dispatch("/command summarize")
        self.assertTrue(result)
        app.client.command.assert_not_called()


# ---------------------------------------------------------------------------
# _device_badge helper
# ---------------------------------------------------------------------------

class TestDeviceBadge(unittest.TestCase):
    def test_mps_badge(self):
        badge = chat_mod._device_badge("mps")
        self.assertIn("MPS", badge)

    def test_cuda_badge(self):
        badge = chat_mod._device_badge("cuda")
        self.assertIn("CUDA", badge)

    def test_cpu_badge(self):
        badge = chat_mod._device_badge("cpu")
        self.assertIn("CPU", badge)

    def test_unknown_device(self):
        badge = chat_mod._device_badge("tpu")
        self.assertIn("tpu", badge)


if __name__ == "__main__":
    unittest.main()
