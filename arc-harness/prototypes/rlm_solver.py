"""Prototype: solve ARC tasks with an RLM (rlms library, alexzhang13/rlm).

The RLM runs a persistent Python REPL against grok-4.6 (xAI OpenAI-compatible
endpoint). The model inspects train pairs, writes+verifies a `transform(g)`
function in the REPL, then applies it to test inputs and submits the predicted
output grids as JSON via answer["ready"]=True.

Compare with the static solver baseline: this is the recursive / self-verifying
loop variant. Test outputs are NOT leaked to the model.

Usage:
    python3 prototypes/rlm_solver.py --tasks 136b0064,1818057f,2ba387bc --iterations 15
    python3 prototypes/rlm_solver.py --limit 3 --iterations 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.core.rlm import RLM  # noqa: E402
from rlm.logger import RLMLogger  # noqa: E402

# compounding: cross-task verified-result store
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
from compounding import CompoundingStore  # noqa: E402


def _force_ipv4() -> None:
    """Prefer IPv4 for all sockets. IPv6 to xAI/Cloudflare frequently STALLS:
    the TCP handshake succeeds, then data trickles (or stops) for many minutes,
    so httpx per-read timeouts never fire and a call hangs well past
    call_timeout. IPv4 calls answer in ~2s. Affects every connection made by
    this process."""
    import socket

    _orig = socket.getaddrinfo

    def _ipv4_first(host, port, family=0, *args, **kwargs):
        if family == socket.AF_INET6:
            return _orig(host, port, socket.AF_INET6, *args, **kwargs)
        res = _orig(host, port, family, *args, **kwargs)
        v4 = [r for r in res if r[0] == socket.AF_INET]
        return v4 if v4 else res

    socket.getaddrinfo = _ipv4_first


_force_ipv4()


def _tolerant_find_code_blocks(text: str) -> list[str]:
    """Find REPL code blocks. grok-4.6 emits ```repl``` (opening+closing
    backticks in one token) then a newline; tolerate both that and the
    standard ```repl<newline> form."""
    pattern = r"```repl[ \t]*(?:```)?\s*\n(.*?)\n```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]


import rlm.core.rlm as _rlm_core  # noqa: E402

_rlm_core.find_code_blocks = _tolerant_find_code_blocks


def _patch_stable_namespace():
    """Make every REPL block (and setup_code) share ONE namespace dict.

    Vanilla LocalREPL execs each block into a fresh merged dict, so redefining
    `transform` in a later block is invisible to helpers defined earlier (their
    __globals__ still point at setup's dict). We stash setup_code on __init__
    and re-exec it into a lazily-created `_stable_ns`, then exec every block
    into that same dict so `check()` always sees the latest `transform`.
    """
    import rlm.core.types as _types
    from rlm.environments import local_repl
    from rlm.environments.local_repl import LocalREPL

    _orig_init = LocalREPL.__init__

    def _init(self, *args, **kwargs):
        self._setup_code = kwargs.get("setup_code")
        _orig_init(self, *args, **kwargs)

    LocalREPL.__init__ = _init

    def _stable_execute_code(self, code: str) -> _types.REPLResult:
        start_time = time.perf_counter()
        self._pending_llm_calls = []
        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                ns = getattr(self, "_stable_ns", None)
                if ns is None:
                    ns = {**self.globals, **self.locals}
                    self._stable_ns = ns
                    setup = getattr(self, "_setup_code", None)
                    if setup:
                        exec(setup, ns, ns)
                exec(code, ns, ns)
                for key, value in ns.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value
                self._restore_scaffold()
                for name in local_repl.RESERVED_TOOL_NAMES:
                    if name in self.globals:
                        ns[name] = self.globals[name]
                    elif name in self.locals:
                        ns[name] = self.locals[name]
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
        final_answer = self._last_final_answer
        self._last_final_answer = None
        return _types.REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    LocalREPL.execute_code = _stable_execute_code


_patch_stable_namespace()


def _patch_consequential_llm_query(max_sub_answer=2000):
    """Make `llm_query` sub-calls CONSEQUENTIAL: if the sub-agent's answer
    defines a `transform`, the harness applies it to the live namespace and
    runs `check()` over all train pairs, returning the verdict. The parent
    RLM then reads the verified-or-rejected outcome instead of free text.
    """
    import io
    import contextlib

    from rlm.environments.local_repl import LocalREPL

    _orig = LocalREPL._llm_query

    def _extract_transform(text: str) -> str | None:
        lines = text.splitlines()
        out, capturing = [], False
        for ln in lines:
            if ln.lstrip().startswith("def transform(") or ln.lstrip().startswith("def  transform("):
                capturing = True
                out = [ln]
            elif capturing:
                if ln.strip() and not (ln.startswith((" ", "\t"))):
                    break
                out.append(ln)
        return "\n".join(out) if out else None

    def _wrapped(self, prompt, model=None):
        resp = _orig(self, prompt, model)
        if not (isinstance(resp, str) and "def transform" in resp):
            return resp
        code = _extract_transform(resp)
        ns = getattr(self, "_stable_ns", None)
        if code is None or ns is None:
            return resp
        try:
            exec(code, ns, ns)
        except Exception as e:
            return resp[:max_sub_answer] + (
                f"\n\n[CONSEQUENCE] Sub-answer defined a transform but it failed "
                f"to load: {type(e).__name__}: {e}"
            )
        check_fn = ns.get("check")
        verdict = ""
        ok = False
        if check_fn is not None:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    ok = bool(check_fn())
            except Exception as e:
                buf.write(f"{type(e).__name__}: {e}\n")
            verdict = buf.getvalue()
        short = resp if len(resp) <= max_sub_answer else resp[:max_sub_answer] + (
            f"\n... [truncated {len(resp) - max_sub_answer} chars]"
        )
        return short + (
            "\n\n[CONSEQUENCE] The sub-agent's transform was APPLIED to this REPL "
            f"and verified with check().\n{verdict}"
            + ("" if ok else "\n=> verdict: NOT all train pairs pass; do not trust this transform yet.")
        )

    LocalREPL._llm_query = _wrapped


_patch_consequential_llm_query()

MODEL = "grok-4.6"
XAI_BASE_URL = "https://api.x.ai/v1"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

INPUT_PER_MTOK = 2.0
OUTPUT_PER_MTOK = 6.0

SYSTEM_PROMPT = """\
You are a Recursive Language Model (RLM) solving an ARC (Abstraction and Reasoning Corpus) task by writing and verifying Python code in a persistent REPL.

A grid is a rectangular list-of-lists of ints, color ids 0-9. Infer the transformation from the train input/output pairs, implement it, verify it on every train pair, then predict the output for every test input. Test outputs are NEVER given to you.

Workflow (each turn: plan briefly in prose, then run ONE ```repl``` block, read the feedback, continue):
1. Inspect with compact diagnostics: `print(pair(i))` and `print(summarize(...))`. Only print full grids when they are small.
2. Hypothesize. Prefer the simplest general rule consistent with ALL train pairs. Record your current rule with `note("...")` and keep it updated — this is your compounding design memory.
3. Implement `transform(g) -> grid`. You may delegate a draft to a sub-agent with `llm_query("Write a Python transform ...")`: if its answer defines `transform`, the harness APPLIES it to the live REPL and runs `check()` automatically (consequential recursion). Read the verdict it returns.
4. Verify with `check()`: it prints PASS/FAIL per train pair, precise cell diffs (coordinates + values) and windows around the first mismatch. Iterate until ALL train pairs pass.
   - Cohesion rule: pairs that ever passed are tracked in `VERIFIED`. If an edit to `transform` breaks a previously-passing pair, `check()` flags a **cohesion break** — you must fix it, never ignore it.
5. Submit when ALL train pairs pass AND you have predictions for every test input: set `answer["content"]` to the final JSON and `answer["ready"] = True` IN THE SAME ```repl``` block as a passing `check()`.

REPL environment:
- `train`: list of {{"input": grid, "output": grid}}. `test_inputs`: list of input grids (outputs NOT provided).
- Helpers: `render(g)`, `check(limit=8)`, `pair(i)`, `summarize(g)`, `note(text)`, `notes()`, `verified_pairs()`, `compound_status()`, `SHOW_VARS()`, `llm_query(prompt, model=None)`, `llm_query_batched(prompts, model=None)`, `rlm_query(...)`, `answer`.
- Only `print(...)` output is shown back between turns; a bare expression is silently discarded.
- Keep diagnostics compact (shapes / value sets / diff coordinates). Printing entire large grids every turn wastes your context budget — the REPL output is appended to your history each turn.
- If you want `check()` to also print a full side-by-side for a specific pair, call `print(render(train[K]['expected']))`-style prints yourself; `check()` stays compact.

FINAL ANSWER FORMAT: set `answer["content"]` to a JSON object mapping each test index (as a string) to its predicted output grid, e.g.
{{"0": [[1,0],[0,1]], "1": [[1,1],[1,0]]}}
Submit only when `check()` passes for ALL train pairs and you have a prediction for every test input (indices 0..__NTEST__-1). Do not submit before inspecting `train`.
{custom_tools_section}
"""


SYSTEM_PROMPT_CCL = """\
You are a Recursive Language Model (RLM) solving a grid-transformation task by writing and verifying Python code in a persistent REPL.

A grid is an 8x8 list-of-lists of ints, color ids 0-9. The transformation is composed of a small set of primitive rules: rotations (rot_cw/rot_ccw), flips (flip_h/flip_v), transpose, color remaps (color_shift/color_swap), gravity (gravity_down/gravity_right), and row/col sorts. Learn the exact composition from the train input/output pairs, implement it, verify it on every train pair, then predict the output for every test input. Test outputs are NEVER given to you.

Workflow (each turn: plan briefly in prose, then run ONE ```repl``` block, read the feedback, continue):
1. Inspect with compact diagnostics: `print(pair(i))` and `print(summarize(...))`. Only print full grids when they are small.
2. Hypothesize the rule composition. Prefer the simplest composition consistent with ALL train pairs. Record your current rule with `note("...")` and keep it updated — this is your compounding design memory.
3. Implement `transform(g) -> grid`. You may delegate a draft to a sub-agent with `llm_query("Write a Python transform ...")`: if its answer defines `transform`, the harness APPLIES it to the live REPL and runs `check()` automatically (consequential recursion). Read the verdict it returns.
4. Verify with `check()`: it prints PASS/FAIL per train pair, precise cell diffs (coordinates + values) and windows around the first mismatch. Iterate until ALL train pairs pass.
   - Cohesion rule: pairs that ever passed are tracked in `VERIFIED`. If an edit to `transform` breaks a previously-passing pair, `check()` flags a **cohesion break** — you must fix it, never ignore it.
5. Submit when ALL train pairs pass AND you have predictions for every test input: set `answer["content"]` to the final JSON and `answer["ready"] = True` IN THE SAME ```repl``` block as a passing `check()`.

REPL environment:
- `train`: list of {{"input": grid, "output": grid}}. `test_inputs`: list of input grids (outputs NOT provided).
- Helpers: `render(g)`, `check(limit=8)`, `pair(i)`, `summarize(g)`, `note(text)`, `notes()`, `verified_pairs()`, `compound_status()`, `SHOW_VARS()`, `llm_query(prompt, model=None)`, `llm_query_batched(prompts, model=None)`, `rlm_query(...)`, `answer`.
- Only `print(...)` output is shown back between turns; a bare expression is silently discarded.
- Keep diagnostics compact (shapes / value sets / diff coordinates). Printing entire large grids every turn wastes your context budget — the REPL output is appended to your history each turn.

FINAL ANSWER FORMAT: set `answer["content"]` to a JSON object mapping each test index (as a string) to its predicted output grid, e.g.
{{"0": [[1,0],[0,1]], "1": [[1,1],[1,0]]}}
Submit only when `check()` passes for ALL train pairs and you have a prediction for every test input (indices 0..__NTEST__-1). Do not submit before inspecting `train`.
{custom_tools_section}
"""


def _load_task(task_id: str, cfg: dict) -> dict:
    root = cfg.get("task_root") or os.path.join(DATA_ROOT, "tasks")
    with open(os.path.join(root, f"{task_id}.json")) as f:
        return json.load(f)


def _task_features(task: dict) -> dict:
    """Extract sparse feature vector from a CCL task for compounding retrieval."""
    features = {}
    grids = [tr["input"] for tr in task.get("train", [])]
    grids += [tr["output"] for tr in task.get("train", [])]
    h_set, w_set, color_set = set(), set(), set()
    for g in grids:
        if g and g[0]:
            h_set.add(len(g))
            w_set.add(len(g[0]))
            for row in g:
                color_set.update(row)
    features["n_colors"] = len(color_set)
    features["max_h"] = max(h_set) if h_set else 0
    features["max_w"] = max(w_set) if w_set else 0
    features["n_train"] = len(task.get("train", []))
    for c in sorted(color_set)[:10]:
        features[f"color_{c}"] = 1.0
    return features


def _compounding_section(store, task_id: str, task: dict, cfg: dict) -> str:
    """Build a prompt section from verified compounding data.

    Injects similar verified solutions as few-shot examples and relevant
    patterns as strategy hints.  Only verified (test-passed) results
    compound into the prompt.
    """
    features = _task_features(task)
    lines = []

    # verified similar solutions
    similar = store.solutions.similar(features, top_k=3)
    if similar:
        lines.append(
            "VERIFIED SOLUTIONS FROM SIMILAR TASKS (for reference only — "
            "do NOT copy, study the approach):"
        )
        for sol in similar:
            code = sol.get("code", "")
            rules = sol.get("rules", [])
            if not code:
                continue
            tag = f"  # task {sol['task_id']}"
            if rules:
                tag += f" ({', '.join(rules)})"
            # take first 15 lines max to save context
            short = "\n".join(code.splitlines()[:15])
            if len(code.splitlines()) > 15:
                short += "\n  # ..."
            lines.append(f"{tag}\n{short}")
        lines.append("")

    # relevant patterns
    patterns = store.patterns.relevant_to(features, top_k=5)
    if patterns:
        pattern_names = [name for name, score in patterns]
        lines.append(
            f"VERIFIED PATTERNS for similar tasks: {', '.join(pattern_names)}"
        )
        lines.append(
            "Consider whether any of these patterns apply. Use them as a "
            "starting hypothesis but verify with check()."
        )
        lines.append("")

    # cross-task capability signal
    enrichment = store.get_enrichment_multiplier()
    cap = store.capability
    if cap > 0:
        lines.append(
            f"Cross-task capability: {store._capability['tasks_solved']}/"
            f"{store._capability['tasks_attempted']} solved so far "
            f"(enrichment x{enrichment:.2f})."
        )
        lines.append("")

    return "\n".join(lines) if lines else ""


def _domain_prompt(cfg: dict, ntest: int, store=None, task_id: str = "", task: dict | None = None) -> str:
    prompt = SYSTEM_PROMPT_CCL if cfg.get("domain") == "ccl" else SYSTEM_PROMPT
    prompt = prompt.replace("__NTEST__", str(ntest))

    # compounding: inject verified similar solutions + relevant patterns
    if store is not None and task is not None:
        section = _compounding_section(store, task_id, task, cfg)
        if section:
            # insert before the final answer format block
            marker = "FINAL ANSWER FORMAT:"
            if marker in prompt:
                prompt = prompt.replace(marker, section + "\n" + marker)
            else:
                prompt = prompt + "\n" + section

    return prompt


def build_setup_code(task: dict) -> str:
    def fmt(grid) -> str:
        return "[" + ",".join("[" + ",".join(str(c) for c in row) + "]" for row in grid) + "]"

    train_lit = "[" + ",".join(
        '{"input": ' + fmt(tr["input"]) + ', "output": ' + fmt(tr["output"]) + "}"
        for tr in task["train"]
    ) + "]"
    test_lit = "[" + ",".join(fmt(t["input"]) for t in task["test"]) + "]"
    return f"""\
train = {train_lit}
test_inputs = {test_lit}

def render(g):
    return "\\n".join("".join(str(c) for c in row) for row in g)

def transform(g):
    raise NotImplementedError("implement transform(g) -> output grid")

def _mismatches(got, exp):
    out = []
    for r in range(min(len(got), len(exp))):
        for c in range(min(len(got[r]), len(exp[r]))):
            if got[r][c] != exp[r][c]:
                out.append((r, c, got[r][c], exp[r][c]))
    return out

def _window(g, r, c, k=1):
    h, w = len(g), len(g[0]) if g else 0
    rows = []
    for rr in range(max(0, r - k), min(h, r + k + 1)):
        rows.append("".join(str(g[rr][cc]) for cc in range(max(0, c - k), min(w, c + k + 1))))
    return "\\n".join(rows) if rows else "(empty grid)"

def summarize(g):
    if not g:
        return "0x0 (empty)"
    flat = [v for row in g for v in row]
    return f"{{len(g)}}x{{len(g[0])}}, vals={{sorted(set(flat))}}, cells={{len(flat)}}"

def pair(i):
    x = train[i]
    print(f"train[{{i}}]: input {{summarize(x['input'])}}  output {{summarize(x['output'])}}")
    if len(x["input"]) * (len(x["input"][0]) if x["input"] else 0) <= 100:
        print("input:"); print(render(x["input"]))
        print("output:"); print(render(x["output"]))

VERIFIED = set()
NOTES = []
COMPOUND_LOG = []

def verified_pairs():
    return sorted(VERIFIED)

def note(text):
    NOTES.append(str(text))
    print(f"NOTE#{{len(NOTES) - 1}} recorded ({{len(NOTES)}} total)")

def notes():
    return "\\n".join(f"#{{i}}: {{t}}" for i, t in enumerate(NOTES))

def _cell_acc(got, exp):
    gh, gw = len(got), len(got[0]) if got else 0
    eh, ew = len(exp), len(exp[0]) if exp else 0
    if (gh, gw) != (eh, ew):
        return 0.0
    if gh * gw == 0:
        return 0.0
    hits = sum(1 for r in range(gh) for c in range(gw) if got[r][c] == exp[r][c])
    return hits / (gh * gw)

def compound_status():
    n = len(COMPOUND_LOG)
    if n == 0:
        return "no check() results recorded yet"
    recent = COMPOUND_LOG[-1]
    conf, wsum, w = 0.0, 0.0, 1.0
    for e in COMPOUND_LOG:
        conf += w * e["confidence"]
        wsum += w
        w *= 0.85
    conf = conf / wsum if wsum else 0.0
    return (f"cohesion {{recent['n_pass']}}/{{recent['n_total']}} train pairs, "
            f"cell_acc {{recent['cell_acc']:.3f}}, confidence(decay) {{conf:.3f}}, "
            f"checks {{n}}")

def check(limit=8):
    all_ok = True
    n_pass = 0
    cell_hits = 0
    cell_total = 0
    for i, x in enumerate(train):
        exp = x["output"]
        try:
            got = transform(x["input"])
            ok = got == exp
        except Exception as e:
            got, ok = None, False
            print(f"train[{{i}}]: ERROR {{type(e).__name__}}: {{e}}")
        if ok:
            n_pass += 1
            VERIFIED.add(i)
            print(f"train[{{i}}]: PASS ({{len(got)}}x{{len(got[0]) if got and got[0] else 0}})")
            continue
        all_ok = False
        if i in VERIFIED:
            print(f"train[{{i}}]: FAIL  ** cohesion break: passed earlier, now fails **")
        else:
            print(f"train[{{i}}]: FAIL")
        if got is None:
            print("  -> transform raised or returned None; fix transform()")
            continue
        gh, gw = len(got), len(got[0]) if got else 0
        eh, ew = len(exp), len(exp[0]) if exp else 0
        if (gh, gw) != (eh, ew):
            print(f"  -> shape mismatch: got {{gh}}x{{gw}}, expected {{eh}}x{{ew}}")
            if gh * gw + eh * ew <= 120:
                print("expected:"); print(render(exp))
                print("got:"); print(render(got))
            continue
        diffs = _mismatches(got, exp)
        print(f"  -> {{len(diffs)}} cell mismatches; showing up to {{limit}}:")
        for r, c, gv, ev in diffs[:limit]:
            print(f"     ({{r}},{{c}}) got={{gv}} expected={{ev}}")
        r0, c0 = diffs[0][0], diffs[0][1]
        print("  got window around first mismatch:"); print(_window(got, r0, c0))
        print("  expected window:"); print(_window(exp, r0, c0))
    for x in train:
        try:
            got = transform(x["input"])
            acc = _cell_acc(got, x["output"])
        except Exception:
            acc = 0.0
        cell_hits += round(acc * (len(x["output"]) * (len(x["output"][0]) if x["output"] else 0)))
        cell_total += len(x["output"]) * (len(x["output"][0]) if x["output"] else 0)
    cell_acc = (cell_hits / cell_total) if cell_total else 0.0
    COMPOUND_LOG.append({{"n_pass": n_pass, "n_total": len(train),
                         "cell_acc": cell_acc,
                         "confidence": (n_pass / len(train)) if train else 0.0}})
    print(f"verified: {{sorted(VERIFIED)}} of {{len(train)}} train pairs")
    print(f"compound: cohesion {{n_pass}}/{{len(train)}}, cell_acc {{cell_acc:.3f}}")
    return all_ok
"""


_SOCKET_TIMEOUT_STATE = {"timeout": 300.0}


def _patch_socket_timeouts() -> dict:
    """Route REPL<->LMHandler sub-call socket timeouts through a process-global.

    Vanilla rlm hardcodes the sub-call socket timeout at 300s (comms_utils),
    independent of call_timeout. We patch the two send_lm_request* helpers so
    every REPL sub-call respects cfg["call_timeout"] instead, keeping the whole
    task's LLM budget coherent. Set _SOCKET_TIMEOUT_STATE["timeout"] per task.
    """
    import rlm.core.comms_utils as _cu

    _orig_req = _cu.send_lm_request
    _orig_req_b = _cu.send_lm_request_batched

    def _send_req(address, request, timeout=None, depth=None):
        if timeout is None:
            timeout = _SOCKET_TIMEOUT_STATE["timeout"]
        return _orig_req(address, request, timeout=timeout, depth=depth)

    def _send_req_b(address, prompts, model=None, timeout=None, depth=0):
        if timeout is None:
            timeout = _SOCKET_TIMEOUT_STATE["timeout"]
        return _orig_req_b(address, prompts, model=model, timeout=timeout, depth=depth)

    _cu.send_lm_request = _send_req
    _cu.send_lm_request_batched = _send_req_b
    return _SOCKET_TIMEOUT_STATE


_SOCKET_TIMEOUT_STATE = _patch_socket_timeouts()


def _heartbeat(task_id: str, interval: float, stop: threading.Event) -> None:
    """Print a liveness line every `interval` seconds so long API calls show up."""
    start = time.perf_counter()
    while not stop.wait(interval):
        print(f"[{task_id}] ... working ({time.perf_counter() - start:.0f}s)", flush=True)


def parse_answer(content: str) -> dict[str, list]:
    """Tolerantly parse the submitted JSON (object of index->grid, or array)."""
    if not content:
        raise ValueError("empty answer content")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        while lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1:
        raise ValueError(f"no JSON object in answer: {content[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if isinstance(parsed, list):
        return {str(i): g for i, g in enumerate(parsed)}
    return {str(k): v for k, v in parsed.items()}


def run_task(task_id: str, cfg: dict, store=None) -> dict:
    task = _load_task(task_id, cfg)
    setup_code = build_setup_code(task)
    prompt = _domain_prompt(cfg, len(task["test"]), store=store, task_id=task_id, task=task)
    meta = task.get("metadata", {})

    def on_iter(itr, depth):
        print(f"[{task_id}] turn {itr}", flush=True)

    def on_complete(itr, depth, dur):
        print(f"[{task_id}] turn {itr} done ({dur:.1f}s)", flush=True)

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL,
            "base_url": XAI_BASE_URL,
            "api_key": os.environ.get("XAI_API_KEY", ""),
            "timeout": cfg["call_timeout"],
        },
        environment="local",
        environment_kwargs={"setup_code": setup_code},
        max_iterations=cfg["iterations"],
        max_timeout=cfg["task_timeout"],
        max_errors=cfg.get("max_errors", 8),
        custom_system_prompt=prompt,
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 8000,
            "extra_body": {"reasoning_effort": "low"},
        },
        sub_sampling_args={
            "extra_body": {"reasoning_effort": cfg.get("sub_reasoning", "medium")}
        },
        on_iteration_start=on_iter,
        on_iteration_complete=on_complete,
        logger=RLMLogger(log_dir=cfg["out"], file_name=f"trace_{task_id}")
        if cfg.get("trace")
        else None,
        verbose=cfg.get("verbose", False),
    )

    t0 = time.perf_counter()
    result = None
    last_err = None
    for attempt in range(cfg.get("retries", 2)):
        try:
            result = rlm.completion("Solve the ARC task by predicting outputs for every test input.")
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[{task_id}] RLM attempt {attempt + 1} failed: {last_err}", flush=True)
            if time.perf_counter() - t0 > cfg["task_timeout"]:
                break
    if result is None:
        print(f"[{task_id}] RLM error: {last_err}", flush=True)
        return {
            "task": task_id,
            "solved": False,
            "n_correct": 0,
            "n_test": len(task["test"]),
            "iterations": cfg["iterations"],
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "parse_error": None,
            "error": last_err,
            "per_index": [{"index": i, "match": False, "reason": "rlm error"}
                          for i in range(len(task["test"]))],
            "trajectory": None,
            "raw_answer": "",
            "level": meta.get("level"),
            "rules": meta.get("rules"),
        }
    elapsed = time.perf_counter() - t0

    content = result.response
    metadata = result.metadata
    if metadata is not None:
        trajectory = {
            "metadata": metadata.get("run_metadata") or metadata,
            "iterations": metadata.get("iterations"),
        }
    else:
        trajectory = None
    try:
        preds = parse_answer(content)
    except ValueError as e:
        preds, err = {}, str(e)
    else:
        err = None

    expected = task["test"]
    per_index = []
    n_correct = 0
    for i, tpair in enumerate(expected):
        pred = preds.get(str(i))
        if pred is None:
            per_index.append({"index": i, "match": False, "reason": "missing prediction"})
            continue
        if pred == tpair["output"]:
            n_correct += 1
            per_index.append({"index": i, "match": True})
        else:
            per_index.append(
                {"index": i, "match": False, "reason": "wrong grid",
                 "expected": tpair["output"], "got": pred}
            )

    usage = result.usage_summary
    in_tok = usage.total_input_tokens if usage else 0
    out_tok = usage.total_output_tokens if usage else 0
    cost = in_tok / 1e6 * INPUT_PER_MTOK + out_tok / 1e6 * OUTPUT_PER_MTOK

    report = {
        "task": task_id,
        "solved": n_correct == len(expected),
        "n_correct": n_correct,
        "n_test": len(expected),
        "iterations": rlm.max_iterations,
        "elapsed_s": round(elapsed, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 4),
        "parse_error": err,
        "per_index": per_index,
        "trajectory": trajectory,
        "raw_answer": content[:2000],
        "level": meta.get("level"),
        "rules": meta.get("rules"),
    }
    print(f"[{task_id}] {'SOLVED' if report['solved'] else 'FAILED'} "
          f"{n_correct}/{len(expected)} test pairs, {elapsed:.0f}s", flush=True)
    return report


def _compound_confidence(log: list[dict], decay: float = 0.85) -> float:
    """Decay-weighted confidence over the compound log (feedback decay)."""
    conf, wsum, w = 0.0, 0.0, 1.0
    for e in log:
        conf += w * e["confidence"]
        wsum += w
        w *= decay
    return (conf / wsum) if wsum else 0.0


def _read_compound_log(rlm) -> tuple[list[dict], dict | None]:
    """Pull COMPOUND_LOG + latest cohesion entry from the persistent REPL."""
    try:
        ns = getattr(rlm._persistent_env, "_stable_ns", None)
        if ns is None:
            return [], None
        log = list(ns.get("COMPOUND_LOG", []))
        return log, (log[-1] if log else None)
    except Exception:
        return [], None


def compound_run_task(task_id: str, cfg: dict, store=None) -> dict:
    """Compound-loop mode: run the RLM in persistent batches where each batch's
    budget and continuation are governed by the compound growth formula
    capability *= (1 + rate * confidence), with decay-weighted feedback and
    convergence detection (port of compound-integration's CompoundLoop)."""
    task = _load_task(task_id, cfg)
    setup_code = build_setup_code(task)
    ntest = len(task["test"])
    base_prompt = _domain_prompt(cfg, ntest, store=store, task_id=task_id, task=task)
    meta = task.get("metadata", {})

    batch_iters = cfg.get("batch_iters", 5)
    max_batches = cfg.get("max_batches", 5)
    rate = cfg.get("compound_rate", 0.2)
    decay = cfg.get("feedback_decay", 0.85)
    patience = cfg.get("patience", 2)
    min_cohesion = cfg.get("min_cohesion", 0.5)
    accept_gate = cfg.get("accept_gate", 0.8)

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL,
            "base_url": XAI_BASE_URL,
            "api_key": os.environ.get("XAI_API_KEY", ""),
            "timeout": cfg["call_timeout"],
        },
        environment="local",
        environment_kwargs={"setup_code": setup_code},
        persistent=True,
        max_iterations=batch_iters,
        max_timeout=cfg["task_timeout"],
        max_errors=cfg.get("max_errors", 8),
        custom_system_prompt=base_prompt,
        sampling_args={
            "temperature": 0.7,
            "max_tokens": 8000,
            "extra_body": {"reasoning_effort": "low"},
        },
        sub_sampling_args={
            "extra_body": {"reasoning_effort": cfg.get("sub_reasoning", "medium")}
        },
        logger=RLMLogger(log_dir=cfg["out"], file_name=f"trace_{task_id}")
        if cfg.get("trace")
        else None,
        verbose=cfg.get("verbose", False),
    )

    t0 = time.perf_counter()
    batches: list[dict] = []
    max_enrichment = cfg.get("max_enrichment", 3.0)
    enrichment = min(store.get_enrichment_multiplier(max_enrichment), max_enrichment) if store else 1.0
    best_cohesion = 0.0
    stall = 0
    rejected = False
    rejected_cohesion = 0.0
    final = None
    depth = batch_iters
    try:
        for b in range(max_batches):
            if time.perf_counter() - t0 > cfg["task_timeout"]:
                break
            if b == 0:
                user = (
                    f"Solve the ARC task by writing transform() so that check() passes ALL "
                    f"{len(task['train'])} train pairs. Do NOT set answer or answer[\"ready\"]=True "
                    f"until check() shows every train pair passing; premature submissions are "
                    f"rejected and wasted."
                )
            else:
                status = _read_compound_log(rlm)[1]
                corrective = ""
                if rejected:
                    corrective = (
                        f"Your previous submission (cohesion {rejected_cohesion:.2f}) was REJECTED "
                        f"because not all train pairs passed. Do NOT set answer until check() shows "
                        f"ALL train pairs passing, then submit in the same repl block as a passing "
                        f"check().\n"
                        f"CRITICAL: preserve cohesion. transform() currently passes some train "
                        f"pairs; do NOT rewrite it wholesale. Keep the logic that already passes "
                        f"and only patch the remaining failing pairs (special-case branches or "
                        f"generalize). Verify with check() after every change so you never drop a "
                        f"previously-passing pair."
                    )
                user = (
                    f"Continue the ARC task (batch {b + 1} of {max_batches}). "
                    f"Your REPL state persists across batches: `transform`, `check`, "
                    f"`notes()`, `history` (prior transcripts) and `answer` are all live. "
                    f"Do NOT re-derive from scratch. Inspect current state, refine "
                    f"transform(), verify with check() until ALL {len(task['train'])} train "
                    f"pairs pass, then set answer and answer[\"ready\"]=True in the SAME "
                    f"repl block as a passing check().\n"
                    f"Last cohesion: {status}. {corrective}"
                )
            print(f"[{task_id}] compound batch {b + 1} (enrichment x{enrichment:.2f}"
                  f"{' [capped]' if enrichment >= max_enrichment else ''})", flush=True)
            depth = max(batch_iters, int(round(batch_iters * enrichment)))
            rlm.max_iterations = depth
            result = rlm.completion(user)
            log, latest = _read_compound_log(rlm)
            cohesion = (latest["n_pass"] / latest["n_total"]) if latest and latest["n_total"] else 0.0
            confidence = _compound_confidence(log, decay) if log else cohesion
            if cohesion > best_cohesion:
                best_cohesion = cohesion
                stall = 0
            else:
                stall += 1
            enrichment *= 1.0 + rate * confidence
            enrichment = min(enrichment, max_enrichment)
            # feedback decay: reduce enrichment when no progress
            if stall > 0:
                enrichment *= decay
            submitted = False
            try:
                preds = parse_answer(result.response)
                submitted = bool(preds)
            except ValueError:
                preds = {}
            batches.append({
                "batch": b + 1,
                "iterations": depth,
                "cohesion": round(cohesion, 3),
                "confidence": round(confidence, 3),
                "enrichment": round(enrichment, 3),
                "stall": stall,
                "submitted": submitted,
                "accepted": bool(submitted and cohesion >= accept_gate),
            })
            print(f"[{task_id}] batch {b + 1}: cohesion {cohesion:.2f} "
                  f"conf {confidence:.2f} enrich x{enrichment:.2f} "
                  f"{'SUBMITTED' if submitted else ''}", flush=True)
            if submitted:
                if cohesion >= accept_gate:
                    final = result
                    print(f"[{task_id}] submission accepted (cohesion {cohesion:.2f})", flush=True)
                    break
                rejected = True
                rejected_cohesion = cohesion
                print(f"[{task_id}] submission REJECTED (cohesion {cohesion:.2f} < "
                      f"gate {accept_gate:.2f}); continuing to compound", flush=True)
                continue
            if best_cohesion >= 1.0:
                print(f"[{task_id}] cohesion reached 1.0 in batch {b + 1}; finalizing", flush=True)
                break
            if stall >= patience and best_cohesion >= min_cohesion:
                print(f"[{task_id}] compound converged after batch {b + 1} "
                      f"(no cohesion gain, best {best_cohesion:.2f})", flush=True)
                break
        if final is None:
            status = _read_compound_log(rlm)[1]
            user = ("SUBMIT NOW: set answer[\"content\"] to your best JSON predictions for "
                    "every test input and answer[\"ready\"]=True in a single repl block, "
                    "then it is submitted. Use your current transform().")
            final = rlm.completion(user)
            batches.append({"batch": max_batches + 1, "iterations": depth,
                            "cohesion": None, "confidence": None, "enrichment": round(enrichment, 3),
                            "stall": stall, "submitted": True,
                            "note": f"forced finalize; last cohesion {status}"})
    except Exception as e:
        print(f"[{task_id}] compound RLM error: {type(e).__name__}: {e}", flush=True)
        final = None
    finally:
        rlm.close()

    elapsed = time.perf_counter() - t0
    content = final.response if final is not None else ""
    metadata = final.metadata if final is not None else None
    if metadata is not None:
        trajectory = {"metadata": metadata.get("run_metadata") or metadata,
                      "iterations": metadata.get("iterations")}
    else:
        trajectory = None
    try:
        preds = parse_answer(content)
    except ValueError as e:
        preds, err = {}, str(e)
    else:
        err = None

    expected = task["test"]
    per_index = []
    n_correct = 0
    for i, tpair in enumerate(expected):
        pred = preds.get(str(i))
        if pred is None:
            per_index.append({"index": i, "match": False, "reason": "missing prediction"})
            continue
        if pred == tpair["output"]:
            n_correct += 1
            per_index.append({"index": i, "match": True})
        else:
            per_index.append({"index": i, "match": False, "reason": "wrong grid",
                              "expected": tpair["output"], "got": pred})

    in_tok = out_tok = 0
    cost = 0.0
    if final is not None and final.usage_summary:
        usage = final.usage_summary
        in_tok = usage.total_input_tokens
        out_tok = usage.total_output_tokens
        cost = in_tok / 1e6 * INPUT_PER_MTOK + out_tok / 1e6 * OUTPUT_PER_MTOK

    report = {
        "task": task_id,
        "solved": n_correct == len(expected),
        "n_correct": n_correct,
        "n_test": len(expected),
        "iterations": batch_iters * len(batches),
        "elapsed_s": round(elapsed, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 4),
        "parse_error": err,
        "per_index": per_index,
        "trajectory": trajectory,
        "raw_answer": content[:2000],
        "level": meta.get("level"),
        "rules": meta.get("rules"),
        "compound": {
            "max_batches": max_batches,
            "batch_iters": batch_iters,
            "rate": rate,
            "decay": decay,
            "batches": batches,
            "final_enrichment": round(enrichment, 3),
            "best_cohesion": round(best_cohesion, 3),
        },
    }
    print(f"[{task_id}] {'SOLVED' if report['solved'] else 'FAILED'} "
          f"{n_correct}/{len(expected)} test pairs, {elapsed:.0f}s, "
          f"{len(batches)} batches", flush=True)
    return report


def run_task_worker(task_id: str, cfg: dict) -> None:
    """Child-process entry point. Runs one task, writes a per-task result JSON,
    and never raises. The parent watches this process for crashes/timeouts."""
    _SOCKET_TIMEOUT_STATE["timeout"] = cfg["call_timeout"]
    # compounding: create a store instance to read verified solutions/patterns
    store = None
    if cfg.get("compounding_dir"):
        try:
            store = CompoundingStore(cfg["compounding_dir"])
        except Exception:
            pass  # non-fatal: run without compounding
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat, args=(task_id, cfg.get("heartbeat", 90), stop), daemon=True
    )
    heartbeat.start()
    try:
        if cfg.get("compound"):
            report = compound_run_task(task_id, cfg, store=store)
        else:
            report = run_task(task_id, cfg, store=store)
    except Exception as e:
        report = {
            "task": task_id,
            "solved": False,
            "n_correct": 0,
            "n_test": 0,
            "iterations": cfg["iterations"],
            "elapsed_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "parse_error": None,
            "error": f"worker crashed: {type(e).__name__}: {e}",
            "per_index": [],
            "trajectory": None,
            "raw_answer": "",
        }
    finally:
        stop.set()
    os.makedirs(cfg["result_dir"], exist_ok=True)
    path = os.path.join(cfg["result_dir"], f"{task_id}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f)
    os.replace(tmp, path)


def _result_path(cfg: dict, task_id: str) -> str:
    return os.path.join(cfg["result_dir"], f"{task_id}.json")


def _load_result(cfg: dict, task_id: str) -> dict:
    with open(_result_path(cfg, task_id)) as f:
        return json.load(f)


def _placeholder_report(task_id: str, cfg: dict, reason: str, elapsed_s: float) -> dict:
    return {
        "task": task_id,
        "solved": False,
        "n_correct": 0,
        "n_test": 0,
        "iterations": cfg["iterations"],
        "elapsed_s": round(elapsed_s, 1),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "parse_error": None,
        "error": reason,
        "per_index": [],
        "trajectory": None,
        "raw_answer": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RLM-based ARC solver prototype")
    ap.add_argument("--tasks", help="comma-separated task ids")
    ap.add_argument("--limit", type=int, default=3, help="first N tasks of the v1 eval set")
    ap.add_argument("--iterations", type=int, default=15, help="max RLM iterations per task")
    ap.add_argument("--timeout", type=float, default=900, help="max seconds per task (watchdog)")
    ap.add_argument("--call-timeout", type=float, default=300, help="max seconds per LLM API call")
    ap.add_argument("--sub-reasoning", default="medium", help="reasoning_effort for depth-1 sub-calls (low/medium/high)")
    ap.add_argument("--workers", type=int, default=2, help="parallel task processes")
    ap.add_argument("--retries", type=int, default=2, help="max attempts per task")
    ap.add_argument("--out", default="prototypes/rlm_runs/", help="output dir")
    ap.add_argument("--verbose", action="store_true", help="print full RLM trajectory to stdout")
    ap.add_argument("--trace", action="store_true", help="log per-task RLM trajectories (jsonl)")
    ap.add_argument("--resume", action="store_true", help="skip tasks that already have a result file")
    ap.add_argument("--heartbeat", type=float, default=90, help="liveness print interval (s)")
    ap.add_argument("--compound", action="store_true",
                    help="compound-loop mode: persistent batched RLM with confidence-gated budget, decay-weighted feedback, convergence detection")
    ap.add_argument("--batch-iters", type=int, default=5, help="max RLM iterations per compound batch")
    ap.add_argument("--max-batches", type=int, default=5, help="max compound batches per task")
    ap.add_argument("--compound-rate", type=float, default=0.2, help="enrichment rate (compound growth)")
    ap.add_argument("--feedback-decay", type=float, default=0.85, help="recency weight decay for confidence")
    ap.add_argument("--patience", type=int, default=2, help="batches without cohesion gain before converged")
    ap.add_argument("--accept-gate", type=float, default=0.8,
                    help="cohesion threshold for accepting a submission (compound)")
    ap.add_argument("--max-enrichment", type=float, default=3.0,
                    help="cap on enrichment multiplier to prevent runaway growth")
    ap.add_argument("--ccl", action="store_true",
                    help="run on the CCL benchmark (Compound Concept Learning, 300 tasks, L1-L3) from data/ccl_tasks")
    args = ap.parse_args()

    ccl_index = None
    if args.ccl:
        with open(os.path.join(DATA_ROOT, "ccl_index.json")) as f:
            ccl_index = {e["id"]: e for e in json.load(f)}
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if not args.ccl and task_ids and all(t.startswith("ccl_") for t in task_ids):
            args.ccl = True
        if args.ccl and ccl_index is None:
            with open(os.path.join(DATA_ROOT, "ccl_index.json")) as f:
                ccl_index = {e["id"]: e for e in json.load(f)}
    elif args.ccl:
        task_ids = list(ccl_index.keys())[: args.limit]
    else:
        with open(os.path.join(DATA_ROOT, "v1_public_evaluation_set.json")) as f:
            all_ids = json.load(f)
        task_ids = all_ids[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    result_dir = os.path.join(args.out, "results")
    os.makedirs(result_dir, exist_ok=True)
    compounding_dir = os.path.join(args.out, "compounding")
    os.makedirs(compounding_dir, exist_ok=True)
    store = CompoundingStore(compounding_dir)
    cfg = {
        "iterations": args.iterations,
        "task_timeout": args.timeout,
        "call_timeout": args.call_timeout,
        "sub_reasoning": args.sub_reasoning,
        "retries": args.retries,
        "trace": args.trace,
        "verbose": args.verbose,
        "out": args.out,
        "result_dir": result_dir,
        "compounding_dir": compounding_dir,
        "heartbeat": args.heartbeat,
        "compound": args.compound,
        "batch_iters": args.batch_iters,
        "max_batches": args.max_batches,
        "compound_rate": args.compound_rate,
        "feedback_decay": args.feedback_decay,
        "patience": args.patience,
        "accept_gate": args.accept_gate,
        "max_enrichment": args.max_enrichment,
        "ccl": args.ccl,
        "domain": "ccl" if args.ccl else "arc",
        "task_root": os.path.join(DATA_ROOT, "ccl_tasks") if args.ccl
        else os.path.join(DATA_ROOT, "tasks"),
    }

    if args.resume:
        task_ids = [t for t in task_ids if not os.path.exists(_result_path(cfg, t))]
        print(f"[sched] resume: {len(task_ids)} tasks remaining after skipping completed", flush=True)
    if not task_ids:
        print("[sched] nothing to run", flush=True)
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pending = list(task_ids)
    active = {}  # pid -> {proc, task, start, attempt}
    attempt = {t: 0 for t in task_ids}
    reports = {}
    while pending or active:
        while pending and len(active) < args.workers:
            tid = pending.pop(0)
            attempt[tid] += 1
            proc = ctx.Process(target=run_task_worker, args=(tid, cfg), daemon=False)
            proc.start()
            active[proc.pid] = {"proc": proc, "task": tid, "start": time.time(), "attempt": attempt[tid]}
            print(f"[sched] start {tid} (pid {proc.pid}, attempt {attempt[tid]})", flush=True)

        if not active:
            break
        time.sleep(5)
        for pid in list(active):
            info = active[pid]
            proc = info["proc"]
            elapsed = time.time() - info["start"]
            task_done = False
            if not proc.is_alive():
                proc.join(2)
                task_done = True
                reason = "ok"
            elif elapsed > cfg["task_timeout"]:
                print(f"[sched] kill {info['task']} (pid {pid}) after {elapsed:.0f}s watchdog", flush=True)
                proc.kill()
                proc.join(2)
                task_done = True
                reason = f"watchdog killed after {elapsed:.0f}s"
            if task_done:
                tid = info["task"]
                del active[pid]
                result_file = _result_path(cfg, tid)
                if os.path.exists(result_file):
                    reports[tid] = _load_result(cfg, tid)
                    # compounding: record verified result
                    r = reports[tid]
                    task_data = _load_task(tid, cfg)
                    features = _task_features(task_data)
                    rules = r.get("rules", [])
                    if not rules and r.get("compound"):
                        rules = [b.get("note", "") for b in r["compound"].get("batches", []) if b.get("accepted")]
                    store.record_task(
                        task_id=tid,
                        solved=r.get("solved", False),
                        code=r.get("raw_answer", "")[:2000] if r.get("solved") else "",
                        features=features,
                        rules=rules,
                        level=r.get("level"),
                        config={"compound": cfg.get("compound"), "batch_iters": cfg.get("batch_iters"),
                                "accept_gate": cfg.get("accept_gate")},
                        cost_usd=r.get("cost_usd", 0),
                        time_s=r.get("elapsed_s", 0),
                        test_passed=r.get("solved", False),
                    )
                elif info["attempt"] < args.retries:
                    reports.pop(tid, None)
                    print(f"[sched] requeue {tid} (attempt {info['attempt']} of {args.retries})", flush=True)
                    pending.append(tid)
                else:
                    reports[tid] = _placeholder_report(tid, cfg, f"died after {elapsed:.0f}s ({reason})", elapsed)
                    print(f"[sched] give up on {tid} after {info['attempt']} attempts", flush=True)

    reports = [reports[t] for t in task_ids if t in reports]
    solved = [r for r in reports if r["solved"]]
    total_cost = sum(r["cost_usd"] for r in reports)
    total_elapsed = sum(r["elapsed_s"] for r in reports)
    stamp = int(time.time())
    out_path = os.path.join(args.out, f"rlm_v1_{stamp}.json")
    payload = {"solved": [r["task"] for r in solved], "reports": reports}
    if args.ccl:
        em = {}
        for lv in (1, 2, 3):
            lv_total = sum(1 for t in task_ids if ccl_index[t]["level"] == lv)
            lv_solved = sum(1 for r in reports if r["solved"] and r.get("level") == lv)
            em[lv] = (lv_solved / lv_total) if lv_total else 0.0
        payload["ccl"] = {
            "n_tasks": len(task_ids),
            "em_l1": round(em[1], 4),
            "em_l2": round(em[2], 4),
            "em_l3": round(em[3], 4),
            "ces": round(em[3] / em[1], 4) if em[1] > 0 else 0.0,
            "total_cost_usd": round(total_cost, 4),
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n== RLM prototype: {len(solved)}/{len(reports)} solved, "
          f"${total_cost:.2f}, {total_elapsed/60:.0f}min total (wall)")
    print(f"solved tasks: {[r['task'] for r in solved]}")
    if args.ccl:
        print(f"CCL: EM@L1 {em[1]*100:.1f}%  EM@L2 {em[2]*100:.1f}%  "
              f"EM@L3 {em[3]*100:.1f}%  CES {em[3]/em[1]:.3f}"
              if em[1] > 0 else "CCL: EM@L1 = 0 (CES undefined)")
    cs = store.summary()
    print(f"compounding: {cs['verified_solutions']} verified solutions, "
          f"{cs['patterns']} patterns, capability {cs['capability']:.3f}, "
          f"enrichment x{cs['enrichment_multiplier']:.2f}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
