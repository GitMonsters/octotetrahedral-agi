"""The refinement-loop synthesizer: generate -> verify -> feedback -> retry."""

from __future__ import annotations

import time

from .config import HarnessConfig, load_models
from .llm import LLMClient
from .prompts import build_generate_prompt, build_refine_prompt, _task_features_for_prompt
from .verifier import extract_code, verify_code


class Synthesizer:
    def __init__(self, cfg: HarnessConfig, store=None):
        self.cfg = cfg
        self.models = load_models()
        self.synth = LLMClient(self.models[cfg.synth_model])
        self.candidate = LLMClient(self.models[cfg.candidate_model])
        self.last_codes: list[str] = []
        self.store = store  # CompoundingStore for cross-task learning

    def solve_task(self, task_id: str, task: dict, budget_steps: int = 0) -> dict:
        """Synthesize a solver for one task. Returns status + best code + score.

        When cfg.consensus is enabled and the task has test inputs, the solver
        must survive the consensus-on-test gate (see _solve_consensus). When
        cfg.kfold is enabled and the task has >= 2 train pairs, the solver must
        survive leave-one-out validation (see _solve_kfold). Otherwise falls
        back to the plain refinement loop.
        """
        if self.cfg.consensus and task.get("test"):
            return self._solve_consensus(task_id, task)
        if self.cfg.kfold and len(task.get("train", [])) >= 2:
            return self._solve_kfold(task_id, task)
        return self._solve_loop(
            task_id, task, steps=budget_steps or self.cfg.max_attempts,
            initial_candidates=self.cfg.initial_candidates,
        )

    def _solve_loop(self, task_id: str, task: dict, steps: int,
                    initial_candidates: int, timeout_s: float | None = None,
                    temp_offset: float = 0.0) -> dict:
        """The refinement-loop core. Enforces a wall-clock deadline: once
        exceeded, no new LLM calls are started and the best-so-far is returned
        with status "timeout"."""
        timeout_s = timeout_s or self.cfg.task_timeout_s

        # compounding: use difficulty signal for dynamic budget allocation
        if self.store is not None:
            features = _task_features_for_prompt(task)
            difficulty = self.store.get_difficulty_signal(task_id, features)
            budget_mult = difficulty.get("estimated_budget_mult", 1.0)
            steps = max(1, int(steps * budget_mult))
            initial_candidates = max(1, int(initial_candidates * budget_mult))
            # compounding: cross-task enrichment multiplier
            enrichment = self.store.get_enrichment_multiplier()
            temp_offset += (enrichment - 1.0) * 0.05  # slight temp boost from capability growth

        best = {"score": -1.0, "code": None, "feedback": ""}
        self.last_codes = []
        history: list[str] = []
        stalls = 0  # consecutive attempts without improving best
        t0 = time.time()
        timed_out = False

        def _run(client, msgs, temp) -> str | None:
            """One LLM call with transient retries; returns extracted code or None."""
            remaining = timeout_s - (time.time() - t0)
            if remaining <= 30:
                return None
            res = None
            for _ in range(3):
                try:
                    res = client.complete(
                        msgs,
                        temperature=temp,
                        max_tokens=self.cfg.max_tokens_generate,
                        timeout=min(self.cfg.llm_timeout_s, remaining),
                        reasoning_effort=self.cfg.reasoning_effort,
                    )
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(f"LLM call failed ({e}), retrying")
                    continue
                if res.text and res.text.strip():
                    break
            if res is None or not res.text or not res.text.strip():
                return None
            code = extract_code(res.text)
            if not code or "def solve" not in code:
                return None
            return code

        def _elapsed() -> float:
            return time.time() - t0

        # Phase 1: diverse first-round candidates. Rank them, keep the best.
        gen_msgs = build_generate_prompt(self.cfg, task, task_id, store=self.store)
        for i in range(initial_candidates):
            if _elapsed() >= timeout_s:
                timed_out = True
                break
            client = self.candidate if i >= 3 else self.synth  # a few cheap ones
            temp = self.cfg.temperature + temp_offset + 0.1 * (i % 3)
            code = _run(client, gen_msgs, temp)
            if code is None:
                continue
            self.last_codes.append(code)
            history.append(code)
            verdict = verify_code(self.cfg, code, task)
            if verdict["score"] > best["score"]:
                best = {"score": verdict["score"], "code": code, "feedback": verdict["feedback"]}
            if verdict["score"] == 1.0:
                return {
                    "task_id": task_id,
                    "status": "solved",
                    "score": 1.0,
                    "code": code,
                    "rules": self._extract_rules(code),
                    "attempts": 1,
                    "history": history,
                }
        if best["score"] == 1.0:
            return {
                "task_id": task_id,
                "status": "solved",
                "score": 1.0,
                "code": best["code"],
                "rules": self._extract_rules(best["code"]),
                "attempts": 1,
                "history": history,
            }

        # Phase 2: refine the best candidate with feedback
        for attempt in range(1, steps):
            if _elapsed() >= timeout_s:
                timed_out = True
                break
            last_code = best.get("code") or (self.last_codes[-1] if self.last_codes else "")
            last_feedback = best.get("feedback", "")
            msgs = build_refine_prompt(self.cfg, task, task_id, last_code, last_feedback, store=self.store)

            improved = False
            for cand in range(self.cfg.candidates_per_attempt):
                if _elapsed() >= timeout_s:
                    timed_out = True
                    break
                temp = self.cfg.refine_temperature + temp_offset
                if stalls >= 2 and best["score"] < 1.0:
                    temp = min(temp, 0.3)  # focus the fix on a stalling near-miss
                code = _run(self.synth, msgs, temp)
                if code is None:
                    continue
                self.last_codes.append(code)
                history.append(code)

                verdict = verify_code(self.cfg, code, task)
                if verdict["score"] > best["score"]:
                    best = {"score": verdict["score"], "code": code, "feedback": verdict["feedback"]}
                    improved = True
                if verdict["score"] == 1.0:
                    return {
                        "task_id": task_id,
                        "status": "solved",
                        "score": 1.0,
                        "code": code,
                        "rules": self._extract_rules(code),
                        "attempts": attempt + 1,
                        "history": history,
                    }

            if not improved:
                stalls += 1
            else:
                stalls = 0

        status = "timed_out" if timed_out else "unsolved"
        return {
            "task_id": task_id,
            "status": status,
            "score": best["score"],
            "code": best["code"],
            "rules": self._extract_rules(best.get("code", "")),
            "feedback": best["feedback"],
            "attempts": steps,
            "history": history,
        }

    def _solve_kfold(self, task_id: str, task: dict) -> dict:
        """Leave-one-out validation.

        A solver is accepted only if, for EVERY fold i, a fresh solver
        synthesized on train-minus-example-i both (a) passes train-minus-i and
        (b) generalizes to the held-out example i. The submitted code is the
        first fold's rule (which provably passes all train pairs and was never
        fit to its own held-out example).
        """
        from .verifier import verify_code

        train = task["train"]
        n = len(train)
        nf = min(n, self.cfg.max_folds)
        fold_timeout = self.cfg.task_timeout_s / max(nf, 1)
        fold_steps = min(self.cfg.max_attempts, self.cfg.fold_budget_attempts)
        fold_cands = min(self.cfg.initial_candidates, self.cfg.fold_initial_candidates)
        fold_codes: list[str] = []
        total_attempts = 0
        t0 = time.time()
        kfold_meta = {"n_folds": nf, "folds_passed": 0, "reject_reason": None, "holdout_scores": []}

        for i in range(nf):
            if time.time() - t0 > self.cfg.task_timeout_s:
                return {
                    "task_id": task_id, "status": "timed_out", "score": 1.0,
                    "code": fold_codes[0] if fold_codes else None, "attempts": total_attempts,
                    "kfold": {**kfold_meta, "reject_reason": "overall deadline hit mid-folds"},
                }
            sub = {"train": [p for j, p in enumerate(train) if j != i]}
            out = self._solve_loop(
                task_id, sub, steps=fold_steps,
                initial_candidates=fold_cands, timeout_s=fold_timeout,
            )
            total_attempts += out.get("attempts", fold_steps)
            if out["score"] < 1.0:
                return {
                    "task_id": task_id, "status": "rejected", "score": out["score"],
                    "code": None, "attempts": total_attempts,
                    "kfold": {**kfold_meta, "reject_reason": (
                        f"fold {i}: no solver passed train-minus-{i} "
                        f"(best {out['score']:.3f})")},
                }
            hv = verify_code(self.cfg, out["code"], {"train": [train[i]]})
            if hv["passed"] != 1:
                return {
                    "task_id": task_id, "status": "rejected", "score": 1.0,
                    "code": None, "attempts": total_attempts,
                    "kfold": {**kfold_meta, "reject_reason": (
                        f"fold {i}: solver passed train-minus-{i} but failed "
                        f"held-out example {i}")},
                }
            fold_codes.append(out["code"])
            kfold_meta["folds_passed"] = i + 1
            kfold_meta["holdout_scores"].append(1.0)

        return {
            "task_id": task_id, "status": "solved", "score": 1.0,
            "code": fold_codes[0], "attempts": total_attempts,
            "kfold": kfold_meta,
        }

    def _solve_consensus(self, task_id: str, task: dict) -> dict:
        """Consensus-on-test gate.

        Runs the full-budget solve (round 0) plus `consensus_rounds-1`
        independent full-train solves with a small budget and varied
        temperature. Each independent solver must pass ALL train pairs. The
        task is solved only if at least `consensus_majority` solvers produce
        IDENTICAL predictions on every test input; the submitted code is the
        majority solver (preferring the highest-budget round-0 code when it is
        in the majority). Solvers that only memorized the training grids
        typically disagree on unseen test inputs, so agreement is a
        generalization signal that needs no withheld examples.
        """
        from collections import Counter

        from .verifier import _run_candidates

        t0 = time.time()
        out0 = self._solve_loop(
            task_id, task, steps=self.cfg.max_attempts,
            initial_candidates=self.cfg.initial_candidates,
            timeout_s=self.cfg.task_timeout_s, temp_offset=0.0,
        )
        attempts = out0.get("attempts", 0)
        if out0["score"] < 1.0:
            out0["attempts"] = attempts
            return out0

        codes: list[str] = [out0["code"]]
        for k in range(1, self.cfg.consensus_rounds):
            remaining = self.cfg.task_timeout_s - (time.time() - t0)
            if remaining < 120:
                break
            out = self._solve_loop(
                task_id, task, steps=self.cfg.consensus_budget_attempts,
                initial_candidates=self.cfg.consensus_initial_candidates,
                timeout_s=min(self.cfg.consensus_round_timeout_s, remaining),
                temp_offset=0.25 * k,
            )
            attempts += out.get("attempts", self.cfg.consensus_budget_attempts)
            if out["score"] >= 1.0:
                codes.append(out["code"])

        test_inputs = task["test"]
        preds = []
        for code in codes:
            results = _run_candidates(self.cfg, code, test_inputs)
            key = tuple(
                tuple(tuple(r) for r in res.get("out"))
                if isinstance(res.get("out"), list) and res.get("out") and isinstance(res["out"][0], list)
                else None
                for res in results
            )
            preds.append(key)

        counts = Counter(preds)
        top_pred, top_n = counts.most_common(1)[0]
        agree = top_n
        meta = {
            "rounds": len(codes),
            "agreeing": agree,
            "needed": self.cfg.consensus_majority,
        }
        if top_n >= self.cfg.consensus_majority:
            idx = preds.index(top_pred)
            return {
                "task_id": task_id, "status": "solved", "score": 1.0,
                "code": codes[idx], "attempts": attempts, "consensus": meta,
            }
        return {
            "task_id": task_id, "status": "rejected", "score": 1.0,
            "code": None, "attempts": attempts,
            "consensus": {**meta, "reject_reason": (
                f"consensus: only {agree}/{len(codes)} independent solvers agreed "
                f"on test predictions (need {self.cfg.consensus_majority})")},
        }

    def apply_solver(self, code: str, test_pairs: list[dict]) -> dict:
        """Run the final solver on test pairs (not part of training signal)."""
        from .verifier import _run_candidates

        results = _run_candidates(self.cfg, code, test_pairs)
        passed = sum(1 for r in results if r.get("ok"))
        return {
            "test_passed": passed,
            "test_total": len(test_pairs),
            "test_results": results,
        }

    @staticmethod
    def _extract_rules(code: str) -> list[str]:
        """Extract transformation rule names from solver code.

        Heuristic: look for common ARC primitives in the code. This is not
        exhaustive but captures the most frequent patterns.
        """
        if not code:
            return []
        rules = []
        code_lower = code.lower()
        # rotation patterns
        if "rot90" in code_lower or "rot_cw" in code_lower or "np.rot90" in code:
            rules.append("rotation")
        if "rot_ccw" in code_lower or "-90" in code and "rot" in code_lower:
            rules.append("rotation_ccw")
        # flip patterns
        if "flip" in code_lower or "::-1" in code or "fliplr" in code_lower or "flipud" in code_lower:
            rules.append("flip")
        # transpose
        if "transpose" in code_lower or ".t " in code or "np.transpose" in code:
            rules.append("transpose")
        # color remapping
        if "color" in code_lower and ("map" in code_lower or "swap" in code_lower or "remap" in code_lower or "shift" in code_lower):
            rules.append("color_remap")
        if "replace" in code_lower and any(c in code_lower for c in ["0", "1", "2"]):
            rules.append("color_replace")
        # gravity
        if "gravity" in code_lower:
            rules.append("gravity")
        # sorting
        if "sort" in code_lower:
            rules.append("sorting")
        # flood fill / connected components
        if "flood" in code_lower or "connected" in code_lower or "bfs" in code_lower or "dfs" in code_lower:
            rules.append("flood_fill")
        # symmetry / reflection
        if "symmetr" in code_lower or "reflect" in code_lower or "mirror" in code_lower:
            rules.append("symmetry")
        # row/col operations
        if "row" in code_lower and ("sum" in code_lower or "count" in code_lower or "max" in code_lower):
            rules.append("row_ops")
        if "col" in code_lower and ("sum" in code_lower or "count" in code_lower or "max" in code_lower):
            rules.append("col_ops")
        # mask / region
        if "mask" in code_lower or "region" in code_lower or "crop" in code_lower:
            rules.append("region_ops")
        return list(set(rules))
