"""
Sensorimotor Loop for ARC Puzzle Solving via Embodied Cognition.

Implements an iterative perceive→think→act→evaluate→adapt cycle where the
model treats grid prediction as an embodied action, observes the consequences,
and refines its output using voxel memory, episodic error memory, and
RNA-editing-driven exploration/exploitation control.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MAX_GRID: int = 30
NUM_COLORS: int = 11  # ARC colors 0-10
PAD_TOKEN: int = NUM_COLORS  # 11 used as padding


@dataclass
class SensorimotorResult:
    """Outcome of a full sensorimotor solving attempt."""

    best_grid: List[List[int]]
    confidence: float
    iterations_used: int
    trajectory: List[List[List[int]]]  # one grid per iteration
    accuracy: Optional[float]  # populated when ground truth available
    converged: bool
    temperature_final: float
    errors_logged: int


class SensorimotorLoop:
    """Iterative perception→think→act→evaluate→adapt loop for ARC solving."""

    def __init__(
        self,
        max_iterations: int = 5,
        temperature_escalation: float = 0.1,
        confidence_threshold: float = 0.85,
        error_memory_weight: float = 2.0,
    ):
        self.max_iterations = max_iterations
        self.temperature_escalation = temperature_escalation
        self.confidence_threshold = confidence_threshold
        self.error_memory_weight = error_memory_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(
        self,
        model: Any,
        input_grid: List[List[int]],
        output_grid: Optional[List[List[int]]] = None,
        device: str = "cpu",
    ) -> SensorimotorResult:
        """Run the sensorimotor loop and return the best predicted grid.

        Args:
            model: OctoTetrahedralModel (or compatible duck-typed object).
            input_grid: 2-D list of ints (ARC input, values 0-10).
            output_grid: Optional ground-truth for training/eval feedback.
            device: Torch device string.

        Returns:
            SensorimotorResult with the best prediction and diagnostics.
        """
        in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_h, out_w = in_h, in_w
        if output_grid is not None:
            out_h = len(output_grid)
            out_w = len(output_grid[0]) if output_grid else 0

        input_tensor = self._list_to_tensor(input_grid, device)
        target_tensor: Optional[torch.Tensor] = None
        if output_grid is not None:
            target_tensor = self._list_to_tensor(output_grid, device)

        # Encode input as token ids for the model
        token_ids = self._grid_to_tokens(input_grid, device)

        trajectory: List[List[List[int]]] = []
        best_grid: Optional[torch.Tensor] = None
        best_accuracy: float = -1.0
        best_confidence: float = 0.0
        prev_grid: Optional[torch.Tensor] = None
        errors_logged: int = 0
        current_temp_offset: float = 0.0
        converged: bool = False

        # ---- Phase 0: seed voxel memory with the input observation ----
        voxel_mem = getattr(model, "voxel_memory", None)
        if voxel_mem is not None:
            voxel_mem.write_grid(input_tensor)

        for iteration in range(self.max_iterations):
            # (a) PERCEIVE ------------------------------------------------
            perception = self._perceive(model, input_tensor, device)

            # (b) THINK ----------------------------------------------------
            outputs = self._think(model, token_ids, device)
            logits = outputs.get("logits")  # [1, seq, vocab]

            # (c) ACT ------------------------------------------------------
            candidate = self._tokens_to_grid(logits, out_h, out_w)  # int tensor [h, w]
            candidate_list = self._tensor_to_list(candidate)
            trajectory.append(candidate_list)

            # (d) EVALUATE -------------------------------------------------
            accuracy: Optional[float] = None
            if target_tensor is not None:
                accuracy = self._compute_accuracy(candidate, target_tensor)

            stability = 0.0
            if prev_grid is not None:
                stability = self._compute_stability(candidate, prev_grid)

            confidence = self._estimate_confidence(
                model, outputs, stability, accuracy
            )

            # Track best result
            improved = False
            if accuracy is not None:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_grid = candidate.clone()
                    best_confidence = confidence
                    improved = True
            elif confidence > best_confidence:
                best_grid = candidate.clone()
                best_confidence = confidence
                improved = True

            if best_grid is None:
                best_grid = candidate.clone()

            # Exit conditions
            if confidence >= self.confidence_threshold:
                converged = True
                break
            if accuracy is not None and accuracy >= 1.0:
                converged = True
                break

            # (e) ADAPT (if not converged and not last iteration) ----------
            if iteration < self.max_iterations - 1:
                # Increase RNA editing temperature to encourage exploration
                current_temp_offset = self._escalate_temperature(
                    model, current_temp_offset
                )

                # Store error signal in episodic memory
                error_tensor = self._compute_error_signal(
                    candidate, target_tensor, input_tensor, device
                )
                self._store_error(model, error_tensor, candidate, iteration)
                errors_logged += 1

                # Write candidate grid to voxel memory (model sees history)
                if voxel_mem is not None:
                    voxel_mem.write_grid(candidate.to(device))
                    voxel_mem.decay()

                # If accuracy decreased, revert to previous best
                if not improved and accuracy is not None and prev_grid is not None:
                    # Feed best grid back as context for next iteration
                    if best_grid is not None:
                        token_ids = self._grid_to_tokens(
                            self._tensor_to_list(best_grid), device
                        )

            prev_grid = candidate.clone()

        # Restore temperature offset we introduced
        self._restore_temperature(model, current_temp_offset)

        assert best_grid is not None
        return SensorimotorResult(
            best_grid=self._tensor_to_list(best_grid),
            confidence=best_confidence,
            iterations_used=len(trajectory),
            trajectory=trajectory,
            accuracy=best_accuracy if best_accuracy >= 0 else None,
            converged=converged,
            temperature_final=self._get_temperature(model) or 0.0,
            errors_logged=errors_logged,
        )

    # ------------------------------------------------------------------
    # Loop step helpers
    # ------------------------------------------------------------------

    def _perceive(
        self,
        model: Any,
        grid_tensor: torch.Tensor,
        device: str,
    ) -> Optional[torch.Tensor]:
        """Encode input grid via the model's perception limb."""
        perception_limb = getattr(model, "perception", None)
        if perception_limb is None:
            return None
        flat = grid_tensor.float().flatten().unsqueeze(0).to(device)  # [1, h*w]
        # Perception limbs typically expect [batch, seq, dim]; pad to embed_dim
        embed_dim = getattr(model, "config", None)
        dim = 256
        if embed_dim is not None:
            dim = getattr(embed_dim, "hidden_dim", 256)
        if flat.shape[-1] < dim:
            flat = F.pad(flat, (0, dim - flat.shape[-1]))
        flat = flat.unsqueeze(1)  # [1, 1, dim]
        try:
            return perception_limb(flat)
        except Exception:
            return None

    def _think(
        self,
        model: Any,
        token_ids: torch.Tensor,
        device: str,
    ) -> Dict[str, Any]:
        """Run the model's forward pass on token ids."""
        token_ids = token_ids.to(device)
        try:
            out = model.forward(input_ids=token_ids, return_dict=True)
            if isinstance(out, dict):
                return out
            # Fallback for models returning a tuple
            return {"logits": out[0]}
        except Exception:
            # Minimal fallback — return random logits shaped for one grid
            seq_len = token_ids.shape[-1]
            return {
                "logits": torch.randn(1, seq_len, NUM_COLORS, device=device)
            }

    def _estimate_confidence(
        self,
        model: Any,
        outputs: Dict[str, Any],
        stability: float,
        accuracy: Optional[float],
    ) -> float:
        """Aggregate a scalar confidence in [0, 1]."""
        scores: List[float] = []

        # Model metacognition confidence
        meta = getattr(model, "metacognition", None)
        if meta is not None and "hidden_states" in outputs:
            try:
                hs = outputs["hidden_states"]
                meta_out = meta(hs)
                if isinstance(meta_out, dict) and "confidence" in meta_out:
                    scores.append(float(meta_out["confidence"].mean()))
                elif isinstance(meta_out, torch.Tensor):
                    scores.append(float(meta_out.mean().sigmoid()))
            except Exception:
                pass

        # Stability is a proxy for confidence: high stability → converging
        scores.append(stability)

        # If we know accuracy, blend it in
        if accuracy is not None:
            scores.append(accuracy)

        return sum(scores) / max(len(scores), 1)

    def _escalate_temperature(
        self, model: Any, current_offset: float
    ) -> float:
        """Bump RNA editing temperature to encourage exploration."""
        rna = getattr(model, "rna_editing", None)
        if rna is None:
            return current_offset

        cap = 2.0
        new_offset = min(current_offset + self.temperature_escalation, cap)
        delta = new_offset - current_offset

        temp_param = getattr(rna, "temperature_base", None)
        if temp_param is not None and isinstance(temp_param, torch.nn.Parameter):
            temp_param.data.add_(delta)
            t_max = getattr(rna, "temperature_max", cap)
            temp_param.data.clamp_(max=t_max)

        return new_offset

    def _restore_temperature(self, model: Any, offset: float) -> None:
        """Undo the cumulative temperature offset applied during the loop."""
        if offset == 0.0:
            return
        rna = getattr(model, "rna_editing", None)
        if rna is None:
            return
        temp_param = getattr(rna, "temperature_base", None)
        if temp_param is not None and isinstance(temp_param, torch.nn.Parameter):
            temp_param.data.sub_(offset)

    def _get_temperature(self, model: Any) -> Optional[float]:
        rna = getattr(model, "rna_editing", None)
        if rna is None:
            return None
        temp_param = getattr(rna, "temperature_base", None)
        if temp_param is not None:
            return float(temp_param.data.mean())
        return None

    def _compute_error_signal(
        self,
        candidate: torch.Tensor,
        target: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Produce a per-cell error tensor for episodic storage."""
        if target is not None:
            # Binary error mask: 1 where wrong, 0 where correct
            h = min(candidate.shape[0], target.shape[0])
            w = min(candidate.shape[1], target.shape[1])
            return (candidate[:h, :w] != target[:h, :w]).float().to(device)
        # Without ground truth, use difference from input as a heuristic
        h = min(candidate.shape[0], input_tensor.shape[0])
        w = min(candidate.shape[1], input_tensor.shape[1])
        return (candidate[:h, :w] != input_tensor[:h, :w]).float().to(device)

    def _store_error(
        self,
        model: Any,
        error_tensor: torch.Tensor,
        candidate: torch.Tensor,
        iteration: int,
    ) -> None:
        """Write an error event into episodic memory."""
        mem_limb = getattr(model, "memory_limb", None)
        if mem_limb is None:
            return
        episodic = getattr(mem_limb, "episodic", None) or getattr(
            mem_limb, "episodic_memory", None
        )
        if episodic is None:
            # memory_limb itself might expose store()
            episodic = mem_limb

        store_fn = getattr(episodic, "store", None)
        if store_fn is None:
            return

        try:
            store_fn(
                memory=error_tensor.flatten(),
                importance=self.error_memory_weight,
                spatial_context=candidate.float().flatten(),
                object_refs=[f"iter_{iteration}"],
                event_type="error",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Grid ↔ token conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_to_tokens(
        grid: List[List[int]], device: str = "cpu"
    ) -> torch.Tensor:
        """Convert a 2-D int grid to a flat token-id tensor [1, max_h*max_w]."""
        tokens = torch.full(
            (MAX_GRID * MAX_GRID,), PAD_TOKEN, dtype=torch.long, device=device
        )
        for i, row in enumerate(grid):
            if i >= MAX_GRID:
                break
            for j, val in enumerate(row):
                if j >= MAX_GRID:
                    break
                tokens[i * MAX_GRID + j] = max(0, min(val, NUM_COLORS - 1))
        return tokens.unsqueeze(0)  # [1, 900]

    @staticmethod
    def _tokens_to_grid(
        logits: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Decode model logits to an integer grid [height, width].

        Args:
            logits: [batch, seq_len, vocab] or [batch, seq_len].
            height: Target grid height.
            width: Target grid width.

        Returns:
            Integer tensor of shape [height, width] with values in [0, 10].
        """
        if logits.dim() == 3:
            # Pick most likely color per cell
            preds = logits[0].argmax(dim=-1)  # [seq_len]
        else:
            preds = logits[0]

        grid = torch.zeros(height, width, dtype=torch.long, device=logits.device)
        for i in range(height):
            for j in range(width):
                idx = i * MAX_GRID + j
                if idx < preds.shape[0]:
                    grid[i, j] = preds[idx].clamp(0, NUM_COLORS - 1)
        return grid

    # ------------------------------------------------------------------
    # Accuracy / stability metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_accuracy(
        predicted: torch.Tensor, expected: torch.Tensor
    ) -> float:
        """Cell-wise accuracy between two integer grids."""
        h = min(predicted.shape[0], expected.shape[0])
        w = min(predicted.shape[1], expected.shape[1])
        if h == 0 or w == 0:
            return 0.0
        p = predicted[:h, :w]
        e = expected[:h, :w]
        return float((p == e).float().mean())

    @staticmethod
    def _compute_stability(
        current: torch.Tensor, previous: torch.Tensor
    ) -> float:
        """Fraction of cells unchanged between two successive predictions."""
        h = min(current.shape[0], previous.shape[0])
        w = min(current.shape[1], previous.shape[1])
        if h == 0 or w == 0:
            return 0.0
        return float((current[:h, :w] == previous[:h, :w]).float().mean())

    # ------------------------------------------------------------------
    # Tensor / list conversion utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _list_to_tensor(
        grid: List[List[int]], device: str = "cpu"
    ) -> torch.Tensor:
        return torch.tensor(grid, dtype=torch.long, device=device)

    @staticmethod
    def _tensor_to_list(t: torch.Tensor) -> List[List[int]]:
        return [[int(t[i, j]) for j in range(t.shape[1])] for i in range(t.shape[0])]


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    import sys

    # --- Helper conversion tests ---
    grid_3x3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    tokens = SensorimotorLoop._grid_to_tokens(grid_3x3)
    assert tokens.shape == (1, MAX_GRID * MAX_GRID)
    assert int(tokens[0, 0]) == 0
    assert int(tokens[0, 1]) == 1
    assert int(tokens[0, MAX_GRID]) == 3  # row 1, col 0

    # Round-trip: grid → tokens → logits(one-hot) → grid
    # Clamp pad tokens to 0 before one-hot (they won't be read back)
    clamped = tokens[0].clamp(0, NUM_COLORS - 1)
    one_hot = F.one_hot(clamped, num_classes=NUM_COLORS).float().unsqueeze(0)
    recovered = SensorimotorLoop._tokens_to_grid(one_hot, 3, 3)
    assert recovered.tolist() == grid_3x3, f"Round-trip failed: {recovered.tolist()}"

    # --- Accuracy / stability ---
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[1, 2], [3, 0]])
    assert SensorimotorLoop._compute_accuracy(a, a) == 1.0
    assert SensorimotorLoop._compute_accuracy(a, b) == 0.75
    assert SensorimotorLoop._compute_stability(a, a) == 1.0

    # --- Minimal duck-typed model for loop test ---
    class _DummyVoxelMemory:
        def write_grid(self, grid: torch.Tensor, **kw: Any) -> None:
            pass

        def decay(self) -> None:
            pass

    class _DummyEpisodic:
        def store(self, **kw: Any) -> None:
            pass

    class _DummyMemoryLimb:
        def __init__(self) -> None:
            self.episodic = _DummyEpisodic()

    class _DummyRNAEditing(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temperature_base = torch.nn.Parameter(torch.tensor(1.0))
            self.temperature_max = 5.0

    class _DummyModel(torch.nn.Module):
        def __init__(self, out_h: int, out_w: int) -> None:
            super().__init__()
            self.out_h = out_h
            self.out_w = out_w
            self.voxel_memory = _DummyVoxelMemory()
            self.memory_limb = _DummyMemoryLimb()
            self.rna_editing = _DummyRNAEditing()
            self.perception = None
            self.metacognition = None
            self._call_count = 0

        def forward(self, input_ids: torch.Tensor, **kw: Any) -> Dict[str, Any]:
            self._call_count += 1
            seq = input_ids.shape[-1]
            # After first call, produce the "correct" answer to test convergence
            if self._call_count > 1:
                logits = torch.zeros(1, seq, NUM_COLORS)
                for i in range(self.out_h):
                    for j in range(self.out_w):
                        logits[0, i * MAX_GRID + j, (i + j) % NUM_COLORS] = 10.0
            else:
                logits = torch.randn(1, seq, NUM_COLORS)
            return {"logits": logits}

    expected = [[(i + j) % NUM_COLORS for j in range(3)] for i in range(3)]
    dummy = _DummyModel(3, 3)
    loop = SensorimotorLoop(max_iterations=4, confidence_threshold=0.99)
    result = loop.solve(dummy, grid_3x3, output_grid=expected, device="cpu")

    assert isinstance(result, SensorimotorResult)
    assert result.iterations_used >= 1
    assert len(result.trajectory) == result.iterations_used
    assert result.best_grid == expected, (
        f"Expected convergence to target grid, got {result.best_grid}"
    )
    assert result.accuracy == 1.0
    assert result.converged
    assert result.temperature_final is not None

    print("SensorimotorLoop self-test passed!")
    sys.exit(0)
