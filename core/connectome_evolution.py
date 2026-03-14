"""Connectome Evolution — evolve tetrahedral neural wiring via genetic algorithms.

Uses tournament selection, uniform crossover, and targeted mutation to optimise
the adjacency graph, synapse strengths, and excitatory/inhibitory balance of a
SpikingTetrahedralLayer.  Genomes encode sparse, symmetric, undirected
connectivity with Dale's-principle-compliant E/I labelling.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """A single candidate wiring configuration for the spiking layer.

    Attributes:
        adjacency_mask: [N, N] binary tensor — which neurons connect.
        weight_scales:  [N, N] positive float tensor — relative synapse strengths.
        ei_labels:      [N] tensor of +1 (excitatory) or -1 (inhibitory).
        fitness:        Scalar fitness assigned during evaluation.
        generation:     Generation in which this genome was created.
    """

    adjacency_mask: torch.Tensor   # [N, N] binary
    weight_scales: torch.Tensor    # [N, N] float ∈ [0, 2]
    ei_labels: torch.Tensor        # [N]    +1 or -1
    fitness: float = 0.0
    generation: int = 0

    def clone(self) -> "Genome":
        """Return a deep copy of this genome."""
        return Genome(
            adjacency_mask=self.adjacency_mask.clone(),
            weight_scales=self.weight_scales.clone(),
            ei_labels=self.ei_labels.clone(),
            fitness=self.fitness,
            generation=self.generation,
        )


# ---------------------------------------------------------------------------
# ConnectomeEvolution
# ---------------------------------------------------------------------------

class ConnectomeEvolution:
    """Genetic-algorithm engine for evolving spiking-layer connectomes.

    Maintains a population of ``Genome`` instances and provides selection,
    crossover, mutation, and elitism operators that respect biological
    constraints (sparse symmetric adjacency, Dale's principle, positive
    weights clamped to [0, 2]).

    Typical usage::

        evo = ConnectomeEvolution()
        evo.seed_from_geometry(adjacency, distances)
        best = evo.run_evolution(model, eval_fn, num_generations=20)
    """

    def __init__(
        self,
        num_neurons: int = 64,
        population_size: int = 20,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        elite_fraction: float = 0.2,
        ei_ratio: float = 0.8,
    ) -> None:
        self.num_neurons: int = num_neurons
        self.population_size: int = population_size
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.elite_fraction: float = elite_fraction
        self.ei_ratio: float = ei_ratio  # target fraction of excitatory neurons

        self.population: List[Genome] = []

    # ------------------------------------------------------------------
    # Seed / initialisation
    # ------------------------------------------------------------------

    def seed_from_geometry(
        self,
        adjacency: torch.Tensor,
        distances: torch.Tensor,
    ) -> None:
        """Create the initial population seeded from tetrahedral geometry.

        Each genome starts from the geometry-derived adjacency but is varied
        by random edge flips and Gaussian weight noise so the population is
        diverse from generation zero.

        Args:
            adjacency: [N, N] binary connectivity from ``TetrahedralGeometry``.
            distances: [N, N] pairwise distances.
        """
        N = self.num_neurons
        assert adjacency.shape == (N, N), f"Expected ({N},{N}), got {adjacency.shape}"

        # Base weight scales from inverse distance (closer → stronger)
        inv_dist = 1.0 / (distances + 1e-6)
        base_weights = (inv_dist / inv_dist.max()).clamp(0.0, 2.0)

        self.population = []
        for i in range(self.population_size):
            # --- adjacency with random edge flips ---
            mask = adjacency.clone().float()
            flip_prob = 0.05 + 0.10 * (i / max(self.population_size - 1, 1))
            flips = (torch.rand(N, N) < flip_prob).float()
            mask = (mask + flips) % 2  # XOR-like flip
            mask = self._enforce_symmetry(mask)
            mask = self._enforce_sparsity(mask)
            mask.fill_diagonal_(0.0)

            # --- weight scales with noise ---
            ws = (base_weights + torch.randn(N, N) * 0.2).abs().clamp(0.0, 2.0)
            ws = ws * mask  # zero out non-connected

            # --- E/I labels ---
            ei = self._random_ei_labels()

            self.population.append(
                Genome(
                    adjacency_mask=mask,
                    weight_scales=ws,
                    ei_labels=ei,
                    fitness=0.0,
                    generation=0,
                )
            )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, genome: Genome, rate: Optional[float] = None) -> Genome:
        """Return a mutated copy of *genome*.

        * Flips random edges in ``adjacency_mask``.
        * Adds Gaussian noise to ``weight_scales``.
        * Randomly flips a small number of E/I labels, then re-enforces the
          target E/I ratio.

        Args:
            genome: Source genome (not modified in-place).
            rate:   Per-element mutation probability (defaults to
                    ``self.mutation_rate``).

        Returns:
            A new ``Genome`` with mutations applied.
        """
        rate = rate if rate is not None else self.mutation_rate
        child = genome.clone()
        N = self.num_neurons

        # --- adjacency edge flips ---
        flip_mask = (torch.rand(N, N) < rate).float()
        child.adjacency_mask = (child.adjacency_mask + flip_mask) % 2
        child.adjacency_mask = self._enforce_symmetry(child.adjacency_mask)
        child.adjacency_mask = self._enforce_sparsity(child.adjacency_mask)
        child.adjacency_mask.fill_diagonal_(0.0)

        # --- weight noise ---
        noise = torch.randn(N, N) * rate * 0.5
        child.weight_scales = (child.weight_scales + noise).abs().clamp(0.0, 2.0)
        child.weight_scales = child.weight_scales * child.adjacency_mask

        # --- E/I label flips ---
        ei_flip = (torch.rand(N) < rate).float()
        child.ei_labels = child.ei_labels * (1 - 2 * ei_flip)  # flip selected
        child.ei_labels = self._enforce_ei_ratio(child.ei_labels)

        child.fitness = 0.0
        return child

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Uniform crossover: independently pick each synapse from either parent.

        For every ``[i, j]`` position a coin is flipped to decide which
        parent donates the adjacency bit and weight scale.  E/I labels are
        similarly crossed over per-neuron.

        Args:
            parent1: First parent genome.
            parent2: Second parent genome.

        Returns:
            A new child ``Genome``.
        """
        N = self.num_neurons

        # Per-synapse mask: True → take from parent1, False → parent2
        syn_sel = torch.rand(N, N) < 0.5

        adj = torch.where(syn_sel, parent1.adjacency_mask, parent2.adjacency_mask)
        adj = self._enforce_symmetry(adj)
        adj = self._enforce_sparsity(adj)
        adj.fill_diagonal_(0.0)

        ws = torch.where(syn_sel, parent1.weight_scales, parent2.weight_scales)
        ws = ws.abs().clamp(0.0, 2.0) * adj

        # Per-neuron E/I selection
        ei_sel = torch.rand(N) < 0.5
        ei = torch.where(ei_sel, parent1.ei_labels, parent2.ei_labels)
        ei = self._enforce_ei_ratio(ei)

        gen = max(parent1.generation, parent2.generation) + 1
        return Genome(
            adjacency_mask=adj,
            weight_scales=ws,
            ei_labels=ei,
            fitness=0.0,
            generation=gen,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        model: nn.Module,
        eval_fn: Callable[[], float],
        device: str = "cpu",
    ) -> None:
        """Score every genome in the population.

        For each genome the method:
        1. Applies its adjacency and weights to *model*'s spiking layer.
        2. Calls ``eval_fn()`` (which should return a scalar fitness).
        3. Stores the result in ``genome.fitness``.

        Args:
            model:   A model containing a ``SpikingTetrahedralLayer`` accessible
                     via ``model.spiking_layer`` **or** as the model itself.
            eval_fn: Zero-argument callable returning a ``float`` fitness.
            device:  Target device string.
        """
        spiking: Optional[nn.Module] = getattr(model, "spiking_layer", None)
        if spiking is None:
            # Allow passing the spiking layer directly for testing
            if hasattr(model, "synapse_weights"):
                spiking = model
            else:
                raise AttributeError(
                    "model has no 'spiking_layer' attribute and is not a "
                    "SpikingTetrahedralLayer itself."
                )

        original_weights = spiking.synapse_weights.data.clone()

        for genome in self.population:
            # Apply genome wiring to the spiking layer
            effective = (
                genome.adjacency_mask.to(device)
                * genome.weight_scales.to(device)
            )
            spiking.synapse_weights.data.copy_(effective)

            genome.fitness = eval_fn()

        # Restore original weights after evaluation
        spiking.synapse_weights.data.copy_(original_weights)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(self, k: int = 2) -> List[Genome]:
        """Tournament selection: pick *k* random individuals, return the best.

        Args:
            k: Tournament size.

        Returns:
            A list containing the single tournament winner (as a one-element
            list for API consistency with multi-parent schemes).
        """
        competitors = random.sample(self.population, min(k, len(self.population)))
        winner = max(competitors, key=lambda g: g.fitness)
        return [winner]

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def evolve_generation(self) -> List[Genome]:
        """Run one generation of the evolutionary loop.

        1. Sort population by fitness (descending).
        2. Carry forward the elite fraction unchanged.
        3. Fill the rest of the population via tournament-selected parents,
           crossover, and mutation.

        Returns:
            The new population list (also stored in ``self.population``).
        """
        self.population.sort(key=lambda g: g.fitness, reverse=True)

        num_elites = max(1, int(self.elite_fraction * self.population_size))
        new_pop: List[Genome] = [g.clone() for g in self.population[:num_elites]]

        while len(new_pop) < self.population_size:
            [p1] = self.select_parents(k=3)
            [p2] = self.select_parents(k=3)

            if random.random() < self.crossover_rate:
                child = self.crossover(p1, p2)
            else:
                child = p1.clone()

            child = self.mutate(child)
            new_pop.append(child)

        self.population = new_pop[: self.population_size]
        return self.population

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_best(self) -> Genome:
        """Return the genome with the highest fitness in the current population.

        Raises:
            ValueError: If the population is empty.
        """
        if not self.population:
            raise ValueError("Population is empty — seed or evolve first.")
        return max(self.population, key=lambda g: g.fitness)

    # ------------------------------------------------------------------
    # Full evolution loop
    # ------------------------------------------------------------------

    def run_evolution(
        self,
        model: nn.Module,
        eval_fn: Callable[[], float],
        num_generations: int = 10,
        device: str = "cpu",
    ) -> Genome:
        """Execute the complete evolution loop.

        For each generation the population is evaluated, the best fitness is
        logged, and a new generation is bred.

        Args:
            model:           Model containing a spiking layer.
            eval_fn:         Zero-argument callable returning fitness ``float``.
            num_generations: How many generations to run.
            device:          Device string for tensor operations.

        Returns:
            The best ``Genome`` found across all generations.
        """
        if not self.population:
            raise ValueError(
                "Population not initialised. Call seed_from_geometry() first."
            )

        best_ever: Optional[Genome] = None

        for gen in range(num_generations):
            self.evaluate_population(model, eval_fn, device=device)

            current_best = self.get_best()
            print(
                f"[ConnectomeEvolution] gen {gen:>3d} | "
                f"best fitness = {current_best.fitness:.4f} | "
                f"mean fitness = {self._mean_fitness():.4f}"
            )

            if best_ever is None or current_best.fitness > best_ever.fitness:
                best_ever = current_best.clone()

            self.evolve_generation()

        # Final evaluation of last generation
        self.evaluate_population(model, eval_fn, device=device)
        final_best = self.get_best()
        if final_best.fitness > (best_ever.fitness if best_ever else -math.inf):
            best_ever = final_best.clone()

        print(
            f"[ConnectomeEvolution] evolution complete | "
            f"best fitness = {best_ever.fitness:.4f}"
        )
        return best_ever

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_symmetry(self, adj: torch.Tensor) -> torch.Tensor:
        """Make adjacency symmetric (undirected graph)."""
        upper = torch.triu(adj, diagonal=1)
        return upper + upper.T

    def _enforce_sparsity(
        self, adj: torch.Tensor, max_density: float = 0.30
    ) -> torch.Tensor:
        """Prune random edges to keep density ≤ *max_density*."""
        N = adj.size(0)
        max_edges = int(max_density * N * (N - 1) / 2)
        # Count current upper-triangle edges
        upper = torch.triu(adj, diagonal=1)
        edge_indices = upper.nonzero(as_tuple=False)
        if len(edge_indices) > max_edges:
            keep = torch.randperm(len(edge_indices))[:max_edges]
            pruned = torch.zeros_like(upper)
            for idx in keep:
                i, j = edge_indices[idx]
                pruned[i, j] = 1.0
            adj = pruned + pruned.T
        return adj

    def _enforce_ei_ratio(self, ei: torch.Tensor) -> torch.Tensor:
        """Clamp E/I labels so that ~``ei_ratio`` fraction are excitatory."""
        N = self.num_neurons
        target_exc = int(round(self.ei_ratio * N))
        target_inh = N - target_exc

        # Sort by current value; highest become excitatory
        _, order = ei.abs().sort(descending=True)
        # Just reassign to hit target counts
        new_ei = torch.ones(N, dtype=ei.dtype)
        # Pick the first target_inh indices in random order to be inhibitory
        inh_candidates = order[torch.randperm(N)][:target_inh]
        new_ei[inh_candidates] = -1.0
        return new_ei

    def _random_ei_labels(self) -> torch.Tensor:
        """Generate random E/I labels respecting the target ratio."""
        N = self.num_neurons
        num_exc = int(round(self.ei_ratio * N))
        labels = torch.ones(N)
        inh_idx = torch.randperm(N)[: N - num_exc]
        labels[inh_idx] = -1.0
        return labels

    def _mean_fitness(self) -> float:
        if not self.population:
            return 0.0
        return sum(g.fitness for g in self.population) / len(self.population)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 64

    # --- Test Genome creation and cloning ---
    g = Genome(
        adjacency_mask=torch.zeros(N, N),
        weight_scales=torch.ones(N, N),
        ei_labels=torch.ones(N),
    )
    g2 = g.clone()
    g2.adjacency_mask[0, 1] = 1.0
    assert g.adjacency_mask[0, 1] == 0.0, "Clone should be independent"

    # --- Test ConnectomeEvolution basics ---
    evo = ConnectomeEvolution(num_neurons=N, population_size=10)

    # Dummy geometry
    pts = torch.randn(N, 3)
    diff = pts.unsqueeze(0) - pts.unsqueeze(1)
    dists = torch.norm(diff, dim=-1)
    adj = ((dists < dists.median()) & ~torch.eye(N, dtype=torch.bool)).float()

    evo.seed_from_geometry(adj, dists)
    assert len(evo.population) == 10, "Population size mismatch"

    # Check symmetry and sparsity
    for genome in evo.population:
        a = genome.adjacency_mask
        assert torch.allclose(a, a.T), "Adjacency must be symmetric"
        assert a.diagonal().sum() == 0, "No self-connections"
        density = a.sum() / (N * (N - 1))
        assert density <= 0.31, f"Too dense: {density:.2f}"
        exc_frac = (genome.ei_labels == 1).float().mean().item()
        assert 0.6 < exc_frac < 1.0, f"Bad E/I ratio: {exc_frac:.2f}"

    # --- Test mutation ---
    original = evo.population[0].clone()
    mutated = evo.mutate(original, rate=0.1)
    assert mutated is not original, "Mutate should return new genome"
    assert torch.allclose(mutated.adjacency_mask, mutated.adjacency_mask.T)

    # --- Test crossover ---
    child = evo.crossover(evo.population[0], evo.population[1])
    assert child.adjacency_mask.shape == (N, N)
    assert torch.allclose(child.adjacency_mask, child.adjacency_mask.T)

    # --- Test evaluation + evolution with a dummy model & fitness fn ---
    class _DummySpiking(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.synapse_weights = nn.Parameter(torch.randn(N, N) * 0.1)

    dummy_model = _DummySpiking()
    call_counter = [0]

    def _dummy_eval() -> float:
        call_counter[0] += 1
        # Fitness = negative mean abs weight (prefers smaller weights)
        return -dummy_model.synapse_weights.data.abs().mean().item()

    evo.evaluate_population(dummy_model, _dummy_eval)
    assert call_counter[0] == 10, f"Expected 10 evals, got {call_counter[0]}"
    assert all(g.fitness != 0.0 for g in evo.population)

    # --- Test select_parents ---
    [winner] = evo.select_parents(k=3)
    assert isinstance(winner, Genome)

    # --- Test evolve_generation ---
    new_pop = evo.evolve_generation()
    assert len(new_pop) == 10

    # --- Test get_best ---
    best = evo.get_best()
    assert isinstance(best, Genome)

    # --- Test run_evolution ---
    evo2 = ConnectomeEvolution(num_neurons=N, population_size=6)
    evo2.seed_from_geometry(adj, dists)

    dummy2 = _DummySpiking()

    def _fitness2() -> float:
        return random.random()

    result = evo2.run_evolution(dummy2, _fitness2, num_generations=3)
    assert isinstance(result, Genome)
    assert result.fitness > 0.0

    print("ConnectomeEvolution self-test passed!")
