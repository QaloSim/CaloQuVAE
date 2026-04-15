"""
Genetic algorithm for orbit (qubit-to-pixel permutation) search over multiple incidence energies.

Fitness = mean Frobenius norm of (QPU latent correlation matrix - classical RBM baseline),
          averaged uniformly over incidence_energies.

Each individual encodes a pair of permutations (vis_mapping, hid_mapping) stored as tuples
so individuals are hashable and a fitness cache avoids re-evaluating identical genomes.

Usage
-----
from utils.dwave.ga import run_ga_permutation_sweep_multi_energy

results = run_ga_permutation_sweep_multi_energy(
    energy_patterns_dict=my_dict,
    rbm=rbm, raw_sampler=sampler,
    conditioning_sets=csets,
    left_chains=lchains, right_chains=rchains,
    hidden_side="right",
    n_cond=53, beta=3.0,
)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Callable, Optional

try:
    from .physics import get_latent_correlation
    from .workflows import sample_expanded_flux_arbitrary
    from .postprocessing import process_analysis_result
except ImportError:
    # Stubs used only when the file is run directly (python ga.py) for self-tests.
    # The real functions are never called by the test helpers below.
    get_latent_correlation = None       # type: ignore[assignment]
    sample_expanded_flux_arbitrary = None  # type: ignore[assignment]
    process_analysis_result = None      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One candidate orbit: a pair of permutation tuples (vis, hid)."""
    vis: tuple          # length n_vis  — ints, tuple for hashability
    hid: tuple          # length n_hid
    fitness: Optional[float] = None      # mean Frobenius error; None until evaluated
    per_energy: Optional[dict] = None   # {energy: frobenius_error}
    aux: dict = field(default_factory=dict)

    def cache_key(self) -> tuple:
        return (self.vis, self.hid)

    def clone(self) -> "Individual":
        return Individual(
            vis=self.vis,
            hid=self.hid,
            fitness=self.fitness,
            per_energy=dict(self.per_energy) if self.per_energy else None,
            aux=dict(self.aux),
        )

    def unset_fitness(self) -> "Individual":
        """Return a copy with fitness cleared (forces re-evaluation / cache lookup)."""
        c = self.clone()
        c.fitness = None
        c.per_energy = None
        c.aux.pop("cache_hit", None)
        return c

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.vis == other.vis and self.hid == other.hid

    def __hash__(self) -> int:
        return hash(self.cache_key())


# ---------------------------------------------------------------------------
# Permutation crossover operators
# ---------------------------------------------------------------------------

def _pmx(p1: tuple, p2: tuple, rng: np.random.Generator) -> tuple[tuple, tuple]:
    """Partially Matched Crossover — preserves absolute positions of the slice."""
    n = len(p1)
    pts = sorted(rng.integers(0, n + 1, size=2).tolist())
    a, b = pts[0], pts[1]

    def _apply(pa: list, pb: list) -> tuple:
        child = list(pb)
        child[a:b] = pa[a:b]
        pos2val_b = {v: i for i, v in enumerate(pb)}
        for i in range(a, b):
            val = pb[i]
            if val not in pa[a:b]:   # val displaced from child; trace chain to find its new home
                pos = i
                while a <= pos < b:
                    pos = pos2val_b[pa[pos]]
                child[pos] = val
        return tuple(child)

    p1l, p2l = list(p1), list(p2)
    return _apply(p1l, p2l), _apply(p2l, p1l)


def _ox(p1: tuple, p2: tuple, rng: np.random.Generator) -> tuple[tuple, tuple]:
    """Order Crossover — preserves relative order of the non-slice elements."""
    n = len(p1)
    pts = sorted(rng.integers(0, n + 1, size=2).tolist())
    a, b = pts[0], pts[1]

    def _apply(pa: tuple, pb: tuple) -> tuple:
        slice_set = set(pa[a:b])
        remaining = [x for x in pb if x not in slice_set]
        return tuple(remaining[:a] + list(pa[a:b]) + remaining[a:])

    return _apply(p1, p2), _apply(p2, p1)


def _cx(p1: tuple, p2: tuple, rng: np.random.Generator) -> tuple[tuple, tuple]:
    """Cycle Crossover — preserves absolute positions of all elements."""
    n = len(p1)
    pos2idx_p2 = {v: i for i, v in enumerate(p2)}

    # Build the cycle mask starting from position 0
    in_cycle = [False] * n
    pos = 0
    while not in_cycle[pos]:
        in_cycle[pos] = True
        pos = pos2idx_p2[p1[pos]]

    child1 = tuple(p1[i] if in_cycle[i] else p2[i] for i in range(n))
    child2 = tuple(p2[i] if in_cycle[i] else p1[i] for i in range(n))
    return child1, child2


# ---------------------------------------------------------------------------
# Permutation mutation operators
# ---------------------------------------------------------------------------

def _swap(p: tuple, rng: np.random.Generator) -> tuple:
    """Exchange two random positions."""
    lst = list(p)
    i, j = rng.integers(0, len(lst), size=2).tolist()
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


def _inversion(p: tuple, rng: np.random.Generator) -> tuple:
    """Reverse a random sub-sequence (2-opt style)."""
    lst = list(p)
    pts = sorted(rng.integers(0, len(lst), size=2).tolist())
    a, b = pts[0], pts[1]
    lst[a:b + 1] = lst[a:b + 1][::-1]
    return tuple(lst)


def _insertion(p: tuple, rng: np.random.Generator) -> tuple:
    """Remove one element and reinsert it at a random position."""
    lst = list(p)
    n = len(lst)
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n - 1))
    if j >= i:
        j += 1
    val = lst.pop(i)
    lst.insert(j, val)
    return tuple(lst)


_CROSSOVER_OPS: dict[str, Callable] = {"pmx": _pmx, "ox": _ox, "cx": _cx}
_MUTATION_OPS: dict[str, Callable] = {"swap": _swap, "inversion": _inversion, "insertion": _insertion}


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------

def _init_population(
    population_size: int,
    n_vis: int,
    n_hid: int,
    rng: np.random.Generator,
    seed_identity: bool = True,
) -> list[Individual]:
    pop: list[Individual] = []
    if seed_identity:
        pop.append(Individual(
            vis=tuple(range(n_vis)),
            hid=tuple(range(n_hid)),
            aux={"type": "IDENTITY", "generation": 0, "op": "init"},
        ))
    while len(pop) < population_size:
        pop.append(Individual(
            vis=tuple(rng.permutation(n_vis).tolist()),
            hid=tuple(rng.permutation(n_hid).tolist()),
            aux={"type": "RANDOM", "generation": 0, "op": "init"},
        ))
    return pop[:population_size]


def _tournament(pop: list[Individual], k: int, rng: np.random.Generator) -> Individual:
    """Tournament selection — minimizes fitness. Draws k contestants without replacement."""
    k = min(k, len(pop))
    idxs = rng.choice(len(pop), size=k, replace=False).tolist()
    best_idx = min(idxs, key=lambda i: pop[i].fitness)
    return pop[best_idx]


# ---------------------------------------------------------------------------
# Classical baseline precomputation
# ---------------------------------------------------------------------------

def precompute_classical_baselines(
    incidence_energies: list,
    energy_patterns_dict: dict,
    rbm,
    n_cond: int,
    rbm_baseline_samples: int = 10000,
    rbm_gibbs_steps: int = 2000,
) -> dict:
    """
    Runs Gibbs sampling for each energy to obtain the classical RBM latent
    correlation matrix used as the fitness reference.

    Returns
    -------
    dict: {energy: np.ndarray of shape (n_latent, n_latent)}
    """
    baselines = {}
    for energy in incidence_energies:
        print(f"  Classical baseline for energy={energy}...")
        primary_pattern = energy_patterns_dict[energy][0].unsqueeze(0)
        cond_vec = primary_pattern.repeat(rbm_baseline_samples, 1)
        v_cl = rbm.sample_v_given_v_clamped(
            clamped_v=cond_vec,
            n_clamped=n_cond,
            gibbs_steps=rbm_gibbs_steps,
            beta=1.0,
        )
        baselines[energy] = get_latent_correlation(v_cl.cpu(), n_cond)
    return baselines


# ---------------------------------------------------------------------------
# Individual evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(
    ind: Individual,
    incidence_energies: list,
    target_batches: dict,
    classical_matrices: dict,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    beta: float,
    num_reads_per_perm: int,
    srt_batches: int,
    base_shims,
    fitness_cache: dict,
) -> Individual:
    """
    Evaluates *ind* against all incidence energies, writing fitness and aux fields.

    Checks *fitness_cache* first; skips the QPU entirely on a cache hit.
    The fitness_cache key is ``ind.cache_key() == (ind.vis, ind.hid)``.

    Modifies *ind* in place and returns it.
    """
    key = ind.cache_key()
    if key in fitness_cache:
        cached = fitness_cache[key]
        ind.fitness = cached["fitness"]
        ind.per_energy = cached["per_energy"]
        ind.aux["chain_break_frac"] = cached["chain_break_frac"]
        ind.aux["matrices"] = cached["matrices"]
        ind.aux["cache_hit"] = True
        return ind

    vis_list = list(ind.vis)
    hid_list = list(ind.hid)

    # Derive the actual reads-per-batch from target_batches shape (the reads_per_batch
    # argument to this function is informational only — the true budget is encoded in
    # the pre-built target_batches tensor passed in via _eval_kwargs).
    _sample_energy = incidence_energies[0]
    actual_reads_per_batch = target_batches[_sample_energy].shape[0]
    total_reads_per_energy = actual_reads_per_batch * srt_batches
    expected_reads_per_batch = num_reads_per_perm // srt_batches
    if actual_reads_per_batch != expected_reads_per_batch:
        print(
            f"  [WARNING] reads mismatch: target_batches has {actual_reads_per_batch} rows "
            f"but num_reads_per_perm//srt_batches={expected_reads_per_batch}. "
            f"Actual total reads per energy = {total_reads_per_energy}."
        )
    print(
        f"  [QPU] reads_per_batch={actual_reads_per_batch}, srt_batches={srt_batches}, "
        f"total_reads_per_energy={total_reads_per_energy}, "
        f"n_energies={len(incidence_energies)}, "
        f"total_reads_this_individual={total_reads_per_energy * len(incidence_energies)}"
    )

    per_energy_errors: dict = {}
    per_energy_matrices: dict = {}
    all_break_fracs: list = []

    for energy in incidence_energies:
        batch_samples: list = []
        batch_breaks: list = []

        for b in range(srt_batches):
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=target_batches[energy],
                hidden_side=hidden_side,
                beta=beta,
                source=f"GA_E{energy}_b{b}",
                use_srt=True,
                logical_srt=True,
                chain_strength=2.0,
                flux_drift_compensation=True,
                additive_flux_offsets=base_shims,
                vis_mapping=vis_list,
                hid_mapping=hid_list,
                perm_seed=None,
            )
            v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
            batch_samples.append(v_sample.cpu())
            if res.break_matrix is not None:
                batch_breaks.append(res.break_matrix)

        full_samples = torch.cat(batch_samples, dim=0)
        qpu_corr = get_latent_correlation(full_samples, n_cond)
        per_energy_errors[energy] = float(np.linalg.norm(qpu_corr - classical_matrices[energy]))
        per_energy_matrices[energy] = qpu_corr

        if batch_breaks:
            all_break_fracs.append(float(np.mean(np.vstack(batch_breaks))))

    ind.fitness = float(np.mean(list(per_energy_errors.values())))
    ind.per_energy = per_energy_errors
    chain_break_frac = float(np.mean(all_break_fracs)) if all_break_fracs else 0.0
    ind.aux["chain_break_frac"] = chain_break_frac
    ind.aux["matrices"] = per_energy_matrices
    ind.aux["cache_hit"] = False

    fitness_cache[key] = {
        "fitness": ind.fitness,
        "per_energy": ind.per_energy,
        "chain_break_frac": chain_break_frac,
        "matrices": per_energy_matrices,
    }
    return ind


# ---------------------------------------------------------------------------
# Main GA
# ---------------------------------------------------------------------------

def run_ga_permutation_sweep_multi_energy(
    energy_patterns_dict: dict,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    beta: float = 3.0,
    # GA config
    population_size: int = 16,
    n_generations: int = 10,
    elite_count: int = 2,
    tournament_size: int = 3,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.3,
    crossover_op: str = "pmx",
    mutation_op: str = "inversion",
    seed_identity_in_init: bool = True,
    # QPU eval config
    num_reads_per_perm: int = 512,
    srt_batches: int = 8,
    rbm_baseline_samples: int = 10000,
    rbm_gibbs_steps: int = 2000,
    # infra
    base_shims=None,
    ga_seed: Optional[int] = None,
) -> dict:
    """
    Genetic algorithm over orbit permutations, scored across multiple incidence energies.

    Parameters
    ----------
    energy_patterns_dict : dict
        Maps each energy value to a tensor of binary conditioning patterns
        (shape ``(n_patterns, n_cond)``).  The primary pattern
        ``energy_patterns_dict[e][0]`` is used for sampling.
    rbm, raw_sampler, conditioning_sets, left_chains, right_chains, hidden_side
        Same as ``run_monte_carlo_permutation_sweep``.
    n_cond : int
        Number of clamped (conditioning) visible units.
    beta : float
        Inverse temperature passed to the QPU sampler.
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Total number of generations (includes generation 0).
    elite_count : int
        Number of best individuals copied unchanged to the next generation.
    tournament_size : int
        Number of contestants drawn per tournament selection.
    crossover_prob : float
        Probability of applying crossover to a pair of parents.
    mutation_prob : float
        Per-individual probability of applying mutation after crossover.
    crossover_op : str
        One of ``"pmx"`` (Partially Matched), ``"ox"`` (Order), ``"cx"`` (Cycle).
    mutation_op : str
        One of ``"inversion"`` (2-opt), ``"swap"``, ``"insertion"``.
    seed_identity_in_init : bool
        If True, the identity permutation is always included in generation 0.
    num_reads_per_perm : int
        Total QPU reads per individual per energy (split across srt_batches).
    srt_batches : int
        Number of SRT batches to accumulate per individual evaluation.
    rbm_baseline_samples : int
        Number of Gibbs samples used to build the classical reference matrices.
    rbm_gibbs_steps : int
        Gibbs chain length for the classical baseline.
    base_shims : optional
        Additive flux offsets forwarded to ``sample_expanded_flux_arbitrary``.
    ga_seed : int or None
        Seed for the GA's own RNG (controls all operator stochasticity).

    Returns
    -------
    dict with keys:
        ``classical_matrices``  — ``{energy: np.ndarray}``
        ``perm_metrics``        — list of evaluation records for all individuals
                                  across all generations (``error_norm`` alias for fitness,
                                  compatible with ``plot_permutation_sweep_analysis``)
        ``default_orbit``       — record for the identity permutation (gen 0, ind 0)
        ``best_orbit``          — globally best record
        ``worst_orbit``         — globally worst record seen
        ``ga_history``          — list of ``{generation, best, mean, median,
                                  cache_hits, cache_size}`` dicts
        ``config``              — full GA configuration snapshot
    """
    incidence_energies = list(energy_patterns_dict.keys())

    if crossover_op not in _CROSSOVER_OPS:
        raise ValueError(f"crossover_op must be one of {list(_CROSSOVER_OPS)}, got '{crossover_op}'")
    if mutation_op not in _MUTATION_OPS:
        raise ValueError(f"mutation_op must be one of {list(_MUTATION_OPS)}, got '{mutation_op}'")

    xover_fn = _CROSSOVER_OPS[crossover_op]
    mut_fn = _MUTATION_OPS[mutation_op]
    rng = np.random.default_rng(ga_seed)
    fitness_cache: dict = {}

    if hidden_side == "right":
        n_vis, n_hid = len(left_chains), len(right_chains)
    else:
        n_vis, n_hid = len(right_chains), len(left_chains)

    config = dict(
        incidence_energies=incidence_energies, n_vis=n_vis, n_hid=n_hid,
        population_size=population_size, n_generations=n_generations,
        elite_count=elite_count, tournament_size=tournament_size,
        crossover_prob=crossover_prob, mutation_prob=mutation_prob,
        crossover_op=crossover_op, mutation_op=mutation_op,
        num_reads_per_perm=num_reads_per_perm, srt_batches=srt_batches,
        rbm_baseline_samples=rbm_baseline_samples, rbm_gibbs_steps=rbm_gibbs_steps,
        ga_seed=ga_seed, beta=beta, n_cond=n_cond,
    )

    reads_per_batch = num_reads_per_perm // srt_batches
    target_batches = {
        e: energy_patterns_dict[e][0].unsqueeze(0).repeat(reads_per_batch, 1)
        for e in incidence_energies
    }

    print("=" * 60)
    print(f"GA Orbit Search | pop={population_size}, gens={n_generations}, seed={ga_seed}")
    print(f"Energies: {incidence_energies} | xover={crossover_op}, mut={mutation_op}")
    print("=" * 60)
    print("\nPrecomputing classical baselines...")
    classical_matrices = precompute_classical_baselines(
        incidence_energies, energy_patterns_dict, rbm, n_cond,
        rbm_baseline_samples=rbm_baseline_samples,
        rbm_gibbs_steps=rbm_gibbs_steps,
    )

    _eval_kwargs = dict(
        incidence_energies=incidence_energies,
        target_batches=target_batches,
        classical_matrices=classical_matrices,
        rbm=rbm, raw_sampler=raw_sampler,
        conditioning_sets=conditioning_sets,
        left_chains=left_chains, right_chains=right_chains,
        hidden_side=hidden_side,
        n_cond=n_cond, beta=beta,
        num_reads_per_perm=num_reads_per_perm, srt_batches=srt_batches,
        base_shims=base_shims, fitness_cache=fitness_cache,
    )

    def _eval_population(pop: list[Individual], gen: int) -> int:
        """Evaluate all unevaluated individuals; return number of cache hits."""
        hits = 0
        for k, ind in enumerate(pop):
            if ind.fitness is not None:
                # Elite carried over with known fitness — counts as a hit
                ind.aux["cache_hit"] = True
                hits += 1
                continue
            tag = f"[gen {gen}, {k+1}/{len(pop)}]"
            print(f"  {tag} evaluating orbit...")
            evaluate_individual(ind, **_eval_kwargs)
            if ind.aux.get("cache_hit"):
                hits += 1
            print(f"    fitness={ind.fitness:.4f}  cbf={ind.aux.get('chain_break_frac', 0):.2%}"
                  f"  {'[cache hit]' if ind.aux.get('cache_hit') else ''}")
        return hits

    def _to_record(ind: Individual, generation: int) -> dict:
        return {
            "vis": ind.vis,
            "hid": ind.hid,
            "fitness": ind.fitness,
            "error_norm": ind.fitness,   # alias for plot_permutation_sweep_analysis
            "per_energy": ind.per_energy,
            "chain_break_frac": ind.aux.get("chain_break_frac", 0.0),
            "matrices": ind.aux.get("matrices", {}),
            "type": ind.aux.get("type", "GA"),
            "op": ind.aux.get("op", ""),
            "generation": generation,
        }

    perm_metrics: list[dict] = []
    ga_history: list[dict] = []
    default_orbit: Optional[dict] = None

    # ---- Generation 0: initial population ----
    print("\nInitializing population (generation 0)...")
    pop = _init_population(population_size, n_vis, n_hid, rng, seed_identity=seed_identity_in_init)
    cache_hits = _eval_population(pop, gen=0)

    for k, ind in enumerate(pop):
        rec = _to_record(ind, generation=0)
        perm_metrics.append(rec)
        if seed_identity_in_init and k == 0:
            default_orbit = rec

    fitnesses = [ind.fitness for ind in pop]
    ga_history.append({
        "generation": 0,
        "best": float(min(fitnesses)),
        "mean": float(np.mean(fitnesses)),
        "median": float(np.median(fitnesses)),
        "cache_hits": cache_hits,
        "cache_size": len(fitness_cache),
    })
    print(f"\nGen 0 | best={ga_history[-1]['best']:.4f}  mean={ga_history[-1]['mean']:.4f}")

    # ---- Generations 1..n_generations-1 ----
    for g in range(1, n_generations):
        print(f"\n--- Generation {g}/{n_generations - 1} ---")

        # Elitism: carry best individuals unchanged (fitness already set → skipped in eval)
        sorted_pop = sorted(pop, key=lambda x: x.fitness)
        elites = []
        for ind in sorted_pop[:elite_count]:
            e = ind.clone()
            e.aux["type"] = "ELITE"
            e.aux["generation"] = g
            elites.append(e)

        # Crossover + mutation to fill the rest of the generation
        offspring: list[Individual] = []
        while len(offspring) < population_size - elite_count:
            p1 = _tournament(pop, tournament_size, rng)
            p2 = _tournament(pop, tournament_size, rng)

            if rng.random() < crossover_prob:
                vis1, vis2 = xover_fn(p1.vis, p2.vis, rng)
                hid1, hid2 = xover_fn(p1.hid, p2.hid, rng)
                c1 = Individual(vis=vis1, hid=hid1,
                                aux={"type": "GA", "generation": g, "op": crossover_op})
                c2 = Individual(vis=vis2, hid=hid2,
                                aux={"type": "GA", "generation": g, "op": crossover_op})
            else:
                c1 = p1.unset_fitness()
                c2 = p2.unset_fitness()
                c1.aux.update({"generation": g, "op": "copy"})
                c2.aux.update({"generation": g, "op": "copy"})

            for c in (c1, c2):
                if rng.random() < mutation_prob:
                    new_vis = mut_fn(c.vis, rng)
                    new_hid = mut_fn(c.hid, rng)
                    if new_vis != c.vis or new_hid != c.hid:
                        c.vis = new_vis
                        c.hid = new_hid
                        c.fitness = None   # genome changed, invalidate fitness
                        c.per_energy = None
                        c.aux["op"] = c.aux.get("op", "") + f"+{mutation_op}"

            offspring += [c1, c2]

        offspring = offspring[:population_size - elite_count]
        pop = elites + offspring
        cache_hits = _eval_population(pop, gen=g)

        for ind in pop:
            perm_metrics.append(_to_record(ind, generation=g))

        fitnesses = [ind.fitness for ind in pop]
        ga_history.append({
            "generation": g,
            "best": float(min(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "median": float(np.median(fitnesses)),
            "cache_hits": cache_hits,
            "cache_size": len(fitness_cache),
        })
        print(f"Gen {g} | best={ga_history[-1]['best']:.4f}  mean={ga_history[-1]['mean']:.4f}"
              f"  cache_hits={cache_hits}/{len(pop)}")

    # ---- Finalize ----
    sorted_metrics = sorted(perm_metrics, key=lambda x: x["error_norm"])
    best_orbit = sorted_metrics[0]
    worst_orbit = sorted_metrics[-1]

    print("\n" + "=" * 60)
    print(f"GA complete | {len(fitness_cache)} unique orbits evaluated")
    if default_orbit:
        print(f"Identity error : {default_orbit['error_norm']:.4f}")
    print(f"Best error     : {best_orbit['error_norm']:.4f}")
    print(f"Worst error    : {worst_orbit['error_norm']:.4f}")
    print("=" * 60)

    return {
        "classical_matrices": classical_matrices,
        "perm_metrics": perm_metrics,
        "default_orbit": default_orbit,
        "best_orbit": best_orbit,
        "worst_orbit": worst_orbit,
        "ga_history": ga_history,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Self-test helpers (run via: python ga.py  or  python -m utils.dwave.ga from repo root)
# ---------------------------------------------------------------------------

def _validate_permutation(p, n: int, label: str = ""):
    assert sorted(p) == list(range(n)), f"Not a valid permutation {label}: {p}"


def _run_operator_tests():
    """Quick smoke tests for all crossover and mutation operators."""
    rng = np.random.default_rng(42)
    n = 10

    for _ in range(200):
        p1 = tuple(rng.permutation(n).tolist())
        p2 = tuple(rng.permutation(n).tolist())

        for name, fn in _CROSSOVER_OPS.items():
            c1, c2 = fn(p1, p2, rng)
            _validate_permutation(c1, n, f"{name} c1")
            _validate_permutation(c2, n, f"{name} c2")
            # Identity parents → identity children
            ident = tuple(range(n))
            ci1, ci2 = fn(ident, ident, rng)
            _validate_permutation(ci1, n, f"{name} ident c1")
            assert ci1 == ident, f"{name}: identity parents should produce identity child"

        for name, fn in _MUTATION_OPS.items():
            m = fn(p1, rng)
            _validate_permutation(m, n, f"{name} mutation")
            if name == "swap":
                diffs = sum(a != b for a, b in zip(p1, m))
                assert diffs <= 2, f"swap should change at most 2 positions, changed {diffs}"

    print("All operator tests passed.")


def _run_cache_test():
    """Verify that Individual hashability and fitness_cache lookups work."""
    vis = tuple(range(10))
    hid = tuple(range(10, 20))
    a = Individual(vis=vis, hid=hid, fitness=1.23)
    b = Individual(vis=vis, hid=hid, fitness=None)
    assert hash(a) == hash(b)
    assert a == b

    cache = {}
    cache[a.cache_key()] = {"fitness": 1.23}
    assert b.cache_key() in cache
    print("Cache test passed.")


def _run_tournament_test():
    """Tournament with size=population always returns the best individual."""
    rng = np.random.default_rng(0)
    n = 8
    pop = [Individual(vis=tuple(range(n)), hid=tuple(range(n)), fitness=float(i)) for i in range(n)]
    for _ in range(50):
        winner = _tournament(pop, k=len(pop), rng=rng)
        assert winner.fitness == 0.0, f"Expected best individual, got fitness={winner.fitness}"
    print("Tournament test passed.")


if __name__ == "__main__":
    _run_operator_tests()
    _run_cache_test()
    _run_tournament_test()
    print("All self-tests passed.")
