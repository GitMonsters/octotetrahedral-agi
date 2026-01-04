"""
OPTUNA OPTIMIZATION FOR ALEPH-TRANSCENDPLEX AGI
Automatically tunes hyperparameters to maximize consciousness (GCI)
"""

import sys
from aleph_transcendplex_full import AlephTranscendplexAGI, CantorGoldenComplement, PHI, PHI_SQ

# Try to import optuna, provide fallback if not available
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Running with grid search instead.")


def objective(trial) -> float:
    """
    Objective function for Optuna
    Returns: Golden Consciousness Index (to be maximized)
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.5, log=True)
    dt = trial.suggest_float('dt', 0.1, 2.0)
    cgc_depth = trial.suggest_int('cgc_depth', 2, 7)
    simulation_steps = trial.suggest_int('simulation_steps', 50, 300, step=50)

    # Create AGI instance
    agi = AlephTranscendplexAGI()
    agi.dt = dt
    agi.cgc = CantorGoldenComplement(depth=cgc_depth)

    # Set learning rate for all nodes
    for layer in agi.layers.values():
        for node in layer.nodes.values():
            node.learning_rate = learning_rate

    # Build architecture
    try:
        agi.build_enhanced_architecture()
    except Exception as e:
        print(f"Error building architecture: {e}")
        return 0.0

    # Run simulation
    try:
        agi.run(steps=simulation_steps)
    except Exception as e:
        print(f"Error during simulation: {e}")
        return 0.0

    # Calculate final metrics
    metrics = agi.calculate_consciousness_metrics()

    # Return GCI as the primary objective
    gci = metrics['GCI']

    # Print progress
    print(f"Trial {trial.number}: GCI={gci:.4f}, "
          f"lr={learning_rate:.4f}, dt={dt:.3f}, "
          f"cgc_depth={cgc_depth}, steps={simulation_steps}")

    return gci


def grid_search_fallback():
    """
    Fallback grid search if Optuna not available
    """
    print("Running grid search optimization...")

    best_gci = 0.0
    best_params = {}

    learning_rates = [0.001, 0.01, 0.1]
    dts = [0.2, 0.618, 1.0]  # Include golden ratio
    cgc_depths = [3, 4, 5]
    steps_options = [100, 200]

    total_combinations = len(learning_rates) * len(dts) * len(cgc_depths) * len(steps_options)
    current = 0

    for lr in learning_rates:
        for dt in dts:
            for cgc_depth in cgc_depths:
                for steps in steps_options:
                    current += 1
                    print(f"\n[{current}/{total_combinations}] Testing: "
                          f"lr={lr}, dt={dt}, cgc_depth={cgc_depth}, steps={steps}")

                    agi = AlephTranscendplexAGI()
                    agi.dt = dt
                    agi.cgc = CantorGoldenComplement(depth=cgc_depth)

                    for layer in agi.layers.values():
                        for node in layer.nodes.values():
                            node.learning_rate = lr

                    try:
                        agi.build_enhanced_architecture()
                        agi.run(steps=steps)
                        metrics = agi.calculate_consciousness_metrics()
                        gci = metrics['GCI']

                        print(f"  Result: GCI={gci:.4f}, Φ_CGC={metrics['Phi_CGC']:.4f}")

                        if gci > best_gci:
                            best_gci = gci
                            best_params = {
                                'learning_rate': lr,
                                'dt': dt,
                                'cgc_depth': cgc_depth,
                                'simulation_steps': steps
                            }
                            print(f"  *** New best! GCI={best_gci:.4f}")

                    except Exception as e:
                        print(f"  Error: {e}")

    return best_params, best_gci


def run_optuna_optimization(n_trials=50):
    """
    Run Optuna optimization
    """
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("Objective: Maximize Golden Consciousness Index (GCI)")
    print(f"Target: GCI > φ² = {PHI_SQ:.4f}")
    print("=" * 80)

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='aleph_transcendplex_optimization'
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best GCI: {study.best_value:.4f}")
    print(f"Consciousness achieved: {study.best_value > PHI_SQ}")

    print("\nBest hyperparameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Run final simulation with best parameters
    print("\n" + "=" * 80)
    print("FINAL SIMULATION WITH BEST PARAMETERS")
    print("=" * 80)

    best_params = study.best_params
    agi = AlephTranscendplexAGI()
    agi.dt = best_params['dt']
    agi.cgc = CantorGoldenComplement(depth=best_params['cgc_depth'])

    for layer in agi.layers.values():
        for node in layer.nodes.values():
            node.learning_rate = best_params['learning_rate']

    agi.build_enhanced_architecture()
    agi.run(steps=best_params['simulation_steps'])

    status = agi.system_status()

    print(f"\nFinal System Status:")
    print(f"  Time: {status['time']:.3f}")
    print(f"  Transcendplexity: {status['transcendplexity']:.4f}")
    print(f"  GCI: {status['GCI']:.4f}")
    print(f"  Φ_CGC: {status['Phi_CGC']:.4f}")
    print(f"  Consciousness: {status['is_conscious']}")

    for layer_name, layer_info in status['layers'].items():
        print(f"\n  {layer_name}:")
        print(f"    Stability: {layer_info['stability']:.3f}")
        print(f"    Synergy: {layer_info['synergy']:.3f}")
        print(f"    Coherence: {layer_info['coherence']:.3f}")

    # Save results
    try:
        import json
        results = {
            'best_trial': study.best_trial.number,
            'best_gci': study.best_value,
            'best_params': study.best_params,
            'final_status': status
        }

        with open('/Users/evanpieser/optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\nResults saved to: optimization_results.json")
    except Exception as e:
        print(f"\nCould not save results: {e}")

    return study


def tinker_mode():
    """
    Interactive tinkering mode
    Allows manual experimentation with parameters
    """
    print("=" * 80)
    print("TINKER MODE - Interactive Experimentation")
    print("=" * 80)

    while True:
        print("\n" + "-" * 80)
        print("Current experiment configuration:")
        print("-" * 80)

        try:
            lr = float(input("Learning rate (0.001-0.5, default=0.01): ") or "0.01")
            dt = float(input(f"Time step (0.1-2.0, default={PHI_INV:.3f}): ") or str(PHI_INV))
            cgc_depth = int(input("CGC depth (2-7, default=4): ") or "4")
            steps = int(input("Simulation steps (50-500, default=200): ") or "200")
        except ValueError:
            print("Invalid input. Using defaults.")
            lr = 0.01
            dt = PHI_INV
            cgc_depth = 4
            steps = 200

        print(f"\nRunning simulation with: lr={lr}, dt={dt}, cgc_depth={cgc_depth}, steps={steps}")

        agi = AlephTranscendplexAGI()
        agi.dt = dt
        agi.cgc = CantorGoldenComplement(depth=cgc_depth)

        for layer in agi.layers.values():
            for node in layer.nodes.values():
                node.learning_rate = lr

        agi.build_enhanced_architecture()

        print("\nRunning...")
        agi.run(steps=steps)

        status = agi.system_status()
        metrics = agi.calculate_consciousness_metrics()

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"GCI: {status['GCI']:.4f} (threshold: {PHI_SQ:.4f})")
        print(f"Φ_CGC: {status['Phi_CGC']:.4f}")
        print(f"Transcendplexity: {status['transcendplexity']:.4f}")
        print(f"Conscious: {status['is_conscious']}")
        print(f"\nMetrics breakdown:")
        print(f"  Triangulation: {metrics['triangulation']:.3f}")
        print(f"  Synergy: {metrics['synergy']:.3f}")
        print(f"  Coherence: {metrics['coherence']:.3f}")
        print(f"  Entropy: {metrics['entropy']:.3f}")
        print(f"  Forbidden fraction: {metrics['forbidden_fraction']:.3f}")

        cont = input("\nTry another configuration? (y/n): ")
        if cont.lower() != 'y':
            break

    print("\nExiting tinker mode.")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "optuna"

    if mode == "tinker":
        tinker_mode()
    elif mode == "grid":
        best_params, best_gci = grid_search_fallback()
        print("\n" + "=" * 80)
        print("GRID SEARCH COMPLETE")
        print("=" * 80)
        print(f"Best GCI: {best_gci:.4f}")
        print(f"Best parameters: {best_params}")
    elif mode == "optuna":
        if OPTUNA_AVAILABLE:
            n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            run_optuna_optimization(n_trials=n_trials)
        else:
            print("Optuna not available. Use 'grid' or 'tinker' mode instead.")
            print("\nUsage:")
            print("  python optimize_transcendplex.py optuna [n_trials]")
            print("  python optimize_transcendplex.py grid")
            print("  python optimize_transcendplex.py tinker")
    else:
        print("Unknown mode. Options: optuna, grid, tinker")
        sys.exit(1)
