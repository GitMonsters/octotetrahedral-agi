#!/usr/bin/env python3
"""
ARC-AGI-3 Runner — Play all available games with the OctoTetra Agent.

Usage:
    python3 arc3/run.py                     # Run all games
    python3 arc3/run.py --game ls20         # Run specific game
    python3 arc3/run.py --game ls20 --verbose  # With debug logging
    python3 arc3/run.py --offline           # Local only (no API)
"""

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc_agi
from arcengine import GameAction, GameState

from arc3.agent import OctoTetraAgent


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy SDK logs unless verbose
    if not verbose:
        logging.getLogger("arc_agi").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_game(arc: arc_agi.Arcade, game_id: str, agent,
             verbose: bool) -> dict:
    """Run a single game and return results."""
    logger = logging.getLogger("arc3.runner")
    logger.info(f"{'='*60}")
    logger.info(f"Playing: {game_id}")
    logger.info(f"{'='*60}")

    env = arc.make(game_id)
    if env is None:
        logger.error(f"Failed to create environment for {game_id}")
        return {'game_id': game_id, 'error': 'Failed to create environment'}

    agent.reset()
    if hasattr(agent, 'play_game'):
        result = agent.play_game(env, GameAction)
    else:
        result = agent.play(env, GameAction)
    result['game_id'] = game_id

    logger.info(f"Result: {result['levels_completed']}/{result['win_levels']} levels, "
               f"{result['total_actions']} actions, "
               f"{'WON' if result['won'] else 'not won'}, "
               f"{result['elapsed_seconds']}s")

    if verbose:
        logger.info(f"World model: {json.dumps(result['world_model'], indent=2)}")
        logger.info(f"Memory stats: {json.dumps(result['memory_stats'], indent=2)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI-3 OctoTetra Agent Runner")
    parser.add_argument("--game", type=str, default=None,
                       help="Specific game ID to play (e.g., ls20)")
    parser.add_argument("--max-actions", type=int, default=500,
                       help="Max actions per level (default: 500)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose/debug logging")
    parser.add_argument("--offline", action="store_true",
                       help="Run in offline mode (local games only)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--mercury-key", type=str, default=None,
                       help="Mercury API key (or set MERCURY_API_KEY env var)")
    parser.add_argument("--no-mercury", action="store_true",
                       help="Disable Mercury reasoning backend")
    parser.add_argument("--computer-use", action="store_true",
                       help="Use Computer Use agent (visual reasoning loop)")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("arc3.runner")

    # Initialize arcade
    if args.offline:
        from arc_agi import OperationMode
        arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
    else:
        arc = arc_agi.Arcade()

    # Get available games
    games = arc.get_environments()
    logger.info(f"Available games: {len(games)}")

    if args.game:
        # Filter to specific game
        matching = [g for g in games if args.game.lower() in g.game_id.lower()]
        if not matching:
            logger.error(f"Game '{args.game}' not found. Available: "
                       f"{[g.game_id for g in games]}")
            sys.exit(1)
        games = matching

    # Set Mercury API key if provided
    if args.mercury_key:
        os.environ["MERCURY_API_KEY"] = args.mercury_key

    # Create agent
    if args.computer_use:
        from arc3.computer_use import ComputerUseAgent
        agent = ComputerUseAgent(
            max_actions=args.max_actions,
            verbose=args.verbose,
            use_llm=not args.no_mercury,
        )
    else:
        agent = OctoTetraAgent(
            max_actions_per_level=args.max_actions,
            verbose=args.verbose,
            use_mercury=not args.no_mercury,
        )

    # Run games
    all_results = []
    total_levels = 0
    total_completed = 0
    total_actions = 0

    for game_info in games:
        result = run_game(arc, game_info.game_id, agent, args.verbose)
        all_results.append(result)

        if 'error' not in result:
            total_levels += result['win_levels']
            total_completed += result['levels_completed']
            total_actions += result['total_actions']

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Games played: {len(all_results)}")
    logger.info(f"Total levels: {total_completed}/{total_levels}")
    logger.info(f"Total actions: {total_actions}")

    for r in all_results:
        status = "WON" if r.get('won') else f"{r.get('levels_completed', 0)}/{r.get('win_levels', '?')}"
        logger.info(f"  {r['game_id']}: {status} ({r.get('total_actions', '?')} actions)")

    # Print scorecard
    scorecard = arc.get_scorecard()
    if scorecard:
        logger.info(f"\nScorecard: {scorecard}")

    # Save results
    if args.output:
        output_data = {
            'summary': {
                'games_played': len(all_results),
                'total_levels': total_levels,
                'levels_completed': total_completed,
                'total_actions': total_actions,
            },
            'results': all_results,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
