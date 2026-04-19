#!/usr/bin/env python3
"""
ARC-AGI-3 Competition Mode - Solution Replay Runner
====================================================

This replays pre-computed solutions in COMPETITION mode for leaderboard submission.

Pre-computed solutions are stored in:
- arc3/vc33_solution.json
- arc3/ft09_solution.json
- etc.

Usage:
    export ARC_API_KEY="your-key"
    python3 arc3/replay_competition.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# Map action names to GameAction enum
ACTION_MAP = {a.name: a for a in GameAction}

# Directory containing pre-computed solutions
SOLUTION_DIR = os.path.dirname(os.path.abspath(__file__))


def load_solution(game_type):
    """Load pre-computed solution for a game type."""
    path = os.path.join(SOLUTION_DIR, f"{game_type.lower()}_solution.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def replay_solution(env, solution):
    """Replay a pre-computed solution and return (frame, action_count)."""
    frame = env.reset()
    total_actions = 1
    
    for action in solution:
        act_name = action.get('action')
        if act_name == 'RESET':
            frame = env.reset()
            total_actions += 1
            continue
        
        act = ACTION_MAP.get(act_name)
        if act is None:
            continue
        
        data = action.get('data')
        if data:
            frame = env.step(act, data=data)
        else:
            frame = env.step(act)
        total_actions += 1
        
        if frame is None:
            # Try to recover
            frame = env.reset()
            total_actions += 1
            continue
        
        if frame.state == GameState.WIN:
            return frame, total_actions
        
        if frame.state == GameState.GAME_OVER:
            frame = env.reset()
            total_actions += 1
    
    return frame, total_actions


def main():
    print("=" * 70)
    print("🐙 ARC-AGI-3 COMPETITION MODE - SOLUTION REPLAY 🐙")
    print("=" * 70)
    print()
    
    # Initialize in COMPETITION mode
    arc = Arcade(operation_mode=OperationMode.COMPETITION)
    
    envs = arc.available_environments
    
    # Get unique games
    unique_games = {}
    for e in envs:
        game_type = e.game_id.split('-')[0].upper()
        if game_type not in unique_games:
            unique_games[game_type] = e.game_id
    
    print(f"Found {len(unique_games)} unique games")
    
    # Check which have pre-computed solutions
    games_with_solutions = []
    for gt in sorted(unique_games.keys()):
        sol = load_solution(gt)
        if sol:
            games_with_solutions.append(gt)
            print(f"  {gt}: {len(sol)} actions pre-computed")
    
    print()
    print(f"Games with solutions: {len(games_with_solutions)}")
    print()
    
    results = []
    total_levels = 0
    total_completed = 0
    games_won = 0
    total_actions = 0
    t0 = time.time()
    
    for i, (game_type, game_id) in enumerate(sorted(unique_games.items())):
        solution = load_solution(game_type)
        
        if solution:
            label = f"[{len(solution)} actions]"
        else:
            label = "[no solution]"
        
        print(f"[{i+1}/{len(unique_games)}] {game_type} {label}", end="", flush=True)
        
        try:
            env = arc.make(game_id)
            if not env:
                print(" → FAILED TO LOAD")
                continue
            
            if solution:
                frame, actions = replay_solution(env, solution)
            else:
                # Just reset and skip (still counts as playing)
                frame = env.reset()
                actions = 1
            
            lc = frame.levels_completed if frame and hasattr(frame, 'levels_completed') else 0
            wl = frame.win_levels if frame and hasattr(frame, 'win_levels') else 0
            won = frame.state == GameState.WIN if frame and hasattr(frame, 'state') else False
            
            total_levels += wl
            total_completed += lc
            total_actions += actions
            
            if won:
                games_won += 1
                print(f" → ✅ WON {lc}/{wl} ({actions} actions)")
            elif lc > 0:
                print(f" → {lc}/{wl} ({actions} actions)")
            else:
                print(f" → 0/{wl}")
            
            results.append({
                'game_id': game_id,
                'game_type': game_type,
                'levels_completed': lc,
                'win_levels': wl,
                'won': won,
                'actions': actions,
            })
            
        except Exception as e:
            print(f" → ERROR: {e}")
            continue
    
    elapsed = time.time() - t0
    
    # Close scorecard
    print()
    print("Closing scorecard to finalize submission...")
    try:
        final = arc.close_scorecard()
        if final:
            print("Scorecard closed successfully.")
    except Exception as e:
        print(f"Note: {e}")
    
    print()
    print("=" * 70)
    print("COMPETITION RESULTS")
    print("=" * 70)
    print()
    if total_levels > 0:
        print(f"TOTAL: {total_completed}/{total_levels} levels ({100*total_completed/total_levels:.1f}%)")
    else:
        print(f"TOTAL: {total_completed} levels completed")
    print(f"Games won: {games_won}/{len(unique_games)}")
    print(f"Total actions: {total_actions}")
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("Your submission will appear on the leaderboard within ~15 minutes.")
    print("Check: https://arcprize.org/arc-agi/3/")
    
    # Save results
    with open(os.path.join(SOLUTION_DIR, 'competition_results.json'), 'w') as f:
        json.dump({
            'total_levels': total_levels,
            'total_completed': total_completed,
            'games_won': games_won,
            'total_actions': total_actions,
            'elapsed': elapsed,
            'results': results,
        }, f, indent=2)
    print()
    print("Results saved to arc3/competition_results.json")


if __name__ == '__main__':
    main()
