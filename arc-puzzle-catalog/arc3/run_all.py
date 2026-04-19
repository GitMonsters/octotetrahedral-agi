#!/usr/bin/env python3
"""
ARC-AGI-3 Full Competition Runner
=================================

Runs ALL games with optimal strategies:
- Dedicated solvers for VC33, LS20 (when compatible)
- OctoTetraAgent with toggle solver for others
- Tracks results and reports final score

Usage:
    python3 arc3/run_all.py
    python3 arc3/run_all.py --verbose
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc_agi
from arcengine import GameAction, GameState

from arc3.agent import OctoTetraAgent
from arc3.solver import Vc33Solver, Wa30Solver, GameAwareSolver

AMAP = {a.value: a for a in GameAction}


def run_vc33(arc, game_id, verbose=False):
    """Run VC33 with dedicated A* solver."""
    print(f'  [VC33] Using dedicated solver...')
    env = arc.make(game_id)
    if not env:
        return 0, 7, False, 0
    
    frame = env.reset()
    total_actions = 1
    
    for level in range(7):
        solver = Vc33Solver(env, verbose=verbose)
        solution = solver.solve_level(timeout=300, max_nodes=500000)
        if solution:
            for dx, dy, label in solution:
                frame = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                total_actions += 1
                if frame.levels_completed > level:
                    break
            if verbose:
                print(f'    Level {level}: {len(solution)} clicks')
        else:
            print(f'    Level {level}: NO SOLUTION')
            break
        if frame.state != GameState.NOT_FINISHED:
            break
    
    won = frame.state == GameState.WIN
    return frame.levels_completed, 7, won, total_actions


def run_wa30(arc, game_id, verbose=False):
    """Run WA30 with dedicated Sokoban solver."""
    print(f'  [WA30] Using dedicated solver...')
    env = arc.make(game_id)
    if not env:
        return 0, 9, False, 0
    
    frame = env.reset()
    total_actions = 1
    
    for level in range(9):
        solver = Wa30Solver(env, verbose=verbose)
        solution = solver.solve_level(max_depth=200, max_nodes=50000, timeout=60)
        if solution:
            for action in solution:
                frame = env.step(AMAP[action])
                total_actions += 1
                if frame.levels_completed > level:
                    break
                if frame.state == GameState.GAME_OVER:
                    break
            if verbose:
                print(f'    Level {level}: {len(solution)} moves')
        else:
            print(f'    Level {level}: NO SOLUTION')
            break
        if frame.state != GameState.NOT_FINISHED:
            break
    
    won = frame.state == GameState.WIN
    return frame.levels_completed, 9, won, total_actions


def run_with_agent(arc, game_id, verbose=False, max_actions=2000):
    """Run any game with OctoTetraAgent."""
    env = arc.make(game_id)
    if not env:
        return 0, 0, False, 0
    
    agent = OctoTetraAgent(
        max_actions_per_level=max_actions,
        verbose=verbose,
        use_mercury=False,  # Mercury needs API key
    )
    result = agent.play_game(env, GameAction)
    return (
        result['levels_completed'],
        result['win_levels'],
        result['won'],
        result['total_actions']
    )


def run_game(arc, game_id, verbose=False):
    """Run a single game with optimal strategy."""
    game_type = game_id.split('-')[0].lower()
    
    # Use dedicated solvers for known games
    if game_type == 'vc33':
        return run_vc33(arc, game_id, verbose)
    elif game_type == 'wa30':
        return run_wa30(arc, game_id, verbose)
    else:
        # Use agent for everything else
        return run_with_agent(arc, game_id, verbose)


def main():
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    # Check API key
    api_key = os.environ.get('ARC_API_KEY')
    if not api_key:
        print('ERROR: ARC_API_KEY environment variable not set')
        print('Get your key from https://three.arcprize.org/')
        sys.exit(1)
    
    print('='*70)
    print('🐙 ARC-AGI-3 FULL COMPETITION RUN 🐙')
    print('='*70)
    print()
    
    # Initialize arcade
    arc = arc_agi.Arcade()
    
    # Get unique games
    games = arc.get_environments()
    seen = set()
    unique_games = []
    for g in games:
        if g.game_id not in seen:
            seen.add(g.game_id)
            unique_games.append(g)
    
    print(f'Found {len(unique_games)} unique games')
    print()
    
    # Run all games
    results = []
    total_completed = 0
    total_levels = 0
    total_actions = 0
    games_won = 0
    
    t0 = time.time()
    
    for i, game_info in enumerate(unique_games):
        gid = game_info.game_id
        game_type = gid.split('-')[0].upper()
        print(f'[{i+1}/{len(unique_games)}] {game_type} ({gid})', end='', flush=True)
        
        gt0 = time.time()
        lc, wl, won, actions = run_game(arc, gid, verbose)
        elapsed = time.time() - gt0
        
        status = '✅ WON' if won else f'{lc}/{wl}'
        print(f' → {status} ({actions} actions, {elapsed:.1f}s)')
        
        results.append({
            'game_id': gid,
            'game_type': game_type,
            'levels_completed': lc,
            'win_levels': wl,
            'won': won,
            'actions': actions,
            'elapsed': elapsed
        })
        
        total_completed += lc
        total_levels += wl
        total_actions += actions
        if won:
            games_won += 1
    
    total_time = time.time() - t0
    
    # Print summary
    print()
    print('='*70)
    print('RESULTS SUMMARY')
    print('='*70)
    print()
    
    # Group by game type
    by_type = {}
    for r in results:
        t = r['game_type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)
    
    for game_type in sorted(by_type.keys()):
        games = by_type[game_type]
        lc = sum(g['levels_completed'] for g in games)
        wl = sum(g['win_levels'] for g in games)
        won = sum(1 for g in games if g['won'])
        status = '✅' if won == len(games) else ('⚠️' if won > 0 else '❌')
        print(f'  {status} {game_type}: {lc}/{wl} levels, {won}/{len(games)} games won')
    
    print()
    print(f'TOTAL: {total_completed}/{total_levels} levels ({100*total_completed/total_levels:.1f}%)')
    print(f'Games won: {games_won}/{len(results)}')
    print(f'Total actions: {total_actions}')
    print(f'Total time: {total_time:.1f}s')
    print()
    
    # Get scorecard
    try:
        sc = arc.get_scorecard()
        print(f'Official Score: {sc.score:.2f}')
        print(f'Official Levels: {sc.total_levels_completed}/{sc.total_levels}')
    except Exception as e:
        print(f'Could not get scorecard: {e}')
    
    # Save results
    output_file = 'arc3_full_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'total_completed': total_completed,
            'total_levels': total_levels,
            'games_won': games_won,
            'total_actions': total_actions,
            'total_time': total_time,
        }, f, indent=2)
    print(f'\nResults saved to {output_file}')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
