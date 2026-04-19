#!/usr/bin/env python3
"""
ARC-AGI-3 Submission Script

This script runs all 3 preview games with optimal solvers and submits to the leaderboard.
Requires ARC_API_KEY environment variable.

Usage:
    python3 arc3/submit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc_agi
from arcengine import GameAction, GameState
from arc3.solver import Ls20Solver, Vc33Solver
from arc3.agent import OctoTetraAgent

AMAP = {a.value: a for a in GameAction}


def run_ls20(arc, game_id='ls20-cb3b57cc'):
    """Run LS20 with dedicated solver or agent fallback."""
    print(f'[LS20] {game_id}')
    env = arc.make(game_id)
    if not env:
        print('  ERROR: Could not create environment')
        return 0, 7, False
    
    frame = env.reset()
    
    # Check if solver is compatible with this variant
    try:
        solver = Ls20Solver(env, verbose=False)
        actions = solver.solve_level()
        solver_works = actions is not None
    except AttributeError:
        solver_works = False
        actions = None
    
    if solver_works:
        # Use dedicated solver
        for level in range(7):
            if level > 0:
                solver = Ls20Solver(env, verbose=False)
                actions = solver.solve_level()
            if not actions:
                print(f'  Level {level}: NO SOLUTION')
                break
            for a in actions:
                frame = env.step(AMAP[a])
                if frame.levels_completed > level:
                    break
            print(f'  Level {level}: OK ({len(actions)} actions)')
            if frame.state != GameState.NOT_FINISHED:
                break
    else:
        # Fallback to agent for incompatible variants
        print('  Using OctoTetraAgent (solver incompatible)')
        agent = OctoTetraAgent(max_actions_per_level=3000, verbose=False)
        result = agent.play_game(env, GameAction)
        frame = type('Frame', (), {
            'levels_completed': result['levels_completed'],
            'state': GameState.WIN if result['won'] else GameState.NOT_FINISHED
        })()
    
    won = frame.state == GameState.WIN
    print(f'  RESULT: {frame.levels_completed}/7, WON={won}')
    return frame.levels_completed, 7, won


def run_ft09(arc, game_id='ft09-9ab2447a'):
    """Run FT09 with OctoTetra agent."""
    print(f'[FT09] {game_id}')
    agent = OctoTetraAgent(max_actions_per_level=2000, verbose=False)
    env = arc.make(game_id)
    if not env:
        print('  ERROR: Could not create environment')
        return 0, 6, False
    
    result = agent.play_game(env, GameAction)
    won = result['won']
    print(f'  RESULT: {result["levels_completed"]}/{result["win_levels"]} levels, WON={won}')
    return result['levels_completed'], result['win_levels'], won


def run_vc33(arc, game_id='vc33-9851e02b'):
    """Run VC33 with dedicated click solver."""
    print(f'[VC33] {game_id}')
    env = arc.make(game_id)
    if not env:
        print('  ERROR: Could not create environment')
        return 0, 7, False
    
    frame = env.reset()
    for level in range(7):
        solver = Vc33Solver(env, verbose=False)
        solution = solver.solve_level(timeout=300, max_nodes=500000)
        if solution:
            for dx, dy, label in solution:
                frame = env.step(GameAction.ACTION6, data={'x': dx, 'y': dy})
                if frame.levels_completed > level:
                    break
            print(f'  Level {level}: OK ({len(solution)} clicks)')
        else:
            print(f'  Level {level}: NO SOLUTION')
            break
        if frame.state != GameState.NOT_FINISHED:
            break
    
    won = frame.state == GameState.WIN
    print(f'  RESULT: {frame.levels_completed}/7, WON={won}')
    return frame.levels_completed, 7, won


def main():
    # Check API key
    api_key = os.environ.get('ARC_API_KEY')
    if not api_key:
        print('ERROR: ARC_API_KEY environment variable not set')
        print('Get your key from https://three.arcprize.org/')
        sys.exit(1)
    
    print('='*60)
    print('ARC-AGI-3 SUBMISSION')
    print('='*60)
    print()
    
    # Initialize arcade (creates scorecard)
    arc = arc_agi.Arcade()
    
    # Get available environments
    envs = arc.get_environments()
    ls20_ids = [e.game_id for e in envs if 'ls20' in e.game_id.lower()]
    ft09_ids = [e.game_id for e in envs if 'ft09' in e.game_id.lower()]
    vc33_ids = [e.game_id for e in envs if 'vc33' in e.game_id.lower()]
    
    print(f'Available games:')
    print(f'  LS20: {ls20_ids}')
    print(f'  FT09: {ft09_ids}')
    print(f'  VC33: {vc33_ids}')
    print()
    
    total_levels = 0
    total_possible = 0
    games_won = 0
    
    # Run LS20 (use cb3b57cc variant which Ls20Solver supports)
    if 'ls20-cb3b57cc' in ls20_ids:
        lc, nl, won = run_ls20(arc, 'ls20-cb3b57cc')
    elif ls20_ids:
        lc, nl, won = run_ls20(arc, ls20_ids[0])
    else:
        lc, nl, won = 0, 7, False
    total_levels += lc
    total_possible += nl
    if won:
        games_won += 1
    print()
    
    # Run FT09
    if ft09_ids:
        lc, nl, won = run_ft09(arc, ft09_ids[0])
        total_levels += lc
        total_possible += nl
        if won:
            games_won += 1
    print()
    
    # Run VC33
    if vc33_ids:
        lc, nl, won = run_vc33(arc, vc33_ids[0])
        total_levels += lc
        total_possible += nl
        if won:
            games_won += 1
    print()
    
    # Get final scorecard
    print('='*60)
    print('FINAL SCORECARD')
    print('='*60)
    sc = arc.get_scorecard()
    print(f'Score: {sc.score:.2f}')
    print(f'Levels: {sc.total_levels_completed}/{sc.total_levels}')
    print(f'Environments: {sc.total_environments_completed}/{sc.total_environments}')
    print(f'Total Actions: {sc.total_actions}')
    print()
    print(f'Games Won: {games_won}/3')
    
    if sc.total_environments_completed == sc.total_environments:
        print()
        print('✅ ALL GAMES COMPLETED! Check the leaderboard at https://three.arcprize.org/')
    
    return 0 if games_won == 3 else 1


if __name__ == '__main__':
    sys.exit(main())
