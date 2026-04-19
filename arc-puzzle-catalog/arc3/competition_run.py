#!/usr/bin/env python3
"""
ARC-AGI-3 Competition Mode Runner
=================================

This runs in COMPETITION mode which is REQUIRED to appear on the leaderboard.

In competition mode, we only get FrameDataRaw (no images), so we use:
1. Random action sequences
2. Level-completion feedback to guide exploration

Usage:
    export ARC_API_KEY="your-key"
    python3 arc3/competition_run.py
"""

import os
import sys
import time
import random
import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

AMAP = {a.value: a for a in GameAction}

# Movement actions
MOVE_ACTIONS = [
    GameAction.ACTION1,  # UP
    GameAction.ACTION2,  # DOWN
    GameAction.ACTION3,  # LEFT
    GameAction.ACTION4,  # RIGHT
]

# All basic actions
BASIC_ACTIONS = MOVE_ACTIONS + [
    GameAction.ACTION5,  # INTERACT
]

GAME_TIMEOUT = 60  # seconds per game
MAX_ACTIONS_PER_LEVEL = 500


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


def play_random_exploration(env, max_actions=2000):
    """Play game using random action exploration."""
    frame = env.reset()
    total_actions = 1
    
    if frame is None:
        return frame, total_actions
    
    best_level = frame.levels_completed
    best_frame = frame
    actions_this_level = 0
    
    while total_actions < max_actions:
        # Choose random action (bias toward movement)
        if random.random() < 0.9:
            action = random.choice(MOVE_ACTIONS)
        else:
            action = random.choice(BASIC_ACTIONS)
        
        try:
            frame = env.step(action)
            total_actions += 1
            actions_this_level += 1
            
            if frame is None:
                # Action failed, try reset
                frame = env.reset()
                total_actions += 1
                actions_this_level = 0
                if frame is None:
                    # Can't reset, game might be over
                    return best_frame, total_actions
                continue
            
            # Track best progress
            if frame.levels_completed > best_level:
                best_level = frame.levels_completed
                best_frame = frame
                actions_this_level = 0
                print(f" L{best_level}", end="", flush=True)
            
            # Check win
            if frame.state == GameState.WIN:
                return frame, total_actions
            
            # Check loss/game over - reset and try again
            if frame.state == GameState.GAME_OVER:
                frame = env.reset()
                total_actions += 1
                actions_this_level = 0
                if frame is None:
                    return best_frame, total_actions
            
            # If stuck on same level too long, reset
            if actions_this_level > MAX_ACTIONS_PER_LEVEL:
                frame = env.reset()
                total_actions += 1
                actions_this_level = 0
                if frame is None:
                    return best_frame, total_actions
            
        except Exception as e:
            # Try to recover with reset
            try:
                frame = env.reset()
                total_actions += 1
                actions_this_level = 0
                if frame is None:
                    return best_frame, total_actions
            except:
                return best_frame, total_actions
    
    return best_frame if best_frame else frame, total_actions


def play_click_exploration(env, max_actions=500):
    """Play game using click actions (ACTION6) at grid positions."""
    frame = env.reset()
    total_actions = 1
    
    if frame is None:
        return frame, total_actions
    
    best_level = frame.levels_completed
    
    # Try clicking at various grid positions
    positions = [(x, y) for x in range(4, 64, 4) for y in range(4, 64, 4)]
    random.shuffle(positions)
    
    for x, y in positions[:max_actions]:
        try:
            frame = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
            total_actions += 1
            
            if frame is None:
                continue
            
            if frame.levels_completed > best_level:
                best_level = frame.levels_completed
            
            if frame.state == GameState.WIN:
                return frame, total_actions
            
            if frame.state in [GameState.LOSS, GameState.GAME_OVER]:
                frame = env.reset()
                total_actions += 1
                
        except Exception:
            continue
    
    return frame, total_actions


def play_game(env, game_type, timeout_sec=GAME_TIMEOUT):
    """Play a game with timeout."""
    frame = env.reset()
    total_actions = 1
    
    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    
    try:
        # Try random exploration
        frame, total_actions = play_random_exploration(env, max_actions=3000)
        
    except TimeoutError:
        pass
    except Exception as e:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    return frame, total_actions


def main():
    print("=" * 70)
    print("🐙 ARC-AGI-3 COMPETITION MODE RUN 🐙")
    print("=" * 70)
    print()
    
    # Initialize in COMPETITION mode
    arc = Arcade(operation_mode=OperationMode.COMPETITION)
    
    envs = arc.available_environments
    
    # Get unique games (deduplicate by game type)
    unique_games = {}
    for e in envs:
        game_type = e.game_id.split('-')[0].upper()
        if game_type not in unique_games:
            unique_games[game_type] = e.game_id
    
    print(f"Found {len(unique_games)} unique games")
    print(f"Timeout per game: {GAME_TIMEOUT}s")
    print()
    
    results = []
    total_levels = 0
    total_completed = 0
    games_won = 0
    total_actions = 0
    t0 = time.time()
    
    for i, (game_type, game_id) in enumerate(sorted(unique_games.items())):
        print(f"[{i+1}/{len(unique_games)}] {game_type}", end="", flush=True)
        
        try:
            env = arc.make(game_id)
            if not env:
                print(" → FAILED TO LOAD")
                continue
            
            frame, actions = play_game(env, game_type)
            
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
    
    # Close scorecard to finalize results
    print()
    print("Closing scorecard to finalize submission...")
    try:
        final = arc.close_scorecard()
        if final:
            print(f"Scorecard closed successfully.")
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


if __name__ == '__main__':
    main()
