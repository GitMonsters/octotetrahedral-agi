# Understand why KDW2 stays at (44,4) and what obstacles exist in level 8
import sys, copy, time
sys.path.insert(0, '/Users/evanpieser/arc-puzzle-catalog')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, ActionInput
from arc3.solver import Wa30Solver, AMAP

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make('wa30-ee6fef47')
obs = env.step(GameAction.RESET)

for level_num in range(8):
    g = env._game
    pre = copy.deepcopy(g)
    solver = Wa30Solver(env, verbose=False)
    sol = solver.solve_level(timeout=55)
    env._game = pre
    for action in sol:
        obs = env.step(AMAP[action])
        if obs.levels_completed > level_num:
            break

g = env._game
print(f"=== Level 8: Obstacle Analysis ===")
lv = g.current_level
all_sprites = lv.get_sprites()
print(f"Total sprites: {len(all_sprites)}")

# Check all sprite positions/tags
for s in sorted(all_sprites, key=lambda s: (s.x, s.y)):
    tags = list(s.tags)
    if s.is_collidable:
        print(f"  Collidable: ({s.x},{s.y}) tags={tags}")

print("\nNon-collidable sprites:")
for s in sorted(all_sprites, key=lambda s: (s.x, s.y)):
    tags = list(s.tags)
    if not s.is_collidable:
        print(f"  ({s.x},{s.y}) tags={tags}")
