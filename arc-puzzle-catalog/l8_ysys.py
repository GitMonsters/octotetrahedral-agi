# Watch ysys behavior in level 8 (player frozen)
import sys, copy, time
sys.path.insert(0, '/Users/evanpieser/arc-puzzle-catalog')
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState, ActionInput
from arc3.solver import Wa30Solver, AMAP

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE)
env = arc.make('wa30-ee6fef47')
obs = env.step(GameAction.RESET)

# Solve levels 0-7 
for level_num in range(8):
    g = env._game
    pre = copy.deepcopy(g)
    solver = Wa30Solver(env, verbose=False)
    sol = solver.solve_level(timeout=55)
    if not sol:
        print(f"FAILED at level {level_num}")
        sys.exit(1)
    env._game = pre
    for action in sol:
        obs = env.step(AMAP[action])
        if obs.levels_completed > level_num:
            break

g = env._game
print(f"=== Level 8 (idx={g.level_index}) — player frozen ===")
lv = g.current_level
blocks = lv.get_sprites_by_tag('geezpjgiyd')
goals = g.wyzquhjerd
block_prev = {id(b): (b.x, b.y) for b in blocks}

for i in range(70):
    # Player does ACTION1 (press up into wall - effectively frozen)
    g._set_action(ActionInput(id=AMAP[1], data={}))
    g.step()
    
    # Track block movements
    for b in blocks:
        new_pos = (b.x, b.y)
        if new_pos != block_prev[id(b)]:
            carrier = g.zmqreragji.get(b)
            carrier_tags = list(carrier.tags) if carrier else []
            in_goal_now = new_pos in goals
            print(f"step {i+1}: block {block_prev[id(b)]} -> {new_pos}, carrier={carrier_tags}, in_goal={in_goal_now}")
            block_prev[id(b)] = new_pos
    
    if i < 3 or i % 5 == 0:
        ysys_sprites = lv.get_sprites_by_tag('ysysltqlke')
        kdw_sprites = lv.get_sprites_by_tag('kdweefinfi')
        in_goal_count = sum(1 for b in blocks if (b.x,b.y) in goals and b not in g.zmqreragji)
        ysys_info = []
        for y in ysys_sprites:
            carry = None
            if y in g.nsevyuople:
                blk = g.nsevyuople[y]
                carry = (blk.x, blk.y)
            ysys_info.append((y.x, y.y, carry))
        kdw_info = [(k.x,k.y) for k in kdw_sprites]
        print(f"  T={i+1}: steps_left={g.kuncbnslnm.current_steps}, ysys={ysys_info}, kdw={kdw_info}, in_goal={in_goal_count}/9")
    
    if g.ymzfopzgbq():
        print(f"WIN at step {i+1}!")
        break
