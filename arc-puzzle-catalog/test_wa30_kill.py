import sys, logging, time
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from arc_agi import Arcade
from arc3.solver import Wa30Solver, AMAP

arcade = Arcade(environments_dir="environment_files")

for test_level in [6, 8]:
    env = arcade.make("wa30-ee6fef47")
    env.reset()
    g = env._game
    g.set_level(test_level)

    ysys = g.current_level.get_sprites_by_tag('ysysltqlke')
    kdw  = g.current_level.get_sprites_by_tag('kdweefinfi')
    print(f"\nLevel {test_level}: ysys={len(ysys)}, kdw={len(kdw)}, steps={g.kuncbnslnm.dbdarsgrbj}")

    solver = Wa30Solver(env, verbose=True)
    t0 = time.time()
    kill_seq = solver._bfs_kill_ysys(g, max_steps=120, timeout=25.0)
    elapsed = time.time() - t0
    if kill_seq:
        print(f"  kill_ysys: {len(kill_seq)} steps in {elapsed:.1f}s")
        # After kill, check remaining blocks
        from arc3.solver import ActionInput
        solver._restore_game_state(g, solver._save_game_state(g))
        init_saved = solver._save_game_state(g)
        solver._restore_game_state(g, init_saved)
        for a in kill_seq:
            g._set_action(ActionInput(id=AMAP[a], data={}))
            g.step()
        remaining_blocks = g.current_level.get_sprites_by_tag('geezpjgiyd')
        goals = list(g.wyzquhjerd)
        steps_used = g.kuncbnslnm.current_steps
        print(f"  After kill: {len(remaining_blocks)} blocks, {g.kuncbnslnm.dbdarsgrbj} step_limit, {steps_used} used")
        print(f"  Goals (first 5): {goals[:5]}")
    else:
        print(f"  kill_ysys: FAILED in {elapsed:.1f}s")
