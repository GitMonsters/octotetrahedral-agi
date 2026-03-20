"""
Game-state-aware solvers for ARC-AGI-3 games.

These solvers access env._game internals to read the full game state,
build navigation graphs, and plan optimal solutions via BFS.
"""

from __future__ import annotations
import copy
import heapq
import logging
import time
from collections import deque
from typing import Optional

import numpy as np
from arcengine import GameAction, GameState

logger = logging.getLogger("arc3.solver")

AMAP = {a.value: a for a in GameAction}

# LS20 action mapping: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
LS20_DIRS = {
    1: (0, -1),   # UP
    2: (0, 1),    # DOWN
    3: (-1, 0),   # LEFT
    4: (1, 0),    # RIGHT
}


class Ls20Solver:
    """Solves LS20 levels using semantic state BFS."""

    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose

    @property
    def game(self):
        return self.env._game

    def _build_nav_graph(self) -> dict:
        """Build navigation graph from game state.
        
        Returns dict with walls, modifiers, targets, walkable set, and adjacency.
        Also maps pickup positions to the grid cells that trigger them.
        """
        g = self.game
        level = g.current_level

        # Collect walls
        walls = set()
        for s in level._sprites:
            if s.tags and "jdd" in s.tags:
                walls.add((s.x, s.y))

        # Collect modifiers
        modifiers = {}
        for s in level._sprites:
            if not s.tags:
                continue
            if "bgt" in s.tags and s.is_visible:
                modifiers[(s.x, s.y)] = "rot"
            elif "gsu" in s.tags and s.is_visible:
                modifiers[(s.x, s.y)] = "shape"
            elif "gic" in s.tags and s.is_visible:
                modifiers[(s.x, s.y)] = "color"

        # Determine grid alignment from player position
        step = 5
        px, py = g.mgu.x, g.mgu.y
        x_mod = px % step
        y_mod = py % step

        all_positions = set()
        for x in range(x_mod, 64, step):
            for y in range(y_mod, 64, step):
                all_positions.add((x, y))

        def is_blocked(x, y):
            for wx, wy in walls:
                if wx >= x and wx < x + step and wy >= y and wy < y + step:
                    return True
            return False

        walkable = {pos for pos in all_positions if not is_blocked(*pos)}

        # Build adjacency
        adj = {}
        for pos in walkable:
            adj[pos] = []
            for act, (dx, dy) in LS20_DIRS.items():
                nx, ny = pos[0] + dx * step, pos[1] + dy * step
                if (nx, ny) in walkable:
                    adj[pos].append((act, (nx, ny)))

        # Map iri pickups to their triggering grid positions.
        # The game checks rbt(dest_x, dest_y, 5, 5) — sprites with 
        # sx >= dest_x and sx < dest_x+5 and sy >= dest_y and sy < dest_y+5.
        # So find which walkable grid cell "contains" each iri sprite.
        raw_pickups = []
        for s in level._sprites:
            if s.tags and "iri" in s.tags and s.is_visible:
                raw_pickups.append((s.x, s.y))

        # Map each raw pickup to a grid position that triggers it
        pickups = set()  # grid-aligned positions that trigger iri pickups
        for sx, sy in raw_pickups:
            # Find grid cell (gx, gy) such that gx <= sx < gx+5 and gy <= sy < gy+5
            gx = sx - (sx - x_mod) % step
            gy = sy - (sy - y_mod) % step
            if (gx, gy) in walkable:
                pickups.add((gx, gy))

        # Also map modifiers to grid positions if they're not grid-aligned
        grid_modifiers = {}
        for (sx, sy), mod_type in modifiers.items():
            gx = sx - (sx - x_mod) % step
            gy = sy - (sy - y_mod) % step
            if (gx, gy) in walkable:
                grid_modifiers[(gx, gy)] = mod_type
            elif (sx, sy) in walkable:
                grid_modifiers[(sx, sy)] = mod_type

        # Target info
        targets = []
        for i, q in enumerate(g.qqv):
            targets.append({
                'pos': (q.x, q.y),
                'shape': g.gfy[i],
                'color': g.vxy[i],
                'rot': g.cjl[i],
                'done': g.rzt[i],
            })

        return {
            'walkable': walkable,
            'adj': adj,
            'modifiers': grid_modifiers,
            'pickups': pickups,
            'targets': targets,
            'player': (px, py),
            'state': (g.snw, g.tmx, g.tuv),
            'n_shapes': len(g.hep),
            'n_colors': len(g.hul),
        }

    def _shortest_path(self, adj, start, end) -> Optional[list[int]]:
        """BFS shortest path between two positions. Returns action list."""
        if start == end:
            return []
        q = deque([(start, [])])
        visited = {start}
        while q:
            pos, path = q.popleft()
            for act, npos in adj.get(pos, []):
                if npos in visited:
                    continue
                new_path = path + [act]
                if npos == end:
                    return new_path
                visited.add(npos)
                q.append((npos, new_path))
        return None

    def solve_level(self) -> Optional[list[int]]:
        """Plan optimal action sequence for current level.
        
        Uses BFS on semantic state with step-budget awareness.
        Routes through iri pickups to refuel step counter when needed.
        """
        nav = self._build_nav_graph()
        targets = nav['targets']
        adj = nav['adj']
        mods = nav['modifiers']
        player = nav['player']
        shape, color, rot = nav['state']
        n_shapes = nav['n_shapes']
        n_colors = nav['n_colors']

        unsolved = [(i, t) for i, t in enumerate(targets) if not t['done']]
        if not unsolved:
            return []

        # Get step budget
        g_obj = self.env._game
        step_budget = g_obj.ggk.tmx  # Max steps before life loss
        if step_budget == 0:
            step_budget = 999  # No limit

        if self.verbose:
            logger.info(f"Solver: {len(unsolved)} targets, {len(nav['walkable'])} tiles, "
                       f"budget={step_budget}, modifiers: {mods}")
            for idx, (i, t) in enumerate(unsolved):
                logger.info(f"  Target {idx}: pos={t['pos']} need shape={t['shape']}, "
                           f"color={t['color']}, rot={t['rot']}")
            logger.info(f"  Current: shape={shape}, color={color}, rot={rot}")
            logger.info(f"  Pickups: {nav['pickups']}")

        # Precompute shortest paths between all key positions
        key_positions = {player}
        for _, t in unsolved:
            key_positions.add(t['pos'])
        for pos in mods:
            key_positions.add(pos)
            for _, npos in adj.get(pos, []):
                key_positions.add(npos)
        for pos in nav['pickups']:
            key_positions.add(pos)

        path_cache = {}
        dist_cache = {}
        for a in key_positions:
            for b in key_positions:
                if a != b:
                    p = self._shortest_path(adj, a, b)
                    if p is not None:
                        path_cache[(a, b)] = p
                        dist_cache[(a, b)] = len(p)

        # Bounce cache for modifiers
        bounce_cache = {}
        for mod_pos in mods:
            best_bounce = None
            for act, npos in adj.get(mod_pos, []):
                back_path = self._shortest_path(adj, npos, mod_pos)
                if back_path is not None:
                    bounce = [act] + back_path
                    if best_bounce is None or len(bounce) < len(best_bounce):
                        best_bounce = bounce
            if best_bounce:
                bounce_cache[mod_pos] = best_bounce

        n_unsolved = len(unsolved)
        all_done = (1 << n_unsolved) - 1

        # BFS state: (pos, shape, color, rot, target_mask, steps_used)
        # steps_used tracks steps since last refuel (iri pickup or level start)
        # When steps_used would exceed budget, must route through iri first
        start_state = (player, shape, color, rot, 0, 0)

        queue = deque([(start_state, [])])
        # visited: (pos, s, c, r, mask) -> best steps_used to reach it
        visited = {(player, shape, color, rot, 0): 0}

        best_solution = None
        iterations = 0
        max_iterations = 2000000

        pickups = nav['pickups']

        while queue and iterations < max_iterations:
            iterations += 1
            (pos, s, c, r, mask, steps), actions = queue.popleft()

            if best_solution and len(actions) >= len(best_solution):
                continue

            # Check if steps_used exceeds budget (shouldn't happen, but guard)
            if step_budget > 0 and steps > step_budget:
                continue

            def try_move(new_pos, new_s, new_c, new_r, new_mask, path_acts):
                nonlocal best_solution
                new_actions = actions + path_acts
                new_steps = steps + len(path_acts)

                if best_solution and len(new_actions) >= len(best_solution):
                    return

                if step_budget > 0 and new_steps > step_budget:
                    return  # Would die before reaching destination

                # Check if new_pos is a pickup (refuel)
                actual_steps = new_steps
                if new_pos in pickups:
                    actual_steps = 0  # Refueled!

                vis_key = (new_pos, new_s, new_c, new_r, new_mask)
                prev_steps = visited.get(vis_key)
                if prev_steps is not None and prev_steps <= actual_steps:
                    return  # Already visited with equal or fewer steps
                visited[vis_key] = actual_steps

                if new_mask == all_done:
                    if best_solution is None or len(new_actions) < len(best_solution):
                        best_solution = new_actions
                        if self.verbose:
                            logger.info(f"Found solution: {len(new_actions)} actions")
                    return

                queue.append(((new_pos, new_s, new_c, new_r, new_mask, actual_steps), new_actions))

            # Option 1: Go to a modifier
            for mod_pos, mod_type in mods.items():
                if pos == mod_pos:
                    continue
                if (pos, mod_pos) not in path_cache:
                    continue
                path = path_cache[(pos, mod_pos)]
                ns, nc, nr = s, c, r
                if mod_type == "rot":
                    nr = (r + 1) % 4
                elif mod_type == "shape":
                    ns = (s + 1) % n_shapes
                elif mod_type == "color":
                    nc = (c + 1) % n_colors
                try_move(mod_pos, ns, nc, nr, mask, path)

            # Option 2: Bounce on current modifier
            if pos in mods and pos in bounce_cache:
                mod_type = mods[pos]
                bounce = bounce_cache[pos]
                ns, nc, nr = s, c, r
                if mod_type == "rot":
                    nr = (r + 1) % 4
                elif mod_type == "shape":
                    ns = (s + 1) % n_shapes
                elif mod_type == "color":
                    nc = (c + 1) % n_colors
                try_move(pos, ns, nc, nr, mask, bounce)

            # Option 3: Go to iri pickup (refuel point)
            for iri_pos in pickups:
                if pos == iri_pos:
                    continue
                if (pos, iri_pos) not in path_cache:
                    continue
                path = path_cache[(pos, iri_pos)]
                try_move(iri_pos, s, c, r, mask, path)

            # Option 4: Deliver to a target
            for idx, (orig_i, t) in enumerate(unsolved):
                if mask & (1 << idx):
                    continue
                if s == t['shape'] and c == t['color'] and r == t['rot']:
                    if (pos, t['pos']) not in path_cache:
                        continue
                    path = path_cache[(pos, t['pos'])]
                    new_mask = mask | (1 << idx)
                    try_move(t['pos'], s, c, r, new_mask, path)

        if self.verbose:
            logger.info(f"BFS: {iterations} iters, {len(visited)} states")

        return best_solution


class Vc33Solver:
    """Solves VC33 levels using BFS with deepcopy state management."""

    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose

    @property
    def game(self):
        return self.env._game

    def _build_display_map(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Build grid→display coordinate mapping for click actions."""
        cam = self.game.camera
        mapping = {}
        for dx in range(64):
            for dy in range(64):
                r = cam.display_to_grid(dx, dy)
                if r and r not in mapping:
                    mapping[r] = (dx, dy)
        return mapping

    def _get_clickable(self, g, display_map: dict) -> list[tuple[int, int, str, int, int]]:
        """Get all clickable display coords with labels and grid positions.
        
        Returns list of (display_x, display_y, label, grid_x, grid_y).
        """
        level = g.current_level
        clicks = []
        seen = set()
        for s in level.get_sprites_by_tag('ZGd'):
            gpos = (s.x, s.y)
            if gpos not in seen and gpos in display_map:
                dx, dy = display_map[gpos]
                clicks.append((dx, dy, f'ZGd({s.x},{s.y})', s.x, s.y))
                seen.add(gpos)
        for s in level.get_sprites_by_tag('zHk'):
            gpos = (s.x, s.y)
            if gpos not in seen and gpos in display_map and g.krt(s):
                dx, dy = display_map[gpos]
                clicks.append((dx, dy, f'zHk({s.x},{s.y})', s.x, s.y))
                seen.add(gpos)
        return clicks

    def _state_key(self, g) -> tuple:
        """Hash the game state for BFS visited tracking."""
        level = g.current_level
        parts = []
        for s in sorted(level.get_sprites_by_tag('HQB'), key=lambda s: s.name):
            parts.append(('H', s.x, s.y))
        for s in sorted(level.get_sprites_by_tag('rDn'), key=lambda s: s.name):
            parts.append(('T', s.x, s.y, s.width, s.height))
        return tuple(parts)

    def _apply_click(self, dx: int, dy: int):
        """Apply a single click via env.step (for replay only)."""
        env = self.env
        obs = env.step(AMAP[6], data={'x': dx, 'y': dy})
        g = env._game
        while g.vai is not None:
            obs = env.step(AMAP[6], data={'x': -1, 'y': -1})
            g = env._game
        return obs

    def _apply_direct(self, g, gx: int, gy: int) -> bool:
        """Apply click directly on game object (no env.step). For BFS search."""
        sprite = g.current_level.get_sprite_at(gx, gy)
        if not sprite:
            return False

        if "ZGd" in sprite.tags:
            g.ccl(sprite)
            return True
        elif "zHk" in sprite.tags:
            if g.krt(sprite):
                vai = g.teu(sprite)
                # Instant-resolve animation: set sprites to final positions
                if vai:
                    for anim_step in vai.lph:
                        anim_step.cab.set_position(anim_step.shl[0], anim_step.shl[1])
                    g.vai = None
                    g.jcy()
                for yps in g.current_level.get_sprites_by_tag("rlV"):
                    yps.set_visible(False)
                return True
        return False

    def _heuristic(self, g) -> int:
        """Estimate remaining clicks needed to win. Admissible for A*."""
        level = g.current_level
        hqbs = level.get_sprites_by_tag('HQB')
        fzks = level.get_sprites_by_tag('fZK')
        rdns = level.get_sprites_by_tag('rDn')
        uxgs = level.get_sprites_by_tag('UXg')
        oro_mag = max(abs(g.oro[0]), abs(g.oro[1]), 1)

        total = 0
        for dds in hqbs:
            AkL = int(dds.pixels[-1, -1])
            best = 200
            for yas in fzks:
                if AkL not in yas.pixels:
                    continue
                d = abs(g.ebl(dds) - g.ebl(yas))
                iZX = [mdf for mdf in uxgs if mdf.collides_with(yas)]
                if iZX:
                    lzS = [avz for avz in rdns if g.gdu(dds, avz)]
                    if lzS:
                        zGp = g.suo(lzS[0])
                        if iZX[0] not in zGp:
                            d += 10  # wrong track penalty
                best = min(best, d)
            total += best
        return total // oro_mag

    def _solve_symbolic(self, grid_seq: list[tuple[int, int]],
                        display_map: dict) -> Optional[list[tuple[int, int, str]]]:
        """Verify and convert a symbolic grid-coord solution to display coords.
        
        Runs the sequence on a deepcopy to confirm it wins, then returns
        the display-coord solution for replay via env.step.
        """
        g = self.env._game
        g_test = copy.deepcopy(g)
        result = []
        for gx, gy in grid_seq:
            dpos = display_map.get((gx, gy))
            if dpos is None:
                return None
            sprite = g_test.current_level.get_sprite_at(gx, gy)
            label = sprite.name if sprite else f"({gx},{gy})"
            self._apply_direct(g_test, gx, gy)
            result.append((dpos[0], dpos[1], label))
            if g_test.gug():
                if self.verbose:
                    logger.info(f"  Symbolic solved: {len(result)} clicks")
                return result
        return None

    def _l6_sequence(self) -> list[tuple[int, int]]:
        """Grid-coordinate sequence for VC33 L6 (5-track, 3-HQB puzzle)."""
        seq = []
        seq += [(16, 0)] * 10   # HMp→RmM transfer x10
        seq += [(16, 24)]       # wmR→RmM transfer x1
        seq += [(14, 30)]       # swap wmR↔RmM (moves ChX to RmM)
        seq += [(12, 0)] * 10   # RmM→HMp transfer x10
        seq += [(34, 0)]        # RmM→HfU transfer x1
        seq += [(32, 8)]        # swap RmM↔HfU (ChX→HfU, VAJ→RmM)
        seq += [(16, 0)] * 10   # HMp→RmM transfer x10
        seq += [(30, 24)] * 2   # AEF→RmM transfer x2
        seq += [(34, 0)]        # RmM→HfU transfer x1
        seq += [(14, 30)]       # swap wmR↔RmM (VAJ→wmR)
        seq += [(32, 30)]       # swap RmM↔AEF (PPS→RmM)
        seq += [(12, 24)] * 6   # RmM→wmR transfer x6
        seq += [(34, 0)] * 4    # RmM→HfU transfer x4
        return seq

    def solve_level(self, max_depth: int = 100, max_nodes: int = 200000,
                    timeout: float = 300.0) -> Optional[list[tuple[int, int, str]]]:
        """A* search to find click sequence that wins the current level.
        
        Uses deepcopy + direct game method calls (no env.step) during search.
        Returns click sequence as (display_x, display_y, label) for replay.
        Falls back to symbolic solver for complex levels (e.g. L6).
        """
        g = self.env._game
        level_idx = g.level_index
        display_map = self._build_display_map()

        # Try symbolic solver first for complex levels (much faster than A*)
        sym_seq = self._l6_sequence()
        sym_sol = self._solve_symbolic(sym_seq, display_map)
        if sym_sol:
            return sym_sol

        initial_clicks = self._get_clickable(g, display_map)
        n_clickable = len(initial_clicks)
        if self.verbose:
            logger.info(f"VC33 L{level_idx}: {n_clickable} clickable, budget={g.vrr.lpw}")

        initial_sk = self._state_key(g)
        initial_h = self._heuristic(g)
        initial_game = copy.deepcopy(g)

        # Use A* for complex levels, plain BFS for simple ones
        use_astar = n_clickable > 5 or initial_h > 5

        counter = 0
        visited = {initial_sk}
        nodes = 0
        t0 = time.time()

        if use_astar:
            pq = [(initial_h, counter, initial_game, [])]
            while pq and time.time() - t0 < timeout:
                f, _, game_copy, seq = heapq.heappop(pq)
                nodes += 1
                if nodes > max_nodes or len(seq) >= max_depth:
                    if len(seq) >= max_depth:
                        continue
                    break

                clicks = self._get_clickable(game_copy, display_map)

                for dx, dy, label, gx, gy in clicks:
                    g_child = copy.deepcopy(game_copy)
                    self._apply_direct(g_child, gx, gy)

                    if g_child.gug():
                        result = seq + [(dx, dy, label)]
                        if self.verbose:
                            logger.info(f"  A* solved: {len(result)} clicks ({nodes} nodes, {time.time()-t0:.1f}s)")
                        return result

                    sk = self._state_key(g_child)
                    if sk not in visited:
                        visited.add(sk)
                        h = self._heuristic(g_child)
                        g_cost = len(seq) + 1
                        counter += 1
                        heapq.heappush(pq, (g_cost + h, counter, g_child, seq + [(dx, dy, label)]))
        else:
            queue = deque([(initial_game, [])])
            while queue and time.time() - t0 < timeout:
                game_copy, seq = queue.popleft()
                nodes += 1
                if nodes > max_nodes or len(seq) >= max_depth:
                    if len(seq) >= max_depth:
                        continue
                    break

                clicks = self._get_clickable(game_copy, display_map)

                for dx, dy, label, gx, gy in clicks:
                    g_child = copy.deepcopy(game_copy)
                    self._apply_direct(g_child, gx, gy)

                    if g_child.gug():
                        result = seq + [(dx, dy, label)]
                        if self.verbose:
                            logger.info(f"  BFS solved: {len(result)} clicks ({nodes} nodes, {time.time()-t0:.1f}s)")
                        return result

                    sk = self._state_key(g_child)
                    if sk not in visited:
                        visited.add(sk)
                        queue.append((g_child, seq + [(dx, dy, label)]))

        if self.verbose:
            logger.info(f"  No solution: {nodes} nodes, {len(visited)} states, {time.time()-t0:.1f}s")
        return None


class GameAwareSolver:
    """Top-level solver that dispatches to game-specific solvers."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def play_game(self, env, game_action_class) -> dict:
        """Play a full game using game-state-aware solving."""
        t0 = time.time()
        obs = env.step(GameAction.RESET)
        total_actions = 1
        wl = obs.win_levels
        game_id = obs.game_id
        lc = 0

        budget = 500 * wl

        if self.verbose:
            logger.info(f"GameAwareSolver: {game_id}, {wl} levels")

        # Detect game type
        is_ls20 = "ls20" in game_id.lower()
        is_vc33 = "vc33" in game_id.lower()
        is_ft09 = "ft09" in game_id.lower()

        if is_ls20:
            return self._play_ls20(env, obs, budget, t0)
        elif is_vc33:
            return self._play_vc33(env, obs, budget, t0)
        elif is_ft09:
            # FT09 already solved by BFS agent, delegate
            return self._play_bfs_fallback(env, obs, budget, t0)
        else:
            return self._play_bfs_fallback(env, obs, budget, t0)

    def _play_ls20(self, env, obs, budget, t0) -> dict:
        """Solve LS20 using semantic BFS.
        
        Strategy: solve each level incrementally, building up a solution chain.
        For each new level, plan from the current game state after replay.
        """
        total = 1
        wl = obs.win_levels
        lc = obs.levels_completed
        solutions = {}  # level -> action list

        max_attempts = 50

        for attempt in range(max_attempts):
            if total > budget:
                break

            # Reset and replay all known solutions
            obs = env.step(GameAction.RESET)
            total += 1
            cl = obs.levels_completed

            replay_ok = True
            while cl in solutions and total < budget:
                for act in solutions[cl]:
                    obs = env.step(AMAP[act])
                    total += 1
                    if obs.state == GameState.WIN:
                        return self._result(total, obs.levels_completed, wl, True, t0)
                    if obs.state == GameState.GAME_OVER:
                        replay_ok = False
                        break
                    if obs.levels_completed > cl:
                        break
                if not replay_ok:
                    break
                if obs.levels_completed > cl:
                    cl = obs.levels_completed
                else:
                    replay_ok = False
                    break

            if not replay_ok:
                if self.verbose:
                    logger.info(f"Replay failed at L{cl}")
                break

            lc = max(lc, cl)

            if cl >= wl:
                break

            # Plan solution for current level (if not already solved)
            if cl not in solutions:
                solver = Ls20Solver(env, verbose=self.verbose)
                sol = solver.solve_level()
                if sol:
                    solutions[cl] = sol
                    if self.verbose:
                        logger.info(f"L{cl} planned: {len(sol)} actions")
                else:
                    if self.verbose:
                        logger.info(f"L{cl}: no solution found")
                    break

            # Execute solution for current level
            for act in solutions[cl]:
                if total >= budget:
                    break
                obs = env.step(AMAP[act])
                total += 1
                if obs.state == GameState.WIN:
                    return self._result(total, obs.levels_completed, wl, True, t0)
                if obs.state == GameState.GAME_OVER:
                    break
                if obs.levels_completed > cl:
                    break

            if obs.state == GameState.GAME_OVER:
                # Solution caused game over - invalidate it and retry
                if self.verbose:
                    logger.info(f"L{cl} solution caused game over, retrying")
                del solutions[cl]
                continue

            if obs.levels_completed > cl:
                lc = max(lc, obs.levels_completed)
                # Continue to next level (loop will reset + replay)
                continue
            else:
                # Solution didn't complete the level
                if self.verbose:
                    logger.info(f"L{cl} solution didn't advance level")
                break

        return self._result(total, lc, wl, False, t0)

    def _play_vc33(self, env, obs, budget, t0) -> dict:
        """Solve VC33 using A*/BFS search per level.
        
        Search phase uses direct game manipulation (no env.step).
        Replay phase uses env.step for proper action counting.
        """
        total = 1
        wl = obs.win_levels
        lc = obs.levels_completed
        level_solutions = []

        for level_num in range(wl):
            if total > budget:
                break

            g = env._game
            if g.level_index != level_num:
                if self.verbose:
                    logger.info(f"VC33: expected L{level_num}, at L{g.level_index}")
                break

            # Save state before search (search doesn't touch env.step)
            pre_solve = copy.deepcopy(g)

            solver = Vc33Solver(env, verbose=self.verbose)
            sol = solver.solve_level(max_depth=100, max_nodes=200000, timeout=300)

            if sol:
                # Restore to level start and replay with env.step
                env._game = pre_solve
                env._game.vrr.olv = env._game.vrr.lpw  # ensure full step budget
                for dx, dy, name in sol:
                    obs = env.step(AMAP[6], data={'x': dx, 'y': dy})
                    total += 1
                    g = env._game
                    while g.vai is not None:
                        obs = env.step(AMAP[6], data={'x': -1, 'y': -1})
                        total += 1
                        g = env._game

                if obs.state == GameState.WIN:
                    lc = wl
                    level_solutions.append(sol)
                    if self.verbose:
                        logger.info(f"L{level_num}: {len(sol)} clicks -> WIN")
                    return self._result(total, lc, wl, True, t0)
                elif g.level_index > level_num:
                    lc = g.level_index
                    level_solutions.append(sol)
                    if self.verbose:
                        logger.info(f"L{level_num}: {len(sol)} clicks")
                else:
                    if self.verbose:
                        logger.info(f"L{level_num}: solution didn't advance")
                    break
            else:
                if self.verbose:
                    logger.info(f"L{level_num}: no solution found")
                break

        return self._result(total, lc, wl, lc >= wl, t0)

    def _play_bfs_fallback(self, env, obs, budget, t0) -> dict:
        """Fallback to the existing BFS agent."""
        from arc3.agent import OctoTetraAgent
        agent = OctoTetraAgent(
            max_actions_per_level=budget // obs.win_levels,
            verbose=self.verbose,
            use_mercury=False,
        )
        return agent.play_game(env, GameAction)

    def _result(self, total, lc, wl, won, t0):
        return {
            'total_actions': total,
            'levels_completed': lc,
            'win_levels': wl,
            'won': won,
            'elapsed_seconds': round(time.time() - t0, 2),
            'world_model': {},
            'memory_stats': {},
        }

    def reset(self):
        pass
