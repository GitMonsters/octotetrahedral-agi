"""
OctoTetra Agent — ARC-AGI-3 BFS + toggle puzzle solver.

Key features:
- Splash screen detection: auto-dismisses level transition screens
- Toggle solver: GF(2) linear algebra for lights-out puzzles
- BFS exploration: systematic state-graph search
"""

from __future__ import annotations
import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger("arc3.agent")


class StateGraph:
    __slots__ = ['trans', 'advance', 'explored']

    def __init__(self):
        self.trans: dict[int, dict[int, int]] = {}
        self.advance: dict[int, set[int]] = {}
        self.explored: dict[int, set[int]] = {}

    def add(self, s, a, ns, adv=False):
        self.trans.setdefault(s, {})[a] = ns
        self.explored.setdefault(s, set()).add(a)
        if adv:
            self.advance.setdefault(s, set()).add(a)

    def solution(self, start):
        q = deque([(start, [])])
        vis = {start}
        while q:
            s, p = q.popleft()
            if len(p) > 300:
                continue
            if s in self.advance:
                return p + [min(self.advance[s])]
            for a in sorted(self.trans.get(s, {})):
                ns = self.trans[s][a]
                if ns not in vis:
                    vis.add(ns)
                    q.append((ns, p + [a]))
        return None

    def next_unexplored(self, start, avail, max_d=80):
        q = deque([(start, [])])
        vis = {start}
        while q:
            s, p = q.popleft()
            if len(p) > max_d:
                continue
            exp = self.explored.get(s, set())
            for a in avail:
                if a not in exp:
                    return (p, a)
            for a in sorted(self.trans.get(s, {})):
                ns = self.trans[s][a]
                if ns not in vis:
                    vis.add(ns)
                    q.append((ns, p + [a]))
        return None

    @property
    def size(self):
        s = set(self.trans.keys())
        for d in self.trans.values():
            s.update(d.values())
        return len(s)


class OctoTetraAgent:
    def __init__(self, max_actions_per_level=5000, verbose=False,
                 use_mercury=True):
        self.max_actions_per_level = max_actions_per_level
        self.verbose = verbose
        self.mercury = None
        if use_mercury:
            try:
                from arc3.mercury import MercuryReasoner
                self.mercury = MercuryReasoner()
                if self.mercury.available and self.verbose:
                    logger.info("Mercury 2 reasoning enabled")
            except Exception:
                pass

    def _h(self, frame):
        return hash(frame[:61].tobytes())

    def play_game(self, env, game_action_class) -> dict:
        from arcengine import GameState
        gac = game_action_class
        AMAP = {a.value: a for a in gac}
        t0 = time.time()

        obs = env.step(gac.RESET)
        total = 1
        wl = obs.win_levels
        raw_avail = sorted(obs.available_actions)
        has_click = 6 in raw_avail
        has_move = any(a in raw_avail for a in [1, 2, 3, 4])

        if self.verbose:
            logger.info(f"Game: {obs.game_id}, levels={wl}, actions={raw_avail}")

        budget = self.max_actions_per_level * wl
        lc = 0
        graphs: dict[int, StateGraph] = {}
        sols: dict[int, list[int]] = {}
        # For BFS solutions, store splash dismiss action if needed
        splash_dismiss: dict[int, int | None] = {}
        click_tgts: list[tuple[int, int]] = []
        toggle_tried: set[int] = set()

        if has_click:
            click_tgts = self._compute_clicks(obs.frame[0])
            avail = [a for a in raw_avail if a != 6] + \
                    [100 + i for i in range(len(click_tgts))]
        else:
            avail = raw_avail

        def do_action(action_id):
            nonlocal total, obs
            if action_id >= 100:
                idx = action_id - 100
                r, c = click_tgts[idx] if idx < len(click_tgts) else (0, 0)
                obs = env.step(gac.ACTION6, data={'x': c, 'y': r})
            elif action_id == 0:
                obs = env.step(gac.RESET)
            else:
                obs = env.step(AMAP[action_id])
            total += 1

        def dismiss_splash():
            """Dismiss splash/transition screen if present."""
            nonlocal total, obs
            if not has_click:
                return
            before = obs.frame[0][:55].copy()
            # Click corner (0,0) to dismiss — avoid hitting game sprites
            obs = env.step(gac.ACTION6, data={'x': 0, 'y': 0})
            total += 1
            after = obs.frame[0][:55]
            diff = np.sum(before != after)
            if diff > 100:
                if self.verbose:
                    logger.info(f"Dismissed splash ({diff} px changed)")

        def replay_to_level(target_cl):
            """Reset and replay solutions to reach target level."""
            nonlocal total, obs, click_tgts, avail
            obs = env.step(gac.RESET)
            total += 1
            cl = obs.levels_completed
            while cl < target_cl and cl in sols:
                sol_acts = sols[cl]
                # Get click targets for this level's solution
                if has_click:
                    click_tgts = self._compute_clicks(obs.frame[0])
                    avail = [a for a in raw_avail if a != 6] + \
                            [100 + i for i in range(len(click_tgts))]
                for a in sol_acts:
                    do_action(a)
                    if obs.state in (GameState.WIN, GameState.GAME_OVER):
                        return obs.levels_completed
                    if obs.levels_completed > cl:
                        break
                if obs.levels_completed > cl:
                    # Dismiss splash screen if present
                    dismiss_splash()
                    cl = obs.levels_completed
                else:
                    break
            # Update click targets for current level
            if has_click:
                click_tgts = self._compute_clicks(obs.frame[0])
                avail = [a for a in raw_avail if a != 6] + \
                        [100 + i for i in range(len(click_tgts))]
            return cl

        h = self._h

        for episode in range(500):
            if total > budget:
                break

            do_action(0)
            cl = obs.levels_completed
            while cl in sols:
                # Recompute click targets for this level
                if has_click:
                    click_tgts = self._compute_clicks(obs.frame[0])
                    avail = [a for a in raw_avail if a != 6] + \
                            [100 + i for i in range(len(click_tgts))]
                for a in sols[cl]:
                    do_action(a)
                    if obs.state in (GameState.WIN, GameState.GAME_OVER):
                        break
                    if obs.levels_completed > cl:
                        break
                if obs.state == GameState.WIN:
                    return self._res(total, obs.levels_completed, wl, True, t0)
                if obs.state == GameState.GAME_OVER:
                    break
                if obs.levels_completed > cl:
                    dismiss_splash()
                    cl = obs.levels_completed
                else:
                    break
            if obs.state == GameState.GAME_OVER:
                continue
            lc = max(lc, cl)

            # Try toggle solver for click games
            if has_click and cl not in sols and cl not in toggle_tried:
                toggle_tried.add(cl)
                click_tgts = self._compute_clicks(obs.frame[0])
                avail = [a for a in raw_avail if a != 6] + \
                        [100 + i for i in range(len(click_tgts))]

                tsol = self._try_toggle_solve(
                    env, gac, click_tgts, sols, cl)
                if tsol['obs'] is not None:
                    total += tsol['actions_used']
                    obs = tsol['obs']

                # Use effective clicks for BFS (much smaller branching)
                if tsol.get('effective_clicks'):
                    eff_tgts = tsol['effective_clicks']
                    if eff_tgts:  # Only override if non-empty
                        click_tgts = eff_tgts
                        avail = [a for a in raw_avail if a != 6] + \
                                [100 + i for i in range(len(click_tgts))]

                if tsol['solution']:
                    sols[cl] = tsol['solution']
                    if self.verbose:
                        logger.info(f"L{cl} TOGGLE-SOLVED: "
                                    f"{len(tsol['solution'])} clicks")
                    if obs and obs.state == GameState.WIN:
                        return self._res(total, obs.levels_completed, wl, True, t0)
                    if obs and obs.levels_completed > cl:
                        dismiss_splash()
                        lc = obs.levels_completed
                        cl = lc
                    continue
                else:
                    if self.verbose:
                        logger.info(f"L{cl}: toggle failed, BFS fallback")
                    cl = replay_to_level(cl)

            g = graphs.setdefault(cl, StateGraph())

            if has_click:
                new_tgts = self._compute_clicks(obs.frame[0])
                if new_tgts:
                    click_tgts = new_tgts
                    avail = [a for a in raw_avail if a != 6] + \
                            [100 + i for i in range(len(click_tgts))]

            start_h = h(obs.frame[0])

            sol = g.solution(start_h)
            if sol:
                sols[cl] = sol
                if self.verbose:
                    logger.info(f"L{cl} SOLVED: {len(sol)} acts ({g.size} states)")
                continue

            while total < budget:
                cur_h = h(obs.frame[0])
                nxt = g.next_unexplored(cur_h, avail)

                if nxt is None:
                    # Try Mercury reasoning when BFS stalls
                    prev_cl_mercury = cl
                    if self.mercury and self.mercury.available and not sol:
                        mercury_actions = self.mercury.plan_exploration(
                            obs.frame[0], g.size, budget, total)
                        if mercury_actions:
                            if self.verbose:
                                logger.info(f"L{cl}: Mercury suggests {len(mercury_actions)} actions")
                            for ma in mercury_actions[:20]:
                                if total >= budget:
                                    break
                                if isinstance(ma, dict):
                                    if ma.get("type") == "click" and has_click:
                                        x, y = ma.get("x", 0), ma.get("y", 0)
                                        obs = env.step(gac.ACTION6, data={'x': x, 'y': y})
                                        total += 1
                                    elif ma.get("type") == "move":
                                        d = ma.get("dir", 1)
                                        if 1 <= d <= 7:
                                            obs = env.step(AMAP.get(d, gac.ACTION1))
                                            total += 1
                                elif isinstance(ma, int) and 1 <= ma <= 7:
                                    obs = env.step(AMAP.get(ma, gac.ACTION1))
                                    total += 1
                                new_h = h(obs.frame[0])
                                g.add(cur_h, ma if isinstance(ma, int) else 0, new_h)
                                cur_h = new_h
                                if obs.state == GameState.WIN:
                                    return self._res(total, obs.levels_completed, wl, True, t0)
                                if obs.levels_completed > cl:
                                    dismiss_splash()
                                    lc = obs.levels_completed
                                    cl = lc
                                    break
                            if cl > prev_cl_mercury:
                                continue
                    
                    sol = g.solution(start_h)
                    if sol:
                        sols[cl] = sol
                        if self.verbose:
                            logger.info(f"L{cl} SOLVED: {len(sol)} acts "
                                        f"({g.size} states)")
                    elif self.verbose and g.size > 0:
                        logger.info(f"L{cl}: exhausted ({g.size} states)")
                    break

                path, action = nxt
                for a in path:
                    do_action(a)
                    if obs.state in (GameState.WIN, GameState.GAME_OVER):
                        break
                    if obs.levels_completed > cl:
                        break

                if obs.state == GameState.WIN:
                    return self._res(total, obs.levels_completed, wl, True, t0)
                if obs.state == GameState.GAME_OVER:
                    break
                if obs.levels_completed > cl:
                    dismiss_splash()
                    lc = obs.levels_completed
                    sol = g.solution(start_h)
                    if sol:
                        sols[cl] = sol
                    if self.verbose:
                        logger.info(f"L{cl}→L{lc} ({g.size} states)")
                    cl = lc
                    g = graphs.setdefault(cl, StateGraph())
                    if has_click:
                        click_tgts = self._compute_clicks(obs.frame[0])
                        avail = [a for a in raw_avail if a != 6] + \
                                [100 + i for i in range(len(click_tgts))]
                    start_h = h(obs.frame[0])
                    continue

                prev_h = h(obs.frame[0])
                prev_lc = obs.levels_completed
                do_action(action)
                new_h = h(obs.frame[0])
                adv = obs.levels_completed > prev_lc
                g.add(prev_h, action, new_h, adv)

                if obs.state == GameState.WIN:
                    return self._res(total, obs.levels_completed, wl, True, t0)
                if obs.state == GameState.GAME_OVER:
                    break
                if adv:
                    dismiss_splash()
                    lc = obs.levels_completed
                    sol = g.solution(start_h)
                    if sol:
                        sols[cl] = sol
                    if self.verbose:
                        logger.info(f"L{cl}→L{lc} ({g.size} states)")
                    cl = lc
                    g = graphs.setdefault(cl, StateGraph())
                    if has_click:
                        click_tgts = self._compute_clicks(obs.frame[0])
                        avail = [a for a in raw_avail if a != 6] + \
                                [100 + i for i in range(len(click_tgts))]
                    start_h = h(obs.frame[0])

            if self.verbose and episode % 10 == 0:
                parts = []
                for lvl in sorted(set(list(graphs.keys()) + list(sols.keys()))):
                    if lvl in sols:
                        parts.append(f"L{lvl}:✓({len(sols[lvl])})")
                    elif lvl in graphs:
                        parts.append(f"L{lvl}:{graphs[lvl].size}s")
                logger.info(f"Ep{episode} [{total}]: {' '.join(parts)}")

        return self._res(total, lc, wl, False, t0)

    def _try_toggle_solve(self, env, gac, click_tgts, sols, target_level):
        """Solve lights-out puzzles via GF(2) linear algebra.
        
        Multi-life scanning with splash screen handling.
        """
        from arcengine import GameState
        AMAP = {a.value: a for a in gac}

        actions_used = 0
        n = len(click_tgts)
        if self.verbose:
            logger.info(f"Toggle solver: {n} click targets for L{target_level}")
        # Skip toggle solver for too many targets (diminishing returns)
        if n == 0 or n > 120:
            return {'solution': None, 'actions_used': 0, 'obs': None}

        def _replay_to(obs):
            nonlocal actions_used
            obs = env.step(gac.RESET)
            actions_used += 1
            cl = obs.levels_completed
            while cl < target_level and cl in sols:
                sol_tgts = self._compute_clicks(obs.frame[0])
                for a in sols[cl]:
                    if a >= 100:
                        idx = a - 100
                        r, c = sol_tgts[idx] if idx < len(sol_tgts) else (0, 0)
                        obs = env.step(gac.ACTION6, data={'x': c, 'y': r})
                    else:
                        obs = env.step(AMAP[a])
                    actions_used += 1
                    if obs.state in (GameState.WIN, GameState.GAME_OVER):
                        return obs, cl
                    if obs.levels_completed > cl:
                        break
                if obs.levels_completed > cl:
                    # Dismiss splash
                    before = obs.frame[0][:55].copy()
                    obs = env.step(gac.ACTION6, data={'x': 0, 'y': 0})
                    actions_used += 1
                    cl = obs.levels_completed
                else:
                    break
            return obs, cl

        # Phase 1: Discover effective clicks
        effective_indices = []
        toggle_effects = {}
        color_transitions = {}  # Maps color_before → color_after
        scan_pos = 0

        scan_budget = min(n * 3 + 200, 1200)  # Cap scan phase

        for life in range(8):
            if scan_pos >= n or actions_used > scan_budget:
                break

            obs, cl = _replay_to(None)
            if cl != target_level or obs.state == GameState.GAME_OVER:
                continue

            # Recompute click targets for this level's actual frame
            level_tgts = self._compute_clicks(obs.frame[0])
            if life == 0:
                click_tgts = level_tgts
                n = len(click_tgts)

            prev_frame = obs.frame[0][:55].copy()
            clicks_this_life = 0

            while scan_pos < n and clicks_this_life < 28:
                r, c = click_tgts[scan_pos]
                obs = env.step(gac.ACTION6, data={'x': c, 'y': r})
                actions_used += 1
                clicks_this_life += 1

                if obs.state == GameState.GAME_OVER:
                    break
                if obs.levels_completed > target_level:
                    return {'solution': [100 + scan_pos],
                            'actions_used': actions_used, 'obs': obs}

                cur_frame = obs.frame[0][:55]
                changed = np.where(cur_frame != prev_frame)
                if len(changed[0]) > 0:
                    effective_indices.append(scan_pos)
                    toggle_effects[scan_pos] = set(
                        zip(changed[0].tolist(), changed[1].tolist()))
                    # Record color transition (first changed pixel)
                    pr, pc = changed[0][0], changed[1][0]
                    color_before = int(prev_frame[pr, pc])
                    color_after = int(cur_frame[pr, pc])
                    if color_before not in color_transitions:
                        color_transitions[color_before] = color_after

                prev_frame = cur_frame.copy()
                scan_pos += 1

        if self.verbose:
            logger.info(f"Toggle scan: {len(effective_indices)}/{n} effective")
            logger.info(f"Color transitions: {color_transitions}")

        # Multi-click probe: click one effective target again to discover full cycle
        if effective_indices and len(color_transitions) > 0:
            probe_idx = effective_indices[0]
            pr, pc = click_tgts[probe_idx]
            # We need a fresh life with the sprite already clicked once
            obs_probe = env.step(gac.RESET)
            actions_used += 1
            # Click the target once to get to state 2
            obs_probe = env.step(gac.ACTION6, data={'x': pc, 'y': pr})
            actions_used += 1
            frame_after1 = obs_probe.frame[0][:55].copy()
            # Click again to see if there's a 3rd color
            obs_probe = env.step(gac.ACTION6, data={'x': pc, 'y': pr})
            actions_used += 1
            frame_after2 = obs_probe.frame[0][:55].copy()
            # Record new color transitions
            for pixel in toggle_effects.get(probe_idx, []):
                c1 = int(frame_after1[pixel])
                c2 = int(frame_after2[pixel])
                if c1 != c2 and c1 not in color_transitions:
                    color_transitions[c1] = c2
                    if self.verbose:
                        logger.info(f"Probe discovered: {c1} → {c2}")

        # Build effective click targets list
        eff_click_tgts = [click_tgts[i] for i in effective_indices]

        if not effective_indices:
            obs, _ = _replay_to(None)
            return {'solution': None, 'actions_used': actions_used, 'obs': obs,
                    'effective_clicks': eff_click_tgts}

        # Phase 2: Build toggle matrix
        all_pixels = set()
        for pixels in toggle_effects.values():
            all_pixels.update(pixels)

        pixel_list = sorted(all_pixels)
        pixel_idx = {p: i for i, p in enumerate(pixel_list)}
        m = len(pixel_list)
        n_eff = len(effective_indices)

        A = np.zeros((m, n_eff), dtype=np.uint8)
        for j, click_idx in enumerate(effective_indices):
            for pixel in toggle_effects[click_idx]:
                if pixel in pixel_idx:
                    A[pixel_idx[pixel], j] = 1

        # Find unique toggle groups
        unique_patterns = {}
        for i in range(m):
            pattern = tuple(A[i])
            if pattern not in unique_patterns and any(A[i]):
                unique_patterns[pattern] = i

        groups = list(unique_patterns.keys())
        ng = len(groups)

        if self.verbose:
            logger.info(f"Toggle: {n_eff} clicks, {ng} groups, {m} pixels")

        A_red = np.zeros((ng, n_eff), dtype=np.uint8)
        for gi, pattern in enumerate(groups):
            A_red[gi] = np.array(pattern, dtype=np.uint8)

        # Phase 3: Infer target from initial frame pixel colors
        # For each toggle group, determine if it needs to toggle
        # Strategy: read initial colors and compare with likely targets
        obs_for_colors, _ = _replay_to(None)
        if obs_for_colors is None:
            return {'solution': None, 'actions_used': actions_used, 
                    'obs': obs_for_colors, 'effective_clicks': eff_click_tgts}

        init_frame = obs_for_colors.frame[0][:55]
        
        # Get initial color for each toggle group
        group_colors = []
        for gi, pattern in enumerate(groups):
            rep_idx = unique_patterns[pattern]
            pr, pc = pixel_list[rep_idx]
            group_colors.append(int(init_frame[pr, pc]))
        
        # Build color cycle from observed transitions
        # color_transitions: {A: B} means clicking goes A → B
        color_cycle = []
        initial_color = group_colors[0] if group_colors else None
        if initial_color is not None:
            c = initial_color
            color_cycle.append(c)
            for _ in range(10):  # Safety limit
                nxt = color_transitions.get(c)
                if nxt is None or nxt == initial_color:
                    break
                color_cycle.append(nxt)
                c = nxt
        n_colors = len(color_cycle)
        
        # Also determine what color each group toggles TO
        toggle_colors = set()
        for gc in group_colors:
            toggle_colors.add(gc)
        # Include ALL colors from the cycle (initial + destinations)
        for c in color_cycle:
            toggle_colors.add(c)
        for src, dst in color_transitions.items():
            toggle_colors.add(src)
            toggle_colors.add(dst)
        
        # Phase 3a: Visual target inference from static indicator sprites
        # Find non-bg pixels NOT affected by any click → indicator sprites
        all_toggle_pixels = set()
        for pixels in toggle_effects.values():
            all_toggle_pixels.update(pixels)
        
        toggle_group_centers = []
        for gi, pattern in enumerate(groups):
            rep_idx = unique_patterns[pattern]
            toggle_group_centers.append(pixel_list[rep_idx])
        
        vis_result = self._infer_visual_target(
            init_frame, all_toggle_pixels, toggle_group_centers, group_colors,
            toggle_colors)
        
        visual_target = None
        target_info = None
        if vis_result is not None:
            visual_target, target_info = vis_result
        
        # Try visual target solution FIRST (highest confidence)
        if visual_target is not None:
            # For multi-color cycles, compute actual click counts per group
            if n_colors > 2 and target_info is not None:
                click_counts = self._compute_multi_color_clicks(
                    target_info, group_colors, color_cycle)
            else:
                click_counts = None  # Use GF(2) for binary
            
            if click_counts is not None:
                # Multi-color: execute exact click counts per group
                total_clicks = sum(click_counts)
                if total_clicks > 0 and total_clicks <= 56:
                    if self.verbose:
                        logger.info(f"Visual target (multi-color): "
                                    f"{total_clicks} clicks, trying immediately")
                    # Build click sequence with repeats
                    click_seq_full = []
                    for gi in range(ng):
                        if click_counts[gi] > 0:
                            # Find which effective click affects this group
                            click_j = None
                            for j in range(n_eff):
                                if A_red[gi, j]:
                                    click_j = j
                                    break
                            if click_j is not None:
                                idx = effective_indices[click_j]
                                for _ in range(click_counts[gi]):
                                    click_seq_full.append(idx)
                    
                    obs_vis, cl_vis = _replay_to(None)
                    if self.verbose:
                        logger.info(f"Replay to L{target_level}: arrived L{cl_vis}, "
                                    f"state={obs_vis.state}, seq_len={len(click_seq_full)}")
                    if cl_vis == target_level and obs_vis.state != GameState.GAME_OVER:
                        for ci, idx in enumerate(click_seq_full):
                            r, c = click_tgts[idx]
                            obs_vis = env.step(gac.ACTION6, data={'x': c, 'y': r})
                            actions_used += 1
                            if self.verbose:
                                logger.info(f"  MC Click {ci+1}/{len(click_seq_full)}: "
                                            f"idx={idx} (r={r},c={c}) "
                                            f"state={obs_vis.state} "
                                            f"lc={obs_vis.levels_completed}")
                            if obs_vis.state == GameState.WIN:
                                sol = [100 + x for x in click_seq_full]
                                return {'solution': sol,
                                        'actions_used': actions_used, 'obs': obs_vis,
                                        'effective_clicks': eff_click_tgts}
                            if obs_vis.levels_completed > target_level:
                                sol = [100 + x for x in click_seq_full]
                                return {'solution': sol,
                                        'actions_used': actions_used, 'obs': obs_vis,
                                        'effective_clicks': eff_click_tgts}
                            if obs_vis.state == GameState.GAME_OVER:
                                if self.verbose:
                                    logger.info(f"  GAME_OVER after click {ci+1}")
                                break
                        obs = obs_vis
            else:
                # Binary toggle: use GF(2) solve
                sol_vis = self._gf2_solve(A_red, visual_target)
                if sol_vis is not None and np.any(sol_vis):
                    n_vis = int(np.sum(sol_vis))
                    if self.verbose:
                        logger.info(f"Visual target: {n_vis} clicks, trying immediately")
                    click_seq = [effective_indices[j] for j in range(n_eff)
                                 if sol_vis[j]]
                    if click_seq and len(click_seq) <= 28:
                        obs_vis, cl_vis = _replay_to(None)
                        if self.verbose:
                            logger.info(f"Replay to L{target_level}: arrived L{cl_vis}, "
                                        f"state={obs_vis.state}")
                        if cl_vis == target_level and obs_vis.state != GameState.GAME_OVER:
                            for ci, idx in enumerate(click_seq):
                                r, c = click_tgts[idx]
                                obs_vis = env.step(gac.ACTION6, data={'x': c, 'y': r})
                                actions_used += 1
                                if self.verbose:
                                    logger.info(f"  Click {ci+1}/{n_vis}: idx={idx} "
                                                f"(r={r},c={c}) state={obs_vis.state} "
                                                f"lc={obs_vis.levels_completed}")
                                if obs_vis.state == GameState.WIN:
                                    return {'solution': [100 + idx for idx in click_seq],
                                            'actions_used': actions_used, 'obs': obs_vis,
                                            'effective_clicks': eff_click_tgts}
                                if obs_vis.levels_completed > target_level:
                                    return {'solution': [100 + idx for idx in click_seq],
                                            'actions_used': actions_used, 'obs': obs_vis,
                                            'effective_clicks': eff_click_tgts}
                                if obs_vis.state == GameState.GAME_OVER:
                                    if self.verbose:
                                        logger.info(f"  GAME_OVER after click {ci+1}")
                                    break
                            obs = obs_vis
        
        solutions = []
        if len(toggle_colors) == 2:
            colors = sorted(toggle_colors)
            # Target: all groups should be color A
            for target_color in colors:
                b = np.array([1 if gc != target_color else 0 
                              for gc in group_colors], dtype=np.uint8)
                sol = self._gf2_solve(A_red, b)
                if sol is not None and np.any(sol):
                    solutions.append(sol)
        
        # Also try: match the most common initial color
        # (groups that are "different" from majority need toggling)
        from collections import Counter
        color_counts = Counter(group_colors)
        majority_color = color_counts.most_common(1)[0][0]
        b_majority = np.array([1 if gc != majority_color else 0 
                               for gc in group_colors], dtype=np.uint8)
        sol_maj = self._gf2_solve(A_red, b_majority)
        if sol_maj is not None and np.any(sol_maj):
            solutions.append(sol_maj)
        
        # Try inverse: minority groups stay, majority toggles
        b_minority = np.array([1 if gc == majority_color else 0
                               for gc in group_colors], dtype=np.uint8)
        sol_min = self._gf2_solve(A_red, b_minority)
        if sol_min is not None and np.any(sol_min):
            solutions.append(sol_min)
        
        # Exhaustive fallback for small ng
        if ng <= 16 and len(solutions) < 4:
            for target_bits in range(1, 2**ng):
                b = np.array([(target_bits >> i) & 1 for i in range(ng)],
                             dtype=np.uint8)
                sol = self._gf2_solve(A_red, b)
                if sol is not None and np.any(sol):
                    solutions.append(sol)

        # Deduplicate and sort by clicks
        seen_sols = set()
        unique_sols = []
        for s in solutions:
            key = tuple(s)
            if key not in seen_sols:
                seen_sols.add(key)
                unique_sols.append(s)
        solutions = sorted(unique_sols, key=lambda s: int(np.sum(s)))

        if self.verbose:
            min_c = int(np.sum(solutions[0])) if solutions else 0
            logger.info(f"Found {len(solutions)} solutions (min={min_c} clicks)")

        # Phase 4: Try solutions (budget limited)
        max_toggle_budget = n_eff * 50 + 500
        toggle_actions_start = actions_used
        for sol_vec in solutions:
            if actions_used - toggle_actions_start > max_toggle_budget:
                break
            click_seq = [effective_indices[j] for j in range(n_eff)
                         if sol_vec[j]]
            if not click_seq or len(click_seq) > 28:
                continue

            obs, cl = _replay_to(None)
            if cl != target_level or obs.state == GameState.GAME_OVER:
                continue

            for idx in click_seq:
                r, c = click_tgts[idx]
                obs = env.step(gac.ACTION6, data={'x': c, 'y': r})
                actions_used += 1

                if obs.state == GameState.WIN:
                    return {'solution': [100 + idx for idx in click_seq],
                            'actions_used': actions_used, 'obs': obs}
                if obs.levels_completed > target_level:
                    return {'solution': [100 + idx for idx in click_seq],
                            'actions_used': actions_used, 'obs': obs}
                if obs.state == GameState.GAME_OVER:
                    break

        obs2, _ = _replay_to(None)
        return {'solution': None, 'actions_used': actions_used, 'obs': obs2,
                'effective_clicks': eff_click_tgts}

    def _infer_visual_target(self, init_frame, all_toggle_pixels,
                             toggle_group_centers, group_colors,
                             toggle_colors=None):
        """Infer toggle target from static indicator sprites in the frame.
        
        Static non-bg pixel clusters near toggle groups encode the win condition:
        - Indicator center pixel = reference color (one of toggle_colors)
        - Each surrounding pixel: bg means "match reference", colored means "differ"
        """
        bg = int(np.argmax(np.bincount(init_frame.flatten().astype(int))))
        bg_set = {bg, 0}  # 0 = transparent in game engine
        h, w = init_frame.shape
        
        # Find static non-bg pixels (not affected by any toggle click)
        static_px = {}
        for r in range(h):
            for c in range(w):
                if (r, c) not in all_toggle_pixels and int(init_frame[r, c]) not in bg_set:
                    static_px[(r, c)] = int(init_frame[r, c])
        
        if not static_px:
            return None
        
        # Cluster static pixels via flood fill (multi-color adjacency)
        visited = set()
        clusters = []
        for (r, c) in static_px:
            if (r, c) in visited:
                continue
            cluster = []
            queue = deque([(r, c)])
            while queue:
                pr, pc = queue.popleft()
                if (pr, pc) in visited:
                    continue
                visited.add((pr, pc))
                cluster.append((pr, pc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = pr + dr, pc + dc
                    if (nr, nc) in static_px and (nr, nc) not in visited:
                        queue.append((nr, nc))
            clusters.append(cluster)
        
        # Filter for indicator-sized clusters and find center by toggle_colors
        indicators = []
        for cluster in clusters:
            rs = [p[0] for p in cluster]
            cs = [p[1] for p in cluster]
            height = max(rs) - min(rs) + 1
            width = max(cs) - min(cs) + 1
            if 3 <= height <= 8 and 3 <= width <= 8:
                # Find center pixel: prefer pixels matching toggle_colors
                tc_pixels = []
                if toggle_colors:
                    tc_pixels = [(r, c) for r, c in cluster
                                 if int(init_frame[r, c]) in toggle_colors]
                if not tc_pixels:
                    # Fallback: use minority color (center vs edge pixels)
                    color_counts = {}
                    for r, c in cluster:
                        v = int(init_frame[r, c])
                        if v not in bg_set:
                            color_counts[v] = color_counts.get(v, 0) + 1
                    if len(color_counts) > 1:
                        min_color = min(color_counts, key=color_counts.get)
                        tc_pixels = [(r, c) for r, c in cluster
                                     if int(init_frame[r, c]) == min_color]
                if tc_pixels:
                    cr = int(np.mean([p[0] for p in tc_pixels]))
                    cc = int(np.mean([p[1] for p in tc_pixels]))
                    ref_color = int(init_frame[cr, cc])
                    if ref_color not in bg_set:
                        indicators.append((cr, cc, ref_color, min(rs), min(cs)))
                        continue
                # Fallback: use geometric center
                cr = (min(rs) + max(rs)) // 2
                cc = (min(cs) + max(cs)) // 2
                ref_color = int(init_frame[cr, cc])
                if ref_color not in bg_set:
                    indicators.append((cr, cc, ref_color, min(rs), min(cs)))
        
        # Phase 2: Find isolated indicator centers not detected by flood fill
        # Some bsT sprites have transparent (0) gaps separating center from edges
        indicator_positions = set()
        for ind in indicators:
            # Mark all positions near this indicator as covered
            ir, ic = ind[0], ind[1]
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    indicator_positions.add((ir + dr, ic + dc))
        
        # Collect all toggle cycle colors + initial group colors
        all_cycle_colors = set()
        if toggle_colors:
            all_cycle_colors.update(toggle_colors)
        # Also check destination colors from transitions
        for px_pos, px_col in static_px.items():
            if px_col in all_cycle_colors and px_pos not in indicator_positions:
                # Check if this could be an indicator center:
                # Must NOT be a toggle pixel and must be in a 6×6 block area
                r, c = px_pos
                # Verify: is there a "frame" of non-bg pixels around this center?
                # Look for at least 2 other static pixels within 4 display pixels
                nearby_count = 0
                for dr in range(-4, 5):
                    for dc in range(-4, 5):
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in static_px and (nr, nc) != (r, c):
                            nearby_count += 1
                if nearby_count >= 2:
                    indicators.append((r, c, px_col, r - 2, c - 2))
                    for dr in range(-4, 5):
                        for dc in range(-4, 5):
                            indicator_positions.add((r + dr, c + dc))
        
        if not indicators:
            return None
        
        if self.verbose:
            logger.info(f"Found {len(indicators)} indicator sprites")
            for ind in indicators:
                logger.info(f"  Indicator at ({ind[0]},{ind[1]}) ref={ind[2]}")
        
        # For each toggle group, collect constraints from ALL applicable indicators
        # Use offset-based matching: each indicator checks Hkx at ±4 game units (±8 display px)
        ng = len(toggle_group_centers)
        target = np.zeros(ng, dtype=np.uint8)
        target_info = [None] * ng
        matched = 0
        
        # Compute scale from frame dimensions (64 display / camera_width)
        # For toggle groups, representative pixel is top-left. Center = +2 display pixels.
        # Build group center lookup (center = rep + 2 for scale=2 / 3x3 sprites)
        offset = 2  # Center offset for 3x3 sprite at scale 2
        group_center_to_idx = {}
        for gi, (gr, gc) in enumerate(toggle_group_centers):
            cr, cc = gr + offset, gc + offset
            group_center_to_idx[(cr, cc)] = gi
        
        # For each indicator, assign constraints to groups at ±8 display offsets
        constraints = [[] for _ in range(ng)]
        step = 8  # 4 game units × scale 2
        
        for ind in indicators:
            ir, ic, ref_color, top_r, top_c = ind
            
            for spr in range(3):
                for spc in range(3):
                    if spr == 1 and spc == 1:
                        continue  # Skip center
                    
                    # Expected Hkx center position
                    exp_r = ir + (spr - 1) * step
                    exp_c = ic + (spc - 1) * step
                    
                    # Find matching group (exact or nearby)
                    gi = group_center_to_idx.get((exp_r, exp_c))
                    if gi is None:
                        for dr in range(-3, 4):
                            for dc in range(-3, 4):
                                gi = group_center_to_idx.get((exp_r + dr, exp_c + dc))
                                if gi is not None:
                                    break
                            if gi is not None:
                                break
                    
                    if gi is None:
                        continue
                    
                    # Read the indicator pixel at this sprite position
                    pix_r = ir + (spr - 1) * 2
                    pix_c = ic + (spc - 1) * 2
                    
                    if 0 <= pix_r < h and 0 <= pix_c < w:
                        pix_val = int(init_frame[pix_r, pix_c])
                        if pix_val in bg_set:
                            constraints[gi].append((True, ref_color))
                        else:
                            constraints[gi].append((False, ref_color))
        
        # Resolve constraints for each group
        for gi in range(ng):
            if not constraints[gi]:
                continue
            
            if len(constraints[gi]) == 1:
                should_match, ref_color = constraints[gi][0]
                target_info[gi] = (should_match, ref_color)
                if should_match:
                    target[gi] = 1 if group_colors[gi] != ref_color else 0
                else:
                    target[gi] = 1 if group_colors[gi] == ref_color else 0
            else:
                # Multiple constraints: store the first for backward compat,
                # but the multi-color solver will use all constraints
                target_info[gi] = constraints[gi]  # Store full list
                # For binary target: need to toggle if ANY constraint requires it
                needs_toggle = False
                for sm, rc in constraints[gi]:
                    if sm and group_colors[gi] != rc:
                        needs_toggle = True
                    elif not sm and group_colors[gi] == rc:
                        needs_toggle = True
                if needs_toggle:
                    target[gi] = 1
            
            if self.verbose:
                sm0, rc0 = constraints[gi][0]
                logger.info(f"  G{gi}({toggle_group_centers[gi][0]},"
                            f"{toggle_group_centers[gi][1]}) col={group_colors[gi]}: "
                            f"{'==' if sm0 else '!='}{rc0} "
                            f"→ t={target[gi]} "
                            f"({len(constraints[gi])} constraints)")
            matched += 1
        
        if matched == 0:
            return None
        
        if self.verbose:
            logger.info(f"Visual target: {matched}/{ng} groups mapped, "
                        f"{int(np.sum(target))} toggles needed")
        
        return target, target_info

    def _compute_multi_color_clicks(self, target_info, group_colors, color_cycle):
        """Compute click counts for multi-color (3+) toggle puzzles.
        
        Uses target_info per group (single tuple or list of tuples) and color_cycle
        to determine exact clicks needed per group.
        """
        n_colors = len(color_cycle)
        if n_colors < 2:
            return None
        
        color_idx = {c: i for i, c in enumerate(color_cycle)}
        
        clicks = []
        for gi in range(len(group_colors)):
            info = target_info[gi]
            if info is None:
                clicks.append(0)
                continue
            
            initial = group_colors[gi]
            init_idx = color_idx.get(initial, 0)
            
            # Normalize to list of constraints
            if isinstance(info, tuple):
                constraint_list = [info]
            else:
                constraint_list = info
            
            # Find which color in cycle satisfies ALL constraints
            best_clicks = None
            for c_offset in range(n_colors):
                candidate_color = color_cycle[(init_idx + c_offset) % n_colors]
                ok = True
                for should_match, ref_color in constraint_list:
                    if should_match and candidate_color != ref_color:
                        ok = False
                        break
                    if not should_match and candidate_color == ref_color:
                        ok = False
                        break
                if ok:
                    best_clicks = c_offset
                    break
            
            clicks.append(best_clicks if best_clicks is not None else 0)
        
        return clicks

    def _gf2_solve(self, A, b):
        """Solve Ax = b over GF(2)."""
        m, n = A.shape
        M = np.zeros((m, n + 1), dtype=np.uint8)
        M[:, :n] = A.copy()
        M[:, n] = b.copy()

        pivot_cols = []
        row = 0
        for col in range(n):
            found = -1
            for r in range(row, m):
                if M[r, col]:
                    found = r
                    break
            if found == -1:
                continue
            M[[row, found]] = M[[found, row]]
            pivot_cols.append(col)
            for r in range(m):
                if r != row and M[r, col]:
                    M[r] = M[r] ^ M[row]
            row += 1

        for r in range(row, m):
            if M[r, n]:
                return None

        x = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            x[col] = M[i, n]
        return x

    def _compute_clicks(self, frame):
        """Compute click targets via connected component analysis."""
        area = frame[:60]
        bg = int(np.argmax(np.bincount(area.flatten().astype(int))))
        skip = {bg, 0}  # Skip background AND transparent (0)
        h, w = area.shape
        visited = np.zeros((h, w), dtype=bool)
        tgts = []
        seen = set()

        for r in range(h):
            for c in range(w):
                if visited[r, c] or int(area[r, c]) in skip:
                    continue
                color = int(area[r, c])
                stack = [(r, c)]
                pixels = []
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if visited[cr, cc] or int(area[cr, cc]) != color:
                        continue
                    visited[cr, cc] = True
                    pixels.append((cr, cc))
                    stack.extend([(cr+1,cc),(cr-1,cc),(cr,cc+1),(cr,cc-1)])
                if len(pixels) >= 4:
                    ys = [p[0] for p in pixels]
                    xs = [p[1] for p in pixels]
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    key = (cy, cx)
                    if key not in seen:
                        tgts.append(key)
                        seen.add(key)
        return tgts

    def _res(self, total, lc, wl, won, t0):
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
