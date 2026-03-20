"""Mercury 2 diffusion LLM integration for ARC-AGI-3 reasoning.

Mercury 2 (Inception Labs) is a diffusion-based language model that generates
tokens in parallel via iterative refinement, achieving 1000+ tok/s with strong
reasoning. Uses an OpenAI-compatible API.

Usage:
    mercury = MercuryReasoner()  # reads MERCURY_API_KEY env var
    actions = mercury.suggest_actions(frame, history, game_info)
"""

import os
import json
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Mercury API config
MERCURY_BASE_URL = os.environ.get(
    "MERCURY_BASE_URL", "https://api.inceptionlabs.ai/v1"
)
MERCURY_API_KEY = os.environ.get("MERCURY_API_KEY", "")
MERCURY_MODEL = os.environ.get("MERCURY_MODEL", "mercury-coder-small")


class MercuryReasoner:
    """Reasoning backend using Mercury 2 diffusion LLM.
    
    Analyzes game frames and action history to suggest next moves
    for levels where pure algorithmic approaches (BFS, toggle solver) fail.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or MERCURY_API_KEY
        self.base_url = base_url or MERCURY_BASE_URL
        self.model = model or MERCURY_MODEL
        self.available = False
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                self.available = True
                logger.info(f"Mercury reasoner initialized (model={self.model})")
            except ImportError:
                logger.warning("openai package not installed, Mercury disabled")
        else:
            logger.debug("No MERCURY_API_KEY set, Mercury reasoning disabled")
    
    def _frame_to_compact(self, frame: np.ndarray, max_rows: int = 55) -> str:
        """Compress a 64×64 frame into a compact text representation.
        
        Uses run-length encoding per row to minimize token usage.
        """
        lines = []
        for r in range(min(max_rows, frame.shape[0])):
            row = frame[r]
            # Run-length encode
            runs = []
            i = 0
            while i < len(row):
                val = int(row[i])
                count = 1
                while i + count < len(row) and int(row[i + count]) == val:
                    count += 1
                if count > 2:
                    runs.append(f"{val}x{count}")
                else:
                    runs.extend([str(val)] * count)
                i += count
            lines.append(",".join(runs))
        return "\n".join(lines)
    
    def _build_system_prompt(self) -> str:
        return """You are an expert ARC-AGI-3 game agent. You analyze pixel frames from 
interactive game environments and determine optimal actions.

Games are played on a 64×64 pixel grid. Available actions:
- ACTION1-4: Directional movement (up/right/down/left)  
- ACTION5: Special action
- ACTION6: Click at coordinates (x, y)
- ACTION7: Alternative action

Your job: analyze the game state, identify patterns/rules, and suggest the best 
sequence of actions to solve the current level. Be concise and actionable.

Output JSON: {"reasoning": "brief analysis", "actions": [{"type": "move", "dir": 1-4} 
or {"type": "click", "x": col, "y": row}], "confidence": 0.0-1.0}"""

    def suggest_actions(self, frame: np.ndarray, 
                        history: list[dict] = None,
                        game_info: dict = None) -> Optional[list[dict]]:
        """Analyze a game frame and suggest next actions.
        
        Args:
            frame: 64×64 numpy array of pixel colors
            history: List of previous {action, result} dicts
            game_info: Optional metadata (game_id, level, actions_remaining)
            
        Returns:
            List of action dicts or None if Mercury unavailable
        """
        if not self.available or not self.client:
            return None
        
        compact_frame = self._frame_to_compact(frame)
        
        user_msg = f"Current frame (RLE-encoded rows):\n{compact_frame}\n\n"
        
        if game_info:
            user_msg += f"Game: {game_info.get('game_id', '?')}, "
            user_msg += f"Level: {game_info.get('level', '?')}, "
            user_msg += f"Actions remaining: {game_info.get('budget', '?')}\n\n"
        
        if history and len(history) > 0:
            recent = history[-10:]  # Last 10 actions
            user_msg += "Recent actions:\n"
            for h in recent:
                user_msg += f"  {h.get('action', '?')} → changed={h.get('changed', 0)} pixels\n"
            user_msg += "\n"
        
        user_msg += "Analyze the game state and suggest the best next actions to solve this level."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            if self.available and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Mercury reasoning: {result.get('reasoning', '')[:100]}")
            
            return result.get("actions")
            
        except Exception as e:
            logger.warning(f"Mercury API call failed: {e}")
            return None
    
    def infer_game_rules(self, observations: list[dict]) -> Optional[str]:
        """Given a sequence of (action, frame_before, frame_after) observations,
        infer the game's rules and mechanics.
        
        Returns a natural language description of inferred rules, or None.
        """
        if not self.available or not self.client:
            return None
        
        obs_text = ""
        for i, obs in enumerate(observations[:20]):
            action = obs.get("action", "?")
            changed = obs.get("changed_pixels", 0)
            obs_text += f"Step {i}: action={action}, {changed} pixels changed\n"
            if "diff_summary" in obs:
                obs_text += f"  Changes: {obs['diff_summary']}\n"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at inferring game rules "
                     "from observations. Analyze the action-effect patterns and describe the "
                     "game mechanics concisely."},
                    {"role": "user", "content": f"Observations:\n{obs_text}\n\n"
                     "What are the rules of this game? How does the player win?"},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Mercury rule inference failed: {e}")
            return None

    def plan_exploration(self, frame: np.ndarray,
                         visited_states: int,
                         total_budget: int,
                         actions_used: int) -> Optional[list[int]]:
        """Suggest an exploration strategy when BFS is too expensive.
        
        Returns a list of action values to try, or None.
        """
        if not self.available or not self.client:
            return None
        
        compact = self._frame_to_compact(frame, max_rows=32)
        
        prompt = (
            f"Game frame (top 32 rows, RLE):\n{compact}\n\n"
            f"BFS explored {visited_states} states. Budget: {actions_used}/{total_budget}.\n"
            f"Suggest a focused exploration sequence (list of action numbers 1-7).\n"
            f"Output JSON: {{\"actions\": [1,2,3,...], \"reasoning\": \"...\"}}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ARC-AGI-3 exploration planner. "
                     "Suggest efficient action sequences to explore new game states."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("actions")
        except Exception as e:
            logger.warning(f"Mercury exploration planning failed: {e}")
            return None
