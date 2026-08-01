"""
VortexDisCode Adapter for OctoAGI Integration
==============================================

Wraps VortexDisCode NVIDIA NIM-powered code generation for use as
the 9th "CodeGen" limb in OctoAGI's compound braided architecture.

Key Features:
- NVIDIA NIM API integration (Llama 3.1 70B, Mistral, CodeLlama)
- Torus geometry code mapping (semantic embeddings)
- Compatible with CompoundBraid cross-attention
- Integrated with CognitiveCohesionBraid skills

Usage:
    from core.vortexdiscode_adapter import VortexDisCodeAdapter
    
    adapter = VortexDisCodeAdapter()
    code = adapter.generate_code("create authentication function", limb_context={})
    result = adapter.debug_code(code, error="IndentationError")
"""

import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import torch

# Add VortexNANO_test and nvidia-nim-setup to path
VORTEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'VortexNANO_test')
NIM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'nvidia-nim-setup')
sys.path.insert(0, VORTEX_PATH)
sys.path.insert(0, NIM_CONFIG_PATH)

try:
    from vortexdiscode_integrated import VortexDisCodeIntegrated, VortexConfig, ModelType
    from vortexdiscode_torus_core import VortexTorusCore, TorusCodeMapping
    import nim_config  # Verify NIM config accessible
    VORTEX_AVAILABLE = True

    # Check if API key is configured
    if "PLACEHOLDER" in getattr(nim_config, 'NVIDIA_API_KEY', 'PLACEHOLDER'):
        print("⚠️  NVIDIA API key not configured - VortexDisCode will run in demo mode")
        print("   Set NVIDIA_API_KEY environment variable to enable full functionality")
        API_KEY_CONFIGURED = False
    else:
        print("✅ NVIDIA API key configured")
        API_KEY_CONFIGURED = True
except ImportError as e:
    print(f"Warning: VortexDisCode not available: {e}")
    VORTEX_AVAILABLE = False
    API_KEY_CONFIGURED = False

    class ModelType(Enum):
        LLAMA_70B = "demo-llama-70b"
        LLAMA_8B = "demo-llama-8b"

    @dataclass
    class VortexConfig:
        cache_enabled: bool = True
        cache_dir: str = ".vortex_cache"
        temperature: float = 0.7
        max_tokens: int = 2048
        default_model: ModelType = ModelType.LLAMA_70B

    class VortexDisCodeIntegrated:
        def __init__(self, config: VortexConfig):
            self.config = config

        def _unavailable(self) -> None:
            raise RuntimeError("Code generation requires optional VortexDisCode dependencies")

        def generate_code(self, *args: Any, **kwargs: Any) -> str:
            self._unavailable()
            return ""

        def debug_code(self, *args: Any, **kwargs: Any) -> str:
            self._unavailable()
            return ""

        def refactor_code(self, *args: Any, **kwargs: Any) -> str:
            self._unavailable()
            return ""

        def optimize_code(self, *args: Any, **kwargs: Any) -> str:
            self._unavailable()
            return ""

        def explain_code(self, *args: Any, **kwargs: Any) -> str:
            self._unavailable()
            return ""

    @dataclass
    class _DemoTorusPosition:
        u: float
        v: float
        R: float
        r: float

        def to_cartesian(self) -> Tuple[float, float, float]:
            x = (self.R + self.r * np.cos(self.v)) * np.cos(self.u)
            y = (self.R + self.r * np.cos(self.v)) * np.sin(self.u)
            z = self.r * np.sin(self.v)
            return float(x), float(y), float(z)

        def geodesic_distance(self, other: Any) -> float:
            du = abs(self.u - other.u)
            dv = abs(self.v - other.v)
            du = min(du, (2 * np.pi) - du)
            dv = min(dv, (2 * np.pi) - dv)
            major = (self.R + self.r * np.cos((self.v + other.v) / 2.0)) * du
            minor = self.r * dv
            return float(np.hypot(major, minor))

    @dataclass
    class TorusCodeMapping:
        file_path: str
        torus_position: _DemoTorusPosition
        content_hash: str
        semantic_embedding: Optional[np.ndarray] = None
        last_updated: float = 0.0

    class VortexTorusCore:
        def __init__(self, config: Optional[VortexConfig] = None):
            self.config = config or VortexConfig()
            self.R = 3.0
            self.r = 1.0
            self.code_mappings: Dict[str, TorusCodeMapping] = {}
            self.embedding_cache: Dict[str, np.ndarray] = {}
            self.embed_dim = 64
            self._u_projection = np.sin(np.linspace(0.5, 3.5 * np.pi, self.embed_dim, dtype=np.float32))
            self._v_projection = np.cos(np.linspace(1.0, 4.0 * np.pi, self.embed_dim, dtype=np.float32))

        def _compute_semantic_embedding(self, content: str) -> np.ndarray:
            cache_key = hashlib.sha256(content.encode('utf-8', errors='ignore')).hexdigest()
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]

            vector = np.zeros(self.embed_dim, dtype=np.float32)
            tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,63}", content.lower())
            for token in tokens:
                digest = hashlib.sha256(token.encode('utf-8')).digest()
                idx = digest[0] % self.embed_dim
                sign = 1.0 if digest[1] % 2 == 0 else -1.0
                vector[idx] += sign * (1.0 + (digest[2] / 255.0))

            scalar_features = np.array([
                len(content) / 4000.0,
                content.count('\n') / 200.0,
                content.count('def ') / 16.0,
                content.count('class ') / 8.0,
                content.count('import ') / 16.0,
                content.count('return ') / 16.0,
                content.count('async ') / 8.0,
                content.count('torch') / 8.0,
            ], dtype=np.float32)
            vector[: len(scalar_features)] += scalar_features

            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                vector[0] = 1.0
                norm = 1.0
            embedding = vector / norm
            self.embedding_cache[cache_key] = embedding
            return embedding

        def _embedding_to_position(self, embedding: np.ndarray) -> _DemoTorusPosition:
            u_signal = float(np.tanh(np.dot(embedding, self._u_projection)))
            v_signal = float(np.tanh(np.dot(embedding, self._v_projection)))
            u = ((u_signal + 1.0) / 2.0) * 2 * np.pi
            v = ((v_signal + 1.0) / 2.0) * 2 * np.pi
            return _DemoTorusPosition(u, v, self.R, self.r)

        def map_code_to_torus(self, file_path: str, content: str) -> _DemoTorusPosition:
            content_hash = hashlib.sha256(content.encode('utf-8', errors='ignore')).hexdigest()
            cached = self.code_mappings.get(file_path)
            if cached and cached.content_hash == content_hash:
                return cached.torus_position

            enriched_content = f"{file_path}\n{content}"
            embedding = self._compute_semantic_embedding(enriched_content)
            position = self._embedding_to_position(embedding)
            self.code_mappings[file_path] = TorusCodeMapping(
                file_path=file_path,
                torus_position=position,
                content_hash=content_hash,
                semantic_embedding=embedding,
                last_updated=time.time(),
            )
            return position


@dataclass
class CodeGenContext:
    """Context from other OctoAGI limbs to inform code generation"""
    spatial_torus_position: Optional[Any] = None  # TorusPosition if available
    memory_patterns: Optional[List[str]] = None   # Past successful code patterns
    reasoning_constraints: Optional[List[str]] = None  # Logical constraints
    metacog_critique: Optional[str] = None  # Self-critique feedback
    coupling_strength: float = 0.15  # Current coupling (0.15-0.95)
    phase: str = "MYRIADPLEXITY"  # MYRIADPLEXITY, COMPOUNDING, TRANSCENDPLEXITY


class VortexDisCodeAdapter:
    """
    Adapter layer between OctoAGI and VortexDisCode.
    
    Provides:
    - Code generation with limb context awareness
    - Torus-based semantic code mapping
    - Coupling-aware quality control
    - Phase transition behavior
    """
    
    def __init__(self, enable_torus: bool = True, cache_dir: str = ".vortex_cache"):
        """
        Initialize VortexDisCode adapter.
        
        Args:
            enable_torus: Enable torus geometry code mapping
            cache_dir: Directory for caching responses and embeddings
        """
        config = VortexConfig(
            cache_enabled=True,
            cache_dir=cache_dir,
            temperature=0.7,
            max_tokens=2048,
            default_model=ModelType.LLAMA_70B,
        )
        self.vortex = VortexDisCodeIntegrated(config)
        self.codegen_available = VORTEX_AVAILABLE

        if not self.codegen_available:
            print("⚠️  VortexDisCode code generation unavailable; torus navigation demo mode enabled")

        # Initialize torus core if enabled
        self.enable_torus = enable_torus
        if enable_torus:
            try:
                self.torus_core = VortexTorusCore(config)
                print("✅ VortexDisCode Torus Core initialized")
            except Exception as e:
                print(f"⚠️  Torus core disabled: {e}")
                self.enable_torus = False
                self.torus_core = None
        else:
            self.torus_core = None
        
        # Track generation history for coupling amplification
        self.generation_count = 0
        self.success_count = 0
        self.codebase_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.code_file_extensions = ('.py',)
        self._torus_map_limit = 256
        self._torus_index_ready = False
        
        print(f"🌀 VortexDisCode Adapter initialized (torus={'enabled' if enable_torus else 'disabled'})")

    def _ensure_codegen_available(self) -> None:
        if not self.codegen_available:
            raise RuntimeError("VortexDisCode code generation is unavailable; torus navigation still works in demo mode")
    
    def _adjust_for_coupling(self, base_temperature: float, coupling: float) -> float:
        """
        Adjust temperature based on coupling strength.
        
        Higher coupling → lower temperature (more focused, production-ready)
        Lower coupling → higher temperature (more creative, exploratory)
        """
        # Coupling range: 0.15 (MYRIAD) → 0.95 (TRANSCEND)
        # Temperature range: 0.9 (creative) → 0.3 (focused)
        temp = 0.9 - (coupling - 0.15) * 0.75
        return max(0.3, min(0.9, temp))
    
    def _select_model_for_task(self, task_complexity: str, coupling: float) -> ModelType:
        """
        Select appropriate NVIDIA NIM model based on task and coupling.
        
        Simple + low coupling → Llama 8B (fast, creative)
        Complex + high coupling → Llama 70B (quality, production)
        """
        if coupling > 0.7:
            # TRANSCENDPLEXITY: Always use best model
            return ModelType.LLAMA_70B
        elif task_complexity == "simple" and coupling < 0.4:
            # MYRIADPLEXITY simple tasks: use fast model
            return ModelType.LLAMA_8B
        else:
            # Default: balanced
            return ModelType.LLAMA_70B
    
    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        task_complexity: str = "medium",
        limb_context: Optional[CodeGenContext] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate code with limb context awareness.
        
        Args:
            prompt: What to generate
            language: Programming language
            task_complexity: "simple", "medium", "complex"
            limb_context: Context from other OctoAGI limbs
        
        Returns:
            (generated_code, metadata) where metadata includes:
                - model_used: Which NVIDIA NIM model
                - torus_position: Position on code torus if available
                - temperature: Temperature used
                - coupling: Current coupling strength
                - phase: Current phase transition
        """
        self._ensure_codegen_available()
        ctx = limb_context or CodeGenContext()
        
        # Adjust temperature based on coupling
        temp = self._adjust_for_coupling(0.7, ctx.coupling_strength)
        model = self._select_model_for_task(task_complexity, ctx.coupling_strength)
        
        # Build enhanced prompt with limb context
        enhanced_prompt = self._build_enhanced_prompt(prompt, ctx)
        
        # Temporarily override config
        original_temp = self.vortex.config.temperature
        original_model = self.vortex.config.default_model
        self.vortex.config.temperature = temp
        self.vortex.config.default_model = model
        
        try:
            # Generate code
            code = self.vortex.generate_code(enhanced_prompt, language, streaming=False)
            
            # Map to torus if enabled
            torus_pos = None
            if self.enable_torus and self.torus_core and code:
                try:
                    torus_pos = self.torus_core.map_code_to_torus(
                        f"generated_{self.generation_count}.{language}",
                        code
                    )
                except Exception as e:
                    print(f"Torus mapping failed: {e}")
            
            # Track for coupling amplification
            self.generation_count += 1
            
            # Build metadata
            metadata = {
                "model_used": model.value,
                "torus_position": torus_pos,
                "temperature": temp,
                "coupling": ctx.coupling_strength,
                "phase": ctx.phase,
                "generation_count": self.generation_count,
                "limb_context_used": bool(limb_context),
            }
            
            return code, metadata
            
        finally:
            # Restore config
            self.vortex.config.temperature = original_temp
            self.vortex.config.default_model = original_model
    
    def debug_code(
        self,
        code: str,
        error: Optional[str] = None,
        limb_context: Optional[CodeGenContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Debug code with metacognition feedback.
        
        If limb_context has metacog_critique, incorporate it into debug prompt.
        """
        self._ensure_codegen_available()
        ctx = limb_context or CodeGenContext()
        
        # Build debug prompt with metacognition critique
        debug_prompt = code
        if ctx.metacog_critique:
            debug_prompt += f"\n\n# MetaCognition Critique:\n# {ctx.metacog_critique}"
        
        result = self.vortex.debug_code(debug_prompt, error)
        
        metadata = {
            "model_used": self.vortex.config.default_model.value,
            "coupling": ctx.coupling_strength,
            "metacog_used": bool(ctx.metacog_critique),
        }
        
        return result, metadata
    
    def refactor_code(
        self,
        code: str,
        instructions: Optional[str] = None,
        limb_context: Optional[CodeGenContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Refactor code with memory patterns and reasoning constraints.
        """
        self._ensure_codegen_available()
        ctx = limb_context or CodeGenContext()
        
        # Enhance instructions with limb context
        enhanced_instructions = instructions or "Improve code quality"
        if ctx.memory_patterns:
            enhanced_instructions += f"\n\nPast successful patterns: {', '.join(ctx.memory_patterns[:3])}"
        if ctx.reasoning_constraints:
            enhanced_instructions += f"\n\nConstraints: {'; '.join(ctx.reasoning_constraints)}"
        
        result = self.vortex.refactor_code(code, enhanced_instructions)
        
        metadata = {
            "model_used": self.vortex.config.default_model.value,
            "coupling": ctx.coupling_strength,
            "memory_patterns_count": len(ctx.memory_patterns) if ctx.memory_patterns else 0,
            "constraints_count": len(ctx.reasoning_constraints) if ctx.reasoning_constraints else 0,
        }
        
        return result, metadata
    
    def optimize_code(
        self,
        code: str,
        target: str = "performance",
        limb_context: Optional[CodeGenContext] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimize code (performance, readability, size)"""
        self._ensure_codegen_available()
        ctx = limb_context or CodeGenContext()
        
        result = self.vortex.optimize_code(code, target)
        
        metadata = {
            "model_used": self.vortex.config.default_model.value,
            "coupling": ctx.coupling_strength,
            "optimization_target": target,
        }
        
        return result, metadata
    
    def explain_code(self, code: str) -> Tuple[str, Dict[str, Any]]:
        """Explain code in detail"""
        self._ensure_codegen_available()
        result = self.vortex.explain_code(code)
        
        metadata = {
            "model_used": self.vortex.config.default_model.value,
        }
        
        return result, metadata
    
    def _build_enhanced_prompt(self, prompt: str, ctx: CodeGenContext) -> str:
        """
        Build enhanced prompt incorporating limb context.
        
        This is where the compound braid integration happens:
        - Spatial torus position suggests code structure patterns
        - Memory patterns inform stylistic choices
        - Reasoning constraints enforce requirements
        - MetaCognition critique guides improvements
        """
        enhanced = prompt
        
        # Add spatial context (torus position)
        if ctx.spatial_torus_position:
            enhanced += f"\n\n[Spatial Context: Code topology position suggests related patterns]"
        
        # Add memory patterns
        if ctx.memory_patterns:
            enhanced += f"\n\n[Memory Patterns: Follow these proven patterns: {', '.join(ctx.memory_patterns[:3])}]"
        
        # Add reasoning constraints
        if ctx.reasoning_constraints:
            enhanced += f"\n\n[Reasoning Constraints: Must satisfy: {'; '.join(ctx.reasoning_constraints)}]"
        
        # Add metacognition critique
        if ctx.metacog_critique:
            enhanced += f"\n\n[MetaCognition: {ctx.metacog_critique}]"
        
        # Add phase-specific guidance
        if ctx.phase == "MYRIADPLEXITY":
            enhanced += "\n\n[Explore creative, diverse solutions]"
        elif ctx.phase == "COMPOUNDING":
            enhanced += "\n\n[Combine best patterns, balance creativity and correctness]"
        elif ctx.phase == "TRANSCENDPLEXITY":
            enhanced += "\n\n[Generate production-ready, optimal solution]"
        
        return enhanced
    
    def mark_success(self):
        """Mark a code generation as successful (for coupling amplification)"""
        self.success_count += 1
    
    def get_success_rate(self) -> float:
        """Get success rate (for coupling calculation)"""
        if self.generation_count == 0:
            return 0.0
        return self.success_count / self.generation_count

    def _iter_codebase_files(
        self,
        root_dir: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> Iterable[str]:
        """Yield code files suitable for torus mapping."""
        search_root = os.path.abspath(root_dir or self.codebase_root)
        max_items = max_files if max_files is not None else self._torus_map_limit
        skipped_dirs = {
            '.git',
            '.venv',
            'venv',
            '__pycache__',
            'node_modules',
            '.pytest_cache',
        }

        yielded = 0
        for current_root, dirnames, filenames in os.walk(search_root):
            dirnames[:] = [
                dirname for dirname in dirnames
                if dirname not in skipped_dirs and not dirname.startswith('.')
            ]
            for filename in sorted(filenames):
                if not filename.endswith(self.code_file_extensions):
                    continue
                file_path = os.path.join(current_root, filename)
                if os.path.getsize(file_path) > 256_000:
                    continue
                yield file_path
                yielded += 1
                if yielded >= max_items:
                    return

    def _get_coupling_search_profile(self, coupling_strength: float) -> Dict[str, Any]:
        """Select torus radius from coupling strength."""
        if coupling_strength < 0.3:
            return {"phase": "MYRIADPLEXITY", "radius": 2.4}
        if coupling_strength < 0.7:
            return {"phase": "COMPOUNDING", "radius": 1.4}
        return {"phase": "TRANSCENDPLEXITY", "radius": 0.8}

    def map_codebase_to_torus(
        self,
        file_paths: Optional[List[str]] = None,
        root_dir: Optional[str] = None,
        max_files: Optional[int] = None,
        force: bool = False,
    ) -> List[Tuple[str, Any]]:
        """Map repository files into torus space for semantic navigation."""
        if not self.enable_torus or not self.torus_core:
            return []

        mapped_positions: List[Tuple[str, Any]] = []
        candidate_files = file_paths or list(self._iter_codebase_files(root_dir=root_dir, max_files=max_files))

        for file_path in candidate_files:
            normalized_path = os.path.abspath(file_path)
            if not force and normalized_path in self.torus_core.code_mappings:
                mapped_positions.append(
                    (normalized_path, self.torus_core.code_mappings[normalized_path].torus_position)
                )
                continue
            try:
                with open(normalized_path, 'r', encoding='utf-8', errors='ignore') as handle:
                    content = handle.read()
                if not content.strip():
                    continue
                enriched_content = f"# file: {os.path.relpath(normalized_path, self.codebase_root)}\n{content}"
                position = self.torus_core.map_code_to_torus(normalized_path, enriched_content)
                mapped_positions.append((normalized_path, position))
            except OSError as e:
                print(f"Skipping torus map for {normalized_path}: {e}")

        if mapped_positions:
            self._torus_index_ready = True

        return mapped_positions

    def _build_query_position(self, query: str) -> Any:
        """Project a natural-language query into torus space."""
        query_embedding = self.torus_core._compute_semantic_embedding(query)
        if hasattr(self.torus_core, '_embedding_to_position'):
            return self.torus_core._embedding_to_position(query_embedding)

        u = (float(query_embedding[0]) % 1.0) * 2 * np.pi
        v = (float(query_embedding[1]) % 1.0) * 2 * np.pi

        mapping_cls = type(next(iter(self.torus_core.code_mappings.values())).torus_position)
        return mapping_cls(u, v, self.torus_core.R, self.torus_core.r)

    def search_related_code(
        self,
        query: str,
        top_k: int = 5,
        coupling_strength: float = 0.5,
        file_paths: Optional[List[str]] = None,
    ) -> List[Tuple[str, float, Any]]:
        """
        Search for related code using torus proximity.

        Returns list of (file_path, distance, torus_position)
        """
        if not self.enable_torus or not self.torus_core:
            return []

        try:
            if file_paths:
                self.map_codebase_to_torus(file_paths=file_paths)
            elif not self.torus_core.code_mappings and not self._torus_index_ready:
                self.map_codebase_to_torus(max_files=self._torus_map_limit)

            if not self.torus_core.code_mappings:
                return []

            query_position = self._build_query_position(query)
            radius_profile = self._get_coupling_search_profile(coupling_strength)
            radius = radius_profile['radius']

            matches: List[Tuple[str, float, Any]] = []
            for file_path, mapping in self.torus_core.code_mappings.items():
                distance = float(query_position.geodesic_distance(mapping.torus_position))
                if distance <= radius:
                    matches.append((file_path, distance, mapping.torus_position))

            matches.sort(key=lambda item: item[1])
            return matches[:top_k]
        except Exception as e:
            print(f"Code search failed: {e}")
            return []

    def semantic_code_navigation(
        self,
        query: str,
        coupling_strength: float = 0.5,
        top_k: int = 5,
        file_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return torus-aware semantic navigation metadata for a query."""
        profile = self._get_coupling_search_profile(coupling_strength)
        results = self.search_related_code(
            query,
            top_k=top_k,
            coupling_strength=coupling_strength,
            file_paths=file_paths,
        )

        return {
            "query": query,
            "coupling_strength": coupling_strength,
            "phase": profile['phase'],
            "search_radius": profile['radius'],
            "mapped_files": len(self.torus_core.code_mappings) if self.torus_core else 0,
            "results": [
                {
                    "file_path": file_path,
                    "distance": distance,
                    "torus_position": position,
                    "u": getattr(position, 'u', None),
                    "v": getattr(position, 'v', None),
                }
                for file_path, distance, position in results
            ],
        }


def semantic_code_navigation(
    query: str,
    coupling_strength: float = 0.5,
    top_k: int = 5,
    file_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience entry point for torus-based semantic code navigation."""
    adapter = VortexDisCodeAdapter(enable_torus=True)
    return adapter.semantic_code_navigation(
        query,
        coupling_strength=coupling_strength,
        top_k=top_k,
        file_paths=file_paths,
    )


# Convenience function for testing
def test_adapter():
    """Test VortexDisCode adapter"""
    adapter = VortexDisCodeAdapter()
    
    print("\n=== Test 1: Simple generation (MYRIADPLEXITY) ===")
    ctx = CodeGenContext(coupling_strength=0.15, phase="MYRIADPLEXITY")
    code, meta = adapter.generate_code(
        "create a function to calculate fibonacci numbers",
        limb_context=ctx
    )
    print(f"Model: {meta['model_used']}, Temp: {meta['temperature']:.2f}")
    print(code)
    
    print("\n=== Test 2: Production generation (TRANSCENDPLEXITY) ===")
    ctx = CodeGenContext(
        coupling_strength=0.85,
        phase="TRANSCENDPLEXITY",
        reasoning_constraints=["Must handle n=0 and n=1", "Must be O(n) time complexity"],
        memory_patterns=["Use memoization", "Include type hints"]
    )
    code, meta = adapter.generate_code(
        "create a function to calculate fibonacci numbers",
        limb_context=ctx
    )
    print(f"Model: {meta['model_used']}, Temp: {meta['temperature']:.2f}")
    print(code)


if __name__ == "__main__":
    test_adapter()
