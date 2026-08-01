#!/usr/bin/env python3
"""
SIMULA-Compound Integration Bridge
===================================

Bridges Google's SIMULA synthetic data generation with OctoTetrahedral's
compound integration system (ngvt_compound_learning, ngvt_compound_orchestrator).

This enables:
1. Data generation as learnable integration experiences
2. Taxonomy design as adaptive workflow paths
3. Quality verification feedback into compound learning engine
4. Cross-model learning on synthetic datasets

Usage:
    from core.simula_compound_bridge import SimulaCompoundBridge
    
    bridge = SimulaCompoundBridge(learning_engine, integration_engine)
    
    # Generate synthetic data with learning tracking
    synthetic_examples = bridge.generate_with_learning(
        domain="arc-puzzle",
        num_examples=100,
        complexity_range=(1, 5)
    )
    
    # Record as learnable workflow
    workflow = bridge.create_integration_workflow(
        name="arc_taxonomy_expansion",
        synthetic_data=synthetic_examples,
        taxonomy=domain_taxonomy
    )
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import json
from datetime import datetime


class TaxonomyLevel(Enum):
    """Hierarchy levels in domain taxonomy."""
    DOMAIN = "domain"          # e.g., "arc-puzzle"
    CATEGORY = "category"      # e.g., "geometric-transform"
    SUBCATEGORY = "subcategory" # e.g., "rotation"
    ATTRIBUTE = "attribute"    # e.g., "angle=90"


@dataclass
class TaxonomyNode:
    """Single node in domain taxonomy tree."""
    id: str
    name: str
    level: TaxonomyLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    coverage_count: int = 0  # How many examples from this node
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'metadata': self.metadata,
            'coverage_count': self.coverage_count,
        }


@dataclass
class MetapromptTemplate:
    """Template for generating varied prompts from taxonomy nodes."""
    id: str
    taxonomy_nodes: List[str]  # IDs of nodes to combine
    prompt_template: str
    variations: int = 5  # How many variations to generate
    complexity_level: int = 1  # 1-5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticDataExample:
    """Single synthetic example with tracking metadata."""
    id: str
    input_data: Any
    output_data: Any
    taxonomy_path: List[str]  # Trace of nodes from this example
    complexity_score: float  # 1.0-5.0
    quality_score: float  # 0.0-1.0 from dual-critic
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DomainTaxonomy:
    """Structured taxonomy of domain (e.g., ARC puzzle space)."""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.nodes: Dict[str, TaxonomyNode] = {}
        self.root_id: Optional[str] = None
        
    def add_node(self, node: TaxonomyNode, parent_id: Optional[str] = None) -> None:
        """Add node to taxonomy tree."""
        self.nodes[node.id] = node
        if parent_id:
            parent = self.nodes.get(parent_id)
            if parent:
                parent.children_ids.append(node.id)
                node.parent_id = parent_id
        else:
            self.root_id = node.id
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get taxonomic path from node to root."""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.insert(0, current.id)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        return path
    
    def sample_nodes_for_coverage(self, num_samples: int) -> List[str]:
        """Sample nodes to maximize taxonomy coverage (prevent mode collapse)."""
        leaf_nodes = [n for n in self.nodes.values() if not n.children_ids]
        if not leaf_nodes:
            return list(self.nodes.keys())[:num_samples]
        
        # Weighted sampling: prioritize under-represented nodes
        total_coverage = sum(n.coverage_count for n in leaf_nodes)
        max_coverage = max((n.coverage_count for n in leaf_nodes), default=0)
        
        weights = []
        for node in leaf_nodes:
            # Lower coverage = higher weight
            weight = (max_coverage + 1) - node.coverage_count
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()
        
        indices = np.random.choice(len(leaf_nodes), size=min(num_samples, len(leaf_nodes)), 
                                   p=weights, replace=True)
        return [leaf_nodes[i].id for i in indices]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain_name': self.domain_name,
            'root_id': self.root_id,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
        }


class SimulaCompoundBridge:
    """
    Bridges synthetic data generation with compound learning system.
    
    Connects:
    - ngvt_compound_learning.CompoundLearningEngine
    - ngvt_compound_orchestrator.AdaptiveWorkflowOrchestrator
    - new synthetic data generation pipeline
    """
    
    def __init__(self, learning_engine=None, integration_engine=None):
        """Initialize bridge (optional: provide compound engines)."""
        self.learning_engine = learning_engine
        self.integration_engine = integration_engine
        self.taxonomies: Dict[str, DomainTaxonomy] = {}
        self.metaprompt_templates: Dict[str, MetapromptTemplate] = {}
        self.generated_examples: List[SyntheticDataExample] = []
        self.dual_critic_history: List[Dict[str, Any]] = []
    
    def create_arc_taxonomy(self) -> DomainTaxonomy:
        """Create taxonomy for ARC puzzle domain."""
        tax = DomainTaxonomy("arc-puzzle")
        
        # Root
        root = TaxonomyNode("arc-root", "ARC Puzzle Space", TaxonomyLevel.DOMAIN)
        tax.add_node(root)
        
        # Categories (common ARC patterns)
        categories = {
            "arc-geom": "Geometric Transforms",
            "arc-color": "Color Transformations",
            "arc-object": "Object Detection & Manipulation",
            "arc-pattern": "Pattern Recognition",
            "arc-symmetry": "Symmetry Operations",
            "arc-fill": "Fill & Connectivity",
        }
        
        for cat_id, cat_name in categories.items():
            cat = TaxonomyNode(cat_id, cat_name, TaxonomyLevel.CATEGORY, parent_id="arc-root")
            tax.add_node(cat, parent_id="arc-root")
            
            # Add subcategories
            if cat_id == "arc-geom":
                for subcat in ["rotation", "scaling", "flipping", "translation"]:
                    sub = TaxonomyNode(f"{cat_id}-{subcat}", subcat.title(), 
                                      TaxonomyLevel.SUBCATEGORY, parent_id=cat_id)
                    tax.add_node(sub, parent_id=cat_id)
            elif cat_id == "arc-color":
                for subcat in ["recolor", "gradient", "blend"]:
                    sub = TaxonomyNode(f"{cat_id}-{subcat}", subcat.title(), 
                                      TaxonomyLevel.SUBCATEGORY, parent_id=cat_id)
                    tax.add_node(sub, parent_id=cat_id)
        
        self.taxonomies["arc"] = tax
        return tax
    
    def generate_metaprompt(self, template: MetapromptTemplate, node_names: Dict[str, str]) -> str:
        """Generate metaprompt from template and node values."""
        prompt = template.prompt_template
        for node_id in template.taxonomy_nodes:
            placeholder = f"{{{{node:{node_id}}}}}"
            value = node_names.get(node_id, node_id)
            prompt = prompt.replace(placeholder, value)
        return prompt
    
    def generate_with_learning(self, 
                              domain: str,
                              num_examples: int = 100,
                              complexity_range: Tuple[int, int] = (1, 5),
                              track_learning: bool = True) -> List[SyntheticDataExample]:
        """
        Generate synthetic examples with compound learning tracking.
        
        Args:
            domain: Domain taxonomy name (e.g., "arc")
            num_examples: How many examples to generate
            complexity_range: (min, max) complexity levels
            track_learning: Record as learning experience if engine available
        
        Returns:
            List of SyntheticDataExample with metadata
        """
        taxonomy = self.taxonomies.get(domain)
        if not taxonomy:
            raise ValueError(f"Taxonomy '{domain}' not found. Create with create_arc_taxonomy().")
        
        examples = []
        
        # Sample nodes to maximize coverage
        sampled_nodes = taxonomy.sample_nodes_for_coverage(num_examples)
        
        for i, node_id in enumerate(sampled_nodes):
            node = taxonomy.nodes[node_id]
            path = taxonomy.get_path_to_root(node_id)
            
            # Complexity: vary across range, with bias toward center
            complexity = np.random.uniform(complexity_range[0], complexity_range[1])
            
            # Create synthetic example (placeholder - real implementation generates actual data)
            example = SyntheticDataExample(
                id=f"{domain}-synthetic-{i}",
                input_data={"node_id": node_id, "complexity": complexity},
                output_data={"placeholder": "real generation TBD"},
                taxonomy_path=path,
                complexity_score=float(complexity),
                quality_score=0.8,  # Placeholder
                metadata={
                    "node_name": node.name,
                    "level": node.level.value,
                }
            )
            
            examples.append(example)
            node.coverage_count += 1
        
        self.generated_examples.extend(examples)
        
        # Record as learning experience if engine available
        if track_learning and self.learning_engine:
            self._record_generation_as_learning(domain, examples, taxonomy)
        
        return examples
    
    def _record_generation_as_learning(self, domain: str, 
                                       examples: List[SyntheticDataExample],
                                       taxonomy: DomainTaxonomy) -> None:
        """Record data generation as learnable integration experience."""
        try:
            from ngvt_compound_learning import LearningExperience
            
            experience = LearningExperience(
                query=f"Generate {len(examples)} synthetic {domain} examples",
                response=f"Generated with taxonomy coverage: {json.dumps({n.id: n.coverage_count for n in taxonomy.nodes.values()})}",
                latency_ms=0.0,  # Placeholder
                success=True,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'domain': domain,
                    'num_examples': len(examples),
                    'avg_complexity': float(np.mean([e.complexity_score for e in examples])),
                    'avg_quality': float(np.mean([e.quality_score for e in examples])),
                    'taxonomy_coverage': len([n for n in taxonomy.nodes.values() if n.coverage_count > 0]),
                }
            )
            
            self.learning_engine.record_experience(experience)
        except ImportError:
            pass  # ngvt_compound_learning not available
    
    def create_integration_workflow(self, 
                                    name: str,
                                    synthetic_data: List[SyntheticDataExample],
                                    taxonomy: Optional[DomainTaxonomy] = None) -> Dict[str, Any]:
        """
        Create an integration workflow from synthetic data generation.
        
        Returns workflow compatible with ngvt_compound_integration_engine.
        """
        try:
            from ngvt_compound_learning import IntegrationWorkflow
            
            workflow = IntegrationWorkflow(
                name=name,
                description=f"Synthetic data generation workflow with {len(synthetic_data)} examples",
                steps=[
                    {
                        'name': 'generate',
                        'input': {'domain': name},
                        'output': {'examples': len(synthetic_data)},
                        'metadata': {
                            'avg_complexity': float(np.mean([e.complexity_score for e in synthetic_data])),
                        }
                    }
                ],
                success_rate=np.mean([e.quality_score for e in synthetic_data]),
                timestamp=datetime.now().isoformat()
            )
            
            return workflow
        except ImportError:
            # Fallback: return plain dict
            return {
                'name': name,
                'num_examples': len(synthetic_data),
                'avg_quality': float(np.mean([e.quality_score for e in synthetic_data])),
            }
    
    def dual_critic_verification(self, example: SyntheticDataExample) -> float:
        """
        Run dual-critic quality verification (SIMULA-inspired).
        
        Two evaluations:
        1. Is this correct?
        2. Is this incorrect?
        
        Returns quality score 0.0-1.0.
        """
        # Placeholder implementation
        # Real version would query two separate critic models
        
        is_correct_score = np.random.uniform(0.6, 1.0)  # Placeholder
        is_incorrect_score = 1.0 - np.random.uniform(0.6, 1.0)  # Inverse
        
        # Quality is agreement between critics
        agreement = (is_correct_score + is_incorrect_score) / 2.0
        
        record = {
            'example_id': example.id,
            'is_correct_score': float(is_correct_score),
            'is_incorrect_score': float(is_incorrect_score),
            'final_quality': float(agreement),
            'timestamp': datetime.now().isoformat(),
        }
        self.dual_critic_history.append(record)
        
        return float(agreement)
    
    def get_taxonomy_coverage_report(self, domain: str) -> Dict[str, Any]:
        """Report on taxonomy coverage to prevent mode collapse."""
        taxonomy = self.taxonomies.get(domain)
        if not taxonomy:
            return {}
        
        all_nodes = taxonomy.nodes.values()
        covered = [n for n in all_nodes if n.coverage_count > 0]
        uncovered = [n for n in all_nodes if n.coverage_count == 0]
        
        return {
            'domain': domain,
            'total_nodes': len(all_nodes),
            'covered_nodes': len(covered),
            'uncovered_nodes': len(uncovered),
            'coverage_percentage': 100.0 * len(covered) / max(len(all_nodes), 1),
            'uncovered_node_ids': [n.id for n in uncovered],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for standalone usage
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SIMULA-Compound Bridge")
    parser.add_argument("--action", default="demo", choices=["demo", "generate", "coverage"])
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--domain", default="arc")
    args = parser.parse_args()
    
    bridge = SimulaCompoundBridge()
    
    if args.action == "demo":
        print("Creating ARC taxonomy...")
        tax = bridge.create_arc_taxonomy()
        print(f"Created taxonomy with {len(tax.nodes)} nodes")
        
        print("\nGenerating synthetic examples with learning...")
        examples = bridge.generate_with_learning(
            domain="arc",
            num_examples=args.num_examples,
            complexity_range=(1, 5)
        )
        
        print(f"Generated {len(examples)} examples")
        for ex in examples[:3]:
            print(f"  - {ex.id}: {ex.metadata}")
        
        print("\nTaxonomy coverage report:")
        report = bridge.get_taxonomy_coverage_report("arc")
        for k, v in report.items():
            print(f"  {k}: {v}")
    
    elif args.action == "generate":
        tax = bridge.create_arc_taxonomy()
        examples = bridge.generate_with_learning("arc", num_examples=args.num_examples)
        print(f"Generated {len(examples)} synthetic examples")
    
    elif args.action == "coverage":
        tax = bridge.create_arc_taxonomy()
        _ = bridge.generate_with_learning("arc", num_examples=100)
        report = bridge.get_taxonomy_coverage_report("arc")
        print(json.dumps(report, indent=2))
