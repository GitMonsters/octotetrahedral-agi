"""
RustyWorm <-> OCTO Bridge
Provides FFI interface for Rust to call OCTO's RNA Editing Layer

This module is called by RustyWorm via PyO3 to get intelligent routing
decisions based on OCTO's octopus-inspired RNA editing mechanism.
"""
import torch
import sys
import os

# Add OCTO path
sys.path.insert(0, '/home/worm/octotetrahedral-agi')

from config import get_config
from adaptation.rna_editing import RNAEditingLayer


class RustyWormBridge:
    """Bridge for RustyWorm Rust FFI calls to OCTO's RNA Editing Layer."""
    
    def __init__(self, hidden_dim: int = 256):
        """
        Initialize the bridge with RNA Editing Layer.
        
        Args:
            hidden_dim: Embedding dimension (must match OCTO config)
        """
        self.config = get_config()
        self.hidden_dim = hidden_dim
        
        # Initialize RNA Editing Layer
        self.rna_editing = RNAEditingLayer(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_pathways=3,  # perception, reasoning, action
            temperature_init=1.0,
            temperature_min=0.1,
            temperature_max=5.0
        )
        self.rna_editing.eval()
        
        # Force CPU for stability (GPU has ROCm kernel issues)
        # Can enable GPU later once ROCm is properly configured
        self.device = 'cpu'
        self.rna_editing.to(self.device)
        
        print(f"[OCTO Bridge] Initialized on {self.device}")
    
    def analyze(self, input_embedding: list) -> dict:
        """
        Analyze input and return RNA editing parameters.
        Called from Rust via PyO3.
        
        Args:
            input_embedding: List of floats [hidden_dim]
            
        Returns:
            Dictionary with routing parameters:
                - temperature: float (0.1 - 5.0)
                - confidence: float (0.0 - 1.0)
                - head_gates: list of 8 floats (0.0 - 1.0)
                - pathway_weights: list of 3 floats summing to 1.0
        """
        x = torch.tensor(input_embedding, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, hidden_dim]
        
        with torch.no_grad():
            result = self.rna_editing(x, return_diagnostics=True)
        
        return {
            'temperature': float(result['temperature'].cpu().item()),
            'confidence': float(result['confidence'].cpu().item()),
            'head_gates': result['head_gates'].cpu().squeeze(0).tolist(),
            'pathway_weights': result['pathway_weights'].cpu().squeeze(0).tolist(),
        }
    
    def get_routing_decision(
        self, 
        input_embedding: list, 
        confidence_threshold: float = 0.65,
        temperature_threshold: float = 1.8
    ) -> dict:
        """
        Get routing decision: System 1 (fast) or System 2 (deep).
        
        Args:
            input_embedding: List of floats [hidden_dim]
            confidence_threshold: Min confidence for System 1
            temperature_threshold: Max temperature for System 1
            
        Returns:
            Dictionary with:
                - route: 'system1' or 'system2'
                - primary_pathway: 0 (perception), 1 (reasoning), 2 (action)
                - All parameters from analyze()
        """
        params = self.analyze(input_embedding)
        
        # High confidence + low temperature = System 1 (fast path)
        # Low confidence + high temperature = System 2 (deep reasoning)
        use_system1 = (
            params['confidence'] > confidence_threshold and 
            params['temperature'] < temperature_threshold
        )
        
        params['route'] = 'system1' if use_system1 else 'system2'
        params['primary_pathway'] = params['pathway_weights'].index(
            max(params['pathway_weights'])
        )
        
        return params
    
    def get_pathway_name(self, index: int) -> str:
        """Get human-readable pathway name."""
        names = ['perception', 'reasoning', 'action']
        return names[index] if 0 <= index < len(names) else 'unknown'
    
    def train_step(
        self, 
        input_embedding: list, 
        target_pathway: int,
        target_confidence: float,
        learning_rate: float = 0.01
    ) -> dict:
        """
        Train the RNA layer with a single example.
        
        Args:
            input_embedding: List of floats [hidden_dim]
            target_pathway: Desired dominant pathway (0=perception, 1=reasoning, 2=action)
            target_confidence: Desired confidence level (0.0 - 1.0)
            learning_rate: Learning rate for this step
            
        Returns:
            Dictionary with loss values
        """
        self.rna_editing.train()
        
        x = torch.tensor(input_embedding, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass
        result = self.rna_editing(x, return_diagnostics=True)
        
        # Create target distributions
        pathway_target = torch.zeros(3, device=self.device)
        pathway_target[target_pathway] = 0.6
        pathway_target[(target_pathway + 1) % 3] = 0.25
        pathway_target[(target_pathway + 2) % 3] = 0.15
        pathway_target = pathway_target.unsqueeze(0)
        
        confidence_target = torch.tensor([[target_confidence]], device=self.device)
        
        # Compute losses
        pathway_loss = torch.nn.functional.mse_loss(
            result['pathway_weights'], pathway_target
        )
        confidence_loss = torch.nn.functional.mse_loss(
            result['confidence'].unsqueeze(-1), confidence_target
        )
        
        # Combined loss
        loss = pathway_loss + 0.5 * confidence_loss
        
        # Simple SGD step
        loss.backward()
        with torch.no_grad():
            for param in self.rna_editing.parameters():
                if param.grad is not None:
                    param -= learning_rate * param.grad
                    param.grad.zero_()
        
        self.rna_editing.eval()
        
        return {
            'total_loss': float(loss.item()),
            'pathway_loss': float(pathway_loss.item()),
            'confidence_loss': float(confidence_loss.item()),
        }
    
    def train_batch(
        self,
        embeddings: list,
        target_pathways: list,
        target_confidences: list,
        epochs: int = 10,
        learning_rate: float = 0.01
    ) -> dict:
        """
        Train the RNA layer with a batch of examples.
        
        Args:
            embeddings: List of embeddings, each [hidden_dim]
            target_pathways: List of target pathways (0, 1, or 2)
            target_confidences: List of target confidences (0.0 - 1.0)
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary with training summary
        """
        self.rna_editing.train()
        
        # Convert to tensors
        x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        batch_size = x.shape[0]
        
        # Create target distributions
        pathway_targets = torch.zeros(batch_size, 3, device=self.device)
        for i, tp in enumerate(target_pathways):
            pathway_targets[i, tp] = 0.6
            pathway_targets[i, (tp + 1) % 3] = 0.25
            pathway_targets[i, (tp + 2) % 3] = 0.15
        
        confidence_targets = torch.tensor(
            target_confidences, dtype=torch.float32, device=self.device
        ).view(batch_size, 1)
        
        total_losses = []
        
        for epoch in range(epochs):
            # Forward pass
            result = self.rna_editing(x, return_diagnostics=True)
            
            # Compute losses
            pathway_loss = torch.nn.functional.mse_loss(
                result['pathway_weights'], pathway_targets
            )
            # Flatten confidence for comparison
            pred_confidence = result['confidence'].view(batch_size, 1)
            confidence_loss = torch.nn.functional.mse_loss(
                pred_confidence, confidence_targets
            )
            
            # Combined loss
            loss = pathway_loss + 0.5 * confidence_loss
            total_losses.append(float(loss.item()))
            
            # Backward and update
            loss.backward()
            with torch.no_grad():
                for param in self.rna_editing.parameters():
                    if param.grad is not None:
                        param -= learning_rate * param.grad
                        param.grad.zero_()
        
        self.rna_editing.eval()
        
        return {
            'epochs': epochs,
            'batch_size': batch_size,
            'initial_loss': total_losses[0] if total_losses else 0,
            'final_loss': total_losses[-1] if total_losses else 0,
            'loss_reduction': (total_losses[0] - total_losses[-1]) if len(total_losses) > 1 else 0,
        }
    
    def save_weights(self, path: str = '/home/worm/octotetrahedral-agi/rna_weights.pt'):
        """Save trained weights to disk."""
        torch.save(self.rna_editing.state_dict(), path)
        return path
    
    def load_weights(self, path: str = '/home/worm/octotetrahedral-agi/rna_weights.pt'):
        """Load trained weights from disk."""
        import os
        if os.path.exists(path):
            self.rna_editing.load_state_dict(torch.load(path, map_location=self.device))
            return True
        return False


# Global instance for FFI - lazily initialized
_bridge_instance = None


def get_bridge(hidden_dim: int = 256) -> RustyWormBridge:
    """Get or create the global bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = RustyWormBridge(hidden_dim)
    return _bridge_instance


def analyze(input_embedding: list) -> dict:
    """
    FFI entry point for analyze.
    Called from Rust via PyO3.
    """
    return get_bridge().analyze(input_embedding)


def get_routing_decision(
    input_embedding: list, 
    confidence_threshold: float = 0.65,
    temperature_threshold: float = 1.8
) -> dict:
    """
    FFI entry point for routing decision.
    Called from Rust via PyO3.
    """
    return get_bridge().get_routing_decision(
        input_embedding, 
        confidence_threshold,
        temperature_threshold
    )


def get_pathway_name(index: int) -> str:
    """FFI entry point for pathway name lookup."""
    return get_bridge().get_pathway_name(index)


def train_step(
    input_embedding: list,
    target_pathway: int,
    target_confidence: float,
    learning_rate: float = 0.01
) -> dict:
    """FFI entry point for single training step."""
    return get_bridge().train_step(
        input_embedding, target_pathway, target_confidence, learning_rate
    )


def train_batch(
    embeddings: list,
    target_pathways: list,
    target_confidences: list,
    epochs: int = 10,
    learning_rate: float = 0.01
) -> dict:
    """FFI entry point for batch training."""
    return get_bridge().train_batch(
        embeddings, target_pathways, target_confidences, epochs, learning_rate
    )


def save_weights(path: str = '/home/worm/octotetrahedral-agi/rna_weights.pt') -> str:
    """FFI entry point for saving weights."""
    return get_bridge().save_weights(path)


def load_weights(path: str = '/home/worm/octotetrahedral-agi/rna_weights.pt') -> bool:
    """FFI entry point for loading weights."""
    return get_bridge().load_weights(path)


# Self-test when run directly
if __name__ == "__main__":
    print("Testing RustyWorm Bridge...")
    
    bridge = get_bridge(256)
    
    # Test with random embedding
    import random
    test_embedding = [random.gauss(0, 1) for _ in range(256)]
    
    result = bridge.analyze(test_embedding)
    print(f"\nAnalysis Result (before training):")
    print(f"  Temperature: {result['temperature']:.3f}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Head Gates: {[f'{g:.2f}' for g in result['head_gates']]}")
    print(f"  Pathway Weights: {[f'{w:.2f}' for w in result['pathway_weights']]}")
    
    routing = bridge.get_routing_decision(test_embedding)
    print(f"\nRouting Decision:")
    print(f"  Route: {routing['route']}")
    print(f"  Primary Pathway: {bridge.get_pathway_name(routing['primary_pathway'])}")
    
    # Test training
    print("\n--- Training Test ---")
    
    # Generate training data: different embeddings for different pathways
    train_embeddings = []
    train_pathways = []
    train_confidences = []
    
    for pathway in range(3):
        for _ in range(10):
            # Create embedding with pathway-specific characteristics
            emb = [random.gauss(0, 1) for _ in range(256)]
            # Add pathway-specific signal
            for i in range(pathway * 85, (pathway + 1) * 85):
                if i < 256:
                    emb[i] += 2.0  # Stronger signal in pathway region
            train_embeddings.append(emb)
            train_pathways.append(pathway)
            # Higher confidence for clearer patterns
            train_confidences.append(0.7 + random.random() * 0.2)
    
    # Train
    print(f"Training with {len(train_embeddings)} samples...")
    result = bridge.train_batch(
        train_embeddings, train_pathways, train_confidences,
        epochs=50, learning_rate=0.05
    )
    print(f"  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss reduction: {result['loss_reduction']:.4f}")
    
    # Test again after training
    print("\n--- Post-Training Analysis ---")
    for pathway in range(3):
        # Create test embedding with pathway-specific characteristics
        test_emb = [random.gauss(0, 1) for _ in range(256)]
        for i in range(pathway * 85, (pathway + 1) * 85):
            if i < 256:
                test_emb[i] += 2.0
        
        result = bridge.analyze(test_emb)
        routing = bridge.get_routing_decision(test_emb)
        pathway_name = bridge.get_pathway_name(pathway)
        detected_name = bridge.get_pathway_name(routing['primary_pathway'])
        
        match = "OK" if routing['primary_pathway'] == pathway else "WRONG"
        print(f"  Expected: {pathway_name}, Got: {detected_name} [{match}] "
              f"(conf={result['confidence']:.2f}, temp={result['temperature']:.2f})")
    
    # Save weights
    save_path = bridge.save_weights()
    print(f"\nWeights saved to: {save_path}")
    
    print("\nBridge test passed!")
