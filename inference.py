"""
OctoTetrahedral AGI - Inference Module
Text generation and model interaction utilities

Features:
- Interactive text generation
- Confidence estimation
- Task-specific inference
- Model introspection
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import logging

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from config import Config, get_config
from model import OctoTetrahedralModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OctoTetrahedralInference:
    """
    Inference wrapper for OctoTetrahedral AGI.
    
    Provides:
    - Text generation with various sampling strategies
    - Confidence estimation
    - Task-specific inference modes
    - Model introspection
    """
    
    def __init__(
        self,
        model: OctoTetrahedralModel,
        tokenizer=None,
        device: str = None
    ):
        self.model = model
        self.model.eval()
        
        self.device = device or model.config.device
        self.model.to(self.device)
        
        # Tokenizer
        if tokenizer is None and HAS_TIKTOKEN:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = tokenizer
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        if self.tokenizer is None:
            # Simple fallback
            tokens = [ord(c) % 1000 for c in text]
        else:
            tokens = self.tokenizer.encode(text)
        return torch.tensor(tokens, device=self.device).unsqueeze(0)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text"""
        tokens = tokens.squeeze().tolist()
        if isinstance(tokens, int):
            tokens = [tokens]
        
        if self.tokenizer is None:
            return ''.join(chr(t % 256) for t in tokens)
        else:
            return self.tokenizer.decode(tokens)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        stop_tokens: Optional[List[str]] = None,
        return_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample vs greedy
            stop_tokens: Tokens that stop generation
            return_confidence: Whether to return confidence
            
        Returns:
            Dict with generated text and optional stats
        """
        input_ids = self.encode(prompt)
        
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
        
        # Decode
        generated_text = self.decode(generated_ids[0])
        new_text = generated_text[len(prompt):]
        
        # Handle stop tokens
        if stop_tokens:
            for stop in stop_tokens:
                if stop in new_text:
                    new_text = new_text[:new_text.index(stop)]
                    break
        
        result = {
            'prompt': prompt,
            'generated': new_text,
            'full_text': prompt + new_text,
            'num_tokens': generated_ids.size(1) - input_ids.size(1)
        }
        
        if return_confidence:
            # Get confidence from model
            output = self.model(
                generated_ids,
                return_confidences=True
            )
            result['confidences'] = output.get('confidences', {})
        
        return result
    
    @torch.no_grad()
    def complete_task(
        self,
        task_text: str,
        max_tokens: int = 20
    ) -> Dict[str, Any]:
        """
        Complete a task (arithmetic, pattern, etc.).
        
        Optimized for short, deterministic answers.
        """
        return self.generate(
            prompt=task_text,
            max_new_tokens=max_tokens,
            temperature=0.1,  # Low temperature for determinism
            do_sample=False,  # Greedy
            stop_tokens=['\n', '.', '!', '?']
        )
    
    @torch.no_grad()
    def get_logits(
        self,
        text: str
    ) -> torch.Tensor:
        """Get raw logits for input text"""
        input_ids = self.encode(text)
        output = self.model(input_ids)
        return output['logits']
    
    @torch.no_grad()
    def get_probability(
        self,
        prompt: str,
        completion: str
    ) -> float:
        """Get probability of completion given prompt"""
        full_text = prompt + completion
        input_ids = self.encode(full_text)
        
        output = self.model(input_ids)
        logits = output['logits']
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sum log probs for completion tokens
        prompt_len = len(self.encode(prompt)[0])
        completion_log_prob = 0.0
        
        for i in range(prompt_len, input_ids.size(1)):
            token_id = input_ids[0, i].item()
            completion_log_prob += log_probs[0, i-1, token_id].item()
        
        return completion_log_prob
    
    @torch.no_grad()
    def get_perplexity(self, text: str) -> float:
        """Compute perplexity of text"""
        input_ids = self.encode(text)
        
        output = self.model(
            input_ids,
            labels=input_ids
        )
        
        # Perplexity = exp(loss)
        return torch.exp(output['loss']).item()
    
    @torch.no_grad()
    def get_confidence(self, text: str) -> Dict[str, float]:
        """Get model confidence for input text"""
        input_ids = self.encode(text)
        
        output = self.model(
            input_ids,
            return_confidences=True
        )
        
        return output.get('confidences', {})
    
    @torch.no_grad()
    def get_attention_patterns(self, text: str) -> torch.Tensor:
        """Get attention patterns from reasoning limb"""
        input_ids = self.encode(text)
        
        # Forward through perception and reasoning
        encoded, _ = self.model.perception(token_ids=input_ids)
        edited, _ = self.model.rna_editing(encoded, confidence=0.5)
        core_out, _ = self.model.core(edited)
        
        self.model.reasoning(core_out, return_confidence=True)
        
        return self.model.reasoning.get_attention_weights()
    
    @torch.no_grad()
    def get_reasoning_state(self, text: str) -> torch.Tensor:
        """Extract reasoning state vector"""
        input_ids = self.encode(text)
        
        output = self.model(input_ids, return_confidences=True)
        return output.get('reasoning_state')
    
    def interactive(self):
        """Interactive generation loop"""
        print("\nOctoTetrahedral AGI - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  /quit - Exit")
        print("  /reset - Reset memory")
        print("  /stats - Show model stats")
        print("  /conf - Show confidence for last input")
        print("=" * 50)
        
        last_confidence = {}
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == '/quit':
                    print("Goodbye!")
                    break
                
                elif user_input == '/reset':
                    self.model.reset_memory()
                    print("Memory reset.")
                    continue
                
                elif user_input == '/stats':
                    stats = self.model.get_stats()
                    print("\nModel Statistics:")
                    print(f"  Parameters: {stats['total_params']:,}")
                    print(f"  Forwards: {stats['forward_count']}")
                    print(f"  Memory util: {stats['memory_utilization']:.4f}")
                    continue
                
                elif user_input == '/conf':
                    if last_confidence:
                        print("\nConfidences:")
                        for k, v in last_confidence.items():
                            print(f"  {k}: {v:.4f}")
                    else:
                        print("No confidence data yet.")
                    continue
                
                # Generate response
                result = self.generate(
                    user_input,
                    max_new_tokens=100,
                    temperature=0.8,
                    return_confidence=True
                )
                
                print(f"\nModel: {result['generated']}")
                last_confidence = result.get('confidences', {})
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def load_model(
    checkpoint_path: Optional[str] = None,
    device: str = None
) -> OctoTetrahedralInference:
    """
    Load model for inference.
    
    Args:
        checkpoint_path: Path to checkpoint (None for fresh model)
        device: Device to use
        
    Returns:
        OctoTetrahedralInference instance
    """
    config = get_config()
    device = device or config.device
    
    if checkpoint_path:
        model, _ = OctoTetrahedralModel.load_checkpoint(checkpoint_path, device)
    else:
        model = OctoTetrahedralModel(config)
        model.to(device)
    
    return OctoTetrahedralInference(model, device=device)


if __name__ == "__main__":
    print("Testing OctoTetrahedral Inference...")
    
    # Create fresh model
    inference = load_model()
    
    # Test encoding/decoding
    text = "Hello, world!"
    encoded = inference.encode(text)
    decoded = inference.decode(encoded)
    print(f"Encode/decode test: '{text}' -> {encoded.shape} -> '{decoded}'")
    
    # Test generation
    print("\nGeneration test:")
    result = inference.generate(
        "Calculate: 5 + 3 = ",
        max_new_tokens=10,
        temperature=0.5
    )
    print(f"  Prompt: {result['prompt']}")
    print(f"  Generated: {result['generated']}")
    
    # Test task completion
    print("\nTask completion test:")
    result = inference.complete_task("What is 7 + 8? ")
    print(f"  Task: What is 7 + 8?")
    print(f"  Answer: {result['generated']}")
    
    # Test confidence
    print("\nConfidence test:")
    conf = inference.get_confidence("The quick brown fox")
    print(f"  Confidences: {conf}")
    
    # Test perplexity
    print("\nPerplexity test:")
    ppl = inference.get_perplexity("The cat sat on the mat")
    print(f"  Perplexity: {ppl:.2f}")
    
    print("\nAll inference tests passed!")
    
    # Optional: interactive mode
    # inference.interactive()
