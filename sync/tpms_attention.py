"""
TPMS-Guided Attention Masks for OctoTetrahedral AGI

Maps Triply Periodic Minimal Surface geometry to attention patterns.
The bicontinuous TPMS channels define optimal information routing topologies.

TPMS-Neural Mapping:
- Gyroid channels (70.5 degree angles) -> attention head directions
- Bicontinuous structure -> forward/backward information flow
- Zero mean curvature -> minimal information loss
- Channel connectivity -> cross-limb attention patterns

Usage:
    tpms_attn = TPMSAttention(hidden_dim=256, num_heads=8, tpms_type='gyroid')
    output = tpms_attn(query, key, value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal

TPMSType = Literal['gyroid', 'schwarzP', 'schwarzD', 'neovius']


class TPMSGeometry:
    """
    Computes TPMS implicit surfaces and their properties.
    
    Equations:
    - Gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0
    - Schwarz P: cos(x) + cos(y) + cos(z) = t
    - Schwarz D: cos(x)cos(y)cos(z) - sin(x)sin(y)sin(z) = t
    - Neovius: 3(cos(x) + cos(y) + cos(z)) + 4cos(x)cos(y)cos(z) = t
    """
    
    @staticmethod
    def gyroid(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: float = 0) -> torch.Tensor:
        """Gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = t"""
        return torch.sin(x) * torch.cos(y) + torch.sin(y) * torch.cos(z) + torch.sin(z) * torch.cos(x) - t
    
    @staticmethod
    def schwarzP(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: float = 0) -> torch.Tensor:
        """Schwarz P: cos(x) + cos(y) + cos(z) = t"""
        return torch.cos(x) + torch.cos(y) + torch.cos(z) - t
    
    @staticmethod
    def schwarzD(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: float = 0) -> torch.Tensor:
        """Schwarz D (Diamond): cos(x)cos(y)cos(z) - sin(x)sin(y)sin(z) = t"""
        return torch.cos(x) * torch.cos(y) * torch.cos(z) - torch.sin(x) * torch.sin(y) * torch.sin(z) - t
    
    @staticmethod
    def neovius(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: float = 0) -> torch.Tensor:
        """Neovius: 3(cos(x) + cos(y) + cos(z)) + 4cos(x)cos(y)cos(z) = t"""
        return 3 * (torch.cos(x) + torch.cos(y) + torch.cos(z)) + 4 * torch.cos(x) * torch.cos(y) * torch.cos(z) - t
    
    @staticmethod
    def get_function(tpms_type: TPMSType):
        """Get TPMS function by name"""
        functions = {
            'gyroid': TPMSGeometry.gyroid,
            'schwarzP': TPMSGeometry.schwarzP,
            'schwarzD': TPMSGeometry.schwarzD,
            'neovius': TPMSGeometry.neovius
        }
        return functions.get(tpms_type, TPMSGeometry.gyroid)
    
    @staticmethod
    def channel_indicator(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
                          tpms_type: TPMSType = 'gyroid', threshold: float = 0.5) -> torch.Tensor:
        """
        Returns 1 where point is in positive channel, 0 in negative channel.
        TPMS divides space into two interweaving channel networks.
        """
        f = TPMSGeometry.get_function(tpms_type)
        value = f(x, y, z)
        return (value > threshold).float()


class TPMSAttentionMask(nn.Module):
    """
    Generates attention masks based on TPMS geometry.
    
    Maps sequence positions to 3D TPMS coordinates and uses
    the bicontinuous channel structure to determine attention patterns.
    """
    
    def __init__(
        self,
        seq_len: int,
        num_heads: int = 8,
        tpms_type: TPMSType = 'gyroid',
        scale: float = 2 * math.pi,
        learnable_phase: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.tpms_type = tpms_type
        self.scale = scale
        
        # Each head can have different orientation in TPMS space
        # This creates diverse attention patterns
        if learnable_phase:
            self.phase = nn.Parameter(torch.randn(num_heads, 3) * 0.5)
        else:
            # Fixed phases at 70.5 degree intervals (gyroid angle)
            angles = torch.linspace(0, 2 * math.pi, num_heads + 1)[:-1]
            self.phase = nn.Parameter(
                torch.stack([
                    torch.cos(angles),
                    torch.sin(angles) * 0.7,  # 70.5 degrees
                    torch.sin(angles + 1.23)   # Phase offset
                ], dim=1),
                requires_grad=False
            )
        
        # Learnable threshold for channel boundary
        self.threshold = nn.Parameter(torch.tensor(0.0))
        
        # Temperature for soft mask
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate TPMS-based attention mask.
        
        Returns:
            mask: [batch, num_heads, seq_len, seq_len]
        """
        # Create position coordinates
        pos = torch.arange(self.seq_len, device=device, dtype=torch.float32)
        pos = pos / self.seq_len * self.scale  # Normalize to [0, scale]
        
        # Create 2D position grid for attention (query, key positions)
        pos_q = pos.unsqueeze(1)  # [seq, 1]
        pos_k = pos.unsqueeze(0)  # [1, seq]
        
        masks = []
        tpms_func = TPMSGeometry.get_function(self.tpms_type)
        
        for h in range(self.num_heads):
            phase = self.phase[h]  # [3]
            
            # Map 2D attention positions to 3D TPMS space
            # x = query position, y = key position, z = their interaction
            x = pos_q + phase[0]
            y = pos_k + phase[1]
            z = (pos_q + pos_k) / 2 + phase[2]  # Symmetric in q,k
            
            # Compute TPMS value
            tpms_value = tpms_func(x, y, z, t=self.threshold.item())
            
            # Soft mask: sigmoid with temperature
            # Positive channel = attend, Negative channel = don't attend
            mask = torch.sigmoid(tpms_value / (self.temperature + 1e-6))
            
            masks.append(mask)
        
        # Stack heads: [num_heads, seq, seq]
        mask = torch.stack(masks, dim=0)
        
        # Expand for batch: [batch, num_heads, seq, seq]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return mask
    
    def get_channel_connectivity(self) -> torch.Tensor:
        """
        Compute connectivity matrix between sequence positions
        based on TPMS channel structure.
        """
        device = self.phase.device
        mask = self.forward(1, device)
        
        # Average over heads to get overall connectivity
        connectivity = mask.mean(dim=1).squeeze(0)
        
        return connectivity


class TPMSAttention(nn.Module):
    """
    Multi-head attention with TPMS-guided attention patterns.
    
    Combines standard scaled dot-product attention with
    TPMS geometric masks for structured information routing.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        tpms_type: TPMSType = 'gyroid',
        tpms_weight: float = 0.3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.tpms_weight = tpms_weight
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # TPMS mask generator (will be created dynamically based on seq_len)
        self.tpms_type = tpms_type
        self.tpms_mask_cache = {}
        
        # Learnable TPMS parameters
        self.tpms_phase = nn.Parameter(torch.randn(num_heads, 3) * 0.5)
        self.tpms_threshold = nn.Parameter(torch.tensor(0.0))
        self.tpms_temperature = nn.Parameter(torch.tensor(1.0))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def get_tpms_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create TPMS mask for given sequence length"""
        if seq_len not in self.tpms_mask_cache:
            mask_gen = TPMSAttentionMask(
                seq_len=seq_len,
                num_heads=self.num_heads,
                tpms_type=self.tpms_type
            )
            mask_gen.phase = self.tpms_phase
            mask_gen.threshold = self.tpms_threshold
            mask_gen.temperature = self.tpms_temperature
            self.tpms_mask_cache[seq_len] = mask_gen
        
        return self.tpms_mask_cache[seq_len](1, device)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with TPMS-guided attention.
        
        Args:
            x: Input tensor [batch, seq, hidden]
            mask: Optional causal/padding mask [batch, seq] or [batch, 1, seq, seq]
            return_attention: Whether to return attention weights
            
        Returns:
            output: [batch, seq, hidden]
            attention: Optional [batch, num_heads, seq, seq]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to multi-head: [batch, num_heads, seq, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, heads, seq, seq]
        
        # Apply TPMS mask (soft modulation)
        tpms_mask = self.get_tpms_mask(seq_len, x.device)  # [1, heads, seq, seq]
        tpms_mask = tpms_mask.expand(batch_size, -1, -1, -1)
        
        # Blend standard attention with TPMS structure
        # TPMS mask acts as a soft bias, not a hard gate
        scores = scores + self.tpms_weight * torch.log(tpms_mask + 1e-6)
        
        # Apply causal/padding mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # [batch, seq] -> [batch, 1, 1, seq]
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)  # [batch, heads, seq, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        
        if return_attention:
            return output, attention
        return output, None


class TPMSLimbRouter(nn.Module):
    """
    Routes information between 8 limbs using TPMS channel topology.
    
    Maps the 8 limbs to positions in TPMS space and uses channel
    connectivity to determine routing weights.
    """
    
    LIMB_NAMES = [
        'perception', 'memory', 'planning', 'language',
        'spatial', 'reasoning', 'metacognition', 'action'
    ]
    
    def __init__(
        self,
        hidden_dim: int,
        num_limbs: int = 8,
        tpms_type: TPMSType = 'gyroid'
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_limbs = num_limbs
        self.tpms_type = tpms_type
        
        # Learnable positions of limbs in TPMS space
        # Initialize on vertices of a cube (maps well to gyroid)
        cube_vertices = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], dtype=torch.float32) * math.pi / 2
        
        self.limb_positions = nn.Parameter(cube_vertices[:num_limbs])
        
        # Channel indicator threshold
        self.channel_threshold = nn.Parameter(torch.tensor(0.3))
        
        # Routing projection
        self.route_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Blend gate
        self.blend_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def get_routing_matrix(self) -> torch.Tensor:
        """
        Compute routing weights between limbs based on TPMS connectivity.
        
        Returns:
            routing: [num_limbs, num_limbs] routing weight matrix
        """
        tpms_func = TPMSGeometry.get_function(self.tpms_type)
        
        routing = torch.zeros(self.num_limbs, self.num_limbs, device=self.limb_positions.device)
        
        for i in range(self.num_limbs):
            for j in range(self.num_limbs):
                if i == j:
                    routing[i, j] = 1.0  # Self-connection
                else:
                    # Check if limbs are in same TPMS channel
                    pos_i = self.limb_positions[i]
                    pos_j = self.limb_positions[j]
                    midpoint = (pos_i + pos_j) / 2
                    
                    # TPMS value at midpoint indicates channel
                    channel_value = tpms_func(
                        midpoint[0:1], midpoint[1:2], midpoint[2:3],
                        t=self.channel_threshold.item()
                    )
                    
                    # Same channel = high routing weight
                    # Opposite channel = low routing weight (cross-talk)
                    routing[i, j] = torch.sigmoid(channel_value * 2).item()
        
        # Normalize rows
        routing = routing / (routing.sum(dim=1, keepdim=True) + 1e-6)
        
        return routing
    
    def forward(self, limb_states: dict) -> dict:
        """
        Route information between limbs using TPMS topology.
        
        Args:
            limb_states: Dict of limb_name -> [batch, seq, hidden]
            
        Returns:
            routed_states: Dict of limb_name -> [batch, seq, hidden]
        """
        device = list(limb_states.values())[0].device
        batch_size = list(limb_states.values())[0].shape[0]
        seq_len = list(limb_states.values())[0].shape[1]
        
        # Get routing matrix
        routing = self.get_routing_matrix().to(device)  # [num_limbs, num_limbs]
        
        # Stack limb states
        ordered_states = []
        for name in self.LIMB_NAMES:
            if name in limb_states:
                ordered_states.append(limb_states[name])
            else:
                ordered_states.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        stacked = torch.stack(ordered_states, dim=2)  # [batch, seq, num_limbs, hidden]
        
        # Project for routing
        projected = self.route_proj(stacked)  # [batch, seq, num_limbs, hidden]
        
        # Apply routing: weighted sum from other limbs
        # routing: [num_limbs, num_limbs]
        # projected: [batch, seq, num_limbs, hidden]
        routed = torch.einsum('ij,bsjd->bsid', routing, projected)
        
        # Blend with original using gate
        output_states = {}
        for idx, name in enumerate(self.LIMB_NAMES):
            if name in limb_states:
                original = limb_states[name]
                routed_state = routed[:, :, idx, :]
                
                gate_input = torch.cat([original, routed_state], dim=-1)
                gate = self.blend_gate(gate_input)
                
                output_states[name] = gate * routed_state + (1 - gate) * original
        
        return output_states


def visualize_tpms_mask(seq_len: int = 64, num_heads: int = 8, tpms_type: str = 'gyroid'):
    """Generate visualization data for TPMS attention mask"""
    mask_gen = TPMSAttentionMask(seq_len, num_heads, tpms_type)
    mask = mask_gen(1, torch.device('cpu'))  # [1, heads, seq, seq]
    
    return {
        'mask': mask.squeeze(0).detach().numpy(),
        'connectivity': mask_gen.get_channel_connectivity().detach().numpy(),
        'tpms_type': tpms_type,
        'num_heads': num_heads,
        'seq_len': seq_len
    }


if __name__ == "__main__":
    print("Testing TPMS Attention...")
    
    batch_size = 4
    seq_len = 32
    hidden_dim = 256
    num_heads = 8
    
    # Test TPMSAttention
    tpms_attn = TPMSAttention(hidden_dim, num_heads, tpms_type='gyroid')
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    output, attention = tpms_attn(x, return_attention=True)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Attention: {attention.shape}")
    
    # Test TPMSLimbRouter
    print("\nTesting TPMS Limb Router...")
    router = TPMSLimbRouter(hidden_dim, num_limbs=8, tpms_type='gyroid')
    
    limb_states = {name: torch.randn(batch_size, seq_len, hidden_dim) 
                   for name in TPMSLimbRouter.LIMB_NAMES}
    
    routed = router(limb_states)
    print(f"Routed {len(routed)} limbs")
    
    # Show routing matrix
    routing = router.get_routing_matrix()
    print(f"\nTPMS Routing Matrix (gyroid):")
    print(routing.detach().numpy().round(2))
    
    print("\nTPMS Attention tests passed!")
