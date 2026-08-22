import torch
import torch.nn as nn


class WorkingMemory(nn.Module):
    def __init__(self, num_slots=4, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.read_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.read_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.write_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.write_content = nn.Linear(hidden_dim, hidden_dim)
        self.erase_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.address_net = nn.Linear(hidden_dim, num_slots)
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.write_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, W, D = x.shape
        pooled = x.mean(dim=1)
        query = self.read_query_proj(pooled)
        keys = self.read_key_proj(self.memory)
        values = self.read_value_proj(self.memory)
        attn = torch.softmax(query @ keys.T / (D ** 0.5), dim=-1)
        read = (attn.unsqueeze(1) @ values.unsqueeze(0)).squeeze(1)
        read = self.read_norm(self.read_out_proj(read))

        gate = self.write_gate(torch.cat([read, pooled], dim=-1))
        content = self.write_content(read)
        erase = self.erase_gate(read)
        self.memory.data = (
            self.memory.data * (1 - erase.mean(0, keepdim=True))
            + gate.mean(0, keepdim=True) * content.mean(0, keepdim=True)
        )
        return read
