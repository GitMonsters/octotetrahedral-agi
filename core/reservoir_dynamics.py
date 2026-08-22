import torch
import torch.nn as nn


class ReservoirDynamics(nn.Module):
    def __init__(self, hidden_dim=256, n_limbs=8, echo_rho=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_limbs = n_limbs
        self._t = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.pacemaker = nn.Module()
        self.pacemaker.mu = nn.Linear(3, hidden_dim, bias=False)
        self.pacemaker.amplitude = nn.Parameter(torch.tensor(1.0))
        self.pacemaker.freqs = nn.Parameter(torch.randn(3))
        self.pacemaker.phases = nn.Parameter(torch.rand(3))

        self.temporal_basis = nn.Module()
        self.temporal_basis.targets = nn.Parameter(torch.randn(n_limbs))

        self.readout = nn.Module()
        self.readout.readout = nn.Linear(hidden_dim * n_limbs, hidden_dim)
        self.readout.norm = nn.LayerNorm(hidden_dim * n_limbs)

    def tick(self):
        self._t += 1

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        flat = stacked.reshape(stacked.size(0), -1)
        normed = self.readout.norm(flat)
        out = self.readout.readout(normed)
        return [out] * self.n_limbs
