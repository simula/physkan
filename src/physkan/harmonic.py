import copy
import inspect
import math
from functools import cached_property
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class KANHarmonic(torch.nn.Module):
    def __init__(self, in_features, out_features, num_harmonics=4, nonlinear_dropout=0.0, bias=True, _quiet_init=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_harmonics = num_harmonics
        self.nonlinear_dropout = nonlinear_dropout

        harmonics = torch.arange(1, num_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", harmonics)
        self.norm = BatchEmaNorm(in_features)

        self.expanded_dim = in_features * num_harmonics * 2
        self.harmonic_weight = nn.Parameter(torch.empty(out_features, self.expanded_dim))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters(_quiet_init)

        self.register_buffer("running_min", torch.full((in_features,), float('inf')))
        self.register_buffer("running_max", torch.full((in_features,), float('-inf')))

    def reset_parameters(self, quiet=False):
        nn.init.xavier_uniform_(self.harmonic_weight)
        nn.init.xavier_uniform_(self.base_weight)
        if quiet:
            with torch.no_grad():
                self.base_weight /= 100.0
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_harmonics={self.num_harmonics}"
        )

    def get_stiffness_loss(self, n=1, lambda_l1=0.0, lambda_l2=0.0):
        """
        Computes the 1st-order Elastic Net (L1 + L2) stiffness penalty,
        scaled by the physical knot step size (h) to maintain invariance across grid sizes.
        """
        weight_tensor = self.harmonic_weight.view(
            self.out_features, self.in_features, 2, self.num_harmonics
        )
        effective_slope = weight_tensor * self.harmonics.pow(n)
        penalty = 0.0
        if lambda_l1 > 0:
            penalty += lambda_l1 * effective_slope.abs().mean()
        if lambda_l2 > 0:
            penalty += lambda_l2 * effective_slope.pow(2).mean()
        return penalty

    def forward(self, x, return_components=False):
        if self.training:
            with torch.no_grad():
                batch_min, batch_max = torch.aminmax(x, dim=0)
                self.running_min.copy_(torch.minimum(self.running_min, batch_min))
                self.running_max.copy_(torch.maximum(self.running_max, batch_max))
                x_clamped = x
        else:
            x_clamped = torch.clamp(x, min=self.running_min, max=self.running_max)
        angles = x_clamped.unsqueeze(-1) * self.harmonics
        sin_terms = torch.sin(angles)
        cos_terms = torch.cos(angles)
        # Flatten the harmonic features: [batch, in_features * num_freqs * 2]
        bases = torch.cat([sin_terms, cos_terms], dim=-1).view(x.size(0), -1)
        harmonic_out = F.linear(bases, self.harmonic_weight)
        if self.nonlinear_dropout > 0.0:
            harmonic_out = F.dropout(harmonic_out, p=self.nonlinear_dropout, training=self.training)
        base_out = F.linear(x, self.base_weight, self.bias)
        out = base_out + harmonic_out

        return {"final": out, "local_damage": 0.0 * out} if return_components else out


class BatchEmaNorm(nn.Module):
    """
    Physics-grade normalizer. 
    Uses EMA running statistics for the forward pass during BOTH training and evaluation.
    Ensures a purely deterministic function f(x) with no batch-coupled noise.
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

        self.weight = nn.Parameter(torch.ones(num_features))
        # Note: Bias (beta) is intentionally omitted for harmonic use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=True)
                if not self.is_initialized:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                    self.is_initialized.fill_(True)
                else:
                    self.running_mean.lerp_(batch_mean, self.momentum)
                    self.running_var.lerp_(batch_var, self.momentum)
        x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.weight * x_norm


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.is_initialized.fill_(False)


