import copy
import inspect
import math
from functools import cached_property
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.Identity,
        grid_range=(-1.0, 1.0),
        nonlinear_dropout=0.0,
        pure_spline_mode=False,
        transition_overlap=0.0,
        bias=True,
        _quiet_init=False,
        _is_hidden_layer=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.nonlinear_dropout = nonlinear_dropout
        self.pure_spline_mode = pure_spline_mode

        if inspect.isclass(base_activation):
            self.base_activation = base_activation()
        elif isinstance(base_activation, nn.Module):
            # Deepcopy guarantees isolation if passed to multiple layers!
            self.base_activation = copy.deepcopy(base_activation)
        elif callable(base_activation):
            self.base_activation = base_activation
        else:
            raise ValueError("base_activation must be a class, module instance, or callable.")

        # Standardize grid_range into a broadcastable tensor
        if not isinstance(grid_range, torch.Tensor):
            grid_range = torch.tensor(grid_range, dtype=torch.float32)
        else:
            grid_range = grid_range.float()
        if grid_range.dim() == 1 and grid_range.size(0) == 2:
            _bounds = grid_range.unsqueeze(0)  # Shape: (1, 2)
        elif grid_range.dim() == 2 and grid_range.size() == (in_features, 2):
            _bounds = grid_range  # Shape: (in_features, 2)
        else:
            raise ValueError(f"grid_range must be shape (2,) or ({in_features}, 2). Got {grid_range.shape}")
        self.register_buffer("grid_bounds", _bounds)

        # 2. Vectorized static grid formulation
        lower, upper = self.grid_bounds.T.unsqueeze(-1)
        h = (upper - lower) / grid_size
        self.register_buffer("h", h)
        # Base steps shape: (1, grid_size + 2*spline_order + 1)
        steps = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32).unsqueeze(0)
        # Broadcasted grid shape: (1, num_knots) OR (in_features, num_knots)
        grid = (steps * h + lower).contiguous()
        self.register_buffer("grid", grid)

        # The two parallel tracks
        if not self.pure_spline_mode:
            self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
            self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Calculate available cushion and trust region
        detach_threshold = 1e-3  # gradient influence at edge of trust region
        loose_fraction = 2.0  #  width of trust region (relative to crumple) at no quarantine
        strict_fraction = 0.5  # width of trust region at full quarantine
        crumple_zone = (self.spline_order * self.h).clamp(min=1e-6)
        cushion = transition_overlap * (crumple_zone - 1e-6)
        self.register_buffer("bounds_cushion", torch.cat([-cushion, cushion]).view(2, -1))
        trust_sigma = crumple_zone / (upper - lower) / math.sqrt(-math.log(detach_threshold))
        trust_padding = transition_overlap * loose_fraction + (1.0 - transition_overlap) * strict_fraction
        self.register_buffer("eff_trust_sigma", trust_sigma * trust_padding)

        self.reset_parameters(_quiet_init)
        self.is_hidden_layer = _is_hidden_layer

    def reset_parameters(self, quiet=False):
        if not self.pure_spline_mode:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            # Force zero intercept error by mean-centering each row; this works
            # in tandem with the effective_weight transform in forward.
            with torch.no_grad():
                self.base_weight -= self.base_weight.mean(dim=1, keepdim=True)
                if quiet:
                    self.base_weight /= 100.0
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        torch.nn.init.zeros_(self.spline_weight)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, "
            f"grid_range={self.grid_range}"
        )

    def b_splines(self, x: torch.Tensor):
        """Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)

        # Determine active spline basis
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def get_stiffness_loss(self, n=1, lambda_l1=0.0, lambda_l2=0.0):
        """
        Computes the 1st-order Elastic Net (L1 + L2) stiffness penalty,
        scaled by the physical knot step size (h) to maintain invariance across grid sizes.
        """
        penalty = 0.0
        h_bcast = self.h.view(1, -1, 1)  # Align dimensions for broadcasting
        diff1 = self.spline_weight.diff(n=n, dim=2)
        slope = diff1 / h_bcast
        if lambda_l1 > 0:
            penalty += lambda_l1 * slope.abs().mean()
        if lambda_l2 > 0:
            penalty += lambda_l2 * slope.pow(2).mean()
        return penalty

    def forward(self, x: torch.Tensor, return_components: bool = False):
        """Internal forward pass computing both primal (physics) and dual (severity)."""
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.pure_spline_mode:
            base_output = 0.0
        else:
            # Physics prior: When base_weight is regularized to 0, the layer defaults to a uniform
            # proportional pass-through of the engineered features, rather than defaulting to 0.0.
            effective_weight = self.base_weight + (1.0 / self.in_features)
            base_output = F.linear(self.base_activation(x), effective_weight, self.bias)

        # 1. Calculate OOB-ness and resulting trust (if required)
        trust = local_damage = None
        if return_components or (torch.is_grad_enabled() and self.spline_weight.requires_grad):
            lower_bound, upper_bound = self.grid_bounds.T
            local_damage = (F.relu(x - upper_bound) + F.relu(lower_bound - x)) / (upper_bound - lower_bound)
            if local_damage.amax() > 1e-6:
                if self.is_hidden_layer:
                    ld = local_damage
                    local_damage = local_damage.amax(dim=-1, keepdim=True)
                trust = torch.exp(-((local_damage.detach() / self.eff_trust_sigma.T) ** 2))

        # 2. Spline track (*clamped* primal only)
        lower_bound_ex, upper_bound_ex = self.grid_bounds.T + self.bounds_cushion
        if self.spline_order == 0:
            # Deg-0 splines lack padding knots, so force the interval open
            upper_bound_ex = upper_bound_ex - 1e-6
        if torch.jit.is_tracing() or (x.amin(dim=0) < lower_bound_ex).any() or (x.amax(dim=0) > upper_bound_ex).any():
            x = x.clamp(min=lower_bound_ex, max=upper_bound_ex)

        bases = self.b_splines(x)
        if trust is None or self.is_hidden_layer:
            # Fast path - hidden layer, or no detachment necessary
            spline_output = F.linear(
                bases.view(x.size(0), -1),
                self.spline_weight.view(self.out_features, -1),
            )
            if trust is not None:
                spline_output = trust * spline_output + (1.0 - trust) * spline_output.detach()
        elif trust is not None:
            # Slow path - input layer with gradients enabled AND there is local damage
            trust = trust.unsqueeze(1)
            unsummed_splines = torch.einsum('bik,oik->boi', bases, self.spline_weight)
            unsummed_splines = trust * unsummed_splines + (1.0 - trust) * unsummed_splines.detach()
            spline_output = unsummed_splines.sum(dim=2)

        if self.nonlinear_dropout > 0.0:
            spline_output = F.dropout(spline_output, p=self.nonlinear_dropout, training=self.training)

        # 3. Combine and return tuple
        x_final = (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)
        if return_components:
            return {
                "final": x_final,
                "local_damage": local_damage,
            }
        else:
            return x_final


