import copy
import inspect
import math
from functools import cached_property
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine

from .interaction import KANInteraction, PolynomialSkip


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.Identity,
        grid_range=(-1.0, 1.0),
        spline_dropout=0.0,
        pure_spline_mode=False,
        transition_overlap=0.0,
        _quiet_init=False,
        _is_hidden_layer=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.spline_dropout = spline_dropout
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
            self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # Calculate available cushion and trust region
        detach_threshold = 1e-3  # gradient influence at edge of trust region
        loose_fraction = 2.0  #  width of trust region (relative to crumple) at no quarantine
        strict_fraction = 0.5  # width of trust region at full quarantine
        crumple_zone = self.spline_order * self.h
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
            # Force zero intercept error by mean-centering each row.
            # This guarantees that the row-sum of base_weight is EXACTLY 0.0,
            # meaning the effective weight row-sum is EXACTLY 1.0 across all channels.
            with torch.no_grad():
                self.base_weight -= self.base_weight.mean(dim=1, keepdim=True)
                if quiet:
                    self.base_weight /= 100.0
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
            effective_weight = self.base_weight + (1.0 / self.in_features)
            base_output = F.linear(self.base_activation(x), effective_weight)

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

        if self.spline_dropout > 0.0:
            spline_output = F.dropout(spline_output, p=self.spline_dropout, training=self.training)

        # 3. Combine and return tuple
        x_final = (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)
        if return_components:
            return {
                "final": x_final,
                "local_damage": local_damage,
            }
        else:
            return x_final


class KAN(torch.nn.Module):
    """Kolmogorov-Arnold Network (KAN) macro-architecture composed of sequentially stacked KANLinear layers.

    Coordinates deep layer propagation by chaining self-contained Bounded KAN blocks. If
    `pure_spline_mode` is False, the underlying layers maintain an internal scale-preserving
    uniform baseline that seamlessly conserves signal magnitude across dimensional changes
    (expansions/contractions) during extreme out-of-bounds anomalies.

    Args:
        layer_dims: Architectural dimensions mapping from input to output features (e.g., [input_dim, hidden_dim, output_dim]).
        grid_size: Number of inner intervals partitioning the spline domain
        spline_order: Polynomial degree of the local B-spline bases.
        base_activation: Activation function applied exclusively to the linear track. Change with caution - see README!
        grid_range: Physical bounds `(lower, upper)` defining the spline evaluation domain. Can be a list-of-tuples or a Tensor,
            for per-feature ranges.
        pure_spline_mode: If True, completely disables the linear track across all child layers, forcing hard
            saturation/clipping at boundaries instead of proportional linear extrapolation.
        spline_dropout: Dropout probability, to encourage asymptote learning.
        interaction_map: Multiplicative feature interaction indices, or lambdas. Use to define explicit cross-terms while preserving
            strict OOB propagation.
    """

    def __init__(
        self,
        layer_dims: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: torch.nn.Module = torch.nn.Identity,
        grid_range: tuple[float, float] | list[tuple[float, float]] | torch.Tensor = (-1.0, 1.0),
        pure_spline_mode: bool = False,
        spline_dropout: float = 0.0,
        interaction_map: list[list[int] | Callable[[torch.Tensor], torch.Tensor]] = [],
        symbolic_order: int = 0,
        transition_overlap: float = 0.0,
    ):
        super().__init__()
        self.interactor = KANInteraction(interaction_map)
        self.layer_dims = layer_dims
        with torch.no_grad():
            synth = torch.ones(0, layer_dims[0], dtype=torch.float32)
            synth_out = self.interactor(synth)
        eff_layer_dims = list(layer_dims)
        eff_layer_dims[0] = synth_out.size(1)
        with torch.no_grad():
            # Standardize grid_range into a broadcastable tensor
            if not isinstance(grid_range, torch.Tensor):
                grid_range = torch.tensor(grid_range, dtype=torch.float32)
            else:
                grid_range = grid_range.float()
            if grid_range.dim() == 1 and grid_range.size(0) == 2:
                grid_range = grid_range.expand(layer_dims[0], 2)
            grid_range = self.interactor.bounds(grid_range, expected_complexity=grid_size)

        self.layers = torch.nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(eff_layer_dims, eff_layer_dims[1:])):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range if i == 0 else (-1.0, 1.0),
                    spline_dropout=spline_dropout,
                    pure_spline_mode=pure_spline_mode,
                    transition_overlap=transition_overlap,
                    _quiet_init=symbolic_order > 0,
                    _is_hidden_layer=bool(i > 0),
                )
            )

        # 2. The Shallow Polynomial Skip Connection
        self.symbolic_order = symbolic_order
        if self.symbolic_order > 0:
            self.poly_skip = PolynomialSkip(
                in_features=eff_layer_dims[0], out_features=eff_layer_dims[-1], order=symbolic_order
            )

    def get_stiffness_loss(self, n=1, lambda_l1=1e-4, lambda_l2=1e-5):
        """
        Computes the 1st-order Elastic Net (L1 + L2) stiffness penalty for all layers,
        scaled by their physical knot step size (h) to maintain invariance across grid sizes.
        """
        return sum(layer.get_stiffness_loss(n=n, lambda_l1=lambda_l1, lambda_l2=lambda_l2) for layer in self.layers)

    @cached_property
    def _sobol(self):
        return SobolEngine(dimension=self.layer_dims[0], scramble=True)

    def get_sobolev_loss(self, lambda_l1=1e-4, lambda_l2=1e-5, num_probes: int | None = None):
        """
        Computes a Sobolev H1 smoothing penalty across a dense, low-discrepancy Sobol grid.
        Forces the macroscopic physical extrapolation to remain smooth, regardless of internal weights.
        """
        N_feats = self.layer_dims[0]
        if num_probes is None:
            # Estimate required depth based on pairwise interaction scaling (D^2)
            needed_bits = (N_feats ** 2 * self.grid_size).bit_length() + 1
            N_probes = 1 << max(5, needed_bits)
        epsilon = 1e-3
        lower, upper = self.layers[0].grid_bounds.T
        x_probes = lower + self._sobol.draw(N_probes).to(lower.device) * (upper - lower)
        x_perturbed = x_probes.unsqueeze(1) + epsilon * torch.eye(N_feats, device=lower.device).unsqueeze(0)
        x_mega = torch.cat([x_probes, x_perturbed.view(-1, N_feats)], dim=0)
        y_mega = self.forward(x_mega)
        y_base = y_mega[:N_probes]
        y_pert = y_mega[N_probes:].view(N_probes, N_feats)
        gradients = (y_pert - y_base) / epsilon
        return lambda_l1 * gradients.abs().mean() + lambda_l2 * gradients.pow(2).mean()

    def get_deep_loss(self, lambda_l1=1e-4, lambda_l2=1e-5):
        loss = 0.0
        for layer in self.layers[1:]:
            loss += lambda_l1 * layer.base_weight.abs().mean() + lambda_l2 * layer.base_weight.pow(2).mean()
        return loss

    def extra_repr(self) -> str:
        # Most information is already in the layers, just add the pre-interactions dim
        return f"in_features={self.layer_dims[0]}"

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, x: torch.Tensor, return_components: bool = False
    ):
        x = interacted = self.interactor(x)
        damages = []

        if self.symbolic_order > 0:
            poly_out = self.poly_skip(x)

        for layer in self.layers:
            layer_comp = layer.forward(x, return_components=True)
            x = layer_comp["final"]
            damages.append(layer_comp["local_damage"])
        if self.symbolic_order > 0:
            x = x + poly_out

        if return_components:
            return {
                "interacted": interacted,
                "kan_out": x,
                "final": x,
                "damages": damages,
            }
        else:
            return x


class KANHybrid(nn.Module):
    def __init__(
        self,
        out_features: int,
        layer_dims: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] | list[tuple[float, float]] | torch.Tensor = (-1.0, 1.0),
        spline_dropout: float = 0.0,
        interaction_map: list[list[int] | Callable[[torch.Tensor], torch.Tensor]] = [],
        symbolic_order: int = 0,
        transition_overlap: float = 0.0,
        mlp_mode: Literal["none", "multiplicative", "additive"] = "none",
        mlp_strength: float = 0.15,
        mlp_hidden_dims: list[int] = [],
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        self.kan = KAN(
            layer_dims=layer_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=grid_range,
            spline_dropout=spline_dropout,
            interaction_map=interaction_map,
            symbolic_order=symbolic_order,
            transition_overlap=transition_overlap,
        )

        self.mixer = nn.Linear(layer_dims[-1], out_features, bias=False)

        self.mlp_mode = mlp_mode
        self.mlp_strength = mlp_strength
        self.mlp_dropout = mlp_dropout
        if mlp_mode != "none":
            mlp_layer_dims = [self.kan.layers[0].in_features + self.kan.layers[-1].out_features] + mlp_hidden_dims + [out_features]
            mlp_layers = []
            for i, (in_dim, out_dim) in enumerate(zip(mlp_layer_dims, mlp_layer_dims[1:])):
                mlp_layers.append(nn.Linear(in_dim, out_dim))
                mlp_layers.append(nn.SiLU())
            self.mlp = nn.Sequential(*mlp_layers[:-1])  # drop final activation

    def get_deep_loss(self, lambda_l1=1e-4, lambda_l2=1e-5):
        return self.kan.get_deep_loss(lambda_l1, lambda_l2)

    def get_stiffness_loss(self, n=1, lambda_l1=1e-4, lambda_l2=1e-5):
        return self.kan.get_stiffness_loss(n, lambda_l1, lambda_l2)

    def forward(
        self, x: torch.Tensor, return_components: bool = False
    ):
        kan_comp = self.kan(x, return_components=True)
        x = kan_comp["final"]
        x = self.mixer(x)

        if self.mlp_mode != "none":
            mlp_in = torch.cat([kan_comp["interacted"], kan_comp["kan_out"]], dim=1)
            mlp_out = self.mlp_strength * torch.tanh(self.mlp(mlp_in.detach()))
            if self.mlp_dropout > 0.0:
                mlp_out = F.dropout(mlp_out, p=self.mlp_dropout, training=self.training)
            if self.mlp_mode == "additive":
                x = x + mlp_out
            elif self.mlp_mode == "multiplicative":
                x = x * (1.0 + mlp_out)
            else:
                raise ValueError(f"Unknown mlp mode '{self.mlp_mode}'")
        else:
            mlp_out = None

        if return_components:
            return kan_comp | {
                "mlp_out": mlp_out,
                "final": x,
            }
        else:
            return x
