import copy
import inspect
import math
from functools import cached_property
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quasirandom import SobolEngine

from .harmonic import KANHarmonic
from .spline import KANLinear
from .interaction import KANInteraction, PolynomialSkip


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
        num_harmonics: Activates harmonic basis for hidden layers instead of splines.
        pure_spline_mode: If True, completely disables the linear track across all child layers, forcing hard
            saturation/clipping at boundaries instead of proportional linear extrapolation.
        nonlinear_dropout: Dropout probability, to encourage asymptote learning.
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
        num_harmonics: int | None = None,
        pure_spline_mode: bool = False,
        nonlinear_dropout: float = 0.0,
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
                    nonlinear_dropout=nonlinear_dropout,
                    pure_spline_mode=pure_spline_mode,
                    transition_overlap=transition_overlap,
                    _quiet_init=symbolic_order > 0,
                    _is_hidden_layer=bool(i > 0),
                ) if i == 0 or num_harmonics is None else KANHarmonic(
                    in_features,
                    out_features,
                    num_harmonics=num_harmonics,
                    nonlinear_dropout=nonlinear_dropout,
                    _quiet_init=symbolic_order > 0,
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
        nonlinear_dropout: float = 0.0,
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
            nonlinear_dropout=nonlinear_dropout,
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

        self.reset_parameters()

    def reset_parameters(self):
        self.kan.reset_parameters()
        # Mixer initialization (interpretable pass-through)
        out_dim, in_dim = self.mixer.weight.shape
        if in_dim == out_dim:
            torch.nn.init.eye_(self.mixer.weight)
        else:
            # Uniform average pooling projection (plus a bit of noise)
            torch.nn.init.normal_(self.mixer.weight, mean=1.0 / in_dim, std=1e-4)
        if self.mlp_mode != "none":
            linear_layers = [m for m in self.mlp if isinstance(m, nn.Linear)]
            for i, layer in enumerate(linear_layers):
                if i < len(linear_layers) - 1:
                    # HIDDEN LAYERS: Kaiming Normal (optimal for SiLU/ReLU)
                    # We use 'relu' as the nonlinearity arg because SiLU's variance
                    # scaling properties are mathematically nearly identical to ReLU.
                    torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
                else:
                    # FINAL LAYER: Zero weights. Open on demand.
                    torch.nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

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
