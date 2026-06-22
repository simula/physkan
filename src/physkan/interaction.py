import itertools
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


InteractionFn = Callable[[*tuple[torch.Tensor, ...]], torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]]

class KANInteraction(torch.nn.Module):
    """Computes explicit feature interactions.

    Args:
        interaction_map: List of index lists defining the multiplicative terms, or a function.
            Example: [[0, 0], [0, 1]] → (feats[:,0]^2 and (feats[:,0] * feats[:,1]).
                     [(lambda x, y: (x**2, x*y), (0, 1)] → same
    """

    def __init__(
            self,
            interaction_map: list[list[int] | tuple[InteractionFn, list[int] | tuple[int, ...]]] = [],
    ):
        super().__init__()
        self.interaction_map = interaction_map

    def extra_repr(self) -> str:
        return f"interaction_map={self.interaction_map}"

    @staticmethod
    def _interact(x, interaction):
        if callable(interaction[0]):
            fn, idx = interaction
            interacted = fn(*[x[:, i] for i in idx])
            out = []
            for new_feat in (interacted if isinstance(interacted, (tuple, list)) else [interacted]):
                if new_feat.dim() == 1:
                    new_feat = new_feat.unsqueeze(-1)
                out.append(new_feat)
            return out
        else:
            return [torch.prod(x[:, interaction], dim=1, keepdim=True)]


    def bounds(self, grid_bounds: torch.Tensor, expected_complexity: int) -> torch.Tensor:
        if not self.interaction_map:
            return grid_bounds

        new_bounds = []
        for interaction in self.interaction_map:
            if callable(interaction[0]):
                # ---------------------------------------------------------
                # EMPIRICAL PATH (Sobol Sampling for Lambdas)
                # ---------------------------------------------------------
                fn, idx = interaction
                num_features = len(idx)
                base_bounds = grid_bounds[idx].view(-1, 2)
                needed_bits = (num_features ** 2 * expected_complexity).bit_length() + 2
                num_samples = 1 << max(10, needed_bits)
                # 1. Generate Sobol samples and move to correct device/dtype
                sobol = torch.quasirandom.SobolEngine(dimension=num_features, scramble=True)
                unit_samples = sobol.draw(num_samples).to(
                    device=grid_bounds.device, dtype=grid_bounds.dtype
                )
                # 2. Scale to physical bounds
                lower_bounds = base_bounds[:, 0]
                upper_bounds = base_bounds[:, 1]
                scaled_samples = lower_bounds + unit_samples * (upper_bounds - lower_bounds)
                # 3. Interact the samples and xtract bounds for each output feature
                interacted = self._interact(scaled_samples, (fn, list(range(num_features))))
                for out_feat in interacted:
                    calc_min = out_feat.min()
                    calc_max = out_feat.max()
                    buffer = (calc_max - calc_min) * 1e-3 + 1e-6
                    new_bounds.append(torch.stack([calc_min - buffer, calc_max + buffer]).unsqueeze(0))

            else:
                # ---------------------------------------------------------
                # ANALYTICAL PATH (Perfect Corners for Pure Products)
                # ---------------------------------------------------------
                idx = interaction
                base_bounds = grid_bounds[idx] # Shape: [num_features, 2]
                
                # cartesian_prod generates all 2^N corner combinations
                corners = torch.cartesian_prod(*[base_bounds[i] for i in range(len(idx))])
                
                # Multiply across the feature dimension and find the true min/max
                prods = corners.prod(dim=1)
                new_bounds.append(torch.stack([prods.min(), prods.max()]).unsqueeze(0))

        grid_bounds = torch.cat([grid_bounds] + new_bounds, dim=0)
        bound_variances = grid_bounds.amax(dim=0) - grid_bounds.amin(dim=0)
        max_grid_width = grid_bounds.diff(dim=1).amax()
        if bound_variances.max() < 2e-3 * max_grid_width:
            grid_bounds = grid_bounds[0]
        return grid_bounds

    def forward(self, x: torch.Tensor):
        if not self.interaction_map:
            return x

        out_x = [x]
        for mapping in self.interaction_map:
            out_x.extend(self._interact(x, mapping))
        return torch.cat(out_x, dim=1)


class PolynomialSkip(nn.Module):
    def __init__(self, in_features, out_features, order=2):
        super().__init__()

        # 1. Generate all multi-indices for polynomials up to 'order'
        # e.g., for inputs [0, 1] order 2: (0,), (1,), (0,0), (0,1), (1,1)
        self.combinations = []
        for d in range(1, order + 1):
            combos = itertools.combinations_with_replacement(range(in_features), d)
            self.combinations.extend(list(combos))

        feature_degrees = [len(c) for c in self.combinations]
        degrees_tensor = torch.tensor(feature_degrees, dtype=torch.float32)
        degree_penalty = 2.0 + 2.0 * degrees_tensor
        self.register_buffer("degree_penalty", degree_penalty.unsqueeze(0))

        num_features = len(self.combinations)

        # 2. The standard linear weights
        self.weights = nn.Parameter(torch.randn(out_features, num_features) * 0.1)

        # 3. The Probationary Gates
        self.gates = nn.Parameter(1.0 * torch.ones(out_features, num_features))

    def forward(self, x, dual_input=None):
        poly_features = []
        poly_duals = []

        # --- A. Compute Polynomials & Interval Duals ---
        for combo in self.combinations:
            term = torch.ones_like(x[:, 0:1])
            term_dual = torch.zeros_like(x[:, 0:1]) if dual_input is not None else None

            for idx in combo:
                x_i = x[:, idx : idx + 1]

                # Interval Multiplication for the Dual Severity
                if dual_input is not None:
                    d_i = dual_input[:, idx : idx + 1]
                    # Severity of a product: (|A| + D_A) * (|B| + D_B) - |A * B|
                    new_dual = (torch.abs(term) + term_dual) * (torch.abs(x_i) + d_i) - torch.abs(
                        term * x_i
                    )
                    term_dual = new_dual

                term = term * x_i

            poly_features.append(term)
            if dual_input is not None:
                poly_duals.append(term_dual)

        P_x = torch.cat(poly_features, dim=1)

        # --- B. Apply the -5.0 Sigmoid Gate ---
        active_weights = self.weights * torch.sigmoid(self.gates - self.degree_penalty)

        # --- C. Route the Physical Prediction ---
        out = F.linear(P_x, active_weights)

        # --- D. Route the Dual Severity (The Abs-Weighted Path) ---
        if dual_input is not None:
            D_x = torch.cat(poly_duals, dim=1)
            # You called it: the dual routes through the absolute value of the active weights!
            out_dual = F.linear(D_x, torch.abs(active_weights))
            return out, out_dual

        return out
