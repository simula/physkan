"""PhysKAN: Physics-Informed Kolmogorov-Arnold Networks.

A neural architecture designed for safety-critical system identification and control.
PhysKAN enforces rigorous physical extrapolation by combining bounded B-splines,
neuro-symbolic polynomial skip-connections, and interval arithmetic to mathematically
track out-of-bounds (OOB) severity.

Key Features:
- Bounded Spline Grids: Mechanical clamping outside the nominal data range.
- Dual Severity Tracking: Mathematical compounding of out-of-bounds errors.
- Hybrid Symbolic Routing: Automated discovery of stable macro-physics via polynomials.
"""

from .kan import KAN as KAN
from .kan import KANHybrid as KANHybrid
from .kan import KANLinear as KANLinear
