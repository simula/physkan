# %% [markdown]
# # PhysKAN: Architecture demonstrations
#
# This suite demonstrates the uncertainty-forwarding and gradient firewall mechanics of the PhysKAN architecture. To make things more interesting, we add a a little noise to the synthetic training set.

# %%
import torch

from physkan import KAN
from physkan.demonstrator import KANDemonstrator

torch.manual_seed(42)

# Helper: Generate raw physical state (x) and angle (theta)
def generate_x_theta_train(steps=600, noise=0.0):
    x = torch.rand(steps, 1) * 2 - 1
    # correlated multiplicative noise + 1/10th floor
    x_noise = torch.randn(steps, 1)
    x = x * (1.0 + x_noise) + 0.1 * x_noise
    # Full phase [-pi, pi] to break collinearity and ensure cos(theta) spans [-1, 1]
    theta = torch.rand(steps, 1) * 2 * torch.pi - torch.pi
    # pure additive noise for theta
    theta_noise = noise * torch.randn(steps, 1)
    theta = theta + theta_noise
    return torch.cat([x, theta], dim=1)


def generate_x_theta_eval(x_min, x_max, steps=200):
    x = torch.linspace(x_min, x_max, steps).unsqueeze(1)
    # Lock theta at 1.5 rad (~85 deg) so cos(theta) is near 0.07.
    # This specifically exposes the naive multiplication trap for evaluation.
    theta = torch.full((steps, 1), 1.5)
    return torch.cat([x, theta], dim=1)

nominal_x_theta = generate_x_theta_train()
eval_x_theta = generate_x_theta_eval(-4.0, 4.0)

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # 4b. Deep network feature discovery
#
# We repeat this final example from `demo.py` for context, just with a bit of added noise in the training set.
# Note that elevated severity does not inherently mean danger, but it
# does mean that the model is increasingly relying on its structural priors.
# If those priors are unconstrained deep networks, extrapolation is unpredictable.
# But if we engineer those priors correctly, we can extrapolate safely.
#
# Note the `hidden_loss` parameter; it adds a penalty to hidden-layer extrapolations using `KAN.get_deep_loss()`, to encourage the input-layer outputs to stay within hidden-layer spline grid bounds. Feel free to increase it to f.x. `1.0` — it will lower the hidden-layer OOB metric, but the real cure is shown below.

# %%
torch.manual_seed(42)
model_4b = KAN(layer_dims=[2, 4, 1], grid_size=5, spline_order=3, nonlinear_dropout=0.1)
demo_4b = KANDemonstrator(
    model=model_4b,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1),
)

demo_4b.train(nominal_x_theta, epochs=1000, hidden_loss=0.0)
demo_4b.plot(eval_x_theta, "4b. Deep discovery")

# %% [markdown]
# # 4c. Deep network feature discovery, a hybrid polynomial approach
#
# To fix the unpredictable extrapolation of the deep network, we introduce the symbolic skip track, bypassing the splines.
# By setting `symbolic_order=3`, the model automatically builds a polynomial expansion of the inputs.
# We apply spline dropout to starve the deep splines, forcing the symbolic track to learn the macro-physics.
# The splines only activate to map local residuals.
# Because the symbolic track provides a predictable structural prior, extrapolation remains stable even when the dual severity ($D$) indicates we have left the training data.
#
# Note that it *might* be beneficial to integrate this via phased training rather than relying only on `nonlinear_dropout`, in effect letting the symbolic track resolve as much as possible of the error before allowing the spline layers to learn anything. Regardless of method, deep discovery of asymptotic behaviour without tail-data anchoring is a high-risk approach that should be treated as **experimental**. The lower-risk strategy is to form the interaction explicitly as `interaction_map=[[0, 0, 1]]` from domain expertise, and create a linear skip connection `symbolic_order=1` with moderate `nonlinear_dropout` in the (0.2, 0.5) range.

# %%
torch.manual_seed(42)
model_4c = KAN(
    layer_dims=[2, 4, 1], grid_size=5, spline_order=3, symbolic_order=3, nonlinear_dropout=0.3
)
demo_4c = KANDemonstrator(
    model=model_4c,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1),
)

demo_4c.train(nominal_x_theta, epochs=1000)
demo_4c.plot(eval_x_theta, "4c. Hybrid deep discovery")

# %% [markdown]
# # 5a. Multi-target severity
#
# **Goal:** Demonstrate that the dual severity tracker is a specific diagnostic tool rather than a global error flag.
#
# We will map a system with two outputs:
# * $y_1$ relies on an $x^2$ anomaly.
# * $y_2$ is insulated, relying on a stable $\cos(\theta)$ feature and a fractional coefficient of $x$.
#
# **The expectation:** When $x$ goes out of bounds, the network should firewall $y_0$ (high severity) while leaving $y_1$ untouched.
# The severity is quarantined because the underlying linear weights strictly dictate the localized interval routing ($|W| \cdot D$).
#
# **The reality:** While $y_1$ is lower than $y_2$, it is still significantly elevated. This is explained in the next case.

# %%
torch.manual_seed(42)


def target_multi(x):
    y1 = x[:, 0:1] ** 2
    y2 = (1e-3 * x[:, 0:1]) + torch.cos(x[:, 1:2])
    return torch.cat([y1, y2], dim=1)


def feature_multi(x):
    return torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1)


model_5a = KAN(
    layer_dims=[2, 4, 2], interaction_map=[[0, 0]], grid_size=5, spline_order=3, symbolic_order=1, nonlinear_dropout=0.8
)

demo_5a = KANDemonstrator(model=model_5a, target_fn=target_multi, feature_fn=feature_multi)

demo_5a.train(nominal_x_theta, epochs=1000)
demo_5a.plot(eval_x_theta, "5a. Multi-target entanglement")

# %% [markdown]
# # 5b. Perfect quarantine (the shallow solution)
#
# **The trap of deep routing:** In the previous plot, the physical extrapolation for $y_1$ (orange line in the top panel) looked flat and safe.
# However, the severity tracker in the bottom panel indicated that $y_1$ was compromised.
#
# Even though the extrapolations were good, the dual estimate caught the network balancing opposing weights.
# This is a phenomenon called cancellation entanglement.
# Because we used a dense deep network (`layer_dims=[3, 4, 2]`), the optimizer did not cleanly sever the connection by setting the weight to exactly zero.
# Instead, it routed the anomaly through multiple hidden nodes using opposing weights that physically canceled each other out (e.g., computing $+5.0x^2$ on one node and $-5.0x^2$ on another).
#
# Because our dual severity firewall mathematically compounds through absolute weights to guarantee safety boundaries, it sees through the cancellation: $|5.0| + |-5.0| = 10.0$.
# The tracker correctly warned us that $y_1$ was balancing opposing out-of-bounds errors.
#
# **The reality:** The only solution is to avoid deep interactions if you require guaranteed causal isolation.
# To achieve perfect surgical detachment, we must remove the dense hidden layers and force the model to directly map inputs to outputs.
#
# Let's drop the hidden layers (`[3, 2]`) and watch the firewall perform a flawless quarantine.


# %%
def target_multi(x):
    y1 = x[:, 0:1] ** 2
    y2 = (1e-3 * x[:, 0:1]) + torch.cos(x[:, 1:2])
    return torch.cat([y1, y2], dim=1)


def feature_multi(x):
    return torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1)

model_5b = KAN(
    layer_dims=[2, 2],
    interaction_map=[[0, 0]],
    grid_size=5,
    spline_order=3,
    symbolic_order=1,
    nonlinear_dropout=0.8,
)

demo_5b = KANDemonstrator(model=model_5b, target_fn=target_multi, feature_fn=feature_multi)
demo_5b.train(nominal_x_theta, epochs=1000)
demo_5b.plot(eval_x_theta, "5b. Multi-target separation")

# %% [markdown]
# # 5a: A fractional cliffhanger
#
# In a perfect world, all physical interactions are governed by clean integer polynomial expansions, and data forms a dense block including zero. Reality is messier.
#
# What happens when the true physics sit halfway between our architectural priors? In the code below, which you'll recognize as a minor modification of case (4c), we push the target function to a worst-case scenario: an exponent 
# of $1.5$. Sitting perfectly equidistant between the linear ($x^1$) and quadratic ($x^2$) symbolic tracks, the network's polynomial routing has an existential crisis. It throws its hands up and pushes the burden of approximation entirely back onto the B-splines.
#
# Because we injected heavy noise into the training data, the splines begin to wildly overfit the jitter in an attempt to bridge the mathematical gap, resulting in a collapse of the extrapolation curve.

# %%
torch.manual_seed(42)
model_5a = KAN(
    layer_dims=[2, 4, 1], grid_size=5, spline_order=3, symbolic_order=3, nonlinear_dropout=0.0
)
demo_5a = KANDemonstrator(
    model=model_5a,
    target_fn=lambda x: (x[:, 0:1].abs() ** 1.5) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1),
)

demo_5a.train(nominal_x_theta, epochs=1000)
demo_5a.plot(eval_x_theta, "5a. Fractional order")

# %% [markdown]
# How do we save a network that is forced into an impossible mathematical compromise? How to keep the data-sparse corners of the nominal-range hypercube in line? And furthermore, how do we handle strictly positive physical bounds (like ship speed data heavily skewed to the nominal operational range of $15-20$ knots) without destroying the network's origin point?
#
# Follow along to the next demonstration in `demo_reality.py` where we introduce structural spline stiffness, per-feature physical grid locking, and a few final tricks to tame the chaos.

# %%
