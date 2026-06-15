# %% [markdown]
# # PhysKAN: Architecture Demonstrations
#
# This suite demonstrates the mathematical vulnerabilities of standard deep learning 
# extrapolations and how the PhysKAN architecture solves them using uncertainty-forwarding 
# and a strict gradient firewall.
#
# We will progress from simple 1D curve fitting to deep feature discovery, watching how 
# the architecture behaves when forced into "data deserts."

# %%
import torch
import torch.nn as nn

from physkan import KAN, KANHybrid
from physkan.demonstrator import KANDemonstrator, TradKAN

torch.manual_seed(42)

# --- Data Generation Helpers ---

def generate_x_data(x_min, x_max, steps=200):
    """Generates 1D physical state data."""
    return torch.linspace(x_min, x_max, steps).unsqueeze(1)

def generate_x_theta_train(x_scale=1.0, steps=400):
    """Generates 2D data: a physical state (x) and an angle (theta)."""
    x = x_scale * (torch.rand(steps, 1) * 2 - 1)
    # Full phase [-pi, pi] breaks collinearity and ensures cos(theta) spans [-1, 1]
    theta = torch.rand(steps, 1) * 2 * torch.pi - torch.pi
    return torch.cat([x, theta], dim=1)

def generate_x_theta_eval(x_min, x_max, steps=200):
    """Generates 2D evaluation data, locking theta to expose multiplication traps."""
    x = torch.linspace(x_min, x_max, steps).unsqueeze(1)
    # Lock theta at 1.5 rad (~85 deg) so cos(theta) is near 0.07.
    theta = torch.full((steps, 1), 1.5)
    return torch.cat([x, theta], dim=1)

# Dataset definitions
nominal_data = generate_x_data(-1.0, 1.0, 100)
dense_data = generate_x_data(-4.0, 4.0, 100)
sparse_data = torch.cat([nominal_data, dense_data[torch.randperm(dense_data.size(0))[:10]]])

nominal_x_theta = generate_x_theta_train()
dense_x_theta = generate_x_theta_train(x_scale=4.0)
eval_x_theta = generate_x_theta_eval(-4.0, 4.0)

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Part 1: The standard KAN vulnerabilities
#
# To understand why PhysKAN is necessary, we first have to look at what happens when a 
# standard KAN hits data it has never seen before.
#
# In this first test, we train a traditional KAN purely on data inside a tight nominal 
# range (`-1.0` to `1.0`). We then ask it to extrapolate out to `-4.0` and `4.0`.
#
# **What to look for in the plot:**
# Notice how the blue prediction line completely ignores the true physical parabola outside 
# the grey box. Because the underlying `SiLU` activation function is asymmetric, the network 
# unpredictably grows on the right side while completely flatlining on the left.
#
# **Try this:**
# Change `nominal_data` to `sparse_data` and then `dense_data` in the `train()` call. What is the difference? Why does this happen? Hint: with `nominal_data`, the entire training set is within the spline bounds (gray area).

# %%
model_1a = TradKAN(
    layer_dims=[1, 1],
    grid_size=5,
    spline_order=3,
    grid_range=(-1.0, 1.0),
)
demo_1a = KANDemonstrator(model=model_1a, target_fn=lambda x: x**2)

demo_1a.train(nominal_data)
demo_1a.plot(dense_data, "1a. Standard KAN: Arbitrary Extrapolation")


# %% [markdown]
# ### The wide grid fallacy
#
# A common, naive fix to the extrapolation problem above is to simply widen the grid bounds. 
# If we know the physics goes out to `4.0`, why not just set the grid to `(-4.0, 4.0)`?
#
# We do that here, increasing the `grid_size` proportionally to maintain the same resolution. 
# We then train it on *sparse* out-of-bounds data.
#
# **What to look for in the plot:**
# The prediction detaches from the physics locally and outputs jagged predictions. B-splines have strictly local support, so the knots far outside the core data region receive 
# few gradient updates. They "collapse," showing that expanding a grid into a data 
# desert is mathematically unsafe.
#
# **Challenge:** Based the previous experiment, what do you think would happen if the model was instead trained on `dense_data` or on `nominal_data`? Guess before trying!

# %%
model_1b = TradKAN(
    layer_dims=[1, 1],
    grid_size=20,
    spline_order=3,
    grid_range=(-4.0, 4.0),
)
demo_1b = KANDemonstrator(model=model_1b, target_fn=lambda x: x**2)

demo_1b.train(sparse_data)
demo_1b.plot(generate_x_data(-4.0, 4.0, 200), "1b. The Wide Grid Fallacy (Untrained Knot Collapse)", nominal_range=(-4.0, 4.0))

# %% [markdown]
# ## Part 2: The PhysKAN mechanical clamp
#
# Now we introduce the core PhysKAN mechanics. Instead of SiLU asymptotes or 
# collapsing wide grids, PhysKAN uses a strict `Identity` linear baseline and a mechanical clamp that stops OOB gradients from reaching the spline.
#
# **What to look for in the plots:**
# Look at the top panel. Inside the grey nominal bounds, the splines perfectly fit the curve. 
# But the moment the data leaves the bounds, the mechanical clamp freezes the splines, preventing 
# oscillation. 
#
# **...but this doesn't look any better does it?** You will notice the extrapolation is a straight line and the in-bounds splines form a 
# parabola — even though it is trained on the sparse (wide) dataset. The splines are protected from the OOB data and free to fit the parabola locally, while the base track 
# absorbs the residual slope. Where the splines shut off, that slope is exposed. We happily trade 
# the wild flatlining of standard networks for a safe, predictable, linear fallback.
# Follow along to see how to turn the linear extrapolation into a perfect fit!

# %%
model_2 = KAN(layer_dims=[1, 1], grid_size=5, spline_order=3)
demo_2 = KANDemonstrator(model=model_2, target_fn=lambda x: x**2)

demo_2.train(sparse_data)
demo_2.plot(dense_data, "2. PhysKAN: Spline Plateau and Linear Asymptote")

# %% [markdown]
# # 3a. Linear recovery via feature engineering
#
# By providing $x^2$ as an engineered feature, the network has the basis it needs to create the perfect asymptote. However, the physical prediction is still not good, as you can see below.
#
# During training, instead of relying purely on the interaction feature, the network put weight some on the raw $x$ feature and used the splines to cancel out the error.
# When extrapolated, the splines clamped, the cancellation stopped, and the raw error emerged.
#
# **Challenge:** Do you think it would make a difference to train on e.g. `sparse_data`? Why?

# %%
model_3a = KAN(layer_dims=[2, 1], grid_size=5, spline_order=3)
demo_3a = KANDemonstrator(model=model_3a, target_fn=lambda x: x**2, feature_fn=lambda x: torch.cat([x, x**2], dim=1))

demo_3a.train(nominal_data)
demo_3a.plot(dense_data, "3a. Basis competition (spline vs asymptote)")

# %% [markdown]
# *Note — it is preferred to provide an explicit `interaction_map` to PhysKAN rather than input-side feature engineering. More on that later.*

# %% [markdown]
# ### The solution: Forcing physical isolation via dropout
#
# To fix this ambiguity, we must force the network to decide who owns the signal. We do 
# this by introducing **spline dropout**. 
#
# By randomly zeroing out the splines during training, we make them unreliable for global 
# macroscopic trends. The optimizer realizes that the only safe, persistent way to minimize 
# the loss is to route the entire $x^2$ signal through the asymptotic base track. 
# The splines are forced to remain flat, reserved only for local, high-frequency residuals.
#
# **What to look for in the plot:**
# With dropout active, the base track assumes 100% of the $x^2$ responsibility. The blue 
# prediction line perfectly traces the physics infinitely out of bounds, while the red 
# severity line confirms our safety tracking remains active.

# %%
model_3b = KAN(
    layer_dims=[2, 1], grid_size=5, spline_order=3, spline_dropout=0.05
)
demo_3b = KANDemonstrator(model=model_3b, target_fn=lambda x: x**2, feature_fn=lambda x: torch.cat([x, x**2], dim=1))

demo_3b.train(nominal_data)
demo_3b.plot(dense_data, "3b. Nominal-range extrapolation (with spline dropout)")


# %% [markdown]
# ## Part 4: Deep interaction discovery
#
# What happens when we remove explicit engineering entirely and ask a deeper network 
# (`[2, 4, 1]` layers) to learn the more complex physics interaction $x^2 \cos(\theta)$ on its own?
#
# **What to look for in the plot:**
# Look at the blue prediction line in the top panel. Because the deep network relies on 
# unconstrained spline combinations across multiple hidden layers to approximate multiplication, 
# its physical extrapolation outside the grey box becomes unpredictable.
#
# **Note:** As we saw above, you can give a hint to the network. Pass `interaction_map=[[0, 0, 1]]` ($x * x * \cos(\theta)$) and the 
# asymptotes will be predicted much better. But that's cheating! We said we wanted interaction *discovery*,
# not interaction *spoon-feeding*.
#
# **Try this:** What do you think will happen if we train it on `dense_x_theta` instead of `nominal_x_theta`? Hint: **Pure mayhem!** See below for more.

# %%
torch.manual_seed(42)
model_4a = KAN(layer_dims=[2, 4, 1], grid_size=10, spline_order=3, spline_dropout=0.1)
demo_4a = KANDemonstrator(
    model=model_4a,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1),
    #mixer=nn.Linear(1, 1),
)

demo_4a.train(nominal_x_theta, epochs=500)
demo_4a.plot(eval_x_theta, "4a. Deep Discovery")

# %% [markdown]
# ### The deep-network amplitude trap
#
# If we try to fix this by training the deep network on the full `[-4.0, 4.0]` dataset, the 
# target amplitude jumps to 16.0. To hit those large numbers, the optimizer violently inflates 
# the internal weights. 
#
# This pushes up the latent amplitudes, and our perfectly safe
# nominal inputs increase beyond the hidden layers' strict `[-1.0, 1.0]` spline bounds. You can 
# actually see this happening in the plot (if changed to train on `dense_x_theta`) — the dotted red "Hidden-layer OOB" line hovers above 
# zero even inside the grey box. The network panicked, clamped all its internal 
# splines shut, and degraded into a rigid linear MLP.
#
# So what if we add a linear mixer at the end, to handle scaling of the output signal? Feel free
# to try, but it's not going to do much good. It will just hit the next problem.
#
# ### Limits of interaction discovery
#
# Even with the output mixer protecting the internal grids, and *even* if you looked at the
# source code and found the `*_loss` arguments to `demo.train()` and dialed them up to a ridiculous
# number like `1e4`, the physical prediction is still a jagged mess. 
# More importantly: **the severity stays largely silent.** Why?
#
# 1. **The theoretical ideal:** Mathematically, a deep network *does* have the capacity 
#    to smoothly multiply. It could use its layers to recreate the algebraic identity 
#    $u \cdot v = \frac{1}{4}((u+v)^2 - (u-v)^2)$. Given enough data points, sufficient 
#    epochs, and painstakingly tuned regularizers guiding the loss landscape, the optimizer 
#    would eventually find this elegant, smooth solution.
# 2. **The reality:** In practice, gradient descent takes the path of 
#    least resistance. Instead of discovering that complex global weight coordination, it 
#    gets trapped in a lazy local minimum. It simply stacks high-frequency 1D spline wiggles 
#    to satisfy the interaction exactly at the training points.
# 3. **The silent alarm:** The OOB damage measure *domain violations*, not smoothness. 
#    Inside the `[-1.0, 1.0]` boxes, the network didn't significantly
#    break any structural bounds. It just drew a physically unnatural shape.
#
# **Takeaway:** While we could theoretically force the optimizer to find the right path 
# with enough attention to training, that is not a practical approach. If your
# interests lie in this direction, look into the `SAM` or even the `L-BFGS` optimizer; because
# greedy optimizers like `AdamW` won't get you there. Instead, to fix this 
# reliably, we must give the architecture a structural prior that natively understands multiplication. 
#
# Proceed to `demo_deep.py` to see the symbolic track (`symbolic_order=N`) solve this perfectly.
# Or, if you're more interested in how to make the model transition successfully from the nominal (spline)
# region to the asymptotic (linear) region in shallow networks, go straight to `demo_splines.py`!
