import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .kan import KAN

class TradKAN(KAN):
    """Mimic a traditional KAN implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **(dict(base_activation=nn.SiLU, transition_overlap=1.0) | kwargs))
        self.trust_scale = torch.inf

    def forward(self, x, return_components=False):
        x = super().forward(x, return_components=return_components)
        if return_components:
            x["damages"] = []
        return x

class KANDemonstrator:
    """A utility class for training and visualizing PhysKAN models,
    specifically designed to analyze out-of-bounds (OOB) dual severity tracking.
    """

    def __init__(self, model, target_fn, feature_fn=None, mixer=None):
        self.model = model
        self.target_fn = target_fn
        # If no feature engineering is provided, pass raw features through
        self.feature_fn = feature_fn if feature_fn else lambda x: x
        self.mixer = mixer

    def train(self, x_raw_train, epochs=500, lr=0.05, weight_decay=1e-4, hidden_loss=0.0, stiffness_loss=0.0, sobolev_loss=0.0, l1_l2=0.5):
        """Trains the model using the provided raw input tensors."""
        y_train = self.target_fn(x_raw_train)
        features = self.feature_fn(x_raw_train)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Expecting the network to return (primal_prediction, dual_severity)
            components = self.model(features, return_components=True)
            y_pred, damages = components["final"], components["damages"]
            if self.mixer:
                y_pred = self.mixer(y_pred)
            loss = criterion(y_pred, y_train)
            loss += hidden_loss * self.model.get_deep_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            loss += stiffness_loss * self.model.get_stiffness_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            if sobolev_loss > 0.0:
                loss += sobolev_loss * self.model.get_sobolev_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            loss.backward()
            optimizer.step()

        return loss.item()

    def predict(self, x_raw):
        """Evaluates the model without tracking gradients."""
        self.model.eval()
        features = self.feature_fn(x_raw)
        with torch.no_grad():
            components = self.model(features, return_components=True)
            if self.mixer:
                components["final"] = self.mixer(components["final"])
            return components

    def plot(self, x_raw_eval, title="PhysKAN Demonstration", feature_idx: int = 0, plot_damage: bool | None = None, nominal_range: tuple[float, float] = (-1.0, 1.0), return_components=False):
        """Plots the physical prediction (primal) and severity tracking (dual).
        feature_idx dictates which raw feature column to plot on the x-axis.
        """
        # Sort by the primary plotting axis for clean lines
        sort_idx = torch.argsort(x_raw_eval[:, feature_idx])
        x_raw_eval = x_raw_eval[sort_idx]

        y_true = self.target_fn(x_raw_eval)
        components = self.predict(x_raw_eval)
        y_pred, damages_pred = components["final"], components["damages"]
        if plot_damage is None:
            plot_damage = any((d != 0.0).any() for d in damages_pred)

        x_plot = x_raw_eval[:, feature_idx].numpy()

        if plot_damage:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 2.8))
        fig.suptitle(title, fontsize=14)

        model_type = "TradKAN" if isinstance(self.model, TradKAN) else "PhysKan"

        # 1. Primal Plot (Physics)
        for i in range(y_true.shape[1]):
            label_true = f"True $y_{i}$" if y_true.shape[1] > 1 else "True physics"
            label_pred = f"Pred $y_{i}$" if y_true.shape[1] > 1 else f"{model_type} prediction"
            ax1.plot(x_plot, y_pred[:, i].numpy(), "-", linewidth=2, label=label_pred)
            ax1.plot(x_plot, y_true[:, i].numpy(), "k--", alpha=0.7, label=label_true)

        ax1.axvspan(*nominal_range, color="gray", alpha=0.1, label="Nominal range")
        ax1.set_ylabel("Target value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if plot_damage:
            # 2. Damage Plot (Severity)
            damage_pred = torch.amax(torch.cat(damages_pred, dim=1), dim=1, keepdim=True)
            num_targets = damage_pred.shape[1]

            # Use a color palette that stands out for warnings (reds, oranges, purples)
            severity_colors = ["#ff0000", "#ff7f0e", "#800080", "#d62728"]

            for i in range(num_targets):
                color = severity_colors[i % len(severity_colors)]
                label_str = "" if num_targets == 1 else f" $y_{i}$"
                severity = damage_pred[:, i].numpy()
                ax2.plot(x_plot, severity, color=color, linestyle="-", linewidth=2, label=f"Damage severity{label_str}")

            ax2.axvspan(*nominal_range, color="gray", alpha=0.1)
            ax2.set_ylabel("OOB severity")
            ax2.set_xlabel(f"Input (feature {feature_idx})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel(f"Input (feature {feature_idx})")

        plt.tight_layout()
        plt.show()
        if return_components:
            return components
