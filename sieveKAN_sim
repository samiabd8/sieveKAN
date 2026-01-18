import os
import uuid
import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math

N_OBS = 1000
DIM = 100
S_SPARSE = 30
THETA = 1.5
N_FOLDS = 5
OUTPUT_DIR = "dml_highdim_results"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def get_seed():
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass

    import os
    env_seed = os.getenv('SIMULATION_SEED')
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            pass

    import time as time_module
    return int(time_module.time() * 1000) % 2**32

SEED = get_seed()
print(f"Using seed: {SEED}")
np.random.seed(SEED)
torch.manual_seed(SEED)

PRUNE_DURING_TRAINING = False
POST_TRAINING_PRUNING = True
PRUNING_THRESHOLD_METHOD = 'delta_normalized'
PRUNING_RELATIVE_FRACTION = 0.01

KAN_N_EPOCHS = 1000
KAN_LR = 0.0001
KAN_BATCH_SIZE = 128
KAN_VAL_SPLIT = 0.2
KAN_PATIENCE = 100
KAN_WEIGHT_DECAY = 1e-9
KAN_L1_REG_SCALE = 0.0
KAN_GROUP_LASSO_REG_SCALE = 1e-3
KAN_ENTROPY_REG_SCALE = 0.0
KAN_GAMMA = 0.4
KAN_SPLINE_ORDER = 3
KAN_GRID_RANGE = [-0.05, 1.05]
KAN_BASE_ACTIVATION = nn.SiLU
KAN_PRUNE_THRESHOLD = 1e-7
KAN_ACTIVE_EDGE_THRESHOLD = 1e-4
KAN_RELATIVE_THRESHOLD_FRACTION = 0.01

KAN_ZETA_DELTA = 0.9
KAN_WIDTH = 201
KAN_DEPTH = 6

LASSO_MAX_ITER = 2000
LASSO_ALPHA = None
LASSO_NONZERO_THRESHOLD = 1e-6

C_SLFN = 10

print("\n" + "="*60)
print("PRUNING CONFIGURATION")
print("="*60)
print(f"  Prune during training: {PRUNE_DURING_TRAINING}")
print(f"  Post-training pruning: {POST_TRAINING_PRUNING}")
print(f"  Threshold method: {PRUNING_THRESHOLD_METHOD}")
if PRUNING_THRESHOLD_METHOD == 'relative_fraction':
    print(f"  Relative fraction: {PRUNING_RELATIVE_FRACTION}")
print("="*60 + "\n")

class EmpiricalCDFTransformer:
    def __init__(self):
        self.sorted_values_ = None
        self.n_train_ = None

    def fit(self, X: np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.sorted_values_ = np.sort(X, axis=0)
        self.n_train_ = X.shape[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.sorted_values_ is None:
            raise ValueError("Must call fit() first")
        n = self.n_train_
        cdf = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[1]):
            cdf[:, i] = np.searchsorted(self.sorted_values_[:, i], X[:, i], side='right') / (n + 1.0)
        return np.clip(cdf, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class BSplineBasis(nn.Module):
    def __init__(self, in_features, grid_size=5, spline_order=KAN_SPLINE_ORDER, grid_range=KAN_GRID_RANGE):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.register_buffer("grid", self._create_grid(grid_range, grid_size))

    def _create_grid(self, grid_range, grid_size):
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(
            -self.spline_order,
            grid_size + self.spline_order + 1
        ) * h + grid_range[0]
        return grid.expand(self.in_features, -1).contiguous()

    def b_splines(self, x: torch.Tensor):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]).clamp_min(1e-8)
            right = (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]).clamp_min(1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        return self.b_splines(x).reshape(*original_shape[:-1], -1)

class SieveKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=KAN_SPLINE_ORDER,
                 base_activation=KAN_BASE_ACTIVATION, grid_range=KAN_GRID_RANGE,
                 use_residual=True, delta_n=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.use_residual = use_residual and (in_features == out_features)
        self.delta_n = delta_n

        self.basis = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        self.num_basis = grid_size + spline_order

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.num_basis)
        )

        self.w_b = nn.Parameter(torch.ones(1))
        self.w_s = nn.Parameter(torch.ones(1))
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.uniform_(self.spline_weight, -0.5/self.grid_size, 0.5/self.grid_size)
        with torch.no_grad():
            self.w_b.fill_(0.0)
            self.w_s.fill_(1.0)

    def set_transition(self, s):
        with torch.no_grad():
            self.w_b.fill_(1 - s)
            self.w_s.fill_(s)

    def project_to_delta_constraint(self):
        if self.delta_n is not None:
            total_coeff_norm = torch.sum(torch.abs(self.spline_weight))
            if total_coeff_norm > self.delta_n:
                scale_factor = self.delta_n / (total_coeff_norm + 1e-8)
                self.spline_weight.data *= scale_factor

    def forward(self, x: torch.Tensor):
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_bases = self.basis(x)
        spline_bases = spline_bases.view(x.size(0), -1)

        spline_out = F.linear(
            spline_bases,
            self.spline_weight.view(self.out_features, -1)
        )

        output = self.w_b * base_out + self.w_s * spline_out

        if self.use_residual:
            output = output + x

        return output

    def l1_regularization_loss(self, l1_reg_scale=KAN_L1_REG_SCALE):
        return l1_reg_scale * torch.sum(torch.abs(self.spline_weight))

    def group_lasso_regularization_loss(self, group_lasso_reg_scale=KAN_GROUP_LASSO_REG_SCALE):
        edge_norms = torch.norm(self.spline_weight, p=2, dim=2)
        group_lasso_loss = group_lasso_reg_scale * torch.sum(edge_norms)
        return group_lasso_loss

    def count_active_edges(self, threshold=KAN_ACTIVE_EDGE_THRESHOLD):
        with torch.no_grad():
            edge_norms = torch.norm(self.spline_weight, p=2, dim=2)
            active_edges = torch.sum(edge_norms > threshold).item()
            return active_edges

    def count_total_edges(self):
        return self.out_features * self.in_features

    def count_active_activations(self, threshold=1e-6):
        with torch.no_grad():
            active_count = torch.sum(torch.abs(self.spline_weight) > threshold).item()
            return active_count

class SieveKAN(nn.Module):
    def __init__(self, input_dim, n_samples=N_OBS, gamma=KAN_GAMMA, kan_width=None, depth=None, 
                 zeta_delta=KAN_ZETA_DELTA, prune_during_training=PRUNE_DURING_TRAINING):
        super().__init__()
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.gamma = gamma
        self.zeta_delta = zeta_delta
        self.prune_during_training = prune_during_training

        self.G = max(5, int(n_samples**gamma))
        self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
        self.W = kan_width if kan_width is not None else (2 * S_SPARSE + 1)

        log_delta = np.log(n_samples)
        poly_delta = n_samples ** zeta_delta if zeta_delta is not None else 1
        self.delta_n = max(5, log_delta, poly_delta)

        print(f"Network parameters: L={self.L}, W={self.W}, G={self.G}, Δ_n={self.delta_n:.2f}")
        print(f"Prune during training: {self.prune_during_training}")

        self.layers = nn.ModuleList()
        self.layers.append(SieveKANLayer(
            input_dim, self.W, grid_size=self.G, use_residual=False, delta_n=self.delta_n / self.L
        ))
        for i in range(self.L - 2):
            self.layers.append(SieveKANLayer(
                self.W, self.W, grid_size=self.G, use_residual=True, delta_n=self.delta_n / self.L
            ))
        self.layers.append(SieveKANLayer(
            self.W, 1, grid_size=self.G, use_residual=False, delta_n=self.delta_n / self.L
        ))

        self.scaler_X = EmpiricalCDFTransformer()
        self.scaler_y = StandardScaler()

    def compute_delta_penalty(self, lambda_delta=1.0):
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += torch.sum(torch.abs(layer.spline_weight))
        violation = total_l1 - self.delta_n
        return lambda_delta * torch.clamp(violation, min=0.0)**2

    def final_delta_projection(self):
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += torch.sum(torch.abs(layer.spline_weight))

        print(f"  Total ℓ₁ before final projection: {total_l1.item():.6f} (Δₙ = {self.delta_n:.3f})")

        if total_l1 > self.delta_n:
            scale = self.delta_n / (total_l1.item() + 1e-12)
            with torch.no_grad():
                for layer in self.layers:
                    layer.spline_weight.data *= scale
            print(f"  → Projected: scaled by {scale:.6f} → ℓ₁ = {self.delta_n:.6f}")
        else:
            print(f"  → Already satisfies Δₙ constraint (no projection needed)")

    def prune_edges(self, threshold_method='delta_over_r', threshold_value=None):
        if threshold_method == 'fixed':
            threshold = threshold_value if threshold_value is not None else KAN_ACTIVE_EDGE_THRESHOLD
            method_str = f"Fixed (threshold={threshold:.6f})"

        elif threshold_method == 'delta_normalized':
            total_edges = self.count_total_edges()
            threshold = self.delta_n / total_edges if total_edges > 0 else 0.0
            method_str = f"Δₙ/total_edges (threshold={threshold:.6f}, Δₙ={self.delta_n:.3f}, total_edges={total_edges})"

        elif threshold_method == 'delta_over_r':
            r_n = self.get_total_potential_activations()
            threshold = self.delta_n / r_n if r_n > 0 else 0.0
            method_str = f"Δₙ/rₙ (threshold={threshold:.6f}, Δₙ={self.delta_n:.3f}, rₙ={r_n})"

        elif threshold_method == 'relative_fraction':
            with torch.no_grad():
                max_edge_norm = 0.0
                for layer in self.layers:
                    edge_norms = torch.norm(layer.spline_weight, p=2, dim=2)
                    layer_max = edge_norms.max().item()
                    if layer_max > max_edge_norm:
                        max_edge_norm = layer_max
                fraction = threshold_value if threshold_value is not None else PRUNING_RELATIVE_FRACTION
                threshold = fraction * max_edge_norm
                method_str = f"Relative fraction (threshold={threshold:.6f}, fraction={fraction}, max_norm={max_edge_norm:.6f})"
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")

        pruned_count = 0
        total_edges_count = 0

        with torch.no_grad():
            for layer in self.layers:
                edge_norms = torch.norm(layer.spline_weight, p=2, dim=2)
                mask = edge_norms > threshold
                layer.spline_weight.data *= mask.unsqueeze(-1).float()
                pruned_count += torch.sum(~mask).item()
                total_edges_count += mask.numel()

        sparsity = pruned_count / total_edges_count if total_edges_count > 0 else 0.0

        print(f"  Pruning method: {method_str}")
        print(f"  Pruned {pruned_count}/{total_edges_count} edges")
        print(f"  Final sparsity: {sparsity:.4f}")

        return pruned_count, total_edges_count, threshold, sparsity

    def apply_post_training_pruning(self, threshold_method=PRUNING_THRESHOLD_METHOD, threshold_value=None):
        print("\n" + "="*60)
        print("APPLYING POST-TRAINING PRUNING")
        print("="*60)

        results = self.prune_edges(threshold_method, threshold_value)

        active_edges = self.count_active_edges()
        total_edges = self.count_total_edges()
        final_sparsity = 1 - (active_edges / total_edges) if total_edges > 0 else 0.0
        print(f"  Final active edges: {active_edges}/{total_edges} (sparsity={final_sparsity:.4f})")

        return results

    def set_transition(self, s):
        for layer in self.layers:
            layer.set_transition(s)

    def project_all_layers(self):
        for layer in self.layers:
            layer.project_to_delta_constraint()

    def l1_regularization_loss(self, l1_reg_scale=KAN_L1_REG_SCALE):
        total_l1_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_l1_loss += layer.l1_regularization_loss(l1_reg_scale)
        return total_l1_loss

    def group_lasso_regularization_loss(self, group_lasso_reg_scale=KAN_GROUP_LASSO_REG_SCALE):
        total_group_lasso_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            total_group_lasso_loss += layer.group_lasso_regularization_loss(group_lasso_reg_scale)
        return total_group_lasso_loss

    def entropy_regularization_loss(self, entropy_lambda=KAN_ENTROPY_REG_SCALE):
        loss = 0.0
        for layer in self.layers:
            coeffs = layer.spline_weight.abs() + 1e-12
            edge_sums = coeffs.sum(dim=2, keepdim=True)
            p = coeffs / edge_sums
            entropy = -torch.sum(p * torch.log(p), dim=2)
            active_mask = edge_sums.squeeze(-1) > 1e-8
            loss += entropy_lambda * torch.sum(entropy[active_mask])
        return loss

    def compute_lambda_reg(self):
        n = self.n_samples
        total_params = self.L * (self.W ** 2) * self.G
        lambda_reg = np.sqrt(np.log(total_params + 1e-8) / n)
        return lambda_reg

    def compute_relative_edge_threshold(self, relative_fraction=KAN_RELATIVE_THRESHOLD_FRACTION):
        with torch.no_grad():
            max_edge_norm = 0.0
            for layer in self.layers:
                edge_norms = torch.norm(layer.spline_weight, p=2, dim=2)
                layer_max = edge_norms.max().item()
                if layer_max > max_edge_norm:
                    max_edge_norm = layer_max
            threshold = relative_fraction * max_edge_norm
            return threshold, max_edge_norm

    def count_active_edges(self, threshold=KAN_ACTIVE_EDGE_THRESHOLD):
        total_active = 0
        for layer in self.layers:
            total_active += layer.count_active_edges(threshold)
        return total_active

    def count_active_edges_relative(self, relative_fraction=KAN_RELATIVE_THRESHOLD_FRACTION):
        threshold, max_edge_norm = self.compute_relative_edge_threshold(relative_fraction)
        active_edges = self.count_active_edges(threshold)
        return active_edges, threshold, max_edge_norm

    def count_active_edges_delta_normalized(self):
        total_potential_edges = self.count_total_edges()
        threshold = self.delta_n / total_potential_edges
        return self.count_active_edges(threshold), threshold, total_potential_edges

    def count_active_edges_delta_over_r(self):
        r_n = self.get_total_potential_activations()
        threshold = self.delta_n / r_n if r_n > 0 else 0.0
        return self.count_active_edges(threshold), threshold, r_n

    def count_total_edges(self):
        total = 0
        for layer in self.layers:
            total += layer.count_total_edges()
        return total

    def count_active_activations(self, threshold=1e-6):
        total_active = 0
        for layer in self.layers:
            total_active += layer.count_active_activations(threshold)
        return total_active

    def get_total_potential_activations(self):
        total = 0
        for layer in self.layers:
            total += layer.out_features * layer.in_features * layer.num_basis
        return total

    def sparsity_ratio(self, relative_fraction=KAN_RELATIVE_THRESHOLD_FRACTION):
        active_edges_fixed = self.count_active_edges()
        total_edges = self.count_total_edges()
        fixed_sparsity = 1 - (active_edges_fixed / total_edges)

        active_edges_relative, threshold_relative, max_edge_norm = self.count_active_edges_relative(relative_fraction)
        relative_sparsity = 1 - (active_edges_relative / total_edges)

        active_edges_delta, threshold_delta, total_potential = self.count_active_edges_delta_normalized()
        delta_sparsity = 1 - (active_edges_delta / total_edges)

        active_edges_delta_over_r, threshold_delta_over_r, r_n = self.count_active_edges_delta_over_r()
        delta_over_r_sparsity = 1 - (active_edges_delta_over_r / total_edges)

        kart_optimal_edges = 2 * S_SPARSE + 1
        kart_theoretical_sparsity = 1 - (kart_optimal_edges / total_edges)

        return {
            'fixed_sparsity': fixed_sparsity,
            'relative_sparsity': relative_sparsity,
            'delta_sparsity': delta_sparsity,
            'delta_over_r_sparsity': delta_over_r_sparsity,
            'kart_theoretical_sparsity': kart_theoretical_sparsity,
            'active_edges_fixed': active_edges_fixed,
            'active_edges_relative': active_edges_relative,
            'active_edges_delta': active_edges_delta,
            'active_edges_delta_over_r': active_edges_delta_over_r,
            'threshold_fixed': KAN_ACTIVE_EDGE_THRESHOLD,
            'threshold_relative': threshold_relative,
            'threshold_delta': threshold_delta,
            'threshold_delta_over_r': threshold_delta_over_r,
            'max_edge_norm': max_edge_norm,
            'total_potential_edges': total_edges,
            'r_n': r_n,
            'delta_n': self.delta_n
        }

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, X_np, y_np, epochs=KAN_N_EPOCHS, lr=KAN_LR, batch_size=KAN_BATCH_SIZE,
            val_split=KAN_VAL_SPLIT, patience=KAN_PATIENCE, weight_decay=KAN_WEIGHT_DECAY,
            l1_reg_scale=KAN_L1_REG_SCALE, group_lasso_reg_scale=KAN_GROUP_LASSO_REG_SCALE,
            model_type='m', fold_idx=1):
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_np, y_np, test_size=val_split, random_state=SEED + fold_idx
        )

        self.scaler_X.fit(X_train_np)
        self.scaler_y.fit(y_train_np)

        X_train = torch.tensor(self.scaler_X.transform(X_train_np), dtype=torch.float32, device=DEVICE)
        y_train = torch.tensor(self.scaler_y.transform(y_train_np), dtype=torch.float32, device=DEVICE)
        X_val = torch.tensor(self.scaler_X.transform(X_val_np), dtype=torch.float32, device=DEVICE)
        y_val = torch.tensor(self.scaler_y.transform(y_val_np), dtype=torch.float32, device=DEVICE)

        self.to(DEVICE)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0
        lambda_reg = self.compute_lambda_reg()
        lambda_delta = 1.0
        entropy_lambda = KAN_ENTROPY_REG_SCALE

        transition_complete_epoch = None

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            epoch_mse_loss = 0.0
            transition = 1.0
            self.set_transition(transition)

            if transition >= 1.0 and transition_complete_epoch is None:
                transition_complete_epoch = epoch
                print(f'  Transition to pure splines complete at epoch {epoch}')

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self(X_batch)
                mse_loss = criterion(pred, y_batch)

                l1_loss_val = self.l1_regularization_loss(l1_reg_scale)
                group_lasso_loss_val = self.group_lasso_regularization_loss(group_lasso_reg_scale)
                reg_loss = lambda_reg * (l1_loss_val + group_lasso_loss_val)

                entropy_loss_val = self.entropy_regularization_loss(entropy_lambda)

                delta_penalty = self.compute_delta_penalty(lambda_delta=lambda_delta)

                total_loss = mse_loss + reg_loss + entropy_loss_val + delta_penalty
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                epoch_loss += total_loss.item() * X_batch.size(0)
                epoch_mse_loss += mse_loss.item() * X_batch.size(0)

            scheduler.step()

            avg_epoch_loss = epoch_loss / len(X_train)
            avg_epoch_mse_loss = epoch_mse_loss / len(X_train)
            train_losses.append(avg_epoch_mse_loss)

            self.eval()
            with torch.no_grad():
                val_pred = self(X_val)
                val_loss = criterion(val_pred, y_val).item()
                val_losses.append(val_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                active_edges_delta, threshold_delta, total_edges = self.count_active_edges_delta_normalized()
                delta_sparsity = 1 - (active_edges_delta / total_edges) if total_edges > 0 else 0.0

                if self.prune_during_training:
                    pruned_count, total_edges_count, prune_threshold, prune_sparsity = self.prune_edges(
                        threshold_method='delta_normalized'
                    )
                else:
                    pruned_count, total_edges_count, prune_threshold, prune_sparsity = (0, 0, 0.0, 0.0)

                current_lr = scheduler.get_last_lr()[0]
                print(f' {model_type} Fold {fold_idx}, Epoch {epoch:4d}/{epochs}: '
                      f'Train Loss = {avg_epoch_mse_loss:.4f}, Val Loss = {val_loss:.4f}, '
                      f'Active Edges (Δₙ/total) = {active_edges_delta}/{total_edges}, '
                      f'Δₙ-Sparsity = {delta_sparsity:.4f}, '
                      f'Pruned during training = {self.prune_during_training}, '
                      f'LR = {current_lr:.6f}')

            if transition_complete_epoch is not None and epoch >= transition_complete_epoch:
                if epoch == transition_complete_epoch:
                    best_val_loss = float('inf')
                    epochs_no_improve = 0
                    print(f'  Resetting best validation loss at start of post-transition phase')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'  Early stopping at epoch {epoch} (transition completed at epoch {transition_complete_epoch})')
                        break
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                epochs_no_improve = 0

        return avg_epoch_mse_loss, best_val_loss

    def predict(self, X_np):
        self.eval()
        self.set_transition(1.0)
        self.to(DEVICE)
        X_t = self.scaler_X.transform(X_np)
        X_tensor = torch.tensor(X_t, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred_uniform = self(X_tensor).cpu().numpy()

        pred_original = self.scaler_y.inverse_transform(pred_uniform)
        return pred_original.flatten()

class SLFN(nn.Module):
    def __init__(self, input_dim: int, sample_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.sample_size = sample_size

        d = input_dim
        alpha = 1.0
        r_n = int((sample_size / np.log(sample_size)) ** (1/(2*(1 + alpha/(d+1)))) * C_SLFN)
        self.hidden_dim = max(r_n, 1)
        self.B_n = np.sqrt(sample_size / np.log(sample_size))

        print(f"SLFN parameters: input_dim={input_dim}, hidden_dim={self.hidden_dim}, "
              f"B_n={self.B_n:.2f}, sample_size={sample_size}")

        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.activation = nn.Sigmoid()
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        W = np.random.randn(self.input_dim, self.hidden_dim)
        norms = np.maximum(np.linalg.norm(W, axis=0), 1.0)
        W = W / norms[None, :]
        scale = self.B_n / np.sum(np.abs(W))
        W = W * min(scale, 1.0)
        self.fc1.weight.data = torch.tensor(W.T, dtype=torch.float32)
        self.fc1.bias.data.zero_()

        v = np.random.randn(self.hidden_dim, 1)
        scale = self.B_n / np.sum(np.abs(v))
        v = v * min(scale, 1.0)
        self.fc2.weight.data = torch.tensor(v.T, dtype=torch.float32)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x_hidden = self.activation(self.fc1(x))
        output = self.fc2(x_hidden)
        return output

def dml_estimator(Y, D, m_hat, g_hat, theta_true=None):
    V_hat = D - m_hat
    U_hat = Y - g_hat

    psi_a = -V_hat * V_hat
    psi_b = V_hat * U_hat

    theta_hat = -np.mean(psi_b) / np.mean(psi_a)

    psi = psi_b + theta_hat * psi_a

    J = np.mean(psi_a)
    sigma2 = np.mean(psi * psi)

    n = len(D)
    var_theta = sigma2 / (J**2 * n)
    std_error = np.sqrt(var_theta)

    conf_low = theta_hat - 1.96 * std_error
    conf_high = theta_hat + 1.96 * std_error

    bias = theta_hat - theta_true if theta_true is not None else None

    return {
        'theta_hat': theta_hat,
        'std_error': std_error,
        'conf_low': conf_low,
        'conf_high': conf_high,
        'J': J,
        'sigma2': sigma2,
        'bias': bias
    }

def extract_sparsity_metrics(sparsity_analyses_m, sparsity_analyses_g):
    avg_delta_sparsity_m = np.mean([sa['delta_sparsity'] for sa in sparsity_analyses_m])
    avg_delta_sparsity_g = np.mean([sa['delta_sparsity'] for sa in sparsity_analyses_g])
    avg_delta_over_r_sparsity_m = np.mean([sa['delta_over_r_sparsity'] for sa in sparsity_analyses_m])
    avg_delta_over_r_sparsity_g = np.mean([sa['delta_over_r_sparsity'] for sa in sparsity_analyses_g])

    first_m = sparsity_analyses_m[0]
    first_g = sparsity_analyses_g[0]

    return {
        'delta_sparsity_m': avg_delta_sparsity_m,
        'delta_sparsity_g': avg_delta_sparsity_g,
        'delta_over_r_sparsity_m': avg_delta_over_r_sparsity_m,
        'delta_over_r_sparsity_g': avg_delta_over_r_sparsity_g,
        'delta_n_m': first_m['delta_n'],
        'delta_n_g': first_g['delta_n'],
        'total_edges_m': first_m['total_potential_edges'],
        'total_edges_g': first_g['total_potential_edges'],
        'r_n_m': first_m['r_n'],
        'r_n_g': first_g['r_n'],
        'threshold_delta_m': first_m['threshold_delta'],
        'threshold_delta_g': first_g['threshold_delta'],
        'threshold_delta_over_r_m': first_m['threshold_delta_over_r'],
        'threshold_delta_over_r_g': first_g['threshold_delta_over_r']
    }

def cross_fitting_slfn(X, D, Y, n_folds=N_FOLDS, epochs=500, lr=0.001, batch_size=16):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    m_hat = np.zeros(n)
    g_hat = np.zeros(n)

    fold_metrics = []

    current_idx = 0
    for fold_idx in range(n_folds):
        start_idx = current_idx
        end_idx = current_idx + fold_sizes[fold_idx]
        current_idx = end_idx

        test_indices = indices[start_idx:end_idx]
        train_indices = np.delete(indices, slice(start_idx, end_idx))

        X_train_all, X_test = X[train_indices], X[test_indices]
        D_train_all, D_test = D[train_indices], D[test_indices]
        Y_train_all, Y_test = Y[train_indices], Y[test_indices]

        X_train, X_val, D_train, D_val, Y_train, Y_val = train_test_split(
            X_train_all, D_train_all, Y_train_all, test_size=0.2, random_state=SEED + fold_idx
        )

        scaler_X_m = StandardScaler()
        scaler_D = StandardScaler()

        X_train_scaled_m = scaler_X_m.fit_transform(X_train)
        X_val_scaled_m = scaler_X_m.transform(X_val)
        X_test_scaled_m = scaler_X_m.transform(X_test)

        D_train_scaled = scaler_D.fit_transform(D_train.reshape(-1, 1)).flatten()
        D_val_scaled = scaler_D.transform(D_val.reshape(-1, 1)).flatten()
        D_test_scaled = scaler_D.transform(D_test.reshape(-1, 1)).flatten()

        model_m = SLFN(input_dim=X.shape[1], sample_size=len(train_indices))
        model_m.to(DEVICE)

        X_train_tensor = torch.tensor(X_train_scaled_m, dtype=torch.float32, device=DEVICE)
        D_train_tensor = torch.tensor(D_train_scaled, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        X_val_tensor = torch.tensor(X_val_scaled_m, dtype=torch.float32, device=DEVICE)
        X_test_tensor = torch.tensor(X_test_scaled_m, dtype=torch.float32, device=DEVICE)

        optimizer_m = torch.optim.Adam(model_m.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            model_m.train()
            optimizer_m.zero_grad()

            indices_batch = torch.randperm(len(X_train_tensor))[:batch_size]
            X_batch = X_train_tensor[indices_batch]
            D_batch = D_train_tensor[indices_batch]

            pred = model_m(X_batch)
            loss = criterion(pred, D_batch)

            loss.backward()
            optimizer_m.step()

            model_m.eval()
            with torch.no_grad():
                val_pred = model_m(X_val_tensor)
                val_loss = criterion(val_pred, torch.tensor(D_val_scaled, dtype=torch.float32, device=DEVICE).reshape(-1, 1))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        model_m.eval()
        with torch.no_grad():
            m_train_pred_scaled = model_m(X_train_tensor).cpu().numpy().flatten()
            m_val_pred_scaled = model_m(X_val_tensor).cpu().numpy().flatten()
            m_test_pred_scaled = model_m(X_test_tensor).cpu().numpy().flatten()

        m_train_pred = scaler_D.inverse_transform(m_train_pred_scaled.reshape(-1, 1)).flatten()
        m_val_pred = scaler_D.inverse_transform(m_val_pred_scaled.reshape(-1, 1)).flatten()
        m_test_pred = scaler_D.inverse_transform(m_test_pred_scaled.reshape(-1, 1)).flatten()

        m_metrics = {
            'train': {
                'mse': mean_squared_error(D_train, m_train_pred),
                'r2': r2_score(D_train, m_train_pred),
                'mse_scaled': mean_squared_error(D_train_scaled, m_train_pred_scaled),
                'r2_scaled': r2_score(D_train_scaled, m_train_pred_scaled)
            },
            'val': {
                'mse': mean_squared_error(D_val, m_val_pred),
                'r2': r2_score(D_val, m_val_pred),
                'mse_scaled': mean_squared_error(D_val_scaled, m_val_pred_scaled),
                'r2_scaled': r2_score(D_val_scaled, m_val_pred_scaled)
            },
            'test': {
                'mse': mean_squared_error(D_test, m_test_pred),
                'r2': r2_score(D_test, m_test_pred),
                'mse_scaled': mean_squared_error(D_test_scaled, m_test_pred_scaled),
                'r2_scaled': r2_score(D_test_scaled, m_test_pred_scaled)
            }
        }

        with torch.no_grad():
            output_weights = model_m.fc2.weight.data.cpu().numpy()
            active_units_m = np.sum(np.abs(output_weights) > 1e-6)

        m_hat[test_indices] = m_test_pred

        scaler_X_g = StandardScaler()
        scaler_Y = StandardScaler()

        X_train_scaled_g = scaler_X_g.fit_transform(X_train)
        X_val_scaled_g = scaler_X_g.transform(X_val)
        X_test_scaled_g = scaler_X_g.transform(X_test)

        Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
        Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
        Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

        model_g = SLFN(input_dim=X.shape[1], sample_size=len(train_indices))
        model_g.to(DEVICE)

        X_train_tensor_g = torch.tensor(X_train_scaled_g, dtype=torch.float32, device=DEVICE)
        Y_train_tensor = torch.tensor(Y_train_scaled, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
        X_val_tensor_g = torch.tensor(X_val_scaled_g, dtype=torch.float32, device=DEVICE)
        X_test_tensor_g = torch.tensor(X_test_scaled_g, dtype=torch.float32, device=DEVICE)

        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=lr)

        best_val_loss_g = float('inf')
        patience_counter_g = 0

        for epoch in range(epochs):
            model_g.train()
            optimizer_g.zero_grad()

            indices_batch = torch.randperm(len(X_train_tensor_g))[:batch_size]
            X_batch = X_train_tensor_g[indices_batch]
            Y_batch = Y_train_tensor[indices_batch]

            pred = model_g(X_batch)
            loss = criterion(pred, Y_batch)

            loss.backward()
            optimizer_g.step()

            model_g.eval()
            with torch.no_grad():
                val_pred = model_g(X_val_tensor_g)
                val_loss = criterion(val_pred, torch.tensor(Y_val_scaled, dtype=torch.float32, device=DEVICE).reshape(-1, 1))

                if val_loss < best_val_loss_g:
                    best_val_loss_g = val_loss
                    patience_counter_g = 0
                else:
                    patience_counter_g += 1

                if patience_counter_g >= patience:
                    break

        model_g.eval()
        with torch.no_grad():
            g_train_pred_scaled = model_g(X_train_tensor_g).cpu().numpy().flatten()
            g_val_pred_scaled = model_g(X_val_tensor_g).cpu().numpy().flatten()
            g_test_pred_scaled = model_g(X_test_tensor_g).cpu().numpy().flatten()

        g_train_pred = scaler_Y.inverse_transform(g_train_pred_scaled.reshape(-1, 1)).flatten()
        g_val_pred = scaler_Y.inverse_transform(g_val_pred_scaled.reshape(-1, 1)).flatten()
        g_test_pred = scaler_Y.inverse_transform(g_test_pred_scaled.reshape(-1, 1)).flatten()

        g_metrics = {
            'train': {
                'mse': mean_squared_error(Y_train, g_train_pred),
                'r2': r2_score(Y_train, g_train_pred),
                'mse_scaled': mean_squared_error(Y_train_scaled, g_train_pred_scaled),
                'r2_scaled': r2_score(Y_train_scaled, g_train_pred_scaled)
            },
            'val': {
                'mse': mean_squared_error(Y_val, g_val_pred),
                'r2': r2_score(Y_val, g_val_pred),
                'mse_scaled': mean_squared_error(Y_val_scaled, g_val_pred_scaled),
                'r2_scaled': r2_score(Y_val_scaled, g_val_pred_scaled)
            },
            'test': {
                'mse': mean_squared_error(Y_test, g_test_pred),
                'r2': r2_score(Y_test, g_test_pred),
                'mse_scaled': mean_squared_error(Y_test_scaled, g_test_pred_scaled),
                'r2_scaled': r2_score(Y_test_scaled, g_test_pred_scaled)
            }
        }

        g_hat[test_indices] = g_test_pred

        with torch.no_grad():
            output_weights_g = model_g.fc2.weight.data.cpu().numpy()
            active_units_g = np.sum(np.abs(output_weights_g) > 1e-6)

        V_hat_train = D_train - m_train_pred
        V_hat_val = D_val - m_val_pred
        V_hat_test = D_test - m_test_pred

        v_hat_stats = {
            'train': {
                'mean': np.mean(V_hat_train),
                'std': np.std(V_hat_train),
                'corr_with_d': np.corrcoef(V_hat_train, D_train)[0,1] if len(D_train) > 1 else 0,
                'denominator': np.sum(V_hat_train * D_train) if len(D_train) > 0 else 0
            },
            'val': {
                'mean': np.mean(V_hat_val),
                'std': np.std(V_hat_val),
                'corr_with_d': np.corrcoef(V_hat_val, D_val)[0,1] if len(D_val) > 1 else 0,
                'denominator': np.sum(V_hat_val * D_val) if len(D_val) > 0 else 0
            },
            'test': {
                'mean': np.mean(V_hat_test),
                'std': np.std(V_hat_test),
                'corr_with_d': np.corrcoef(V_hat_test, D_test)[0,1] if len(D_test) > 1 else 0,
                'denominator': np.sum(V_hat_test * D_test) if len(D_test) > 0 else 0
            }
        }

        fold_metrics.append({
            'fold': fold_idx + 1,
            'model_m_metrics': m_metrics,
            'model_g_metrics': g_metrics,
            'active_units_m': active_units_m,
            'active_units_g': active_units_g,
            'hidden_dim_m': model_m.hidden_dim,
            'hidden_dim_g': model_g.hidden_dim,
            'v_hat_stats': v_hat_stats
        })

        print(f"SLFN Fold {fold_idx + 1}: "
              f"m active units = {active_units_m}/{model_m.hidden_dim} (R²: train={m_metrics['train']['r2']:.3f}, val={m_metrics['val']['r2']:.3f}, test={m_metrics['test']['r2']:.3f}), "
              f"g active units = {active_units_g}/{model_g.hidden_dim} (R²: train={g_metrics['train']['r2']:.3f}, val={g_metrics['val']['r2']:.3f}, test={g_metrics['test']['r2']:.3f})")

    aggregated = {
        'fold_metrics': fold_metrics,
        'overall': {
            'm_train_r2_mean': np.mean([fm['model_m_metrics']['train']['r2'] for fm in fold_metrics]),
            'm_val_r2_mean': np.mean([fm['model_m_metrics']['val']['r2'] for fm in fold_metrics]),
            'm_test_r2_mean': np.mean([fm['model_m_metrics']['test']['r2'] for fm in fold_metrics]),
            'g_train_r2_mean': np.mean([fm['model_g_metrics']['train']['r2'] for fm in fold_metrics]),
            'g_val_r2_mean': np.mean([fm['model_g_metrics']['val']['r2'] for fm in fold_metrics]),
            'g_test_r2_mean': np.mean([fm['model_g_metrics']['test']['r2'] for fm in fold_metrics]),
            'active_units_m_mean': np.mean([fm['active_units_m'] for fm in fold_metrics]),
            'active_units_g_mean': np.mean([fm['active_units_g'] for fm in fold_metrics]),
            'v_hat_test_std_mean': np.mean([fm['v_hat_stats']['test']['std'] for fm in fold_metrics]),
            'v_hat_test_corr_mean': np.mean([fm['v_hat_stats']['test']['corr_with_d'] for fm in fold_metrics])
        }
    }

    return m_hat, g_hat, aggregated

def cross_fitting_lasso(X, D, Y, n_folds=N_FOLDS, lambda_reg=LASSO_ALPHA):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    m_hat = np.zeros(n)
    g_hat = np.zeros(n)

    fold_metrics = []

    current_idx = 0
    for fold_idx in range(n_folds):
        start_idx = current_idx
        end_idx = current_idx + fold_sizes[fold_idx]
        current_idx = end_idx

        test_indices = indices[start_idx:end_idx]
        train_indices = np.delete(indices, slice(start_idx, end_idx))

        X_train_all, X_test = X[train_indices], X[test_indices]
        D_train_all, D_test = D[train_indices], D[test_indices]
        Y_train_all, Y_test = Y[train_indices], Y[test_indices]

        X_train, X_val, D_train, D_val, Y_train, Y_val = train_test_split(
            X_train_all, D_train_all, Y_train_all, test_size=0.2, random_state=SEED + fold_idx
        )

        scaler_X_m = StandardScaler()
        scaler_D = StandardScaler()

        X_train_scaled_m = scaler_X_m.fit_transform(X_train)
        X_val_scaled_m = scaler_X_m.transform(X_val)
        X_test_scaled_m = scaler_X_m.transform(X_test)

        D_train_scaled = scaler_D.fit_transform(D_train.reshape(-1, 1)).flatten()
        D_val_scaled = scaler_D.transform(D_val.reshape(-1, 1)).flatten()
        D_test_scaled = scaler_D.transform(D_test.reshape(-1, 1)).flatten()

        model_m = Lasso(alpha=lambda_reg, max_iter=LASSO_MAX_ITER, random_state=SEED)
        model_m.fit(X_train_scaled_m, D_train)

        m_train_pred = model_m.predict(X_train_scaled_m)
        m_val_pred = model_m.predict(X_val_scaled_m)
        m_test_pred = model_m.predict(X_test_scaled_m)

        m_train_pred_scaled = scaler_D.transform(m_train_pred.reshape(-1, 1)).flatten()
        m_val_pred_scaled = scaler_D.transform(m_val_pred.reshape(-1, 1)).flatten()
        m_test_pred_scaled = scaler_D.transform(m_test_pred.reshape(-1, 1)).flatten()

        m_metrics = {
            'train': {
                'mse': mean_squared_error(D_train, m_train_pred),
                'r2': r2_score(D_train, m_train_pred),
                'mse_scaled': mean_squared_error(D_train_scaled, m_train_pred_scaled),
                'r2_scaled': r2_score(D_train_scaled, m_train_pred_scaled)
            },
            'val': {
                'mse': mean_squared_error(D_val, m_val_pred),
                'r2': r2_score(D_val, m_val_pred),
                'mse_scaled': mean_squared_error(D_val_scaled, m_val_pred_scaled),
                'r2_scaled': r2_score(D_val_scaled, m_val_pred_scaled)
            },
            'test': {
                'mse': mean_squared_error(D_test, m_test_pred),
                'r2': r2_score(D_test, m_test_pred),
                'mse_scaled': mean_squared_error(D_test_scaled, m_test_pred_scaled),
                'r2_scaled': r2_score(D_test_scaled, m_test_pred_scaled)
            }
        }

        m_hat[test_indices] = m_test_pred
        nonzero_count_m = np.sum(np.abs(model_m.coef_) > LASSO_NONZERO_THRESHOLD)

        scaler_X_g = StandardScaler()
        scaler_Y = StandardScaler()

        X_train_scaled_g = scaler_X_g.fit_transform(X_train)
        X_val_scaled_g = scaler_X_g.transform(X_val)
        X_test_scaled_g = scaler_X_g.transform(X_test)

        Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
        Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
        Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

        model_g = Lasso(alpha=lambda_reg, max_iter=LASSO_MAX_ITER, random_state=SEED)
        model_g.fit(X_train_scaled_g, Y_train)

        g_train_pred = model_g.predict(X_train_scaled_g)
        g_val_pred = model_g.predict(X_val_scaled_g)
        g_test_pred = model_g.predict(X_test_scaled_g)

        g_train_pred_scaled = scaler_Y.transform(g_train_pred.reshape(-1, 1)).flatten()
        g_val_pred_scaled = scaler_Y.transform(g_val_pred.reshape(-1, 1)).flatten()
        g_test_pred_scaled = scaler_Y.transform(g_test_pred.reshape(-1, 1)).flatten()

        g_metrics = {
            'train': {
                'mse': mean_squared_error(Y_train, g_train_pred),
                'r2': r2_score(Y_train, g_train_pred),
                'mse_scaled': mean_squared_error(Y_train_scaled, g_train_pred_scaled),
                'r2_scaled': r2_score(Y_train_scaled, g_train_pred_scaled)
            },
            'val': {
                'mse': mean_squared_error(Y_val, g_val_pred),
                'r2': r2_score(Y_val, g_val_pred),
                'mse_scaled': mean_squared_error(Y_val_scaled, g_val_pred_scaled),
                'r2_scaled': r2_score(Y_val_scaled, g_val_pred_scaled)
            },
            'test': {
                'mse': mean_squared_error(Y_test, g_test_pred),
                'r2': r2_score(Y_test, g_test_pred),
                'mse_scaled': mean_squared_error(Y_test_scaled, g_test_pred_scaled),
                'r2_scaled': r2_score(Y_test_scaled, g_test_pred_scaled)
            }
        }

        g_hat[test_indices] = g_test_pred
        nonzero_count_g = np.sum(np.abs(model_g.coef_) > LASSO_NONZERO_THRESHOLD)

        V_hat_train = D_train - m_train_pred
        V_hat_val = D_val - m_val_pred
        V_hat_test = D_test - m_test_pred

        v_hat_stats = {
            'train': {
                'mean': np.mean(V_hat_train),
                'std': np.std(V_hat_train),
                'corr_with_d': np.corrcoef(V_hat_train, D_train)[0,1] if len(D_train) > 1 else 0,
                'denominator': np.sum(V_hat_train * D_train) if len(D_train) > 0 else 0
            },
            'val': {
                'mean': np.mean(V_hat_val),
                'std': np.std(V_hat_val),
                'corr_with_d': np.corrcoef(V_hat_val, D_val)[0,1] if len(D_val) > 1 else 0,
                'denominator': np.sum(V_hat_val * D_val) if len(D_val) > 0 else 0
            },
            'test': {
                'mean': np.mean(V_hat_test),
                'std': np.std(V_hat_test),
                'corr_with_d': np.corrcoef(V_hat_test, D_test)[0,1] if len(D_test) > 1 else 0,
                'denominator': np.sum(V_hat_test * D_test) if len(D_test) > 0 else 0
            }
        }

        fold_metrics.append({
            'fold': fold_idx + 1,
            'model_m_metrics': m_metrics,
            'model_g_metrics': g_metrics,
            'nonzero_coefs_m': nonzero_count_m,
            'nonzero_coefs_g': nonzero_count_g,
            'v_hat_stats': v_hat_stats
        })

        print(f"LASSO Fold {fold_idx + 1}: "
              f"m nonzero coefs = {nonzero_count_m}/{X.shape[1]} (R²: train={m_metrics['train']['r2']:.3f}, val={m_metrics['val']['r2']:.3f}, test={m_metrics['test']['r2']:.3f}), "
              f"g nonzero coefs = {nonzero_count_g}/{X.shape[1]} (R²: train={g_metrics['train']['r2']:.3f}, val={g_metrics['val']['r2']:.3f}, test={g_metrics['test']['r2']:.3f})")

    aggregated = {
        'fold_metrics': fold_metrics,
        'overall': {
            'm_train_r2_mean': np.mean([fm['model_m_metrics']['train']['r2'] for fm in fold_metrics]),
            'm_val_r2_mean': np.mean([fm['model_m_metrics']['val']['r2'] for fm in fold_metrics]),
            'm_test_r2_mean': np.mean([fm['model_m_metrics']['test']['r2'] for fm in fold_metrics]),
            'g_train_r2_mean': np.mean([fm['model_g_metrics']['train']['r2'] for fm in fold_metrics]),
            'g_val_r2_mean': np.mean([fm['model_g_metrics']['val']['r2'] for fm in fold_metrics]),
            'g_test_r2_mean': np.mean([fm['model_g_metrics']['test']['r2'] for fm in fold_metrics]),
            'nonzero_coefs_m_mean': np.mean([fm['nonzero_coefs_m'] for fm in fold_metrics]),
            'nonzero_coefs_g_mean': np.mean([fm['nonzero_coefs_g'] for fm in fold_metrics]),
            'v_hat_test_std_mean': np.mean([fm['v_hat_stats']['test']['std'] for fm in fold_metrics]),
            'v_hat_test_corr_mean': np.mean([fm['v_hat_stats']['test']['corr_with_d'] for fm in fold_metrics])
        }
    }

    return m_hat, g_hat, aggregated

def cross_fitting_kan(X, D, Y, n_folds=N_FOLDS, n_epochs=KAN_N_EPOCHS, lr=KAN_LR,
                                  gamma=KAN_GAMMA, zeta_delta=KAN_ZETA_DELTA, kan_width=None, depth=None,
                                  patience=KAN_PATIENCE, weight_decay=KAN_WEIGHT_DECAY,
                                  l1_reg_scale=KAN_L1_REG_SCALE, group_lasso_reg_scale=KAN_GROUP_LASSO_REG_SCALE):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_sizes = [n // n_folds] * n_folds
    for i in range(n % n_folds):
        fold_sizes[i] += 1

    m_hat = np.zeros(n)
    g_hat = np.zeros(n)

    fold_metrics = []

    current_idx = 0
    for fold_idx in range(n_folds):
        start_idx = current_idx
        end_idx = current_idx + fold_sizes[fold_idx]
        current_idx = end_idx

        test_indices = indices[start_idx:end_idx]
        train_indices = np.delete(indices, slice(start_idx, end_idx))

        X_train_all, X_test = X[train_indices], X[test_indices]
        D_train_all, D_test = D[train_indices], D[test_indices]
        Y_train_all, Y_test = Y[train_indices], Y[test_indices]

        model_m = SieveKAN(
            input_dim=X.shape[1],
            n_samples=len(train_indices),
            gamma=gamma,
            zeta_delta=zeta_delta,
            kan_width=kan_width,
            depth=depth,
            prune_during_training=PRUNE_DURING_TRAINING
        )

        train_mse_m, val_mse_m = model_m.fit(X_train_all, D_train_all, epochs=n_epochs, lr=lr,
                                             patience=patience, weight_decay=weight_decay,
                                             l1_reg_scale=l1_reg_scale,
                                             group_lasso_reg_scale=group_lasso_reg_scale,
                                             model_type='m', fold_idx=fold_idx + 1)

        if POST_TRAINING_PRUNING:
            model_m.apply_post_training_pruning(
                threshold_method=PRUNING_THRESHOLD_METHOD,
                threshold_value=None
            )

        model_m.final_delta_projection()

        X_train_all_tensor = torch.tensor(model_m.scaler_X.transform(X_train_all), dtype=torch.float32, device=DEVICE)
        X_test_tensor = torch.tensor(model_m.scaler_X.transform(X_test), dtype=torch.float32, device=DEVICE)

        model_m.eval()
        with torch.no_grad():
            m_train_pred_scaled = model_m(X_train_all_tensor).cpu().numpy().flatten()
            m_test_pred_scaled = model_m(X_test_tensor).cpu().numpy().flatten()

        m_train_pred = model_m.scaler_y.inverse_transform(m_train_pred_scaled.reshape(-1, 1)).flatten()
        m_test_pred = model_m.scaler_y.inverse_transform(m_test_pred_scaled.reshape(-1, 1)).flatten()

        D_train_scaled = model_m.scaler_y.transform(D_train_all.reshape(-1, 1)).flatten()
        D_test_scaled = model_m.scaler_y.transform(D_test.reshape(-1, 1)).flatten()

        m_metrics = {
            'train': {
                'mse': mean_squared_error(D_train_all, m_train_pred),
                'r2': r2_score(D_train_all, m_train_pred),
                'mse_scaled': mean_squared_error(D_train_scaled, m_train_pred_scaled),
                'r2_scaled': r2_score(D_train_scaled, m_train_pred_scaled)
            },
            'val': {
                'mse': val_mse_m,
                'r2': None,
                'mse_scaled': val_mse_m,
                'r2_scaled': None
            },
            'test': {
                'mse': mean_squared_error(D_test, m_test_pred),
                'r2': r2_score(D_test, m_test_pred),
                'mse_scaled': mean_squared_error(D_test_scaled, m_test_pred_scaled),
                'r2_scaled': r2_score(D_test_scaled, m_test_pred_scaled)
            }
        }

        m_hat[test_indices] = m_test_pred

        active_edges_m = model_m.count_active_edges()
        total_edges_m = model_m.count_total_edges()

        model_g = SieveKAN(
            input_dim=X.shape[1],
            n_samples=len(train_indices),
            gamma=gamma,
            zeta_delta=zeta_delta,
            kan_width=kan_width,
            depth=depth,
            prune_during_training=PRUNE_DURING_TRAINING
        )

        train_mse_g, val_mse_g = model_g.fit(X_train_all, Y_train_all, epochs=n_epochs, lr=lr,
                                             patience=patience, weight_decay=weight_decay,
                                             l1_reg_scale=l1_reg_scale,
                                             group_lasso_reg_scale=group_lasso_reg_scale,
                                             model_type='g', fold_idx=fold_idx + 1)

        if POST_TRAINING_PRUNING:
            model_g.apply_post_training_pruning(
                threshold_method=PRUNING_THRESHOLD_METHOD,
                threshold_value=None
            )

        model_g.final_delta_projection()

        X_train_all_tensor_g = torch.tensor(model_g.scaler_X.transform(X_train_all), dtype=torch.float32, device=DEVICE)
        X_test_tensor_g = torch.tensor(model_g.scaler_X.transform(X_test), dtype=torch.float32, device=DEVICE)

        model_g.eval()
        with torch.no_grad():
            g_train_pred_scaled = model_g(X_train_all_tensor_g).cpu().numpy().flatten()
            g_test_pred_scaled = model_g(X_test_tensor_g).cpu().numpy().flatten()

        g_train_pred = model_g.scaler_y.inverse_transform(g_train_pred_scaled.reshape(-1, 1)).flatten()
        g_test_pred = model_g.scaler_y.inverse_transform(g_test_pred_scaled.reshape(-1, 1)).flatten()

        Y_train_scaled = model_g.scaler_y.transform(Y_train_all.reshape(-1, 1)).flatten()
        Y_test_scaled = model_g.scaler_y.transform(Y_test.reshape(-1, 1)).flatten()

        g_metrics = {
            'train': {
                'mse': mean_squared_error(Y_train_all, g_train_pred),
                'r2': r2_score(Y_train_all, g_train_pred),
                'mse_scaled': mean_squared_error(Y_train_scaled, g_train_pred_scaled),
                'r2_scaled': r2_score(Y_train_scaled, g_train_pred_scaled)
            },
            'val': {
                'mse': val_mse_g,
                'r2': None,
                'mse_scaled': val_mse_g,
                'r2_scaled': None
            },
            'test': {
                'mse': mean_squared_error(Y_test, g_test_pred),
                'r2': r2_score(Y_test, g_test_pred),
                'mse_scaled': mean_squared_error(Y_test_scaled, g_test_pred_scaled),
                'r2_scaled': r2_score(Y_test_scaled, g_test_pred_scaled)
            }
        }

        g_hat[test_indices] = g_test_pred

        active_edges_g = model_g.count_active_edges()
        total_edges_g = model_g.count_total_edges()

        V_hat_train = D_train_all - m_train_pred
        V_hat_test = D_test - m_test_pred

        v_hat_stats = {
            'train': {
                'mean': np.mean(V_hat_train),
                'std': np.std(V_hat_train),
                'corr_with_d': np.corrcoef(V_hat_train, D_train_all)[0,1] if len(D_train_all) > 1 else 0,
                'denominator': np.sum(V_hat_train * D_train_all) if len(D_train_all) > 0 else 0
            },
            'test': {
                'mean': np.mean(V_hat_test),
                'std': np.std(V_hat_test),
                'corr_with_d': np.corrcoef(V_hat_test, D_test)[0,1] if len(D_test) > 1 else 0,
                'denominator': np.sum(V_hat_test * D_test) if len(D_test) > 0 else 0
            }
        }

        sparsity_analysis_m = model_m.sparsity_ratio()
        sparsity_analysis_g = model_g.sparsity_ratio()

        fold_metrics.append({
            'fold': fold_idx + 1,
            'model_m_metrics': m_metrics,
            'model_g_metrics': g_metrics,
            'active_edges_m': active_edges_m,
            'active_edges_g': active_edges_g,
            'total_edges_m': total_edges_m,
            'total_edges_g': total_edges_g,
            'sparsity_analysis_m': sparsity_analysis_m,
            'sparsity_analysis_g': sparsity_analysis_g,
            'v_hat_stats': v_hat_stats
        })

        print(f"KAN Fold {fold_idx + 1}: "
              f"m active edges = {active_edges_m}/{total_edges_m} (R²: train={m_metrics['train']['r2']:.3f}, test={m_metrics['test']['r2']:.3f}), "
              f"g active edges = {active_edges_g}/{total_edges_g} (R²: train={g_metrics['train']['r2']:.3f}, test={g_metrics['test']['r2']:.3f})")

    aggregated = {
        'fold_metrics': fold_metrics,
        'overall': {
            'm_train_r2_mean': np.mean([fm['model_m_metrics']['train']['r2'] for fm in fold_metrics]),
            'm_test_r2_mean': np.mean([fm['model_m_metrics']['test']['r2'] for fm in fold_metrics]),
            'g_train_r2_mean': np.mean([fm['model_g_metrics']['train']['r2'] for fm in fold_metrics]),
            'g_test_r2_mean': np.mean([fm['model_g_metrics']['test']['r2'] for fm in fold_metrics]),
            'active_edges_m_mean': np.mean([fm['active_edges_m'] for fm in fold_metrics]),
            'active_edges_g_mean': np.mean([fm['active_edges_g'] for fm in fold_metrics]),
            'v_hat_test_std_mean': np.mean([fm['v_hat_stats']['test']['std'] for fm in fold_metrics]),
            'v_hat_test_corr_mean': np.mean([fm['v_hat_stats']['test']['corr_with_d'] for fm in fold_metrics])
        }
    }

    return m_hat, g_hat, aggregated

def dml_highdim_simulation(n=N_OBS, d=DIM, s=S_SPARSE, theta=THETA, n_folds=N_FOLDS,
                                                zeta_delta=KAN_ZETA_DELTA, kan_width=None, depth=None):
    start_time = time.time()
    print(f"Generating DML data: n={n}, d={d}, s={s}, theta={theta}, ζ_δ={zeta_delta}")

    X = np.random.uniform(-3, 3, size=(n, d))

    inner_m = np.sum([1 / (1 + np.exp(-X[:, j])) for j in range(s)], axis=0)
    m0 = np.sin(inner_m) + 0.1 * np.sum([X[:, j]**2 for j in range(s)], axis=0)

    inner_g = np.sum([np.sin(X[:, j]) for j in range(s)], axis=0)
    g0 = 1 / (1 + np.exp(-inner_g)) + 0.1 * np.sum([np.abs(X[:, j]) for j in range(s)], axis=0)

    D = m0 + np.random.normal(0, 0.5, n)
    Y = theta * D + g0 + np.random.normal(0, 0.5, n)

    print(f"Generated data shapes: X={X.shape}, D={D.shape}, Y={Y.shape}")
    print(f"True m0 stats: mean={m0.mean():.3f}, std={m0.std():.3f}")
    print(f"True g0 stats: mean={g0.mean():.3f}, std={g0.std():.3f}")

    all_results = {}

    print("\n" + "="*80)
    print("KAN CROSS-FITTING DML")
    print("="*80)

    kan_start_time = time.time()
    m_hat_kan, g_hat_kan, kan_metrics = cross_fitting_kan(
        X, D, Y, n_folds=n_folds, n_epochs=KAN_N_EPOCHS, lr=KAN_LR,
        zeta_delta=zeta_delta, kan_width=kan_width, depth=depth
    )
    kan_time = time.time() - kan_start_time

    dml_results_kan = dml_estimator(Y, D, m_hat_kan, g_hat_kan, theta_true=theta)

    all_results['kan'] = {
        'm_hat': m_hat_kan,
        'g_hat': g_hat_kan,
        'dml_results': dml_results_kan,
        'metrics': kan_metrics,
        'runtime': kan_time
    }

    print("\n" + "="*80)
    print("LASSO CROSS-FITTING DML")
    print("="*80)

    lasso_start_time = time.time()
    lambda_reg = 0.1 * np.sqrt(np.log(d) / n)
    m_hat_lasso, g_hat_lasso, lasso_metrics = cross_fitting_lasso(
        X, D, Y, n_folds=n_folds, lambda_reg=lambda_reg
    )
    lasso_time = time.time() - lasso_start_time

    dml_results_lasso = dml_estimator(Y, D, m_hat_lasso, g_hat_lasso, theta_true=theta)

    all_results['lasso'] = {
        'm_hat': m_hat_lasso,
        'g_hat': g_hat_lasso,
        'dml_results': dml_results_lasso,
        'metrics': lasso_metrics,
        'runtime': lasso_time
    }

    print("\n" + "="*80)
    print("SLFN CROSS-FITTING DML")
    print("="*80)

    slfn_start_time = time.time()
    m_hat_slfn, g_hat_slfn, slfn_metrics = cross_fitting_slfn(
        X, D, Y, n_folds=n_folds, epochs=500, lr=0.001, batch_size=32
    )
    slfn_time = time.time() - slfn_start_time

    dml_results_slfn = dml_estimator(Y, D, m_hat_slfn, g_hat_slfn, theta_true=theta)

    all_results['slfn'] = {
        'm_hat': m_hat_slfn,
        'g_hat': g_hat_slfn,
        'dml_results': dml_results_slfn,
        'metrics': slfn_metrics,
        'runtime': slfn_time
    }

    print("\n" + "="*80)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("="*80)

    print("\nDML ESTIMATOR PERFORMANCE:")
    print(f"{'Method':<10} {'θ̂':<10} {'Bias':<10} {'Std Error':<12} {'95% CI':<25} {'Coverage':<10}")
    print("-" * 80)

    for method, results in all_results.items():
        dml = results['dml_results']
        coverage = dml['conf_low'] <= theta <= dml['conf_high']
        ci_str = f"[{dml['conf_low']:.4f}, {dml['conf_high']:.4f}]"
        print(f"{method.upper():<10} {dml['theta_hat']:<10.4f} {dml['bias']:<10.4f} {dml['std_error']:<12.4f} {ci_str:<25} {str(coverage):<10}")

    print("\n" + "="*80)
    print("NUISANCE FUNCTION PERFORMANCE (Test Set R²)")
    print("="*80)

    print(f"\n{'Method':<10} {'m(X) Train R²':<15} {'m(X) Test R²':<15} {'g(X) Train R²':<15} {'g(X) Test R²':<15}")
    print("-" * 80)

    for method, results in all_results.items():
        metrics = results['metrics']['overall']
        if method == 'kan':
            print(f"{method.upper():<10} {metrics['m_train_r2_mean']:<15.4f} {metrics['m_test_r2_mean']:<15.4f} {metrics['g_train_r2_mean']:<15.4f} {metrics['g_test_r2_mean']:<15.4f}")
        else:
            print(f"{method.upper():<10} {metrics['m_train_r2_mean']:<15.4f} {metrics['m_test_r2_mean']:<15.4f} {metrics['g_train_r2_mean']:<15.4f} {metrics['g_test_r2_mean']:<15.4f}")

    print("\n" + "="*80)
    print("DML DIAGNOSTICS (V̂ = D - m̂(X) Statistics on Test Set)")
    print("="*80)

    print(f"\n{'Method':<10} {'V̂ Mean':<12} {'V̂ Std':<12} {'Corr(V̂,D)':<12} {'∑V̂D':<15} {'Runtime (s)':<12}")
    print("-" * 80)

    for method, results in all_results.items():
        metrics = results['metrics']['overall']
        runtime = results['runtime']
        if method == 'kan':
            v_stats = results['metrics']['fold_metrics'][0]['v_hat_stats']['test']
            print(f"{method.upper():<10} {v_stats['mean']:<12.4f} {v_stats['std']:<12.4f} {v_stats['corr_with_d']:<12.4f} {v_stats['denominator']:<15.4f} {runtime:<12.2f}")
        else:
            v_stats = results['metrics']['fold_metrics'][0]['v_hat_stats']['test']
            print(f"{method.upper():<10} {v_stats['mean']:<12.4f} {v_stats['std']:<12.4f} {v_stats['corr_with_d']:<12.4f} {v_stats['denominator']:<15.4f} {runtime:<12.2f}")

    print("\n" + "="*80)
    print("MODEL COMPLEXITY")
    print("="*80)

    print(f"\n{'Method':<10} {'Active Params m':<20} {'Active Params g':<20} {'Total Capacity':<20}")
    print("-" * 80)

    for method, results in all_results.items():
        metrics = results['metrics']
        if method == 'kan':
            kan_metrics = metrics['fold_metrics'][0]
            active_m = kan_metrics['active_edges_m']
            active_g = kan_metrics['active_edges_g']
            total_m = kan_metrics['total_edges_m']
            total_g = kan_metrics['total_edges_g']
            print(f"{method.upper():<10} {active_m}/{total_m} ({active_m/total_m*100:.1f}%){' ':10} {active_g}/{total_g} ({active_g/total_g*100:.1f}%){' ':10} {total_m + total_g:<20}")
        elif method == 'lasso':
            lasso_metrics = metrics['fold_metrics'][0]
            active_m = lasso_metrics['nonzero_coefs_m']
            active_g = lasso_metrics['nonzero_coefs_g']
            total = 2 * d
            print(f"{method.upper():<10} {active_m}/{d} ({active_m/d*100:.1f}%){' ':13} {active_g}/{d} ({active_g/d*100:.1f}%){' ':13} {total:<20}")
        elif method == 'slfn':
            slfn_metrics = metrics['fold_metrics'][0]
            active_m = slfn_metrics['active_units_m']
            active_g = slfn_metrics['active_units_g']
            total_m = slfn_metrics['hidden_dim_m']
            total_g = slfn_metrics['hidden_dim_g']
            print(f"{method.upper():<10} {active_m}/{total_m} ({active_m/total_m*100:.1f}%){' ':10} {active_g}/{total_g} ({active_g/total_g*100:.1f}%){' ':10} {total_m + total_g:<20}")

    return all_results

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== COMPREHENSIVE HIGH-DIMENSIONAL DML EVALUATION ===")
    print(f"Parameters: n={N_OBS}, d={DIM}, s={S_SPARSE}, θ={THETA}, ζ_δ={KAN_ZETA_DELTA}")
    print(f"Training: epochs={KAN_N_EPOCHS}, lr={KAN_LR}, weight_decay={KAN_WEIGHT_DECAY}")
    print(f"KAN Network: L={KAN_DEPTH}, W={KAN_WIDTH}, G={max(5, int(N_OBS**KAN_GAMMA))}")
    print(f"LASSO: lambda={0.1 * np.sqrt(np.log(DIM) / N_OBS):.4f}")
    print(f"SLFN (Chen & White 1999): C_SLFN={C_SLFN}")
    print(f"Using seed: {SEED}")
    print(f"Pruning: during_training={PRUNE_DURING_TRAINING}, post_training={POST_TRAINING_PRUNING}, method={PRUNING_THRESHOLD_METHOD}")

    print("\n" + "="*80)
    print("RUNNING COMPOSITIONAL DGP:")
    print("m(X) = sin(Σ logistic(X_j)) + 0.1ΣX_j²")
    print("g(X) = logistic(Σ sin(X_j)) + 0.1Σ|X_j|")
    print("="*80)

    results = dml_highdim_simulation(
        n=N_OBS, d=DIM, s=S_SPARSE, theta=THETA, n_folds=N_FOLDS,
        zeta_delta=KAN_ZETA_DELTA, kan_width=KAN_WIDTH, depth=KAN_DEPTH
    )

    all_rows = []

    for method, method_results in results.items():
        metrics = method_results['metrics']['overall']
        dml = method_results['dml_results']

        v_stats = method_results['metrics']['fold_metrics'][0]['v_hat_stats']['test']

        row = {
            'method': method,
            'seed': SEED,

            'theta_hat': dml['theta_hat'],
            'bias': dml['bias'],
            'std_error': dml['std_error'],
            'conf_low': dml['conf_low'],
            'conf_high': dml['conf_high'],
            'coverage': dml['conf_low'] <= THETA <= dml['conf_high'],
            'J_gradient': dml['J'],
            'sigma2': dml.get('sigma2', 0),

            'm_train_r2': metrics.get('m_train_r2_mean', np.nan),
            'm_val_r2': metrics.get('m_val_r2_mean', np.nan),
            'm_test_r2': metrics.get('m_test_r2_mean', np.nan),
            'g_train_r2': metrics.get('g_train_r2_mean', np.nan),
            'g_val_r2': metrics.get('g_val_r2_mean', np.nan),
            'g_test_r2': metrics.get('g_test_r2_mean', np.nan),

            'v_hat_mean': v_stats['mean'],
            'v_hat_std': v_stats['std'],
            'v_hat_corr_d': v_stats['corr_with_d'],
            'v_hat_denominator': v_stats['denominator'],

            'active_params_m': metrics.get('active_edges_m_mean', 
                                          metrics.get('nonzero_coefs_m_mean',
                                                     metrics.get('active_units_m_mean', np.nan))),
            'active_params_g': metrics.get('active_edges_g_mean',
                                          metrics.get('nonzero_coefs_g_mean',
                                                     metrics.get('active_units_g_mean', np.nan))),

            'runtime': method_results['runtime'],

            'n_obs': N_OBS,
            'dim': DIM,
            's_sparse': S_SPARSE,
            'theta_true': THETA,
            'n_folds': N_FOLDS,
            'zeta_delta': KAN_ZETA_DELTA if method == 'kan' else np.nan,
            'kan_width': KAN_WIDTH if method == 'kan' else np.nan,
            'kan_depth': KAN_DEPTH if method == 'kan' else np.nan,
            'prune_during_training': PRUNE_DURING_TRAINING if method == 'kan' else np.nan,
            'post_training_pruning': POST_TRAINING_PRUNING if method == 'kan' else np.nan,
            'pruning_threshold_method': PRUNING_THRESHOLD_METHOD if method == 'kan' else np.nan,
            'simulation_id': str(uuid.uuid4())[:8],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    file_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"dml_comprehensive_results_{file_id}.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nComprehensive results saved to {output_path}")

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    summary = results_df[['method', 'theta_hat', 'bias', 'm_test_r2', 'g_test_r2', 
                         'v_hat_std', 'v_hat_corr_d', 'active_params_m', 'runtime']].copy()


    print(summary.to_string(index=False))
