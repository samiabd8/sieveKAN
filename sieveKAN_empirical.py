import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time
from scipy.stats import trim_mean, boxcox
from sklearn.linear_model import LassoCV, Lasso

os.chdir("C:/Users/samab/OneDrive/Documents/Research/SparseMacroFinanceFactors")

KAN_VAL_SPLIT = 0.1
KAN_TEST_SPLIT = 0.1

KAN_N_EPOCHS = 5000
KAN_LR = 0.001
KAN_BATCH_SIZE = 16
KAN_PATIENCE = 500
KAN_WEIGHT_DECAY = 1e-5
KAN_L1_REG_SCALE = 0.0
KAN_GROUP_LASSO_REG_SCALE = 5e-3
KAN_GAMMA = 0.5
KAN_SPLINE_ORDER = 3
KAN_GRID_RANGE = [-0.5, 1.5]
KAN_BASE_ACTIVATION = nn.SiLU
KAN_ACTIVE_EDGE_THRESHOLD = 1e-4
KAN_RELATIVE_THRESHOLD_FRACTION = 0.01

LASSO_MAX_ITER = 5000
LASSO_ALPHAS = np.logspace(-6, 1, 50)

KAN_ZETA_DELTA = 0.9
KAN_WIDTH = 243
KAN_DEPTH = 6

C_SLFN = 1.0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("="*50)
print("LOADING EMPIRICAL MACRO-FINANCE DATA")
print("="*50)

try:
    data_macro = pd.read_csv('dataMacro.csv', parse_dates=['date'])

    data_factors = pd.read_csv('SparseMacroFinanceFactorReturnAug2025.csv', parse_dates=['date'])

    data_test = pd.read_csv('dataTestCz.csv', parse_dates=['date'])

    size01_data = data_test[['date', 'Size01']].copy()

    data = data_macro.merge(data_factors, on='date').merge(size01_data, on='date')

    print("Successfully loaded data with Size01 from dataTestCz only")

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the following files exist in the current directory:")
    print("  - dataMacro.csv")
    print("  - SparseMacroFinanceFactorReturnAug2025.csv")
    print("  - dataTestCz.csv")
    exit(1)

y_col = 'Size01'
Y = data[y_col].values

feature_cols = [col for col in data.columns if col not in ['date', 'Size01']]
X_full = data[feature_cols].values
feature_names = feature_cols
d = X_full.shape[1]
n = len(Y)

print(f"\nData Dimensions:")
print(f"  Total observations (n): {n}")
print(f"  Number of covariates (d): {d}")
print(f"  Y shape: {Y.shape}")
print(f"  X shape: {X_full.shape}")

print(f"\n" + "="*50)
print("TIME SERIES SPLITTING")
print("="*50)

n_test = int(n * KAN_TEST_SPLIT)
n_val = int(n * KAN_VAL_SPLIT)
n_train = n - n_test - n_val

print(f"Split sizes:")
print(f"  Training set: {n_train} observations ({n_train/n*100:.1f}%)")
print(f"  Validation set: {n_val} observations ({n_val/n*100:.1f}%)")
print(f"  Test set: {n_test} observations ({n_test/n*100:.1f}%)")

X_train = X_full[:n_train]
Y_train = Y[:n_train]

X_val = X_full[n_train:n_train+n_val]
Y_val = Y[n_train:n_train+n_val]

X_test = X_full[n_train+n_val:]
Y_test = Y[n_train+n_val:]

print(f"\nDataset shapes:")
print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"  X_val: {X_val.shape}, Y_val: {Y_val.shape}")
print(f"  X_test: {X_test.shape}, Y_test: {Y_test.shape}")

print(f"\n" + "="*50)
print("CDF TRANSFORMATION")
print("="*50)

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

print("Fitting CDF transformer on training data...")
cdf_transformer = EmpiricalCDFTransformer()
X_train_cdf = cdf_transformer.fit_transform(X_train)

X_val_cdf = cdf_transformer.transform(X_val)
X_test_cdf = cdf_transformer.transform(X_test)

print(f"CDF transformation complete:")
print(f"  X_train_cdf range: [{X_train_cdf.min():.3f}, {X_train_cdf.max():.3f}]")
print(f"  X_val_cdf range: [{X_val_cdf.min():.3f}, {X_val_cdf.max():.3f}]")
print(f"  X_test_cdf range: [{X_test_cdf.min():.3f}, {X_test_cdf.max():.3f}]")

scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_val_scaled = scaler_Y.transform(Y_val.reshape(-1, 1)).flatten()
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

print(f"\nY standardization (using training statistics):")
print(f"  Y_train mean (original): {Y_train.mean():.4f}, std: {Y_train.std():.4f}")
print(f"  Y_val mean (original): {Y_val.mean():.4f}, std: {Y_val.std():.4f}")
print(f"  Y_test mean (original): {Y_test.mean():.4f}, std: {Y_test.std():.4f}")

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

def train_slfn_model(X_train_cdf, Y_train_scaled, X_val_cdf, Y_val_scaled, input_dim, n_samples):
    print(f"\nTraining SLFN (Chen & White, 1999) model...")
    start_time = time.time()

    slfn_model = SLFN(input_dim=input_dim, sample_size=n_samples)
    slfn_model.to(DEVICE)

    X_train_tensor = torch.tensor(X_train_cdf, dtype=torch.float32, device=DEVICE)
    Y_train_tensor = torch.tensor(Y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    X_val_tensor = torch.tensor(X_val_cdf, dtype=torch.float32, device=DEVICE)
    Y_val_tensor = torch.tensor(Y_val_scaled.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.AdamW(slfn_model.parameters(), lr=KAN_LR, weight_decay=KAN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=KAN_BATCH_SIZE, shuffle=True
    )

    for epoch in range(KAN_N_EPOCHS):
        slfn_model.train()
        epoch_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()

            pred = slfn_model(X_batch)
            loss = criterion(pred, Y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(slfn_model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_train_loss = epoch_loss / len(X_train_tensor)
        train_losses.append(avg_train_loss)

        slfn_model.eval()
        with torch.no_grad():
            val_pred = slfn_model(X_val_tensor)
            val_loss = criterion(val_pred, Y_val_tensor).item()
            val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in slfn_model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch == KAN_N_EPOCHS - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f'  Epoch {epoch:4d}/{KAN_N_EPOCHS}: Train Loss = {avg_train_loss:.4f}, '
                  f'Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}')

        if patience_counter >= KAN_PATIENCE:
            print(f'  Early stopping at epoch {epoch}')
            break

    slfn_training_time = time.time() - start_time
    print(f"SLFN training completed in {slfn_training_time:.2f} seconds")

    if best_model_state is not None:
        slfn_model.load_state_dict(best_model_state)
        slfn_model.to(DEVICE)

    return slfn_model, slfn_training_time, best_val_loss

print(f"\n" + "="*50)
print("LASSO REGRESSION WITH CDF-TRANSFORMED DATA")
print("="*50)

print("Fitting LassoCV using validation set for alpha selection...")
start_time = time.time()

lasso_cv = LassoCV(
    alphas=LASSO_ALPHAS,
    cv=None,
    random_state=SEED,
    max_iter=LASSO_MAX_ITER
)

lasso_cv.fit(X_train_cdf, Y_train_scaled)

val_predictions = lasso_cv.predict(X_val_cdf)
val_mse = mean_squared_error(Y_val_scaled, val_predictions)

lasso_time = time.time() - start_time
print(f"Lasso fitting completed in {lasso_time:.2f} seconds")
print(f"Optimal alpha (λ): {lasso_cv.alpha_:.6f}")
print(f"Validation MSE (scaled): {val_mse:.6f}")

print("Fitting final Lasso model with optimal alpha...")
lasso_final = Lasso(alpha=lasso_cv.alpha_, max_iter=LASSO_MAX_ITER, random_state=SEED)
lasso_final.fit(X_train_cdf, Y_train_scaled)

Y_train_pred_lasso_scaled = lasso_final.predict(X_train_cdf)
Y_val_pred_lasso_scaled = lasso_final.predict(X_val_cdf)
Y_test_pred_lasso_scaled = lasso_final.predict(X_test_cdf)

Y_train_pred_lasso = scaler_Y.inverse_transform(Y_train_pred_lasso_scaled.reshape(-1, 1)).flatten()
Y_val_pred_lasso = scaler_Y.inverse_transform(Y_val_pred_lasso_scaled.reshape(-1, 1)).flatten()
Y_test_pred_lasso = scaler_Y.inverse_transform(Y_test_pred_lasso_scaled.reshape(-1, 1)).flatten()

train_mse_lasso = mean_squared_error(Y_train, Y_train_pred_lasso)
val_mse_lasso = mean_squared_error(Y_val, Y_val_pred_lasso)
test_mse_lasso = mean_squared_error(Y_test, Y_test_pred_lasso)

def calculate_r2(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0

train_r2_lasso = calculate_r2(Y_train, Y_train_pred_lasso)
val_r2_lasso = calculate_r2(Y_val, Y_val_pred_lasso)
test_r2_lasso = calculate_r2(Y_test, Y_test_pred_lasso)

coefficients = lasso_final.coef_
nonzero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
nonzero_count_lasso = len(nonzero_indices)
sparsity_ratio_lasso = 1 - (nonzero_count_lasso / d)

print(f"\nLASSO Results:")
print(f"  Training MSE (original): {train_mse_lasso:.6f}, R²: {train_r2_lasso:.4f}")
print(f"  Validation MSE (original): {val_mse_lasso:.6f}, R²: {val_r2_lasso:.4f}")
print(f"  Test MSE (original): {test_mse_lasso:.6f}, R²: {test_r2_lasso:.4f}")
print(f"  Non-zero coefficients: {nonzero_count_lasso}/{d} ({sparsity_ratio_lasso*100:.1f}% sparse)")

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

class ResidualKANLayer(nn.Module):
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

    def prune_edges(self, threshold):
        with torch.no_grad():
            edge_norms = torch.norm(self.spline_weight, p=2, dim=2)
            mask = (edge_norms > threshold).unsqueeze(2).expand(-1, -1, self.num_basis)
            self.spline_weight.data = self.spline_weight.data * mask.float()

class ResidualSieveKAN(nn.Module):
    def __init__(self, input_dim, n_samples, gamma=KAN_GAMMA, kan_width=None, depth=None, zeta_delta=KAN_ZETA_DELTA):
        super().__init__()
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.gamma = gamma
        self.zeta_delta = zeta_delta

        self.G = max(5, int(n_samples**gamma))
        self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
        self.W = kan_width if kan_width is not None else min(50, input_dim)

        log_delta = np.log(n_samples)
        poly_delta = n_samples ** zeta_delta if zeta_delta is not None else 1
        self.delta_n = max(5, log_delta, poly_delta)

        print(f"  Network parameters: L={self.L}, W={self.W}, G={self.G}, Δ_n={self.delta_n:.2f}")

        self.layers = nn.ModuleList()

        self.layers.append(ResidualKANLayer(
            input_dim, self.W, grid_size=self.G, use_residual=True, delta_n=self.delta_n / self.L
        ))

        for i in range(self.L - 2):
            self.layers.append(ResidualKANLayer(
                self.W, self.W, grid_size=self.G, use_residual=True, delta_n=self.delta_n / self.L
            ))

        self.layers.append(ResidualKANLayer(
            self.W, 1, grid_size=self.G, use_residual=False, delta_n=self.delta_n / self.L
        ))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_lambda_reg(self):
        n = self.n_samples
        total_params = self.L * (self.W ** 2) * self.G
        lambda_reg = np.sqrt(np.log(total_params + 1e-8) / n)
        return lambda_reg

    def count_active_edges(self, threshold=KAN_ACTIVE_EDGE_THRESHOLD):
        total_active = 0
        for layer in self.layers:
            total_active += layer.count_active_edges(threshold)
        return total_active

    def count_total_edges(self):
        total = 0
        for layer in self.layers:
            total += layer.count_total_edges()
        return total

    def get_sparsity_ratio(self, threshold=KAN_ACTIVE_EDGE_THRESHOLD):
        active_edges = self.count_active_edges(threshold)
        total_edges = self.count_total_edges()
        return 1 - (active_edges / total_edges) if total_edges > 0 else 0

    def get_delta_sparsity_ratio(self):
        total_edges = self.count_total_edges()
        delta_threshold = self.delta_n / total_edges if total_edges > 0 else 0
        active_edges = self.count_active_edges(delta_threshold)
        delta_sparsity = 1 - (active_edges / total_edges) if total_edges > 0 else 0
        return delta_sparsity, delta_threshold, active_edges

    def prune_with_delta_threshold(self):
        total_edges = self.count_total_edges()
        delta_threshold = self.delta_n / total_edges if total_edges > 0 else 0

        print(f"  Pruning with Δ_n/total_edges threshold: {delta_threshold:.6f}")

        for layer in self.layers:
            layer.prune_edges(delta_threshold)

        return delta_threshold

print(f"\n" + "="*50)
print("TRAINING SLFN MODEL (Chen & White, 1999)")
print("="*50)

slfn_model, slfn_training_time, slfn_val_loss = train_slfn_model(
    X_train_cdf, Y_train_scaled, X_val_cdf, Y_val_scaled,
    input_dim=d, n_samples=n_train
)

X_test_tensor = torch.tensor(X_test_cdf, dtype=torch.float32, device=DEVICE)

slfn_model.eval()
with torch.no_grad():
    Y_train_pred_slfn_scaled = slfn_model(torch.tensor(X_train_cdf, dtype=torch.float32, device=DEVICE)).cpu().numpy().flatten()
    Y_val_pred_slfn_scaled = slfn_model(torch.tensor(X_val_cdf, dtype=torch.float32, device=DEVICE)).cpu().numpy().flatten()
    Y_test_pred_slfn_scaled = slfn_model(X_test_tensor).cpu().numpy().flatten()

Y_train_pred_slfn = scaler_Y.inverse_transform(Y_train_pred_slfn_scaled.reshape(-1, 1)).flatten()
Y_val_pred_slfn = scaler_Y.inverse_transform(Y_val_pred_slfn_scaled.reshape(-1, 1)).flatten()
Y_test_pred_slfn = scaler_Y.inverse_transform(Y_test_pred_slfn_scaled.reshape(-1, 1)).flatten()

train_mse_slfn = mean_squared_error(Y_train, Y_train_pred_slfn)
val_mse_slfn = mean_squared_error(Y_val, Y_val_pred_slfn)
test_mse_slfn = mean_squared_error(Y_test, Y_test_pred_slfn)

train_r2_slfn = calculate_r2(Y_train, Y_train_pred_slfn)
val_r2_slfn = calculate_r2(Y_val, Y_val_pred_slfn)
test_r2_slfn = calculate_r2(Y_test, Y_test_pred_slfn)

print(f"\nSLFN Results:")
print(f"  Training MSE (original): {train_mse_slfn:.6f}, R²: {train_r2_slfn:.4f}")
print(f"  Validation MSE (original): {val_mse_slfn:.6f}, R²: {val_r2_slfn:.4f}")
print(f"  Test MSE (original): {test_mse_slfn:.6f}, R²: {test_r2_slfn:.4f}")
print(f"  Model parameters: hidden_dim={slfn_model.hidden_dim}, total_params={(d * slfn_model.hidden_dim) + slfn_model.hidden_dim + 1}")

print(f"\n" + "="*50)
print("SIEVE KAN NONPARAMETRIC REGRESSION")
print("="*50)

X_train_tensor = torch.tensor(X_train_cdf, dtype=torch.float32, device=DEVICE)
Y_train_tensor = torch.tensor(Y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
X_val_tensor = torch.tensor(X_val_cdf, dtype=torch.float32, device=DEVICE)
Y_val_tensor = torch.tensor(Y_val_scaled.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

print(f"Initializing Sieve KAN...")
kan_model = ResidualSieveKAN(
    input_dim=d,
    n_samples=n_train,
    gamma=KAN_GAMMA,
    kan_width=KAN_WIDTH,
    depth=KAN_DEPTH,
    zeta_delta=KAN_ZETA_DELTA
)
kan_model.to(DEVICE)

optimizer = torch.optim.AdamW(kan_model.parameters(), lr=KAN_LR, weight_decay=KAN_WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
criterion = nn.MSELoss()

print(f"Training Sieve KAN for {KAN_N_EPOCHS} epochs...")
start_time = time.time()

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=KAN_BATCH_SIZE, shuffle=True
)

for epoch in range(KAN_N_EPOCHS):
    kan_model.train()
    epoch_loss = 0.0

    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()

        pred = kan_model(X_batch)
        mse_loss = criterion(pred, Y_batch)

        reg_loss = 0
        for layer in kan_model.layers:
            reg_loss += layer.group_lasso_regularization_loss(KAN_GROUP_LASSO_REG_SCALE)

        lambda_reg = kan_model.compute_lambda_reg()
        total_loss = mse_loss + lambda_reg * reg_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(kan_model.parameters(), 5.0)
        optimizer.step()

        for layer in kan_model.layers:
            layer.project_to_delta_constraint()

        epoch_loss += total_loss.item() * X_batch.size(0)

    scheduler.step()
    avg_train_loss = epoch_loss / len(X_train_tensor)
    train_losses.append(avg_train_loss)

    kan_model.eval()
    with torch.no_grad():
        val_pred = kan_model(X_val_tensor)
        val_loss = criterion(val_pred, Y_val_tensor).item()
        val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = {k: v.cpu() for k, v in kan_model.state_dict().items()}
    else:
        patience_counter += 1

    if epoch % 50 == 0 or epoch == KAN_N_EPOCHS - 1:
        active_edges = kan_model.count_active_edges()
        total_edges = kan_model.count_total_edges()
        sparsity = kan_model.get_sparsity_ratio()
        current_lr = scheduler.get_last_lr()[0]

        print(f'  Epoch {epoch:4d}/{KAN_N_EPOCHS}: Train Loss = {avg_train_loss:.4f}, '
              f'Val Loss = {val_loss:.4f}, Active Edges = {active_edges}/{total_edges} '
              f'({sparsity:.2%} sparse), LR = {current_lr:.6f}')

    if patience_counter >= KAN_PATIENCE:
        print(f'  Early stopping at epoch {epoch}')
        break

kan_training_time = time.time() - start_time
print(f"Sieve KAN training completed in {kan_training_time:.2f} seconds")

if best_model_state is not None:
    kan_model.load_state_dict(best_model_state)
    kan_model.to(DEVICE)

print(f"\n" + "="*50)
print("PRUNING INACTIVE EDGES WITH Δ_n/total_edges THRESHOLD")
print("="*50)

fixed_sparsity = kan_model.get_sparsity_ratio()
delta_sparsity, delta_threshold, active_edges_delta = kan_model.get_delta_sparsity_ratio()

print(f"Before pruning:")
print(f"  Fixed threshold sparsity ({KAN_ACTIVE_EDGE_THRESHOLD:.6f}): {fixed_sparsity:.4f}")
print(f"  Δ_n/total_edges sparsity (Δ_n={kan_model.delta_n:.2f}): {delta_sparsity:.4f}")
print(f"  Δ_n/total_edges threshold: {delta_threshold:.6f}")

kan_model.prune_with_delta_threshold()

fixed_sparsity_after = kan_model.get_sparsity_ratio()
delta_sparsity_after, delta_threshold_after, active_edges_delta_after = kan_model.get_delta_sparsity_ratio()

print(f"\nAfter pruning:")
print(f"  Fixed threshold sparsity: {fixed_sparsity_after:.4f}")
print(f"  Δ_n/total_edges sparsity: {delta_sparsity_after:.4f}")
print(f"  Active edges after pruning: {active_edges_delta_after}/{kan_model.count_total_edges()}")

kan_model.eval()
with torch.no_grad():
    Y_train_pred_kan_scaled = kan_model(X_train_tensor).cpu().numpy().flatten()
    Y_val_pred_kan_scaled = kan_model(X_val_tensor).cpu().numpy().flatten()
    Y_test_pred_kan_scaled = kan_model(X_test_tensor).cpu().numpy().flatten()

Y_train_pred_kan = scaler_Y.inverse_transform(Y_train_pred_kan_scaled.reshape(-1, 1)).flatten()
Y_val_pred_kan = scaler_Y.inverse_transform(Y_val_pred_kan_scaled.reshape(-1, 1)).flatten()
Y_test_pred_kan = scaler_Y.inverse_transform(Y_test_pred_kan_scaled.reshape(-1, 1)).flatten()

train_mse_kan = mean_squared_error(Y_train, Y_train_pred_kan)
val_mse_kan = mean_squared_error(Y_val, Y_val_pred_kan)
test_mse_kan = mean_squared_error(Y_test, Y_test_pred_kan)

train_r2_kan = calculate_r2(Y_train, Y_train_pred_kan)
val_r2_kan = calculate_r2(Y_val, Y_val_pred_kan)
test_r2_kan = calculate_r2(Y_test, Y_test_pred_kan)

active_edges_fixed = kan_model.count_active_edges()
total_edges = kan_model.count_total_edges()
sparsity_ratio_kan_fixed = kan_model.get_sparsity_ratio()
sparsity_ratio_kan_delta, _, active_edges_delta_final = kan_model.get_delta_sparsity_ratio()

print(f"\nSIEVE KAN Results (after pruning):")
print(f"  Training MSE (original): {train_mse_kan:.6f}, R²: {train_r2_kan:.4f}")
print(f"  Validation MSE (original): {val_mse_kan:.6f}, R²: {val_r2_kan:.4f}")
print(f"  Test MSE (original): {test_mse_kan:.6f}, R²: {test_r2_kan:.4f}")
print(f"\nSparsity Analysis:")
print(f"  Fixed threshold ({KAN_ACTIVE_EDGE_THRESHOLD}): {active_edges_fixed}/{total_edges} edges ({sparsity_ratio_kan_fixed*100:.1f}% sparse)")
print(f"  Δ_n/total_edges (Δ_n={kan_model.delta_n:.2f}): {active_edges_delta_final}/{total_edges} edges ({sparsity_ratio_kan_delta*100:.1f}% sparse)")
print(f"  Δ_n/total_edges threshold: {delta_threshold_after:.6f}")
print(f"  Theoretical KART optimal: ~{2*8+1} edges (assuming 8 relevant factors)")

print(f"\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"\nPerformance Comparison (Test Set):")
print(f"{'Metric':<25} {'LASSO':<12} {'SLFN':<12} {'Sieve KAN':<12}")
print("-" * 61)
print(f"{'Test MSE':<25} {test_mse_lasso:<12.6f} {test_mse_slfn:<12.6f} {test_mse_kan:<12.6f}")
print(f"{'Test R²':<25} {test_r2_lasso:<12.4f} {test_r2_slfn:<12.4f} {test_r2_kan:<12.4f}")
print(f"{'Training Time (s)':<25} {lasso_time:<12.2f} {slfn_training_time:<12.2f} {kan_training_time:<12.2f}")
print(f"{'Fixed Sparsity (%)':<25} {sparsity_ratio_lasso*100:<12.1f} {'N/A':<12} {sparsity_ratio_kan_fixed*100:<12.1f}")
print(f"{'Δ_n Sparsity (%)':<25} {'N/A':<12} {'N/A':<12} {sparsity_ratio_kan_delta*100:<12.1f}")
print(f"{'Model Size':<25} {nonzero_count_lasso:<12} {slfn_model.hidden_dim:<12} {active_edges_delta_final:<12}")

print(f"\nRelative Improvement vs LASSO:")
improvement_kan_mse = (test_mse_lasso - test_mse_kan) / test_mse_lasso * 100 if test_mse_lasso > 0 else 0
improvement_slfn_mse = (test_mse_lasso - test_mse_slfn) / test_mse_lasso * 100 if test_mse_lasso > 0 else 0
improvement_kan_r2 = (test_r2_kan - test_r2_lasso) * 100
improvement_slfn_r2 = (test_r2_slfn - test_r2_lasso) * 100

print(f"  KAN MSE Improvement: {improvement_kan_mse:.2f}% (R²: +{improvement_kan_r2:.2f} pp)")
print(f"  SLFN MSE Improvement: {improvement_slfn_mse:.2f}% (R²: +{improvement_slfn_r2:.2f} pp)")

print(f"\nRelative Improvement (KAN vs SLFN):")
kan_vs_slfn_mse = (test_mse_slfn - test_mse_kan) / test_mse_slfn * 100 if test_mse_slfn > 0 else 0
kan_vs_slfn_r2 = (test_r2_kan - test_r2_slfn) * 100
print(f"  KAN vs SLFN MSE Improvement: {kan_vs_slfn_mse:.2f}%")
print(f"  KAN vs SLFN R² Improvement: {kan_vs_slfn_r2:.2f} percentage points")

print(f"\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

print("\nTop 15 Features by Sieve KAN Edge Norms (Input Layer):")
kan_model.eval()
with torch.no_grad():
    input_layer = kan_model.layers[0]
    edge_norms = torch.norm(input_layer.spline_weight, p=2, dim=2)

    if edge_norms.dim() == 2:
        if edge_norms.size(0) > 1:
            feature_importance = edge_norms.mean(dim=0).cpu().numpy()
        else:
            feature_importance = edge_norms.squeeze(0).cpu().numpy()
    else:
        feature_importance = edge_norms.cpu().numpy()

    top_k = min(15, len(feature_importance))
    if len(feature_importance) < d:
        print(f"  Warning: Feature importance array has length {len(feature_importance)} but expected {d}")
        top_k = min(top_k, len(feature_importance))

    top_indices = np.argsort(feature_importance)[::-1][:top_k]

    for i, idx in enumerate(top_indices):
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            norm_value = feature_importance[idx]
            print(f"  {i+1:2d}. {feature_name}: {norm_value:.6f}")
        else:
            print(f"  {i+1:2d}. Index {idx} out of bounds for feature_names")

print("\nTop 15 Features by LASSO Coefficients (Absolute):")
abs_coefficients = np.abs(coefficients)
top_lasso_indices = np.argsort(abs_coefficients)[::-1][:15]

for i, idx in enumerate(top_lasso_indices):
    if idx < len(feature_names):
        feature_name = feature_names[idx]
        coef_value = coefficients[idx]
        print(f"  {i+1:2d}. {feature_name}: {coef_value:.6f}")
    else:
        print(f"  {i+1:2d}. Index {idx} out of bounds for feature_names")

print("\nTop 15 Features by SLFN First Layer Weights (Absolute):")
slfn_model.eval()
with torch.no_grad():
    slfn_weights = slfn_model.fc1.weight.data.cpu().numpy()
    slfn_feature_importance = np.mean(np.abs(slfn_weights), axis=0)

    top_slfn_indices = np.argsort(slfn_feature_importance)[::-1][:15]

    for i, idx in enumerate(top_slfn_indices):
        if idx < len(feature_names):
            feature_name = feature_names[idx]
            weight_value = slfn_feature_importance[idx]
            print(f"  {i+1:2d}. {feature_name}: {weight_value:.6f}")
        else:
            print(f"  {i+1:2d}. Index {idx} out of bounds for feature_names")

if len(feature_importance) >= 15 and len(abs_coefficients) >= 15:
    kan_top_set = set([feature_names[i] for i in top_indices if i < len(feature_names)])
    lasso_top_set = set([feature_names[i] for i in top_lasso_indices if i < len(feature_names)])
    slfn_top_set = set([feature_names[i] for i in top_slfn_indices if i < len(feature_names)])

    overlap_all = kan_top_set.intersection(lasso_top_set).intersection(slfn_top_set)
    overlap_kan_lasso = kan_top_set.intersection(lasso_top_set)

    print(f"\nOverlap in top 15 features:")
    print(f"  All three models: {len(overlap_all)}/15")
    print(f"  KAN & LASSO: {len(overlap_kan_lasso)}/15")

    if overlap_all:
        print("  Common to all models:", ", ".join(sorted(overlap_all)))

print(f"\nKey Financial Variables Analysis:")
key_vars = ['yield', 'creditSpread', 'inflation', 'unemployment', 'gdp_growth']
for var in key_vars:
    if var in feature_names:
        idx = feature_names.index(var)

        if idx < len(coefficients):
            lasso_coef = coefficients[idx]
            lasso_status = "SELECTED" if np.abs(lasso_coef) > 1e-6 else "NOT SELECTED"
        else:
            lasso_coef = 0
            lasso_status = "NOT FOUND"

        if idx < len(feature_importance):
            kan_norm = feature_importance[idx]
            kan_rank = np.where(top_indices == idx)[0][0] + 1 if idx in top_indices else "Not in top 15"
        else:
            kan_norm = 0
            kan_rank = "Not found"

        if idx < len(slfn_feature_importance):
            slfn_weight = slfn_feature_importance[idx]
            slfn_rank = np.where(top_slfn_indices == idx)[0][0] + 1 if idx in top_slfn_indices else "Not in top 15"
        else:
            slfn_weight = 0
            slfn_rank = "Not found"

        print(f"  {var:<15} LASSO: {lasso_coef:.6f} ({lasso_status}) | "
              f"KAN: {kan_norm:.6f} (Rank {kan_rank}) | "
              f"SLFN: {slfn_weight:.6f} (Rank {slfn_rank})")

print(f"\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Data: n={n}, d={d}, target={y_col}")
print(f"Splits: Train={n_train}, Val={n_val}, Test={n_test}")
print(f"\nLASSO Summary:")
print(f"  Test MSE: {test_mse_lasso:.6f}, Test R²: {test_r2_lasso:.4f}")
print(f"  Selected features: {nonzero_count_lasso}/{d} ({sparsity_ratio_lasso*100:.1f}% sparse)")
print(f"\nSLFN Summary (Chen & White, 1999):")
print(f"  Test MSE: {test_mse_slfn:.6f}, Test R²: {test_r2_slfn:.4f}")
print(f"  Hidden units: {slfn_model.hidden_dim}")
print(f"  Total parameters: {(d * slfn_model.hidden_dim) + slfn_model.hidden_dim + 1}")
print(f"\nSieve KAN Summary (after pruning with Δ_n/total_edges):")
print(f"  Test MSE: {test_mse_kan:.6f}, Test R²: {test_r2_kan:.4f}")
print(f"  Fixed threshold sparsity: {sparsity_ratio_kan_fixed*100:.1f}%")
print(f"  Δ_n/total_edges sparsity: {sparsity_ratio_kan_delta*100:.1f}%")
print(f"  Δ_n/total_edges threshold: {delta_threshold_after:.6f}")
print(f"  Active edges: {active_edges_delta_final}/{total_edges}")
print(f"  Network: L={kan_model.L}, W={kan_model.W}, G={kan_model.G}, Δ_n={kan_model.delta_n:.2f}")
print(f"\nImprovement vs LASSO:")
print(f"  SLFN: {improvement_slfn_mse:.1f}% lower MSE, +{improvement_slfn_r2:.2f} pp R²")
print(f"  Sieve KAN: {improvement_kan_mse:.1f}% lower MSE, +{improvement_kan_r2:.2f} pp R²")
print(f"  KAN vs SLFN: {kan_vs_slfn_mse:.1f}% lower MSE, +{kan_vs_slfn_r2:.2f} pp R²")