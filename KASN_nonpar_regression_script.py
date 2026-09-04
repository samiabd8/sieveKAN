"""
KASN nonparametric regression — Monte Carlo replication (note: this can be run directly via SLURM) 
"""

import os
import gc
import sys
import time
import copy
import uuid
import math
import socket
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



def get_seed():
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass

    env_seed = os.getenv('SIMULATION_SEED')
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            pass

    task_id = os.getenv('SLURM_ARRAY_TASK_ID') or os.getenv('SLURM_PROCID')
    if task_id is not None:
        try:
            job_id = int(os.getenv('SLURM_ARRAY_JOB_ID')
                         or os.getenv('SLURM_JOB_ID') or 0)
            seed = (job_id * 100_003 + int(task_id) * 1_000_003) % 2 ** 31
            print(f"  [seed] from SLURM: job {job_id}, task {task_id} -> {seed}")
            return seed
        except ValueError:
            pass

    base = int(time.time() * 1000) % 2 ** 31
    salt = (hash(socket.gethostname()) ^ os.getpid()) % 1_000_003
    seed = (base + salt * 1_000_003) % 2 ** 31
    print(f"  [seed] no SLURM task id; clock+host+pid fallback -> {seed}")
    return seed

SEED = get_seed()
np.random.seed(SEED)
torch.manual_seed(SEED)


def _env(name, default, cast=float):
    """Read an override, treating '', 'none' and 'null' as an explicit None."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in ('', 'none', 'null'):
        return None
    try:
        return cast(raw)
    except (TypeError, ValueError):
        print(f"  [config] WARNING: could not read {name}={raw!r}, "
              f"keeping default {default!r}")
        return default


def _env_int(name, default):
    return _env(name, default, cast=lambda v: int(float(v)))


def _env_str(name, default):
    return _env(name, default, cast=str)


def _env_bool(name, default):
    return _env(name, default,
                cast=lambda v: v.strip().lower() in ('1', 'true', 'yes', 'on'))


def _env_list(name, default, cast=float):
    """Comma-separated list override, e.g. SHARED_PENALTY_GRID=1e-3,1e-2."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw.lower() in ('', 'none', 'null'):
        return None
    try:
        return [cast(p) for p in raw.split(',') if p.strip() != '']
    except (TypeError, ValueError):
        print(f"  [config] WARNING: could not read {name}={raw!r}, "
              f"keeping default {default!r}")
        return default


SPEC_LABEL = _env_str('SPEC_LABEL', 'default')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision('high')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

gc.collect()
if DEVICE.type == 'cuda':
    torch.cuda.empty_cache()

RUN_ID = str(uuid.uuid4())[:8]

OUTPUT_DIR = _env_str('OUTPUT_DIR', "nonparametric_regression_results")

TARGET_FUNCTION = _env_str('TARGET_FUNCTION', 'compositional') #TARGET_FUNCTION = _env_str('TARGET_FUNCTION', 'additive')
TARGET_STANDARDISE = _env_bool('TARGET_STANDARDISE', True)

N_OBS      = _env_int('N_OBS', 2000)
DIM        = _env_int('DIM', 500)        # d
S_SPARSE   = _env_int('S_SPARSE', 50)    # S: covariates f_0 actually depends on
NOISE_SD   = _env('NOISE_SD', 0.5)       # sd, i.e. variance 0.25
NOISE_RHO  = _env('NOISE_RHO', 0.0)

DGP_DEPENDENCE = _env_str('DGP_DEPENDENCE', 'var1')
RHO_X          = _env('RHO_X', 0.7)      # AR coefficient; ignored under 'iid'
MARGINAL       = _env_str('MARGINAL', 'uniform')   # 'uniform' | 'normal'
X_LO, X_HI     = -3.0, 3.0               # support under MARGINAL='uniform'
BURN_IN        = _env_int('BURN_IN', 500)

TRAIN_RATIO = _env('TRAIN_RATIO', 0.50)
VAL_RATIO   = _env('VAL_RATIO', 0.25)
TEST_RATIO  = _env('TEST_RATIO', 0.25)
SPLIT_MODE  = _env_str('SPLIT_MODE', 'chronological')   # 'chronological'|'random'

ALL_LEARNERS = ('kasn', 'oracle', 'lasso', 'slfn', 'dnn', 'gam', 'ppr')
RUN_ONLY   = _env_str('RUN_ONLY', None)
RUN_EXCEPT = _env_str('RUN_EXCEPT', None)

VAL_FRACTION = 0.20
VAL_BUFFER   = 0

_DEFAULT_KASN_GAMMA = 0.3999
_DEFAULT_KASN_DEPTH = 4 #4 #6
_DEFAULT_C_W        = 15
_DEFAULT_KASN_WIDTH = None
_DEFAULT_PENALTY_GRID = [5e-2, 1e-1, 1.0, 5.0] #[1e-3, 1e-2, 5e-2, 1e-1, 1.0, 5.0]
_DEFAULT_LR_GRID      = [1e-3] #[1e-5, 1e-3, 1e-1] #[1e-5, 1e-4, 1e-3] #[1e-3, 5e-3]
_DEFAULT_C_W_GRID     = [5,15] #None

# ============================================================================
# CONFIGURATION — KASN 
# ============================================================================
M_SMOOTH_EDGE     = _env_int('M_SMOOTH_EDGE', 2)      # m, smoothness of the edges
KASN_SPLINE_ORDER = _env_int('KASN_SPLINE_ORDER', 4)  # k
KASN_GRID_RANGE   = [-0.5, 1.5]                       # I_eps: [0,1], padded
KASN_DEPTH        = _env_int('KASN_DEPTH', _DEFAULT_KASN_DEPTH)   # L_n
C_W               = _env_int('C_W', _DEFAULT_C_W)     # W_n = C_W * floor(log n)
KASN_WIDTH        = _env_int('KASN_WIDTH', _DEFAULT_KASN_WIDTH)   # pin W_n
KASN_GAMMA        = _env('KASN_GAMMA', _DEFAULT_KASN_GAMMA)       # G_n = n^gamma
KASN_ZETA_DELTA   = _env('KASN_ZETA_DELTA', 0.40)     # zeta_delta
KASN_BASE_ACTIVATION = nn.SiLU

DELTA_MODE        = _env_str('DELTA_MODE', 'dual')          # 'dual' | 'theoretical'
DELTA_SCALE       = _env('DELTA_SCALE', 1.0)
DELTA_ENFORCEMENT = _env_str('DELTA_ENFORCEMENT', 'none')   # 'none'|'project'|'penalty'
LAMBDA_SELECTION  = _env_str('LAMBDA_SELECTION', 'validation')

KASN_N_EPOCHS     = _env_int('KASN_N_EPOCHS', 1000)
KASN_PATIENCE     = _env_int('KASN_PATIENCE', 500)
KASN_WEIGHT_DECAY = _env('KASN_WEIGHT_DECAY', 1e-19)   # off; the group lasso works
KASN_L1_REG_SCALE = _env('KASN_L1_REG_SCALE', 0.0)
KASN_ACTIVE_EDGE_THRESHOLD = 1e-4                      # fixed-threshold diagnostic
GRAD_CLIP_NORM    = 500.0

POST_TRAINING_PRUNING        = _env_bool('POST_TRAINING_PRUNING', True)
PRUNING_THRESHOLD_METHOD     = _env_str('PRUNING_THRESHOLD_METHOD', 'delta_over_r')
PRUNING_RELATIVE_FRACTION    = 0.01
PRUNE_DURING_TRAINING        = _env_bool('PRUNE_DURING_TRAINING', False)
POST_PRUNE_FINETUNE_EPOCHS   = _env_int('POST_PRUNE_FINETUNE_EPOCHS', 30)
POST_PRUNE_FINETUNE_LR_SCALE = 0.5
POST_PRUNE_FINETUNE_PATIENCE = 15
PRUNE_AWARE_SELECTION        = _env_bool('PRUNE_AWARE_SELECTION', True)

HP_SELECTION_MODE = _env_str('HP_SELECTION_MODE', 'pilot')

KASN_LR         = _env('KASN_LR', None)
KASN_BATCH_SIZE = _env_int('KASN_BATCH_SIZE', None)
KASN_GROUP_LASSO_REG_SCALE = _env('KASN_GROUP_LASSO_REG_SCALE', None)

KASN_LR_GRID         = _env_list('KASN_LR_GRID', _DEFAULT_LR_GRID)
KASN_BATCH_SIZE_GRID = _env_list('KASN_BATCH_SIZE_GRID', [16, 32, 64], cast=int) #KASN_BATCH_SIZE_GRID = _env_list('KASN_BATCH_SIZE_GRID', [128, 256], cast=int)
KASN_GAMMA_GRID      = _env_list('KASN_GAMMA_GRID', [KASN_GAMMA])
C_W_GRID             = _env_list('C_W_GRID', _DEFAULT_C_W_GRID or [C_W], cast=int)
SHARED_PENALTY_GRID  = _env_list('SHARED_PENALTY_GRID', _DEFAULT_PENALTY_GRID)
KASN_TUNING_EPOCHS   = _env_int('KASN_TUNING_EPOCHS', 50)

SHUFFLE_BATCHES = _env_bool('SHUFFLE_BATCHES', False)   # time series: keep order
FULL_BATCH      = _env_bool('FULL_BATCH', False)
USE_AMP         = _env_bool('USE_AMP', True)
INFERENCE_CHUNK_SIZE = 8192

# ------------------------------------------------- comparison learners ------
LASSO_MAX_ITER          = 10000
LASSO_NONZERO_THRESHOLD = 1e-6

SHARED_LR_GRID         = _env_list('SHARED_LR_GRID', [1e-3, 5e-3])
SHARED_BATCH_SIZE_GRID = _env_list('SHARED_BATCH_SIZE_GRID', [128, 256], cast=int)

# SLFN: the Chen and White (1999) shallow sieve network
C_SLFN, C_N_SLFN, C_OUT_SLFN = 10, 25.0, 25.0
M_SMOOTH, BN_MODE, ALPHA_SLFN = 1, "log", 1.0
SLFN_EPOCHS    = _env_int('SLFN_EPOCHS', 1000)
SLFN_PATIENCE  = _env_int('SLFN_PATIENCE', 50)
SLFN_LR, SLFN_BATCH_SIZE, SLFN_WEIGHT_DECAY = None, None, None
SLFN_TUNING_EPOCHS = _env_int('SLFN_TUNING_EPOCHS', 30)

# DNN: the Farrell, Liang and Misra (2021) ReLU architecture
DNN_HIDDEN_LAYER_SIZES   = _env_list('DNN_HIDDEN_LAYER_SIZES',
                                     [81, 81, 81, 81], cast=int)
DNN_MAX_EPOCHS           = _env_int('DNN_MAX_EPOCHS', 1000)
DNN_MAX_EPOCHS_NO_CHANGE = _env_int('DNN_MAX_EPOCHS_NO_CHANGE', 50)
DNN_ALPHA_REG, DNN_R_PAR = 0.0, 0.2
DNN_LR, DNN_BATCH_SIZE, DNN_WEIGHT_DECAY = None, None, None
DNN_TUNING_EPOCHS = _env_int('DNN_TUNING_EPOCHS', 30)

GAM_N_SPLINES    = _env_int('GAM_N_SPLINES', None)   # None -> n^gamma, i.e. G_n
GAM_SPLINE_ORDER = 3

PPR_R_GRID    = _env_list('PPR_R_GRID', [2, 4, 6, 10], cast=int)
PPR_FIT_TYPE  = 'spline'
PPR_SPLINE_DF = 3


def _grid_or_fixed(val, grid, name):
    return f"{val}" if val is not None else f"search {grid} [{name}]"


def _as_model_set(v):
    """Normalise a toggle value to a lower-cased set, or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return {p.strip().lower() for p in v.split(',') if p.strip()}
    return {str(x).strip().lower() for x in v}


def enabled_learners():
    """Resolve RUN_ONLY / RUN_EXCEPT into the list of learners to run."""
    only = _as_model_set(globals().get('RUN_ONLY'))
    exc = _as_model_set(globals().get('RUN_EXCEPT')) or set()
    unknown = ((only or set()) | exc) - set(ALL_LEARNERS)
    assert not unknown, (f"unknown learner name(s) {sorted(unknown)}; "
                         f"valid: {list(ALL_LEARNERS)}")
    out = (set(ALL_LEARNERS) if only is None else set(only)) - exc
    assert out, "RUN_ONLY/RUN_EXCEPT leave no learners to run"
    return [m for m in ALL_LEARNERS if m in out]   


def learner_enabled(name):
    return name in enabled_learners()


def _check_theory_conditions(n, d, verbose=True):
    """Verify the parameter restrictions of eq. (2.6)-(2.8) at this sample size.

    These are the conditions the rate theorems are stated under, so a run that
    violates them is not estimating the object the theory describes.  They are
    asserted rather than warned about: silently drifting outside the admissible
    region is exactly the failure that is hard to notice once 1000 replications
    have already been written to disk.
    """
    m = M_SMOOTH_EDGE
    lo, hi = 1.0 / (4.0 * (m + 1)), 0.5
    assert lo < KASN_GAMMA < hi, (
        f"gamma={KASN_GAMMA} violates 1/(4(m+1))={lo:.4f} < gamma < 1/2 for m={m}")
    assert KASN_ZETA_DELTA > KASN_GAMMA / 2.0, (
        f"zeta_delta={KASN_ZETA_DELTA} must exceed gamma/2={KASN_GAMMA/2:.4f}")
    assert KASN_SPLINE_ORDER >= m + 1, (
        f"spline order k={KASN_SPLINE_ORDER} must be >= m+1={m+1}")
    assert DELTA_MODE in ('theoretical', 'dual')
    assert DELTA_ENFORCEMENT in ('project', 'penalty', 'none')
    assert LAMBDA_SELECTION in ('validation', 'delta_constraint')
    assert not (DELTA_MODE == 'dual' and DELTA_ENFORCEMENT != 'none'), \
        "DELTA_MODE='dual' leaves Delta_n unset, so set DELTA_ENFORCEMENT='none'"
    assert not (DELTA_MODE == 'theoretical' and DELTA_ENFORCEMENT == 'none'), \
        "an explicit Delta_n with no enforcement is not the estimator over K_n"
    if LAMBDA_SELECTION == 'delta_constraint':
        assert DELTA_MODE == 'theoretical' and DELTA_ENFORCEMENT == 'penalty', \
            "eq. (2.5) needs an explicit Delta_n and the penalty form"
    if verbose:
        G_n = max(5, int(n ** KASN_GAMMA))
        L_n = KASN_DEPTH if KASN_DEPTH is not None else max(3, int(np.log(n)))
        W_n = (KASN_WIDTH if KASN_WIDTH is not None
               else max(1, C_W * int(np.floor(np.log(n)))))
        r_n = d * W_n + (L_n - 2) * W_n ** 2 + W_n
        print(f"  Theory check (n={n}, d={d}): gamma in ({lo:.4f}, {hi}) -> "
              f"{KASN_GAMMA};  zeta_delta {KASN_ZETA_DELTA} > gamma/2 "
              f"{KASN_GAMMA/2:.4f};  k={KASN_SPLINE_ORDER} >= m+1={m+1}")
        print(f"  Implied architecture: L_n={L_n}  W_n={W_n}  G_n={G_n}  "
              f"r_n={r_n:,}  n^zeta_delta={n**KASN_ZETA_DELTA:.2f}  "
              f"s_n~(log n)^2={int(np.ceil(np.log(n)**2))}")


def _should_shuffle():
    return bool(SHUFFLE_BATCHES)


def _effective_batch_size(n_train, requested_bs):
    if FULL_BATCH or requested_bs is None:
        return n_train
    return min(int(requested_bs), n_train)


class EmpiricalCDFTransformer:
    """Marginal empirical-CDF map onto [0,1], fitted on the training fold only.

    This is the normalisation of eq. (2.3): X~_{t,j} = F^_n^j(X_{t,j}).  It
    induces compact support, which is what lets the spline grid stay fixed, and
    contributes an O(n^{-1/2} log n) approximation error that is negligible
    relative to the KASN approximation error of Theorem 1.
    """

    def __init__(self):
        self.sorted_values_ = None
        self.n_train_ = None

    def fit(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.sorted_values_ = np.sort(X, axis=0)
        self.n_train_ = X.shape[0]
        return self

    def transform(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = self.n_train_
        cdf = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[1]):
            cdf[:, i] = np.searchsorted(
                self.sorted_values_[:, i], X[:, i], side='right') / (n + 1.0)
        return np.clip(cdf, 0.0, 1.0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# KASN ARCHITECTURE
# ============================================================================

KASN_BASE_BRANCH = False

class BSplineBasis(nn.Module):

    def __init__(self, in_features, grid_size=5,
                 spline_order=KASN_SPLINE_ORDER, grid_range=KASN_GRID_RANGE):
        super().__init__()
        self.in_features = in_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.register_buffer("grid", self._create_grid(grid_range, grid_size))

    def _create_grid(self, grid_range, grid_size):
        h = (grid_range[1] - grid_range[0]) / grid_size
        g = torch.arange(-self.spline_order,
                         grid_size + self.spline_order + 1) * h + grid_range[0]
        return g.expand(self.in_features, -1).contiguous()

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (
                grid[:, k:-1] - grid[:, :-(k + 1)]).clamp_min(1e-8)
            right = (grid[:, k + 1:] - x) / (
                grid[:, k + 1:] - grid[:, 1:(-k)]).clamp_min(1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x):
        s = x.shape
        return self.b_splines(x.reshape(-1, self.in_features)).reshape(*s[:-1], -1)


class KASNLayer(nn.Module):
    """One layer Phi_l of the KASN: a matrix of univariate B-spline edges.

    Activations live on edges, not nodes, so a node is a plain summation.  With
    in_features = p and out_features = q the layer holds p*q univariate
    functions, each parameterised by G_n + k spline coefficients c_j, and the
    group-lasso penalty operates on those coefficient vectors as groups.
    """

    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=KASN_SPLINE_ORDER,
                 base_activation=KASN_BASE_ACTIVATION,
                 grid_range=KASN_GRID_RANGE, use_residual=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # h(x) = x, the parameter-free identity residual connection of eq. (2.4).
        self.use_residual = use_residual and (in_features == out_features)
        self.basis = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        self.num_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.num_basis))
        self.grid_size = grid_size
        self.use_base = KASN_BASE_BRANCH
        if self.use_base:
            self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.base_activation = base_activation()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.spline_weight,
                         -0.5 / self.grid_size, 0.5 / self.grid_size)
        if self.use_base:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x):
        out = F.linear(self.basis(x).view(x.size(0), -1),
                       self.spline_weight.view(self.out_features, -1))
        if self.use_base:
            out = out + F.linear(self.base_activation(x), self.base_weight)
        return out + x if self.use_residual else out

    # -- regularisation -------------------------------------------------------
    def edge_norms(self):
        """||c_j||_2 for every edge j, shape (out_features, in_features)."""
        return torch.norm(self.spline_weight, p=2, dim=2)

    def l12_norm(self):
        """sum_j ||c_j||_2 over this layer's edges."""
        return self.edge_norms().sum()

    def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
        return s * torch.sum(torch.abs(self.spline_weight))

    def group_lasso_regularization_loss(self, gl):
        return gl * self.l12_norm()

    def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
        with torch.no_grad():
            return int(torch.sum(self.edge_norms() > thr).item())

    def count_total_edges(self):
        return self.out_features * self.in_features


class KASN(nn.Module):

    def __init__(self, input_dim, n_samples, gamma=None, kasn_width=None,
                 depth=None, zeta_delta=KASN_ZETA_DELTA,
                 prune_during_training=PRUNE_DURING_TRAINING,
                 delta_mode=None, verbose=True):
        super().__init__()
        gamma = KASN_GAMMA if gamma is None else gamma
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.gamma = gamma
        self.zeta_delta = zeta_delta
        self.prune_during_training = prune_during_training
        self.delta_mode = DELTA_MODE if delta_mode is None else delta_mode

        self.G = max(5, int(n_samples ** gamma))                 # G_n
        self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
        if kasn_width is not None:
            self.W = int(kasn_width)
        elif KASN_WIDTH is not None:
            self.W = int(KASN_WIDTH)
        else:
            self.W = max(1, C_W * int(np.floor(np.log(n_samples))))   # W_n

        self.delta_n = (float('inf') if self.delta_mode == 'dual'
                        else float(max(5.0, DELTA_SCALE * n_samples ** zeta_delta)))

        self.layers = nn.ModuleList()
        self.layers.append(KASNLayer(input_dim, self.W, grid_size=self.G,
                                     use_residual=False))
        for _ in range(self.L - 2):
            self.layers.append(KASNLayer(self.W, self.W, grid_size=self.G,
                                         use_residual=True))
        self.layers.append(KASNLayer(self.W, 1, grid_size=self.G,
                                     use_residual=False))

        self.scaler_X = EmpiricalCDFTransformer()
        self.scaler_y = StandardScaler()
        self._assert_r_n_identity()
        if verbose:
            print(f"  KASN: L_n={self.L}  W_n={self.W}  G_n={self.G}  "
                  f"gamma={gamma}  d={input_dim}  r_n={self.r_n():,}  "
                  f"Delta_n={'inf (dual)' if not np.isfinite(self.delta_n) else f'{self.delta_n:.2f}'}")

    def r_n(self):
        """Total potential edges, eq. (2.1)."""
        return self.count_total_edges()

    def _assert_r_n_identity(self):
        """The stacked layers must hold exactly d W + (L-2) W^2 + W edges.

        This is the identity that makes the pruning threshold Delta_n / r_n the
        object defined in Section 2.3 rather than an arbitrary rescaling, so it
        is checked rather than assumed.
        """
        theoretical = (self.input_dim * self.W
                       + (self.L - 2) * self.W ** 2
                       + self.W)
        actual = self.count_total_edges()
        assert theoretical == actual, (
            f"r_n mismatch: eq.(2.1) gives {theoretical}, layers hold {actual}")

    def s_n_cap(self):
        """Deterministic sparsity bound s_n = ceil(C_s sbar_0) = O((log n)^2)."""
        return int(np.ceil(np.log(max(self.n_samples, 3)) ** 2))

    def count_total_edges(self):
        return sum(l.count_total_edges() for l in self.layers)

    def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
        return sum(l.count_active_edges(thr) for l in self.layers)

    def get_total_potential_activations(self):
        return sum(l.out_features * l.in_features * l.num_basis
                   for l in self.layers)

    # -- GROUP LASSO regularisation -------------------------------------------------------
    def l12_norm(self):
        """R_n = ||c_n||_{1,2} = sum_{j<=r_n} ||c_j||_2, as a float."""
        with torch.no_grad():
            return float(sum(l.l12_norm().item() for l in self.layers))

    def l12_norm_tensor(self):
        return sum(l.l12_norm() for l in self.layers)

    def group_lasso_regularization_loss(self, gl):
        return sum(l.group_lasso_regularization_loss(gl) for l in self.layers)

    def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
        return sum(l.l1_regularization_loss(s) for l in self.layers)

    def compute_lambda_reg(self):
        """sqrt(log(L W^2) / n): the rate factor multiplying the penalty."""
        return float(np.sqrt(np.log(self.L * self.W ** 2 + 1e-8) / self.n_samples))

    def compute_delta_penalty(self, lam=1.0):
        """Quadratic hinge softly enforcing ||c_n||_{1,2} <= Delta_n.

        Only used under DELTA_ENFORCEMENT='penalty'; under projection the
        constraint holds exactly at every step and the hinge is redundant.
        """
        if DELTA_ENFORCEMENT != 'penalty' or not np.isfinite(self.delta_n):
            return torch.zeros((), device=next(self.parameters()).device)
        return lam * torch.clamp(self.l12_norm_tensor() - self.delta_n,
                                 min=0.0) ** 2

    @staticmethod
    def _project_l1_ball(v, radius):
        """Euclidean projection of a non-negative vector onto {v : sum v <= R}.

        The standard sort-and-threshold algorithm: sort descending, find the
        largest rho with u_rho - (cumsum_rho - R)/rho > 0, and soft-threshold at
        theta = (cumsum_rho - R)/rho.  Returns v unchanged when it is already
        feasible.
        """
        s = float(v.sum())
        if s <= radius:
            return v
        u, _ = torch.sort(v, descending=True)
        css = torch.cumsum(u, dim=0)
        rho_idx = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        cond = u - (css - radius) / rho_idx > 0
        rho = int(torch.nonzero(cond).max().item()) + 1
        theta = (css[rho - 1] - radius) / rho
        return torch.clamp(v - theta, min=0.0)

    def project_to_delta_ball(self):
        """Project the coefficients onto {c : ||c||_{1,2} <= Delta_n}.

        The l_{1,2} ball projection factorises: project the vector of group
        norms (||c_1||_2, ..., ||c_{r_n}||_2) onto the l_1 ball of radius
        Delta_n, then rescale each group to its projected norm.  Groups whose
        norm is soft-thresholded to zero are removed outright, so the projection
        is simultaneously the feasibility step and a selection step — it is the
        primal counterpart of the Group Lasso penalty.

        Returns the number of edges the projection zeroed.
        """
        if not np.isfinite(self.delta_n):
            return 0
        with torch.no_grad():
            norms = torch.cat([l.edge_norms().reshape(-1) for l in self.layers])
            if float(norms.sum()) <= self.delta_n:
                return 0
            projected = self._project_l1_ball(norms, self.delta_n)
            # scale factor per group; groups at zero norm stay at zero
            scale = torch.where(norms > 1e-12, projected / norms.clamp_min(1e-12),
                                torch.zeros_like(norms))
            off, zeroed = 0, 0
            for l in self.layers:
                k = l.out_features * l.in_features
                s = scale[off:off + k].view(l.out_features, l.in_features, 1)
                l.spline_weight.data.mul_(s)
                zeroed += int((s.reshape(-1) == 0).sum().item())
                off += k
            assert off == norms.numel()
        return zeroed

    def final_delta_projection(self, verbose=True):
        total = self.l12_norm()
        if not np.isfinite(self.delta_n):
            self.delta_n = total
            if verbose:
                print(f"    Dual mode: implied Delta_n = R_n(lambda) = {total:.4f}")
            return total
        if total > self.delta_n:
            zeroed = self.project_to_delta_ball()
            shrink = self.l12_norm() / max(total, 1e-12)
            if verbose:
                print(f"    Feasibility projection: ||c||_1,2 {total:.4f} -> "
                      f"{self.l12_norm():.4f} (Delta_n={self.delta_n:.4f}), "
                      f"{zeroed:,} edges zeroed")
            if shrink < 0.9 and DELTA_ENFORCEMENT == 'penalty':
                print(f"    ** WARNING: the feasibility projection rescaled the "
                      f"fitted network to {shrink:.2f}x its norm. lambda_n is "
                      f"too small for Delta_n={self.delta_n:.2f}; the reported "
                      f"estimator is not the one that was validated. Use "
                      f"LAMBDA_SELECTION='delta_constraint', raise "
                      f"SHARED_PENALTY_GRID, raise DELTA_SCALE, or switch to "
                      f"DELTA_ENFORCEMENT='project'.")
        elif verbose:
            print(f"    ||c||_1,2 = {total:.4f} <= Delta_n = {self.delta_n:.4f} "
                  f"(feasible, no projection needed)")
        return total

    def _eff_delta(self):
        """Delta_n, or its realised counterpart when the dual form is in use."""
        return self.l12_norm() if not np.isfinite(self.delta_n) else self.delta_n

    # -- pruning --------------------------------------------------------------
    def active_edge_threshold(self, method=None, val=None):
        """The threshold below which an edge is declared inactive.

        'delta_over_r' is the adaptive criterion of Section 2.3: Delta_n / r_n
        is the edge norm of a hypothetical network spreading its whole
        coefficient budget evenly over all r_n potential edges, so an edge below
        it carries less than an even share of the mass.
        """
        method = PRUNING_THRESHOLD_METHOD if method is None else method
        if method == 'delta_over_r':
            r_n = self.r_n()
            return self._eff_delta() / r_n if r_n > 0 else 0.0
        if method == 'fixed':
            return val if val is not None else KASN_ACTIVE_EDGE_THRESHOLD
        if method == 'relative_fraction':
            frac = val if val is not None else PRUNING_RELATIVE_FRACTION
            with torch.no_grad():
                mx = max(l.edge_norms().max().item() for l in self.layers)
            return frac * mx
        raise ValueError(f"Unknown pruning method: {method}")

    def prune_edges(self, method=None, val=None):
        thr = self.active_edge_threshold(method, val)
        pruned = total = 0
        with torch.no_grad():
            for layer in self.layers:
                mask = layer.edge_norms() <= thr
                layer.spline_weight.data[mask] = 0.0
                pruned += int(mask.sum().item())
                total += mask.numel()
        return pruned, total, thr

    def apply_post_training_pruning(self, method=None, val=None, verbose=True):
        p, t, thr = self.prune_edges(method, val)
        if verbose:
            print(f"    Pruning ({method or PRUNING_THRESHOLD_METHOD}, "
                  f"threshold={thr:.3e}): {p:,}/{t:,} edges zeroed "
                  f"({100 * p / t:.1f}%)")
        return p, t, thr

    def set_prune_mask(self):
        """Freeze the current zero pattern so later training cannot revive it.

        Stored as a plain per-layer attribute rather than a buffer, so it never
        enters state_dict and cannot collide with checkpoint restoration.
        """
        n_masked = n_total = 0
        with torch.no_grad():
            for layer in self.layers:
                m = (layer.edge_norms() == 0)
                layer._prune_mask = m
                n_masked += int(m.sum().item())
                n_total += m.numel()
        return n_masked, n_total

    def apply_prune_mask(self):
        """Re-zero masked edges; called after every optimiser step."""
        with torch.no_grad():
            for layer in self.layers:
                m = getattr(layer, '_prune_mask', None)
                if m is not None:
                    layer.spline_weight.data[m] = 0.0

    def clear_prune_mask(self):
        for layer in self.layers:
            if hasattr(layer, '_prune_mask'):
                del layer._prune_mask

    def masked_edge_counts(self):
        """(active, total) implied by the frozen mask, or by exact zeros."""
        act = tot = 0
        with torch.no_grad():
            for layer in self.layers:
                m = getattr(layer, '_prune_mask', None)
                if m is None:
                    m = (layer.edge_norms() == 0)
                act += int((~m).sum().item())
                tot += m.numel()
        return act, tot

    def sparsity_report(self):
        """Sparsity ratio at each threshold, plus the theoretical benchmarks."""
        tot = self.count_total_edges()
        thr_dr = self.active_edge_threshold('delta_over_r')
        ae_dr = self.count_active_edges(thr_dr)
        ae_fix = self.count_active_edges(KASN_ACTIVE_EDGE_THRESHOLD)
        act_masked, _ = self.masked_edge_counts()
        s_n = self.s_n_cap()
        return {
            'r_n': tot,
            'delta_n': self._eff_delta(),
            'threshold_delta_over_r': thr_dr,
            'active_edges_delta_over_r': ae_dr,
            'active_edges_fixed': ae_fix,
            'active_edges_masked': act_masked,
            'sparsity_ratio': 1.0 - ae_dr / tot if tot else np.nan,
            'sparsity_ratio_masked': 1.0 - act_masked / tot if tot else np.nan,
            's_n_cap': s_n,
            's_n_cap_respected': bool(ae_dr <= s_n),
            'l12_norm': self.l12_norm(),
            'G_n': self.G, 'L_n': self.L, 'W_n': self.W,
        }

    # -- forward / fit / predict ---------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, X_train_np, y_train_np, X_val_np, y_val_np,
            epochs=None, lr=None, batch_size=None, patience=KASN_PATIENCE,
            weight_decay=KASN_WEIGHT_DECAY, l1_reg_scale=KASN_L1_REG_SCALE,
            group_lasso_reg_scale=None, resume_scalers=False, verbose=True,
            log_every=25):
        epochs = KASN_N_EPOCHS if epochs is None else epochs
        if lr is None:
            lr = KASN_LR if KASN_LR is not None else KASN_LR_GRID[0]
        if batch_size is None:
            batch_size = (KASN_BATCH_SIZE if KASN_BATCH_SIZE is not None
                          else KASN_BATCH_SIZE_GRID[0])
        if group_lasso_reg_scale is None:
            group_lasso_reg_scale = (KASN_GROUP_LASSO_REG_SCALE
                                     or SHARED_PENALTY_GRID[0])

        y_train_np = np.asarray(y_train_np, dtype=np.float64).reshape(-1, 1)
        y_val_np = np.asarray(y_val_np, dtype=np.float64).reshape(-1, 1)

        if not resume_scalers:
            self.scaler_X.fit(X_train_np)
            self.scaler_y.fit(y_train_np)

        X_train_t = torch.tensor(self.scaler_X.transform(X_train_np),
                                 dtype=torch.float32)
        y_train_t = torch.tensor(self.scaler_y.transform(y_train_np),
                                 dtype=torch.float32)
        X_val_t = torch.tensor(self.scaler_X.transform(X_val_np),
                               dtype=torch.float32)
        y_val_t = torch.tensor(self.scaler_y.transform(y_val_np),
                               dtype=torch.float32)

        pin = (DEVICE.type == 'cuda')
        bs_actual = _effective_batch_size(len(X_train_t), batch_size)
        drop_last = len(X_train_t) > bs_actual

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=bs_actual, shuffle=_should_shuffle(), drop_last=drop_last,
            pin_memory=pin, num_workers=0)

        self.to(DEVICE)
        X_val_gpu = X_val_t.to(DEVICE, non_blocking=True)
        y_val_gpu = y_val_t.to(DEVICE, non_blocking=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500,
                                                    gamma=0.5)
        criterion = nn.MSELoss()
        amp_enabled = USE_AMP and (DEVICE.type == 'cuda')
        scaler_amp = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        lambda_reg = self.compute_lambda_reg()
        best_val_loss, best_state, best_epoch = float('inf'), None, 0
        epochs_no_improve = 0
        train_losses, val_losses, epoch_log = [], [], []

        for epoch in range(epochs):
            self.train()
            epoch_mse, seen = 0.0, 0
            for Xb, yb in train_loader:
                Xb = Xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=amp_enabled):
                    pred = self(Xb)
                    mse_loss = criterion(pred, yb)
                    gl_loss = self.group_lasso_regularization_loss(
                        group_lasso_reg_scale)
                    l1_loss = self.l1_regularization_loss(l1_reg_scale)
                    total = (mse_loss + lambda_reg * (gl_loss + l1_loss)
                             + self.compute_delta_penalty())
                scaler_amp.scale(total).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP_NORM)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                if DELTA_ENFORCEMENT == 'project':
                    self.project_to_delta_ball()
                self.apply_prune_mask()
                epoch_mse += mse_loss.item() * Xb.size(0)
                seen += Xb.size(0)

            scheduler.step()
            avg_mse = epoch_mse / max(seen, 1)
            train_losses.append(avg_mse)

            self.eval()
            with torch.no_grad():
                parts = []
                for i in range(0, X_val_gpu.shape[0], INFERENCE_CHUNK_SIZE):
                    with torch.amp.autocast(device_type=DEVICE.type,
                                            enabled=amp_enabled):
                        parts.append(self(X_val_gpu[i:i + INFERENCE_CHUNK_SIZE]))
                val_loss = criterion(torch.cat(parts).float(), y_val_gpu).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                thr = self.active_edge_threshold()
                ae = self.count_active_edges(thr)
                te = self.count_total_edges()
                if self.prune_during_training:
                    self.prune_edges()
                print(f"      ep {epoch:4d}/{epochs}  train={avg_mse:.6f}  "
                      f"val={val_loss:.6f}  best={best_val_loss:.6f} "
                      f"(ep {best_epoch})  active={ae}/{te}  "
                      f"sparsity={1 - ae / te:.4f}  R_n={self.l12_norm():.2f}")
                epoch_log.append({'epoch': epoch, 'train_loss': avg_mse,
                                  'val_mse': val_loss,
                                  'best_val_mse': best_val_loss,
                                  'active_edges': ae, 'total_edges': te,
                                  'l12_norm': self.l12_norm(),
                                  'lr': scheduler.get_last_lr()[0]})

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    print(f"      early stopping at ep {epoch} "
                          f"(best {best_epoch}, val {best_val_loss:.6f})")
                break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.to(DEVICE)

        if (verbose and len(train_losses) > 5
                and train_losses[0] > 0
                and (train_losses[0] - min(train_losses)) / train_losses[0] < 0.01):
            print(f"    ** WARNING: training loss moved <1% "
                  f"({train_losses[0]:.4f} -> {min(train_losses):.4f}); the fit "
                  f"is essentially constant. "
                  + (f"Delta_n={self.delta_n:.2f} over r_n={self.r_n():,} edges "
                     f"is likely too tight — raise DELTA_SCALE or use "
                     f"DELTA_MODE='dual'."
                     if DELTA_ENFORCEMENT == 'project'
                     else "Check the penalty scale and learning rate."))
        return train_losses, val_losses, best_val_loss, epoch_log

    def predict(self, X_np):
        self.eval()
        self.to(DEVICE)
        amp_enabled = USE_AMP and (DEVICE.type == 'cuda')
        X_t = torch.tensor(self.scaler_X.transform(X_np), dtype=torch.float32)
        parts = []
        with torch.inference_mode():
            for i in range(0, X_t.shape[0], INFERENCE_CHUNK_SIZE):
                chunk = X_t[i:i + INFERENCE_CHUNK_SIZE].to(DEVICE,
                                                           non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=amp_enabled):
                    parts.append(self(chunk).float().cpu())
        return self.scaler_y.inverse_transform(
            torch.cat(parts).numpy()).flatten()


# ============================================================================
# COMPARISON ARCHITECTURES
# ============================================================================


class SLFN(nn.Module):
    """Chen and White (1999) single-hidden-layer sieve network.

    The hidden width r_n = C (n / log n)^{1/(2(1 + alpha/d*))} and the norm
    constraints on the input and output weights are the sieve restrictions that
    give this class its entropy bound; they are imposed by rescaling rather than
    by penalisation, so the network stays inside the sieve at every step.
    """

    def __init__(self, input_dim, sample_size, m=M_SMOOTH, c_n=C_N_SLFN,
                 C=C_SLFN, bn_mode=BN_MODE, alpha=ALPHA_SLFN, C_OUT=C_OUT_SLFN,
                 verbose=False):
        super().__init__()
        self.m = m
        self.c_weight = c_n / 2
        self.c_bias = c_n / 2
        d_star = input_dim + 1
        exponent = 1.0 / (2.0 * (1.0 + alpha / d_star))
        r_n = int((sample_size / np.log(sample_size)) ** exponent * C)
        self.hidden_dim = max(r_n, 1)
        self.B_n = (float(np.log(self.hidden_dim)) * C_OUT
                    if bn_mode == "log" else C_OUT)
        if verbose:
            print(f"  SLFN: hidden_dim={self.hidden_dim}, B_n={self.B_n:.4f}")
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1, bias=False)
        self.activation = nn.Sigmoid()

    def get_constrained_params(self):
        wl = self.fc1.weight.abs().sum(dim=1) + 1e-8
        ba = self.fc1.bias.abs() + 1e-8
        sc = torch.min(torch.clamp(self.c_weight / wl, max=1.0),
                       torch.clamp(self.c_bias / ba, max=1.0))
        w1 = self.fc1.weight * sc.unsqueeze(1)
        b1 = self.fc1.bias * sc
        ol = self.fc2.weight.abs().sum() + 1e-8
        w2 = self.fc2.weight * torch.clamp(self.B_n / ol, max=1.0)
        return w1, b1, w2

    def forward(self, x):
        w1, b1, w2 = self.get_constrained_params()
        an = torch.norm(w1, p=2, dim=1).clamp(min=1.0)
        h = self.activation(F.linear(x, w1, b1)) * torch.pow(an, -self.m)
        return F.linear(h, w2)


class FarrellDNN(nn.Module):
    """Farrell, Liang and Misra (2021) feedforward ReLU network."""

    def __init__(self, input_dim, hidden_layer_sizes=None,
                 alpha_reg=DNN_ALPHA_REG, r_par=DNN_R_PAR):
        super().__init__()
        if hidden_layer_sizes is None:
            hidden_layer_sizes = DNN_HIDDEN_LAYER_SIZES
        self.alpha_reg = alpha_reg
        self.r_par = r_par
        sizes = [input_dim] + list(hidden_layer_sizes) + [1]
        layers = []
        for i in range(len(sizes) - 1):
            lin = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self._linears = [m for m in self.network.modules()
                         if isinstance(m, nn.Linear)]

    def forward(self, x):
        return self.network(x)

    def l1l2_penalty(self):
        if self.alpha_reg == 0.0:
            return torch.zeros((), device=next(self.parameters()).device)
        l1c = self.alpha_reg * self.r_par
        l2c = self.alpha_reg * (1.0 - self.r_par)
        p = torch.zeros((), device=next(self.parameters()).device)
        for lin in self._linears:
            p = p + l1c * lin.weight.abs().sum() + l2c * (lin.weight ** 2).sum()
        return p

    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def _amp_epoch(model, X_tr, y_tr, optimizer, criterion, batch_size,
               amp_enabled, extra_loss_fn=None, scaler_amp=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    eff_bs = _effective_batch_size(len(X_tr), batch_size)
    idx = (torch.arange(len(X_tr), device=DEVICE) if eff_bs >= len(X_tr)
           else torch.randperm(len(X_tr), device=DEVICE)[:eff_bs])
    Xb, yb = X_tr[idx], y_tr[idx]
    with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
        pred = model(Xb)
        mse = criterion(pred, yb)
        total = mse + (extra_loss_fn() if extra_loss_fn else 0.0)
    if scaler_amp is not None:
        scaler_amp.scale(total).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
    else:
        total.backward()
        optimizer.step()
    return mse.item()


def _neural_hp_search(ModelClass, model_kwargs, X_tr_t, y_tr_t, X_va_t, y_va_t,
                      base_lr, base_batch_size, tuning_epochs, amp_enabled,
                      label, verbose=True):
    """Greedy lr -> batch size -> weight decay search on the fold's holdout."""
    criterion = nn.MSELoss()
    lr_grid = SHARED_LR_GRID if base_lr is None else [base_lr]
    bs_grid = ([None] if FULL_BATCH else
               (SHARED_BATCH_SIZE_GRID if base_batch_size is None
                else [base_batch_size]))
    cur_lr, cur_bs, cur_wd = lr_grid[0], bs_grid[0], SHARED_PENALTY_GRID[0]

    def _eval(lr_v, bs_v, wd_v):
        cand = ModelClass(**model_kwargs).to(DEVICE)
        opt = torch.optim.Adam(cand.parameters(), lr=lr_v, weight_decay=wd_v)
        sc = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        extra = cand.l1l2_penalty if hasattr(cand, 'l1l2_penalty') else None
        for _ in range(tuning_epochs):
            _amp_epoch(cand, X_tr_t, y_tr_t, opt, criterion, bs_v, amp_enabled,
                       extra_loss_fn=extra, scaler_amp=sc)
        cand.eval()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=amp_enabled):
                vl = criterion(cand(X_va_t).float(), y_va_t).item()
        del cand, opt, sc
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return vl

    for grid, name in ((lr_grid, 'lr'), (bs_grid, 'bs'),
                       (SHARED_PENALTY_GRID, 'wd')):
        if name != 'wd' and len(grid) <= 1:
            continue
        best_vl = float('inf')
        for v in grid:
            vl = _eval(v if name == 'lr' else cur_lr,
                       v if name == 'bs' else cur_bs,
                       v if name == 'wd' else cur_wd)
            if vl < best_vl:
                best_vl = vl
                if name == 'lr':
                    cur_lr = v
                elif name == 'bs':
                    cur_bs = v
                else:
                    cur_wd = v
        if verbose:
            chosen = {'lr': cur_lr, 'bs': cur_bs, 'wd': cur_wd}[name]
            print(f"      [{label}] {name} -> {chosen}  (val {best_vl:.6f})")
    return cur_wd, cur_lr, cur_bs


def _train_neural_generic(model_obj, label, X_train, y_train, X_val, y_val,
                          epochs, patience, best_lr, best_bs, best_wd,
                          amp_enabled, scaler_X, scaler_y, extra_loss_fn=None,
                          verbose=True):
    X_train_s = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler_X.transform(X_val).astype(np.float32)
    y_train_s = scaler_y.fit_transform(
        np.asarray(y_train).reshape(-1, 1)).flatten().astype(np.float32)
    y_val_s = scaler_y.transform(
        np.asarray(y_val).reshape(-1, 1)).flatten().astype(np.float32)

    X_tr_t = torch.tensor(X_train_s, device=DEVICE)
    y_tr_t = torch.tensor(y_train_s, device=DEVICE).reshape(-1, 1)
    X_va_t = torch.tensor(X_val_s, device=DEVICE)
    y_va_t = torch.tensor(y_val_s, device=DEVICE).reshape(-1, 1)

    model_obj.to(DEVICE)
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=best_lr,
                                 weight_decay=best_wd)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    criterion = nn.MSELoss()

    best_val_loss, best_state, best_epoch, patience_counter = float('inf'), None, 0, 0
    for epoch in range(epochs):
        _amp_epoch(model_obj, X_tr_t, y_tr_t, optimizer, criterion, best_bs,
                   amp_enabled, extra_loss_fn=extra_loss_fn,
                   scaler_amp=scaler_amp)
        model_obj.eval()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=amp_enabled):
                val_loss = criterion(model_obj(X_va_t).float(), y_va_t).item()
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model_obj.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience and patience_counter >= patience:
            break
    if best_state:
        model_obj.load_state_dict(best_state)
    if verbose:
        print(f"      [{label}] best epoch {best_epoch}, val {best_val_loss:.6f}")

    def _predict(X_np):
        Xs = torch.tensor(scaler_X.transform(X_np).astype(np.float32),
                          device=DEVICE)
        model_obj.eval()
        with torch.inference_mode():
            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=amp_enabled):
                out = model_obj(Xs).float().cpu().numpy()
        return scaler_y.inverse_transform(out).flatten()

    return _predict, best_val_loss


# ============================================================================
# KASN PRUNING / FINE-TUNING / HYPERPARAMETER SELECTION
# ============================================================================


def _scaled_val_loss(model, X_val, y_val):
    yv = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
    yv_s = model.scaler_y.transform(yv).ravel()
    pr_s = model.scaler_y.transform(
        np.asarray(model.predict(X_val), dtype=np.float64).reshape(-1, 1)).ravel()
    return float(mean_squared_error(yv_s, pr_s))


def prune_and_finetune(model, X_train, y_train, X_val, y_val, lr, batch_size,
                       group_lasso_reg_scale, label='', verbose=True):
    val_pre = _scaled_val_loss(model, X_val, y_val)
    if not POST_TRAINING_PRUNING:
        return val_pre, val_pre, val_pre

    model.apply_post_training_pruning(verbose=verbose)
    val_post = _scaled_val_loss(model, X_val, y_val)
    val_ft = val_post

    if POST_PRUNE_FINETUNE_EPOCHS > 0:
        n_masked, n_total = model.set_prune_mask()
        if verbose:
            print(f"    {label}fine-tuning {POST_PRUNE_FINETUNE_EPOCHS} epochs "
                  f"with {n_masked:,}/{n_total:,} edges frozen at zero "
                  f"({100 * n_masked / n_total:.1f}%)")

        pruned_state = copy.deepcopy(model.state_dict())
        model.fit(X_train, y_train, X_val, y_val,
                  epochs=POST_PRUNE_FINETUNE_EPOCHS,
                  lr=lr * POST_PRUNE_FINETUNE_LR_SCALE, batch_size=batch_size,
                  patience=POST_PRUNE_FINETUNE_PATIENCE,
                  weight_decay=KASN_WEIGHT_DECAY,
                  l1_reg_scale=KASN_L1_REG_SCALE,
                  group_lasso_reg_scale=group_lasso_reg_scale,
                  resume_scalers=True, verbose=False)
        model.apply_prune_mask()          # restoration may reintroduce values
        val_ft = _scaled_val_loss(model, X_val, y_val)

        reverted = False
        if val_ft > val_post:
            model.load_state_dict(pruned_state)
            model.to(DEVICE)
            model.apply_prune_mask()
            val_ft = _scaled_val_loss(model, X_val, y_val)
            reverted = True

        act, tot = model.masked_edge_counts()     
        assert tot - act == n_masked, (
            f"prune mask not respected: {tot - act} zeroed vs {n_masked} masked")
        if verbose and reverted:
            print(f"    {label}fine-tune made validation worse; reverted to the "
                  f"pruned fit ({val_ft:.6f})")
        if verbose:
            damage = val_post - val_pre
            rec = (f"recovered {100 * (val_post - val_ft) / damage:.0f}% of the "
                   f"pruning loss" if damage > 1e-9 * max(val_pre, 1e-12)
                   else "pruning cost nothing to recover")
            print(f"    {label}val {val_pre:.6f} pre-prune -> {val_post:.6f} "
                  f"pruned -> {val_ft:.6f} fine-tuned ({rec})  "
                  f"active {act:,}/{tot:,} ({100 * act / tot:.1f}%)")
    return val_pre, val_post, val_ft


def select_kasn_hyperparams(X_train, y_train, X_val, y_val, n_samples,
                            depth=None, label='', verbose=True):
    t0 = time.time()
    hp_log = []
    depth = KASN_DEPTH if depth is None else depth

    gamma_grid = KASN_GAMMA_GRID if KASN_GAMMA is None else [KASN_GAMMA]
    lr_grid = KASN_LR_GRID if KASN_LR is None else [KASN_LR]
    bs_grid = ([None] if FULL_BATCH else
               (KASN_BATCH_SIZE_GRID if KASN_BATCH_SIZE is None
                else [KASN_BATCH_SIZE]))
    gl_grid = (SHARED_PENALTY_GRID if KASN_GROUP_LASSO_REG_SCALE is None
               else [KASN_GROUP_LASSO_REG_SCALE])
    log_n = int(np.floor(np.log(n_samples)))
    cw_grid = C_W_GRID if KASN_WIDTH is None else [None]

    cur_gamma, cur_lr, cur_bs = gamma_grid[0], lr_grid[0], bs_grid[0]
    cur_gl = gl_grid[0]
    cur_w = KASN_WIDTH if KASN_WIDTH is not None else max(1, C_W_GRID[0] * log_n)

    if verbose:
        n_cands = (len(gamma_grid) + len(lr_grid) + len(bs_grid)
                   + len(gl_grid) * len(cw_grid))
        scored = ("pruned + fine-tuned" if (PRUNE_AWARE_SELECTION
                                            and POST_TRAINING_PRUNING)
                  else "unpruned")
        print(f"\n  {label}KASN hyperparameter selection: {n_cands} candidates, "
              f"{KASN_TUNING_EPOCHS} tuning epochs each, scoring the {scored} fit")

    def _run(gamma, lr, bs, gl, width, phase):
        cand = KASN(input_dim=X_train.shape[1], n_samples=n_samples,
                    gamma=gamma, kasn_width=width, depth=depth,
                    prune_during_training=False, verbose=False)
        _, _, vl_unpruned, _ = cand.fit(
            X_train, y_train, X_val, y_val, epochs=KASN_TUNING_EPOCHS, lr=lr,
            batch_size=bs, patience=None, group_lasso_reg_scale=gl,
            verbose=False)
        R_n = cand.l12_norm()
        vl, spars = vl_unpruned, np.nan
        if PRUNE_AWARE_SELECTION and POST_TRAINING_PRUNING:
            _, _, vl = prune_and_finetune(cand, X_train, y_train, X_val, y_val,
                                          lr=lr, batch_size=bs,
                                          group_lasso_reg_scale=gl,
                                          verbose=False)
            ae, te = cand.masked_edge_counts()
            spars = 1.0 - ae / te if te else np.nan
        delta_n = cand.delta_n if np.isfinite(cand.delta_n) else np.nan
        del cand
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        hp_log.append({'phase': phase, 'gamma': gamma, 'lr': lr, 'bs': bs,
                       'gl': gl, 'width': width, 'val_loss': vl,
                       'val_loss_unpruned': vl_unpruned, 'R_n': R_n,
                       'delta_n': delta_n, 'sparsity': spars,
                       'G_n': max(5, int(n_samples ** gamma))})
        return vl, R_n

    for grid, name in ((gamma_grid, 'gamma'), (lr_grid, 'lr'), (bs_grid, 'bs')):
        if len(grid) <= 1:
            continue
        best_vl = float('inf')
        for v in grid:
            vl, _ = _run(v if name == 'gamma' else cur_gamma,
                         v if name == 'lr' else cur_lr,
                         v if name == 'bs' else cur_bs,
                         cur_gl, cur_w, phase=name)
            if vl < best_vl:
                best_vl = vl
                if name == 'gamma':
                    cur_gamma = v
                elif name == 'lr':
                    cur_lr = v
                else:
                    cur_bs = v
        if verbose:
            chosen = {'gamma': cur_gamma, 'lr': cur_lr, 'bs': cur_bs}[name]
            print(f"      {name} -> {chosen}  (val {best_vl:.6f})")

    delta_target = float(max(5.0, n_samples ** KASN_ZETA_DELTA))
    best_vl = float('inf')
    best_gl, best_w = cur_gl, cur_w
    feasible = []
    for gl in sorted(gl_grid):
        for cw in cw_grid:
            w = cur_w if cw is None else max(1, cw * log_n)
            vl, R_n = _run(cur_gamma, cur_lr, cur_bs, gl, w, phase='gl_width')
            if R_n <= delta_target:
                feasible.append((gl, w, vl, R_n))
            if vl < best_vl:
                best_vl, best_gl, best_w = vl, gl, w
    if LAMBDA_SELECTION == 'delta_constraint':
        assert feasible, (
            f"no penalty in {sorted(gl_grid)} brings R_n below "
            f"Delta_n={delta_target:.2f}; widen SHARED_PENALTY_GRID upward")
        cur_gl, cur_w, _vl, _R = feasible[0]     # smallest lambda that is feasible
        if verbose:
            print(f"      lambda_n by eq.(2.5): smallest penalty with "
                  f"R_n <= Delta_n={delta_target:.2f}  ->  lambda={cur_gl:.0e} "
                  f"(R_n={_R:.2f}, val {_vl:.6f})")
    else:
        cur_gl, cur_w = best_gl, best_w
        if verbose:
            print(f"      (lambda, W_n) -> ({cur_gl:.0e}, {cur_w})  "
                  f"(val {best_vl:.6f})")

    sel_time = time.time() - t0
    hp = {'gamma': cur_gamma, 'lr': cur_lr, 'batch_size': cur_bs,
          'group_lasso': cur_gl, 'width': cur_w, 'depth': depth,
          'selection_time': sel_time, 'log': hp_log}
    if verbose:
        print(f"    Selected: gamma={cur_gamma}  lr={cur_lr:.0e}  "
              f"bs={cur_bs}  lambda={cur_gl:.0e}  W_n={cur_w}   "
              f"({sel_time:.1f}s)")
      
        for val, grid, name in ((cur_gl, sorted(gl_grid), 'group-lasso lambda'),
                                (cur_lr, sorted(lr_grid), 'learning rate'),
                                (cur_gamma, sorted(gamma_grid), 'gamma')):
            if len(grid) > 1 and val in (grid[0], grid[-1]):
                edge = 'lower' if val == grid[0] else 'upper'
                print(f"    ** WARNING: the selected {name} ({val:g}) is at the "
                      f"{edge} edge of {grid}. Extend the grid in that "
                      f"direction and re-run; the search may want to go "
                      f"further.")
    return hp


def default_kasn_hp(n_samples):
    """The pinned hyperparameters, used when HP_SELECTION_MODE == 'fixed'."""
    log_n = int(np.floor(np.log(n_samples)))
    return {
        'gamma': KASN_GAMMA,
        'lr': KASN_LR if KASN_LR is not None else KASN_LR_GRID[0],
        'batch_size': (KASN_BATCH_SIZE if KASN_BATCH_SIZE is not None
                       else KASN_BATCH_SIZE_GRID[0]),
        'group_lasso': (KASN_GROUP_LASSO_REG_SCALE
                        if KASN_GROUP_LASSO_REG_SCALE is not None
                        else SHARED_PENALTY_GRID[0]),
        'width': KASN_WIDTH if KASN_WIDTH is not None else max(1, C_W * log_n),
        'depth': KASN_DEPTH,
        'selection_time': 0.0,
        'log': [],
    }


def _tail_holdout(n_fit, val_fraction=None, buffer=None):
    vf = VAL_FRACTION if val_fraction is None else val_fraction
    buf = VAL_BUFFER if buffer is None else buffer
    n_val = max(1, int(round(n_fit * vf)))
    n_val = min(n_val, n_fit - 2)
    cut = n_fit - n_val
    tr = np.arange(0, max(1, cut - buf))
    va = np.arange(cut, n_fit)
    assert len(tr) > 0 and len(va) > 0, "fit fold too small to hold out a tail"
    return tr, va

def _out_path(stem, ext, output_dir=None):
    """Artifact path stamped with this replication's seed and run identifier."""
    d = OUTPUT_DIR if output_dir is None else output_dir
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{stem}_seed{SEED}_{RUN_ID}.{ext}")


def free_learner(info):
    info.pop('_model', None)
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def _fit_kasn(X_tr, y_tr, X_va, y_va, hp=None, n_samples=None, label='',
              verbose=True):
    n_samples = (len(y_tr) + len(y_va)) if n_samples is None else n_samples
    if hp is None:
        hp = select_kasn_hyperparams(X_tr, y_tr, X_va, y_va,
                                     n_samples=n_samples, label=label,
                                     verbose=verbose)
    t0 = time.time()
    model = KASN(input_dim=X_tr.shape[1], n_samples=n_samples,
                 gamma=hp['gamma'], kasn_width=hp['width'], depth=hp['depth'],
                 prune_during_training=PRUNE_DURING_TRAINING, verbose=verbose)
    _, _, best_val, _ = model.fit(
        X_tr, y_tr, X_va, y_va, epochs=KASN_N_EPOCHS, lr=hp['lr'],
        batch_size=hp['batch_size'], patience=KASN_PATIENCE,
        group_lasso_reg_scale=hp['group_lasso'], verbose=verbose,
        log_every=max(25, KASN_N_EPOCHS // 8))
    val_pre, val_post, val_ft = prune_and_finetune(
        model, X_tr, y_tr, X_va, y_va, lr=hp['lr'],
        batch_size=hp['batch_size'], group_lasso_reg_scale=hp['group_lasso'],
        label=label, verbose=verbose)
    # Feasibility (Lemma 6): the reported estimator must lie in K_n.
    model.final_delta_projection(verbose=verbose)

    rep = model.sparsity_report()
    info = {'_model': model, 'val_mse': val_ft, 'val_mse_pre_prune': val_pre,
            'val_mse_post_prune': val_post, 'val_mse_fit': best_val,
            'fit_seconds': time.time() - t0,
            'selection_time': hp.get('selection_time', 0.0),
            'selected_gamma': hp['gamma'], 'selected_lr': hp['lr'],
            'selected_bs': hp['batch_size'], 'selected_gl': hp['group_lasso'],
            'selected_width': hp['width'], 'selected_depth': hp['depth']}
    info.update(rep)
    return model.predict, info


def _fit_torch(kind, X_tr, y_tr, X_va, y_va, verbose=True):
    """SLFN or Farrell DNN."""
    amp_enabled = USE_AMP and (DEVICE.type == 'cuda')
    scaler_X, scaler_y = EmpiricalCDFTransformer(), StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler_X.transform(X_va).astype(np.float32)
    y_tr_s = scaler_y.fit_transform(
        np.asarray(y_tr).reshape(-1, 1)).flatten().astype(np.float32)
    y_va_s = scaler_y.transform(
        np.asarray(y_va).reshape(-1, 1)).flatten().astype(np.float32)
    X_tr_t = torch.tensor(X_tr_s, device=DEVICE)
    y_tr_t = torch.tensor(y_tr_s, device=DEVICE).reshape(-1, 1)
    X_va_t = torch.tensor(X_va_s, device=DEVICE)
    y_va_t = torch.tensor(y_va_s, device=DEVICE).reshape(-1, 1)

    if kind == 'slfn':
        cls, kwargs = SLFN, {'input_dim': X_tr.shape[1],
                             'sample_size': len(y_tr)}
        base_lr, base_bs, base_wd = SLFN_LR, SLFN_BATCH_SIZE, SLFN_WEIGHT_DECAY
        epochs, patience, tune = SLFN_EPOCHS, SLFN_PATIENCE, SLFN_TUNING_EPOCHS
    else:
        cls, kwargs = FarrellDNN, {'input_dim': X_tr.shape[1]}
        base_lr, base_bs, base_wd = DNN_LR, DNN_BATCH_SIZE, DNN_WEIGHT_DECAY
        epochs, patience = DNN_MAX_EPOCHS, DNN_MAX_EPOCHS_NO_CHANGE
        tune = DNN_TUNING_EPOCHS

    t0 = time.time()
    sel_t0 = time.time()
    if any(v is None for v in (base_lr, base_bs, base_wd)):
        wd, lr, bs = _neural_hp_search(cls, kwargs, X_tr_t, y_tr_t, X_va_t,
                                       y_va_t, base_lr, base_bs, tune,
                                       amp_enabled, kind.upper(), verbose)
    else:
        wd, lr, bs = base_wd, base_lr, base_bs
    sel_time = time.time() - sel_t0

    model = cls(**kwargs)
    extra = model.l1l2_penalty if hasattr(model, 'l1l2_penalty') else None
    predict_fn, best_val = _train_neural_generic(
        model, kind.upper(), X_tr, y_tr, X_va, y_va, epochs, patience, lr, bs,
        wd, amp_enabled, EmpiricalCDFTransformer(), StandardScaler(),
        extra_loss_fn=extra, verbose=verbose)

    info = {'_model': model, 'val_mse': best_val,
            'fit_seconds': time.time() - t0, 'selection_time': sel_time,
            'selected_lr': lr, 'selected_bs': bs, 'selected_wd': wd}
    if kind == 'slfn':
        with torch.no_grad():
            out_w = model.fc2.weight.detach().cpu().numpy()
        info['hidden_dim'] = model.hidden_dim
        info['active_units'] = int(np.sum(np.abs(out_w) > 1e-6))
    else:
        info['n_params'] = model.total_params()
    return predict_fn, info


def _fit_lasso(X_tr, y_tr, X_va, y_va, verbose=True):
    scaler_X, scaler_y = EmpiricalCDFTransformer(), StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_va_s = scaler_X.transform(X_va)
    y_tr_s = scaler_y.fit_transform(np.asarray(y_tr).reshape(-1, 1)).flatten()
    y_va_s = scaler_y.transform(np.asarray(y_va).reshape(-1, 1)).flatten()

    t0 = time.time()
    best = (float('inf'), None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in SHARED_PENALTY_GRID:
            cand = Lasso(alpha=a, max_iter=LASSO_MAX_ITER, random_state=SEED)
            cand.fit(X_tr_s, y_tr_s)
            vm = mean_squared_error(y_va_s, cand.predict(X_va_s))
            if vm < best[0]:
                best = (vm, a, cand)
    vm, alpha, model = best
    nz = int(np.sum(np.abs(model.coef_) > LASSO_NONZERO_THRESHOLD))
    if verbose:
        print(f"      [LASSO] alpha={alpha:.0e}  nonzero={nz}/{X_tr.shape[1]}"
              f"  val {vm:.6f}")

    def _predict(X):
        return scaler_y.inverse_transform(
            model.predict(scaler_X.transform(X)).reshape(-1, 1)).flatten()

    return _predict, {'_model': model, 'val_mse': vm, 'selected_alpha': alpha,
                      'nonzero_coefs': nz, 'total_features': X_tr.shape[1],
                      'sparsity_ratio': 1.0 - nz / X_tr.shape[1],
                      'fit_seconds': time.time() - t0, 'selection_time': 0.0}


def _fit_gam(X_tr, y_tr, X_va, y_va, verbose=True):
    try:
        from pygam import LinearGAM, s as gam_s
    except ImportError:
        return None, {'skipped': 'pygam not installed (pip install pygam)'}
    scaler_X, scaler_y = EmpiricalCDFTransformer(), StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_va_s = scaler_X.transform(X_va)
    y_tr_s = scaler_y.fit_transform(np.asarray(y_tr).reshape(-1, 1)).flatten()
    y_va_s = scaler_y.transform(np.asarray(y_va).reshape(-1, 1)).flatten()

    d = X_tr.shape[1]
    n_sp = (GAM_N_SPLINES if GAM_N_SPLINES is not None
            else max(5, int(len(y_tr) ** KASN_GAMMA)))
    terms = gam_s(0, n_splines=n_sp, spline_order=GAM_SPLINE_ORDER)
    for j in range(1, d):
        terms = terms + gam_s(j, n_splines=n_sp, spline_order=GAM_SPLINE_ORDER)

    t0 = time.time()
    best = (float('inf'), None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lam in SHARED_PENALTY_GRID:
            try:
                cand = LinearGAM(terms, lam=lam).fit(X_tr_s, y_tr_s)
                vm = mean_squared_error(y_va_s, cand.predict(X_va_s))
            except Exception as exc:        # pygam raises on ill-conditioning
                if verbose:
                    print(f"      [GAM] lam={lam:.0e} failed: "
                          f"{type(exc).__name__}")
                continue
            if vm < best[0]:
                best = (vm, lam, cand)
    vm, lam, model = best
    if model is None:
        return None, {'skipped': 'every GAM candidate failed to fit'}
    if verbose:
        print(f"      [GAM] lam={lam:.0e}  n_splines={n_sp}  val {vm:.6f}")
    try:
        edf = float(model.statistics_['edof'])
    except Exception:
        edf = float(n_sp * d)

    def _predict(X):
        return scaler_y.inverse_transform(
            model.predict(scaler_X.transform(X)).reshape(-1, 1)).flatten()

    return _predict, {'_model': model, 'val_mse': vm, 'selected_lam': lam,
                      'n_splines': n_sp, 'edf': edf,
                      'fit_seconds': time.time() - t0, 'selection_time': 0.0}


def _fit_ppr(X_tr, y_tr, X_va, y_va, verbose=True):
    try:
        from skpp import ProjectionPursuitRegressor
    except ImportError:
        return None, {'skipped': 'skpp not installed (pip install skpp)'}
    scaler_X, scaler_y = EmpiricalCDFTransformer(), StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    X_va_s = scaler_X.transform(X_va)
    y_tr_s = scaler_y.fit_transform(np.asarray(y_tr).reshape(-1, 1)).flatten()
    y_va_s = scaler_y.transform(np.asarray(y_va).reshape(-1, 1)).flatten()

    t0 = time.time()
    best = (float('inf'), None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in PPR_R_GRID:
            try:
                cand = ProjectionPursuitRegressor(r=r, fit_type=PPR_FIT_TYPE,
                                                  degree=PPR_SPLINE_DF)
                cand.fit(X_tr_s, y_tr_s)
                vm = mean_squared_error(y_va_s, cand.predict(X_va_s))
            except Exception as exc:
                if verbose:
                    print(f"      [PPR] r={r} failed: {type(exc).__name__}")
                continue
            if vm < best[0]:
                best = (vm, r, cand)
    vm, r, model = best
    if model is None:
        return None, {'skipped': 'every PPR candidate failed to fit'}
    if verbose:
        print(f"      [PPR] r={r}  val {vm:.6f}")

    def _predict(X):
        return scaler_y.inverse_transform(
            model.predict(scaler_X.transform(X)).reshape(-1, 1)).flatten()

    return _predict, {'_model': model, 'val_mse': vm, 'selected_r': r,
                      'n_terms': r, 'fit_seconds': time.time() - t0,
                      'selection_time': 0.0}


def fit_learner(name, X_tr, y_tr, X_va, y_va, hp=None, n_samples=None,
                label='', verbose=True):
    """Fit one learner.  Returns (predict_fn, info); predict_fn is None if the
    learner is unavailable in this environment, with info['skipped'] saying why.
    """
    if name == 'kasn':
        return _fit_kasn(X_tr, y_tr, X_va, y_va, hp, n_samples, label, verbose)
    if name in ('slfn', 'dnn'):
        return _fit_torch(name, X_tr, y_tr, X_va, y_va, verbose)
    if name == 'lasso':
        return _fit_lasso(X_tr, y_tr, X_va, y_va, verbose)
    if name == 'gam':
        return _fit_gam(X_tr, y_tr, X_va, y_va, verbose)
    if name == 'ppr':
        return _fit_ppr(X_tr, y_tr, X_va, y_va, verbose)
    raise ValueError(f"unknown learner {name!r}")


def _stationary_ar1(n, d, rho, sd, rng, burn_in=None):
    """Gaussian AR(1) array with an N(0, sd^2) marginal for every rho.

    Scaling the innovation by sqrt(1-rho^2) keeps the stationary variance equal
    to sd^2 whatever the persistence, so changing rho changes the dependence and
    nothing else about the design.  Discarding a burn-in makes the returned
    block stationary rather than transient.
    """
    burn_in = BURN_IN if burn_in is None else burn_in
    if rho == 0.0:
        return rng.normal(0.0, sd, size=(n, d))
    total = n + burn_in
    innov = rng.normal(0.0, sd * np.sqrt(1.0 - rho ** 2), size=(total, d))
    out = np.empty((total, d))
    out[0] = rng.normal(0.0, sd, size=d)
    for t in range(1, total):
        out[t] = rho * out[t - 1] + innov[t]
    return out[burn_in:]


def _ar1_autocorr(x, lag=1):
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    denom = np.mean(xc ** 2)
    if denom <= 0:
        return 0.0
    return float(np.mean(xc[lag:] * xc[:-lag]) / denom)

# ============================================================================
# DATA GENERATING PROCESS 
# ============================================================================


def _f_additive(Xs):
    return np.sin(Xs).sum(axis=1)


def _f_compositional(Xs):
    return (1.0 / (1.0 + np.exp(-np.sin(Xs).sum(axis=1)))
            + 0.1 * np.abs(Xs).sum(axis=1))


def _f_deep(Xs):
    s = Xs.shape[1]
    inner = np.sin(Xs).sum(axis=1) / np.sqrt(max(s, 1))   # phi_j = sin
    psi2 = np.tanh(0.8 * inner)
    return np.sin(2.0 * psi2) + 0.5 * psi2 ** 2


_TARGETS = {'additive': _f_additive,
            'compositional': _f_compositional,
            'deep': _f_deep}


_SCALE_CACHE = {}


def _target_scale(name, s):
    key = (name, s, MARGINAL)
    if key not in _SCALE_CACHE:
        rng = np.random.default_rng(20240817)
        Z = rng.standard_normal((200_000, s))
        Xr = (X_LO + (X_HI - X_LO) * norm.cdf(Z) if MARGINAL == 'uniform' else Z)
        raw = _TARGETS[name](Xr)
        _SCALE_CACHE[key] = (float(raw.mean()), float(raw.std()),
                             float(_f_additive(Xr).std()))
    return _SCALE_CACHE[key]


def f_0(X, s_rel=None, target=None):
    s = S_SPARSE if s_rel is None else s_rel
    name = TARGET_FUNCTION if target is None else target
    assert name in _TARGETS, (f"unknown TARGET_FUNCTION {name!r}; "
                              f"expected one of {sorted(_TARGETS)}")
    out = _TARGETS[name](X[:, :s])
    if TARGET_STANDARDISE and name != 'additive':
        mu, sd, ref_sd = _target_scale(name, s)
        if sd > 1e-12:
            out = (out - mu) / sd * ref_sd
    return out


def simulate_nonparametric(n=None, d=None, s_rel=None, dependence=None,
                           rho_x=None, noise_sd=None, seed=None, verbose=True):
    n = N_OBS if n is None else n
    d = DIM if d is None else d
    s_rel = S_SPARSE if s_rel is None else s_rel
    dep = DGP_DEPENDENCE if dependence is None else dependence
    rho_x = RHO_X if rho_x is None else rho_x
    noise_sd = NOISE_SD if noise_sd is None else noise_sd
    rng = np.random.default_rng(SEED if seed is None else seed)

    assert s_rel <= d, "S_SPARSE cannot exceed DIM"
    assert MARGINAL in ('uniform', 'normal'), \
        f"unknown MARGINAL {MARGINAL!r}; use 'uniform' or 'normal'"
    if dep == 'iid':
        rho_x = 0.0
    elif dep != 'var1':
        raise ValueError(f"unknown DGP_DEPENDENCE {dep!r}; use 'var1' or 'iid'")

    Z = _stationary_ar1(n, d, rho_x, 1.0, rng)
    if MARGINAL == 'uniform':
        X = X_LO + (X_HI - X_LO) * norm.cdf(Z)
    else:
        X = Z

    f0 = f_0(X, s_rel)
    eps = _stationary_ar1(n, 1, NOISE_RHO, noise_sd, rng).ravel()
    y = f0 + eps

    if verbose:
        print(f"  [DGP] target={TARGET_FUNCTION}"
              + ("  (rescaled to the additive target's sd)"
                 if TARGET_STANDARDISE and TARGET_FUNCTION != 'additive' else ""))
        print(f"  [DGP] n={n:,}  d={d}  S={s_rel}  dependence={dep}  "
              f"rho_X={rho_x}  marginal={MARGINAL}  noise_sd={noise_sd}"
              + (f"  noise_rho={NOISE_RHO}" if NOISE_RHO else ""))
        print(f"  [DGP] sd(f_0)={f0.std():.4f}  sd(y)={y.std():.4f}  "
              f"noise floor R^2 = {1 - eps.var() / y.var():.4f}")
        print(f"  [DGP] lag-1 autocorrelation of X_1: "
              f"{_ar1_autocorr(X[:, 0]):+.4f}   of f_0: {_ar1_autocorr(f0):+.4f}")
    return {'X': X, 'y': y, 'f0': f0, 'eps': eps, 'rho_x': rho_x}


def split_data(X, y, f0, verbose=True):
    n = len(y)
    mode = SPLIT_MODE
    if DGP_DEPENDENCE == 'var1' and mode != 'chronological':
        raise ValueError(
            "SPLIT_MODE='random' is invalid under DGP_DEPENDENCE='var1': a "
            "random holdout is surrounded by its own training neighbours and "
            "the reported test error is optimistic. Use 'chronological', or "
            "set DGP_DEPENDENCE='iid'.")

    if mode == 'chronological':
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)
        idx = np.arange(n)
        i_tr, i_va, i_te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    elif mode == 'random':
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n)
        n_tr = int(n * TRAIN_RATIO)
        n_va = int(n * VAL_RATIO)
        i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    else:
        raise ValueError(f"unknown SPLIT_MODE {mode!r}")

    if verbose:
        print(f"  [split] {mode}: train={len(i_tr):,}  val={len(i_va):,}  "
              f"test={len(i_te):,}")
    return {'train': i_tr, 'val': i_va, 'test': i_te}


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_learner(name, data, idx, hp=None, verbose=True):
    X, y, f0 = data['X'], data['y'], data['f0']
    i_tr, i_va, i_te = idx['train'], idx['val'], idx['test']
    t0 = time.time()

    if name == 'oracle':
        predict_fn, info = (lambda Z: f_0(Z)), {'val_mse': np.nan,
                                                'fit_seconds': 0.0,
                                                'selection_time': 0.0}
    else:
        predict_fn, info = fit_learner(
            name, X[i_tr], y[i_tr], X[i_va], y[i_va], hp=hp,
            n_samples=len(i_tr), label=f'{name}: ', verbose=verbose)
        if predict_fn is None:
            return None

    pred_tr = np.asarray(predict_fn(X[i_tr]), dtype=np.float64)
    pred_va = np.asarray(predict_fn(X[i_va]), dtype=np.float64)
    pred_te = np.asarray(predict_fn(X[i_te]), dtype=np.float64)
    free_learner(info)

    n_tr = len(i_tr)
    l2_err = float(np.sqrt(np.mean((pred_te - f0[i_te]) ** 2)))
    out = {
        'method': name,
        'train_mse': float(mean_squared_error(y[i_tr], pred_tr)),
        'train_r2': float(r2_score(y[i_tr], pred_tr)),
        'val_mse': float(mean_squared_error(y[i_va], pred_va)),
        'val_r2': float(r2_score(y[i_va], pred_va)),
        'test_mse': float(mean_squared_error(y[i_te], pred_te)),
        'test_r2': float(r2_score(y[i_te], pred_te)),
        'train_rmse': float(np.sqrt(mean_squared_error(y[i_tr], pred_tr))),
        'test_rmse': float(np.sqrt(mean_squared_error(y[i_te], pred_te))),
        # ||f_hat - f_0||_{L2(mu)} on held-out data, and the rate diagnostic
        'l2_error': l2_err,
        'l2_error_x_n14': l2_err * n_tr ** 0.25,
        'r2_vs_f0': float(r2_score(f0[i_te], pred_te)),
        'runtime': time.time() - t0,
    }
    out.update({k: v for k, v in info.items() if not k.startswith('_')})
    return out


def print_configuration():
    print("\n" + "=" * 78)
    print("KASN — NONPARAMETRIC REGRESSION SIMULATION")
    print("=" * 78)
    print(f"  Spec label:         {SPEC_LABEL}   seed={SEED}  run_id={RUN_ID}")
    print(f"  Device:             {DEVICE}")
    print(f"  DGP:                n={N_OBS:,}  d={DIM}  S={S_SPARSE}  "
          f"f_0(X) = sum_j sin(X_j)")
    print(f"                      dependence={DGP_DEPENDENCE}"
          + (f"  rho_X={RHO_X}" if DGP_DEPENDENCE == 'var1' else "")
          + f"  marginal={MARGINAL}  noise_sd={NOISE_SD}")
    print(f"  Split:              {SPLIT_MODE}  "
          f"{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Learners:           {enabled_learners()}")
    print(f"  Delta_n mode:       {DELTA_MODE}  enforcement={DELTA_ENFORCEMENT}"
          f"  lambda_n via {LAMBDA_SELECTION}")
    print(f"  KASN gamma:         {_grid_or_fixed(KASN_GAMMA, KASN_GAMMA_GRID, 'KASN_GAMMA_GRID')}")
    print(f"  KASN lr:            {_grid_or_fixed(KASN_LR, KASN_LR_GRID, 'KASN_LR_GRID')}")
    print(f"  KASN batch_size:    {_grid_or_fixed(KASN_BATCH_SIZE, KASN_BATCH_SIZE_GRID, 'KASN_BATCH_SIZE_GRID')}")
    print(f"  KASN group lasso:   {_grid_or_fixed(KASN_GROUP_LASSO_REG_SCALE, SHARED_PENALTY_GRID, 'SHARED_PENALTY_GRID')}")
    print(f"  KASN width:         {_grid_or_fixed(KASN_WIDTH, C_W_GRID, 'C_W_GRID')}  (C_W={C_W})")
    print(f"  HP selection:       {HP_SELECTION_MODE}  "
          f"({'prune-aware' if PRUNE_AWARE_SELECTION else 'unpruned'} scoring)")
    print(f"  Pruning:            {PRUNING_THRESHOLD_METHOD}  "
          f"post-training={POST_TRAINING_PRUNING}  "
          f"finetune_epochs={POST_PRUNE_FINETUNE_EPOCHS}")
    print(f"  Epochs:             KASN={KASN_N_EPOCHS} (patience "
          f"{KASN_PATIENCE}, tuning {KASN_TUNING_EPOCHS})")
    _check_theory_conditions(int(N_OBS * TRAIN_RATIO), DIM)
    print("=" * 78 + "\n")


def config_columns():
    return {
        'spec_label': SPEC_LABEL,
        'seed': SEED,
        'run_id': RUN_ID,
        'n_obs': N_OBS, 'dim': DIM, 's_sparse': S_SPARSE,
        'target_function': TARGET_FUNCTION,
        'target_standardise': TARGET_STANDARDISE,
        'dgp_dependence': DGP_DEPENDENCE, 'rho_x': RHO_X,
        'marginal': MARGINAL, 'noise_sd': NOISE_SD, 'noise_rho': NOISE_RHO,
        'split_mode': SPLIT_MODE, 'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO, 'test_ratio': TEST_RATIO,
        'kasn_gamma': KASN_GAMMA, 'zeta_delta': KASN_ZETA_DELTA,
        'kasn_depth': KASN_DEPTH, 'kasn_width_cfg': KASN_WIDTH, 'c_w': C_W,
        'spline_order': KASN_SPLINE_ORDER, 'm_smooth_edge': M_SMOOTH_EDGE,
        'delta_mode': DELTA_MODE, 'delta_enforcement': DELTA_ENFORCEMENT,
        'delta_scale': DELTA_SCALE, 'lambda_selection': LAMBDA_SELECTION,
        'hp_selection_mode': HP_SELECTION_MODE,
        'prune_during_training': PRUNE_DURING_TRAINING,
        'post_training_pruning': POST_TRAINING_PRUNING,
        'pruning_threshold_method': PRUNING_THRESHOLD_METHOD,
        'post_prune_finetune_epochs': POST_PRUNE_FINETUNE_EPOCHS,
        'kasn_n_epochs': KASN_N_EPOCHS, 'kasn_patience': KASN_PATIENCE,
        'slfn_c': C_SLFN,
        'simulation_id': RUN_ID,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    t_start = time.time()
    print_configuration()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = simulate_nonparametric(verbose=True)
    idx = split_data(data['X'], data['y'], data['f0'], verbose=True)

    learners = enabled_learners()
    hp = None
    if 'kasn' in learners:
        if HP_SELECTION_MODE == 'fixed':
            hp = default_kasn_hp(len(idx['train']))
            print(f"\n  KASN hyperparameters pinned: "
                  f"{ {k: v for k, v in hp.items() if k != 'log'} }")
        else:
            hp = select_kasn_hyperparams(
                data['X'][idx['train']], data['y'][idx['train']],
                data['X'][idx['val']], data['y'][idx['val']],
                n_samples=len(idx['train']), label='', verbose=True)

    rows, cfg = [], config_columns()
    for name in learners:
        print(f"\n{'=' * 78}\n  {name.upper()}\n{'=' * 78}")
        res = evaluate_learner(name, data, idx,
                               hp=(hp if name == 'kasn' else None), verbose=True)
        if res is None:
            print(f"  [{name}] unavailable in this environment — skipped")
            continue
        row = dict(cfg)
        row.update(res)
        rows.append(row)
        print(f"  {name.upper()}: test RMSE={res['test_rmse']:.4f}  "
              f"test R2={res['test_r2']:.4f}  "
              f"||f_hat - f_0||={res['l2_error']:.4f}  "
              f"runtime={res['runtime']:.1f}s"
              + (f"  sparsity={res['sparsity_ratio']:.4f}"
                 if res.get('sparsity_ratio') is not None
                 and np.isfinite(res.get('sparsity_ratio', np.nan)) else ""))

    if not rows:
        print("\n  No learner produced a result; nothing written.")
        return None

    df = pd.DataFrame(rows)
    path = os.path.join(
        OUTPUT_DIR, f"nonparametric_regression_results_{SEED}_{RUN_ID}.csv")
    df.to_csv(path, index=False)

    print(f"\n{'=' * 78}\n  REPLICATION SUMMARY  (seed {SEED})\n{'=' * 78}")
    cols = ['method', 'train_rmse', 'train_r2', 'test_rmse', 'test_r2',
            'l2_error', 'l2_error_x_n14', 'sparsity_ratio', 'runtime']
    print(df[[c for c in cols if c in df.columns]].to_string(
        index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\n  Written to {path}")
    print(f"  Total replication time: {time.time() - t_start:.1f}s")
    return df


if __name__ == '__main__':
    main()
