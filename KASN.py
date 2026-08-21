class EmpiricalCDFTransformer:
    def __init__(self):
        self.sorted_values_ = None
        self.n_train_       = None
 
    def fit(self, X):
        if X.ndim == 1: X = X.reshape(-1, 1)
        self.sorted_values_ = np.sort(X, axis=0)
        self.n_train_       = X.shape[0]
        return self
 
    def transform(self, X):
        if X.ndim == 1: X = X.reshape(-1, 1)
        n   = self.n_train_
        cdf = np.zeros_like(X, dtype=np.float64)
        for i in range(X.shape[1]):
            cdf[:, i] = np.searchsorted(
                self.sorted_values_[:, i], X[:, i], side='right') / (n + 1.0)
        return np.clip(cdf, 0.0, 1.0)
 
    def fit_transform(self, X):
        self.fit(X); return self.transform(X)
 
 
def _oslo_projection_matrix(G_old, G_new, k=KASN_SPLINE_ORDER,
                              grid_range=KASN_GRID_RANGE,
                              n_eval=OSLO_N_EVAL):
    if G_old == G_new:
        return torch.eye(G_old + k)
 
    x_eval = torch.linspace(
        grid_range[0] + 1e-4, grid_range[1] - 1e-4, n_eval
    ).unsqueeze(-1)
 
    with torch.no_grad():
        old_basis_m = BSplineBasis(1, G_old, k, grid_range)
        new_basis_m = BSplineBasis(1, G_new, k, grid_range)
        old_B = old_basis_m.b_splines(x_eval).squeeze(1)
        new_B = new_basis_m.b_splines(x_eval).squeeze(1)
 
    M = torch.linalg.lstsq(new_B.float(), old_B.float()).solution
    return M.float()
 
 
class BSplineBasis(nn.Module):
    def __init__(self, in_features, grid_size=5,
                 spline_order=KASN_SPLINE_ORDER, grid_range=KASN_GRID_RANGE):
        super().__init__()
        self.in_features  = in_features
        self.grid_size    = grid_size
        self.spline_order = spline_order
        self.register_buffer("grid", self._create_grid(grid_range, grid_size))
 
    def _create_grid(self, grid_range, grid_size):
        h = (grid_range[1] - grid_range[0]) / grid_size
        g = torch.arange(-self.spline_order,
                         grid_size + self.spline_order + 1) * h + grid_range[0]
        return g.expand(self.in_features, -1).contiguous()
 
    def b_splines(self, x):
        grid  = self.grid
        x     = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left  = (x - grid[:, :-(k+1)]) / (
                grid[:, k:-1] - grid[:, :-(k+1)]).clamp_min(1e-8)
            right = (grid[:, k+1:] - x) / (
                grid[:, k+1:] - grid[:, 1:(-k)]).clamp_min(1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()
 
    def forward(self, x):
        s = x.shape
        return self.b_splines(x.reshape(-1, self.in_features)).reshape(*s[:-1], -1)
 
 
class KASNLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=KASN_SPLINE_ORDER,
                 base_activation=KASN_BASE_ACTIVATION,
                 grid_range=KASN_GRID_RANGE,
                 use_residual=True, delta_n=None):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.use_residual = use_residual and (in_features == out_features)
        self.basis     = BSplineBasis(in_features, grid_size, spline_order, grid_range)
        self.num_basis = grid_size + spline_order
        self.base_weight   = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.num_basis))
        self.w_b             = nn.Parameter(torch.ones(1))
        self.w_s             = nn.Parameter(torch.ones(1))
        self.w_b.requires_grad_(False)
        self.w_s.requires_grad_(False)
        self.base_activation = base_activation()
        self.grid_size       = grid_size
        self.reset_parameters()
 
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.uniform_(self.spline_weight, -0.5/self.grid_size, 0.5/self.grid_size)
        with torch.no_grad():
            self.w_b.fill_(0.0); self.w_s.fill_(1.0)
 
    def set_transition(self, s):
        with torch.no_grad(): self.w_b.fill_(1-s); self.w_s.fill_(s)
 
    def forward(self, x):
        base_out   = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(self.basis(x).view(x.size(0), -1),
                              self.spline_weight.view(self.out_features, -1))
        out = self.w_b * base_out + self.w_s * spline_out
        return out + x if self.use_residual else out
 
    def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
        return s * torch.sum(torch.abs(self.spline_weight))
 
    def group_lasso_regularization_loss(self, gl):
        return gl * torch.norm(self.spline_weight, p=2, dim=2).sum()
 
    def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
        with torch.no_grad():
            return int(torch.sum(
                torch.norm(self.spline_weight, p=2, dim=2) > thr).item())
 
    def count_total_edges(self):
        return self.out_features * self.in_features
 
    def count_active_activations(self, thr=1e-6):
        with torch.no_grad():
            return int(torch.sum(torch.abs(self.spline_weight) > thr).item())
 
 
class KASN(nn.Module):
    def __init__(self, input_dim, n_samples, gamma=None, kasn_width=None,
                 depth=None, zeta_delta=KASN_ZETA_DELTA,
                 prune_during_training=PRUNE_DURING_TRAINING, kart_s=None,
                 dual_reg=KASN_DUAL_REG):
        super().__init__()
        gamma = gamma if gamma is not None else (KASN_GAMMA if KASN_GAMMA is not None
                                                  else KASN_GAMMA_GRID[0])
        self.input_dim             = input_dim
        self.n_samples             = n_samples
        self.gamma                 = gamma
        self.zeta_delta            = zeta_delta
        self.prune_during_training = prune_during_training
        self.kart_s                = kart_s if kart_s is not None else input_dim
        self.dual_reg              = dual_reg
 
        self.G = max(5, int(n_samples ** gamma))
        self.L = depth if depth is not None else max(3, int(np.log(n_samples)))
        self.W = kasn_width if kasn_width is not None else \
                 (2 * input_dim + 1)          # KART width
 
        self.delta_n = float('inf') if dual_reg else \
            max(5, np.log(n_samples),
                n_samples ** zeta_delta if zeta_delta is not None else 1)
 
        if ADAPTIVE_GRID:
            G0 = max(5, int(KASN_GRID_PHASE_SCALES[0] * self.G))
            G1 = max(5, int(KASN_GRID_PHASE_SCALES[1] * self.G))
            _phase_candidates = [G0, G1, self.G]
            _phases = []
            for g in _phase_candidates:
                if not _phases or g > _phases[-1]:
                    _phases.append(g)
            self.G_phases = _phases
        else:
            self.G_phases = [self.G]
        G_init = self.G_phases[0]
 
        _phases_str = (f"  adaptive phases: {self.G_phases}  "
                       f"transitions at {[int(f*100) for f in KASN_GRID_PHASE_FRACS]}% of epochs"
                       if ADAPTIVE_GRID and len(self.G_phases) > 1
                       else "")
        print(f"KASN: L={self.L}, W={self.W}, G_n={self.G} (init G={G_init}), "
              f"gamma={gamma}, "
              f"Δ_n={'∞ (dual)' if dual_reg else f'{self.delta_n:.2f}'}, "
              f"d={input_dim}")
        if _phases_str:
            print(f"  {_phases_str}")
 
        self.layers = nn.ModuleList()
        self.layers.append(KASNLayer(input_dim, self.W, grid_size=G_init,
                                     use_residual=False))
        for _ in range(self.L - 2):
            self.layers.append(KASNLayer(self.W, self.W, grid_size=G_init,
                                         use_residual=True))
        self.layers.append(KASNLayer(self.W, 1, grid_size=G_init,
                                     use_residual=False))
        self.scaler_X = EmpiricalCDFTransformer()
        self.scaler_y = StandardScaler()
 
    def _extend_to_G(self, G_new):
        extended = False
        for layer in self.layers:
            G_old = layer.grid_size
            if G_old == G_new:
                continue
            k   = layer.basis.spline_order
            dev = layer.spline_weight.device
 
            M = _oslo_projection_matrix(
                G_old, G_new, k, KASN_GRID_RANGE, OSLO_N_EVAL
            ).to(dev)
 
            with torch.no_grad():
                old_w = layer.spline_weight.data
                new_w = old_w @ M.t()
 
            layer.basis     = BSplineBasis(
                layer.in_features, G_new, k, KASN_GRID_RANGE).to(dev)
            layer.num_basis = G_new + k
            layer.grid_size = G_new
            layer.spline_weight = nn.Parameter(new_w)
            extended = True
 
        return extended
 
    def _rebuild_at_G(self, G_target):
        for layer in self.layers:
            if layer.grid_size == G_target:
                continue
            k   = layer.basis.spline_order
            dev = layer.spline_weight.device
            layer.basis = BSplineBasis(layer.in_features, G_target, k,
                                       KASN_GRID_RANGE).to(dev)
            layer.num_basis     = G_target + k
            layer.spline_weight = nn.Parameter(torch.zeros(
                layer.out_features, layer.in_features, G_target + k,
                device=dev))
            layer.grid_size = G_target
 
    def compute_current_total_l1(self):
        return sum(torch.norm(l.spline_weight, p=2, dim=2).sum().item()
                   for l in self.layers)
 
    def compute_delta_penalty(self, lam=1.0):
        if self.dual_reg:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        total = sum(torch.sum(torch.abs(l.spline_weight)) for l in self.layers)
        return lam * torch.clamp(total - self.delta_n, min=0.0) ** 2
 
    def l1_regularization_loss(self, s=KASN_L1_REG_SCALE):
        return sum(l.l1_regularization_loss(s) for l in self.layers)
 
    def group_lasso_regularization_loss(self, gl):
        return sum(l.group_lasso_regularization_loss(gl) for l in self.layers)
 
    def compute_lambda_reg(self):
        return np.sqrt(np.log(self.L * self.W**2 + 1e-8) / self.n_samples)
 
    def final_delta_projection(self):
        total = sum(torch.sum(torch.abs(l.spline_weight)).item()
                    for l in self.layers)
        if self.dual_reg:
            print(f"  Dual reg: implied Δₙ = {total:.6f}")
            self.delta_n = total; return
        if total > self.delta_n:
            scale = self.delta_n / (total + 1e-12)
            with torch.no_grad():
                for l in self.layers: l.spline_weight.data *= scale
            print(f"  Projected ℓ₁={total:.4f} → {self.delta_n:.4f}")
        else:
            print(f"  ℓ₁={total:.4f} ≤ Δₙ={self.delta_n:.4f} (no projection)")
 
    def set_transition(self, s):
        for l in self.layers: l.set_transition(s)
 
    def _eff_delta(self):
        return self.compute_current_total_l1() \
            if (self.dual_reg and not np.isfinite(self.delta_n)) else self.delta_n
 
    def prune_edges(self, method='delta_normalized', val=None):
        if method == 'delta_normalized':
            tot = self.count_total_edges()
            thr = self._eff_delta() / tot if tot > 0 else 0.0
        elif method == 'fixed':
            thr = val or KASN_ACTIVE_EDGE_THRESHOLD
        elif method == 'relative_fraction':
            frac = val or PRUNING_RELATIVE_FRACTION
            with torch.no_grad():
                mx = max(torch.norm(l.spline_weight, p=2, dim=2).max().item()
                         for l in self.layers)
            thr = frac * mx
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        pruned = total = 0
        with torch.no_grad():
            for layer in self.layers:
                en   = torch.norm(layer.spline_weight, p=2, dim=2)
                mask = en <= thr
                layer.spline_weight.data[mask] = 0.0
                pruned += mask.sum().item(); total += en.numel()
        return pruned, total
 
    def apply_post_training_pruning(self, threshold_method='delta_normalized',
                                    threshold_value=None):
        p, t = self.prune_edges(threshold_method, threshold_value)
        print(f"  Post-training pruning ({threshold_method}): "
              f"{p}/{t} edges zeroed ({p/t*100:.1f}%)")
 
    def set_prune_mask(self):
        n_masked = n_total = 0
        with torch.no_grad():
            for layer in self.layers:
                m = (torch.norm(layer.spline_weight, p=2, dim=2) == 0)
                layer._prune_mask = m
                n_masked += int(m.sum().item()); n_total += m.numel()
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
                    m = (torch.norm(layer.spline_weight, p=2, dim=2) == 0)
                act += int((~m).sum().item()); tot += m.numel()
        return act, tot
 
    def count_active_edges(self, thr=KASN_ACTIVE_EDGE_THRESHOLD):
        return sum(l.count_active_edges(thr) for l in self.layers)
 
    def count_total_edges(self):
        return sum(l.count_total_edges() for l in self.layers)
 
    def count_active_edges_delta_normalized(self):
        tot = self.count_total_edges()
        thr = self._eff_delta() / tot if tot > 0 else 0.0
        return self.count_active_edges(thr), thr, tot
 
    def count_active_edges_relative(self, rel=PRUNING_RELATIVE_FRACTION):
        with torch.no_grad():
            mx = max(torch.norm(l.spline_weight, p=2, dim=2).max().item()
                     for l in self.layers)
        thr = rel * mx
        return self.count_active_edges(thr), thr, mx
 
    def count_active_edges_delta_over_r(self):
        r_n = sum(l.out_features * l.in_features * l.num_basis for l in self.layers)
        thr = self._eff_delta() / r_n if r_n > 0 else 0.0
        return self.count_active_edges(thr), thr, r_n
 
    def get_total_potential_activations(self):
        return sum(l.out_features * l.in_features * l.num_basis for l in self.layers)
 
    def analyze_representation_sparsity_enhanced(self, rel=PRUNING_RELATIVE_FRACTION):
        tot                    = self.count_total_edges()
        ae_fix                 = self.count_active_edges()
        ae_rel, tr_rel, mx     = self.count_active_edges_relative(rel)
        ae_dn,  tr_dn,  _      = self.count_active_edges_delta_normalized()
        ae_dr,  tr_dr, r_n     = self.count_active_edges_delta_over_r()
        kart                   = 2 * self.kart_s + 1
        return {
            'fixed_sparsity':            1 - ae_fix / tot,
            'relative_sparsity':         1 - ae_rel / tot,
            'delta_sparsity':            1 - ae_dn  / tot,
            'delta_over_r_sparsity':     1 - ae_dr  / tot,
            'kart_theoretical_sparsity': 1 - kart   / tot,
            'active_edges_fixed':        ae_fix,
            'active_edges_relative':     ae_rel,
            'active_edges_delta':        ae_dn,
            'active_edges_delta_over_r': ae_dr,
            'threshold_fixed':           KASN_ACTIVE_EDGE_THRESHOLD,
            'threshold_relative':        tr_rel,
            'threshold_delta':           tr_dn,
            'threshold_delta_over_r':    tr_dr,
            'max_edge_norm':             mx,
            'total_potential_edges':     tot,
            'r_n':                       r_n,
            'delta_n':                   self.delta_n,
        }
 
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x
 
    def fit(self, X_train_np, y_train_np, X_val_np, y_val_np,
            epochs=1000, lr=None, batch_size=None,
            patience=KASN_PATIENCE,
            weight_decay=KASN_WEIGHT_DECAY, l1_reg_scale=KASN_L1_REG_SCALE,
            group_lasso_reg_scale=None,
            sample_weights=None, resume_scalers=False):
        if lr is None:
            lr = KASN_LR if KASN_LR is not None else KASN_LR_GRID[0]
        if batch_size is None:
            batch_size = KASN_BATCH_SIZE if KASN_BATCH_SIZE is not None \
                         else KASN_BATCH_SIZE_GRID[0]
        if y_train_np.ndim == 1: y_train_np = y_train_np.reshape(-1, 1)
        if y_val_np.ndim   == 1: y_val_np   = y_val_np.reshape(-1, 1)
 
        if not resume_scalers:
            self.scaler_X.fit(X_train_np)
            self.scaler_y.fit(y_train_np)
 
        X_train_t = torch.tensor(self.scaler_X.transform(X_train_np),
                                 dtype=torch.float32)
        y_train_t = torch.tensor(self.scaler_y.transform(y_train_np),
                                 dtype=torch.float32)
        X_val_t   = torch.tensor(self.scaler_X.transform(X_val_np),
                                 dtype=torch.float32)
        y_val_t   = torch.tensor(self.scaler_y.transform(y_val_np),
                                 dtype=torch.float32)
 
        pin       = (DEVICE.type == 'cuda')
        bs_actual = _effective_batch_size(len(X_train_t), batch_size)
 
        if sample_weights is not None:
            w_t = torch.tensor(sample_weights, dtype=torch.float32)
            train_dataset = torch.utils.data.TensorDataset(
                X_train_t, y_train_t, w_t)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader  = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs_actual,
            shuffle=_should_shuffle(), drop_last=True,
            pin_memory=pin, num_workers=4 if pin else 0,
            persistent_workers=pin, prefetch_factor=2 if pin else None,
        )
 
        self.to(DEVICE)
        X_val_gpu = X_val_t.to(DEVICE, non_blocking=True)
        y_val_gpu = y_val_t.to(DEVICE, non_blocking=True)
 
        optimizer  = torch.optim.AdamW(self.parameters(), lr=lr,
                                       weight_decay=weight_decay)
        scheduler  = torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=500, gamma=0.5)
        criterion  = nn.MSELoss()
        amp_enabled = USE_AMP and (DEVICE.type == 'cuda')
        scaler_amp  = torch.amp.GradScaler('cuda', enabled=amp_enabled)
 
        if group_lasso_reg_scale is None:
            group_lasso_reg_scale = KASN_GROUP_LASSO_REG_SCALE or SHARED_PENALTY_GRID[0]
 
        lambda_reg = self.compute_lambda_reg()
 
        best_val_loss, best_state, best_epoch = float('inf'), None, 0
        # The per-phase best above drives patience; the global best below
        # survives grid extensions so the run is not forced to keep whichever
        # phase happened to be last, even when an earlier phase was better.
        gbest = {'val': float('inf'), 'state': None, 'epoch': 0,
                 'G': self.layers[0].grid_size}
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        epoch_log  = []
        _weighted  = sample_weights is not None
 
        if ADAPTIVE_GRID and len(self.G_phases) > 1:
            _n_trans = len(self.G_phases) - 1
            _grid_milestones = {
                int(KASN_GRID_PHASE_FRACS[i] * epochs): self.G_phases[i + 1]
                for i in range(min(_n_trans, len(KASN_GRID_PHASE_FRACS)))
                if self.G_phases[i + 1] > self.G_phases[i]
            }
        else:
            _grid_milestones = {}
 
        for epoch in range(epochs):
            if epoch in _grid_milestones:
                G_old    = self.layers[0].grid_size
                G_target = _grid_milestones[epoch]
                t_ext    = time.time()
                print(f"\n  [AdaptiveGrid] ep {epoch}: G {G_old} → {G_target}  "
                      f"(Oslo projection, n_eval={OSLO_N_EVAL})...")
                self._extend_to_G(G_target)
                print(f"  [AdaptiveGrid] Extension complete in "
                      f"{time.time()-t_ext:.1f}s  "
                      f"new basis_fns/edge={G_target + self.layers[0].basis.spline_order}")
                _phase_idx = sum(1 for e in _grid_milestones if e <= epoch)
                lr_phase   = lr * (0.5 ** _phase_idx)
                print(f"  [AdaptiveGrid] lr → {lr_phase:.2e}  "
                      f"(phase {_phase_idx + 1}/{len(self.G_phases)}, "
                      f"0.5^{_phase_idx} × {lr:.2e})")
                optimizer  = torch.optim.AdamW(self.parameters(), lr=lr_phase,
                                               weight_decay=weight_decay)
                scheduler  = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=500, gamma=0.5)
                scaler_amp = torch.amp.GradScaler('cuda', enabled=amp_enabled)
                epochs_no_improve = 0
                best_val_loss     = float('inf')
                best_state        = None
 
            self.train(); self.set_transition(1.0)
            epoch_mse = 0.0
 
            for batch in train_loader:
                if _weighted:
                    Xb, yb, wb = [t.to(DEVICE, non_blocking=True) for t in batch]
                    wb = wb.unsqueeze(1)
                else:
                    Xb, yb = [t.to(DEVICE, non_blocking=True) for t in batch]
                    wb = None
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    pred     = self(Xb)
                    if wb is not None:
                        mse_loss = (F.mse_loss(pred, yb, reduction='none') * wb
                                    ).mean()
                    else:
                        mse_loss = criterion(pred, yb)
                    gl_loss  = self.group_lasso_regularization_loss(
                        group_lasso_reg_scale)
                    l1_loss  = self.l1_regularization_loss(l1_reg_scale)
                    total    = mse_loss + lambda_reg * (gl_loss + l1_loss) + \
                               self.compute_delta_penalty()
                scaler_amp.scale(total).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 500.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
              
                self.apply_prune_mask()
                epoch_mse += mse_loss.item() * Xb.size(0)
 
            scheduler.step()
            avg_mse = epoch_mse / len(X_train_t)
            train_losses.append(avg_mse)
 
            self.eval()
            with torch.no_grad():
                parts = []
                for i in range(0, X_val_gpu.shape[0], INFERENCE_CHUNK_SIZE):
                    chunk = X_val_gpu[i:i+INFERENCE_CHUNK_SIZE]
                    with torch.amp.autocast(device_type=DEVICE.type,
                                            enabled=amp_enabled):
                        parts.append(self(chunk))
                val_loss = criterion(torch.cat(parts), y_val_gpu).item()
            val_losses.append(val_loss)
 
            if val_loss < best_val_loss:
                best_val_loss = val_loss; best_epoch = epoch
                best_state    = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
 
            if val_loss < gbest['val']:
                gbest = {'val': val_loss, 'epoch': epoch,
                         'G': self.layers[0].grid_size,
                         'state': copy.deepcopy(self.state_dict())}
 
            if epoch % 10 == 0 or epoch == epochs - 1:
                ae, _, te = self.count_active_edges_delta_normalized()
                ds = 1 - ae / te if te > 0 else 0.0
                if self.prune_during_training:
                    self.prune_edges('delta_normalized')
                print(f'Epoch {epoch:4d}/{epochs}: '
                      f'Train={avg_mse:.6f}, Val={val_loss:.6f}, '
                      f'BestVal={best_val_loss:.6f} (ep {best_epoch}), '
                      f'ActiveEdges={ae}/{te}, Δ-Sparsity={ds:.4f}, '
                      f'LR={scheduler.get_last_lr()[0]:.6f}'
                      + _eval_monitor_str(self))
                epoch_log.append({
                    'epoch':            epoch,
                    'train_loss':       avg_mse,
                    'train_is_weighted': _weighted,
                    'val_mse':          val_loss,
                    'best_val_mse':     best_val_loss,
                    'best_epoch':       best_epoch,
                    'active_edges':     ae,
                    'total_edges':      te,
                    'delta_sparsity':   round(ds, 6),
                    'lr':               scheduler.get_last_lr()[0],
                    'epochs_no_improve': epochs_no_improve,
                    'early_stopped':    False,
                    'grid_size':        self.layers[0].grid_size,
                })
 
            _final_phase_reached = (not _grid_milestones or
                                     epoch >= max(_grid_milestones.keys()))
 
            if (patience is not None
                    and epochs_no_improve >= patience
                    and _final_phase_reached):
                print(f'  Early stopping ep {epoch} ...')
                ...
                break
 
        if GLOBAL_BEST_RESTORE and gbest['state'] is not None:
            cur_G = self.layers[0].grid_size
            if gbest['G'] != cur_G:
                print(f"  Global best came from grid phase G={gbest['G']} "
                      f"(final phase G={cur_G}); resizing to restore it.")
                self._rebuild_at_G(gbest['G'])
                self.G = gbest['G']
            self.load_state_dict(gbest['state'])
            self.to(DEVICE)
            if best_state is not None and best_val_loss > gbest['val']:
                print(f"  Note: final-phase best was {best_val_loss:.6f} "
                      f"(ep {best_epoch}); global best is "
                      f"{gbest['val']:.6f} (ep {gbest['epoch']}), an "
                      f"improvement of "
                      f"{100*(1 - gbest['val']/max(best_val_loss,1e-30)):.1f}%")
            print(f"  Restored GLOBAL best model from epoch {gbest['epoch']} "
                  f"(val loss: {gbest['val']:.6f} PRE-PRUNING, G={gbest['G']})")
            best_val_loss, best_epoch = gbest['val'], gbest['epoch']
        elif best_state is not None:
            self.load_state_dict(best_state)
            print(f'  Restored best model from epoch {best_epoch} '
                  f'(val loss: {best_val_loss:.6f} PRE-PRUNING)')
 
        return train_losses, val_losses, best_val_loss, epoch_log
 
    def predict(self, X_np):
        self.eval(); self.set_transition(1.0); self.to(DEVICE)
        amp_enabled = USE_AMP and (DEVICE.type == 'cuda')
        X_t = torch.tensor(self.scaler_X.transform(X_np), dtype=torch.float32)
        parts = []
        with torch.inference_mode():
            for i in range(0, X_t.shape[0], INFERENCE_CHUNK_SIZE):
                chunk = X_t[i:i+INFERENCE_CHUNK_SIZE].to(DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
                    parts.append(self(chunk).cpu())
        return self.scaler_y.inverse_transform(
            torch.cat(parts).numpy()).flatten()
