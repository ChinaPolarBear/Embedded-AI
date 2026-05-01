import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plot_output_utils import make_figure_output_dir, save_figure

# ============================================================
# 1) Global configuration and physical parameters
#    (NLSE coefficients and training hyperparameters)
# ============================================================

# Automatically select GPU when available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Fibre / NLSE parameters ---
alpha = 0.081      # loss coefficient
beta2 = -1.0       # group velocity dispersion (GVD) coefficient
gamma = 1.0        # Kerr nonlinearity coefficient

# --- Link / sampling settings ---
L = 1.0            # fibre length / propagation distance
T_max = 1.0        # time window is [-T_max, +T_max]
N_t = 256          # number of time samples

# --- QAM / RRC waveform-generation settings for the input A(0,t) ---
num_symbols = 64   # 64-QAM symbol count
sps = 8            # samples per symbol
rrc_beta = 0.2     # RRC roll-off
rrc_span = 10      # RRC span in symbols

# --- Dataset sizes ---
N_train = 1024
N_test  = 128

# --- Two-stage training schedule ---
epochs_pretrain = 1500   # Phase 1: physics-first (PDE + IC)
epochs_hybrid   = 3500   # Phase 2: physics + anchors at z=L

# --- Number of functions per DeepONet batch
#     (each function corresponds to one full A0(t) waveform) ---
batch_size_fns = 64

# --- PINN sampling counts
#     Larger values improve accuracy but also increase runtime. ---
N_ic   = 1024     # initial-condition samples at z = 0
N_data = 512      # endpoint anchor samples at z = L
N_pde  = 4096     # PDE residual samples inside the domain

# Learning rate used together with StepLR.
lr = 8e-4

# --- Noise level used during training and evaluation ---
snr_db_train = 25.0
snr_db_eval  = 25.0

# --- DeepONet architecture hyperparameters ---
P_dim         = 128  # latent basis size; the complex representation uses 2 * P_dim outputs
branch_hidden = 128
trunk_hidden  = 256
fourier_dim   = 64   # Fourier feature dimension for trunk input (t,z)

# Model save path.
MODEL_PATH = "hybrid_pinn_deeponet.pth"
SSFM_N_STEPS = 200
DATASET_CACHE_DIR = Path("dataset_cache")
TRAIN_DATASET_CACHE_KEY = "train"
TEST_DATASET_CACHE_KEY = "test"
QAT_DATASET_CACHE_KEY = "qat_supervised"


# ============================================================
# 2) Time axis and SSFM solver
#    Used to generate reference labels.
# ============================================================

# Time grid t in [-T_max, T_max].
t_grid = torch.linspace(-T_max, T_max, N_t, device=device)

# Sampling interval used for the frequency-domain grid.
dt = (t_grid[1] - t_grid[0]).item()

# Frequency grid f and angular frequency omega for the linear SSFM step.
freq  = torch.fft.fftfreq(N_t, d=dt).to(device)
omega = 2 * np.pi * freq


def ssfm_propagate(A0_complex, L_dist, n_steps=SSFM_N_STEPS):
    """
    Split-Step Fourier Method (SSFM) for the NLSE.

    Inputs:
      - A0_complex: shape [N_t] complex tensor, A(0,t)
      - L_dist: propagation distance
      - n_steps: number of z-steps

    Output:
      - A(L,t): shape [N_t] complex tensor
    """
    # Step size along the propagation axis z.
    dz = L_dist / n_steps

    # Clone the input to avoid in-place modification.
    A = A0_complex.clone()

    # Linear half-step operator: exp((j*beta2/2 * omega^2 - alpha/2) * dz/2)
    linear_half = torch.exp((0.5j * beta2 * omega**2 - 0.5 * alpha) * (dz / 2))

    # Nonlinear phase rotation coefficient: exp(j * gamma * |A|^2 * dz)
    nonlinear_coef = 1j * gamma * dz

    # SSFM sequence: linear half-step -> nonlinear full-step -> linear half-step
    for _ in range(n_steps):
        # Linear update in the frequency domain, then transform back to time.
        A = torch.fft.ifft(torch.fft.fft(A) * linear_half)
        # Nonlinear phase rotation driven by the instantaneous amplitude.
        A = A * torch.exp(nonlinear_coef * torch.abs(A)**2)
        # Apply the second linear half-step.
        A = torch.fft.ifft(torch.fft.fft(A) * linear_half)

    return A


# ============================================================
# 3) Generate QAM + RRC initial waveforms A(0,t)
#    and build datasets
# ============================================================

def _rrc_taps_torch(beta: float, sps: int, span: int, device):
    """
    Generate Root-Raised-Cosine (RRC) FIR taps with energy normalisation.

    Output:
      taps: shape [span * sps + 1], float tensor on the target device
    """
    if beta < 0 or beta > 1:
        raise ValueError("beta must be in [0, 1].")
    if sps <= 0 or span <= 0:
        raise ValueError("sps and span must be positive integers.")

    N = span * sps
    # Time is expressed in symbol periods, so divide by sps.
    t = torch.arange(-N / 2, N / 2 + 1, device=device, dtype=torch.float32) / float(sps)
    taps = torch.zeros_like(t)

    eps = 1e-12  # Avoid division by zero.
    for i in range(t.numel()):
        ti = t[i].item()

        # Special case: t = 0
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + (4 * beta / np.pi)

        # Special case: t = +/- 1 / (4 * beta)
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )

        # General closed-form expression.
        else:
            ti_t = t[i]
            numerator = torch.sin(np.pi * ti_t * (1 - beta)) + 4 * beta * ti_t * torch.cos(np.pi * ti_t * (1 + beta))
            denominator = (np.pi * ti_t * (1 - (4 * beta * ti_t) ** 2))
            taps[i] = numerator / (denominator + eps)

    # Energy normalisation: sum(taps^2) = 1.
    taps = taps / torch.sqrt(torch.sum(taps**2) + eps)
    return taps


def _qam64_symbols_torch(num_symbols: int, device):
    """
    Generate random 64-QAM complex symbols and normalise the average power to 1.

    Output:
      syms: shape [num_symbols], complex64 tensor
    """
    # 64-QAM uses 8 amplitude levels on each I/Q branch.
    levels = torch.tensor([-7., -5., -3., -1., 1., 3., 5., 7.], device=device)
    I_idx = torch.randint(0, 8, (num_symbols,), device=device)
    Q_idx = torch.randint(0, 8, (num_symbols,), device=device)

    I = levels[I_idx]
    Q = levels[Q_idx]
    syms = (I + 1j * Q).to(torch.complex64)

    # Normalise to unit average power: E{|syms|^2} = 1.
    syms = syms / torch.sqrt(torch.mean(torch.abs(syms) ** 2) + 1e-12)
    return syms


def generate_qam_waveform(
    sps: int = 8,
    beta: float = 0.2,
    span: int = 10,
    normalize_power: bool = True,
    center_crop: bool = True,
):
    """
    Generate a realistic 64-QAM + RRC-shaped waveform A0(t) = A(0,t).

    Returns:
      - A0: [N_t] complex tensor on device
      - symbols_np: numpy complex array for constellation plotting
      - centers_np: numpy array of symbol-centre time coordinates for plotting/debugging
    """
    # (1) Generate the 64-QAM symbol sequence.
    syms = _qam64_symbols_torch(num_symbols, device=device)  # [num_symbols]
    symbols_np = syms.detach().cpu().numpy()

    # (2) Upsample by inserting each symbol every sps samples.
    up = torch.zeros(num_symbols * sps, dtype=torch.complex64, device=device)
    up[::sps] = syms  # All other samples are zero, equivalent to an impulse train.

    # (3) Apply RRC shaping with FIR convolution, real and imaginary parts separately.
    h = _rrc_taps_torch(beta=beta, sps=sps, span=span, device=device)  # real taps
    x = up.view(1, 1, -1)   # conv1d expects [B=1, C=1, N]
    h1 = h.view(1, 1, -1)

    # Use same-style padding so the output length stays close to the input length.
    pad = (h.numel() - 1) // 2

    y_re = torch.nn.functional.conv1d(x.real, h1, padding=pad)
    y_im = torch.nn.functional.conv1d(x.imag, h1, padding=pad)

    shaped = (y_re + 1j * y_im).view(-1)  # [num_symbols * sps]

    # (4) Crop or pad to the fixed model length N_t.
    if shaped.numel() < N_t:
        pad_len = N_t - shaped.numel()
        left = pad_len // 2
        right = pad_len - left
        shaped = torch.nn.functional.pad(shaped, (left, right))
    elif shaped.numel() > N_t:
        start = (shaped.numel() - N_t) // 2 if center_crop else 0
        shaped = shaped[start:start + N_t]

    A0 = shaped[:N_t].clone()

    # (5) Apply a second average-power normalisation to stabilise training scale.
    if normalize_power:
        power = torch.sqrt(torch.mean(torch.abs(A0) ** 2) + 1e-12)
        A0 = A0 / (power + 1e-12)

    # (6) Return symbol centres for plotting only; they are not used during training.
    centers = torch.linspace(t_grid.min(), t_grid.max(), num_symbols, device=device)
    centers_np = centers.detach().cpu().numpy()

    return A0, symbols_np, centers_np


def add_awgn(signal: torch.Tensor, snr_db: float):
    """
    Add proper complex AWGN to a complex-valued signal.

    Inputs:
      - signal: complex tensor [...], for example [N_t]
      - snr_db: SNR in dB

    Output:
      - signal + noise
    """
    # Average signal power E{|x|^2}.
    power = torch.mean(torch.abs(signal) ** 2)

    # Convert SNR from dB to linear scale.
    snr_linear = 10 ** (snr_db / 10)

    # Noise power = signal power / SNR.
    noise_power = power / snr_linear

    # Proper complex noise: real and imaginary parts each take half the variance.
    noise = torch.sqrt(noise_power / 2) * (
        torch.randn_like(signal.real) +
        1j * torch.randn_like(signal.real)
    )

    return signal + noise


def _dataset_cache_metadata(n_samples, snr_db):
    return {
        "n_samples": int(n_samples),
        "snr_db": float(snr_db),
        "N_t": int(N_t),
        "L": float(L),
        "num_symbols": int(num_symbols),
        "sps": int(sps),
        "rrc_beta": float(rrc_beta),
        "rrc_span": int(rrc_span),
        "ssfm_n_steps": int(SSFM_N_STEPS),
    }


def get_dataset_cache_path(n_samples, snr_db, cache_key="dataset", cache_dir=DATASET_CACHE_DIR):
    safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(cache_key))
    snr_token = str(float(snr_db)).replace("-", "m").replace(".", "p")
    return Path(cache_dir) / f"{safe_key}_n{int(n_samples)}_snr{snr_token}.pt"


def load_dataset_cache(n_samples, snr_db, cache_key="dataset", cache_dir=DATASET_CACHE_DIR):
    cache_path = get_dataset_cache_path(n_samples, snr_db, cache_key=cache_key, cache_dir=cache_dir)
    if not cache_path.is_file():
        return None

    payload = torch.load(cache_path, map_location="cpu")
    expected_meta = _dataset_cache_metadata(n_samples, snr_db)
    cached_meta = payload.get("meta", {})

    for key, value in expected_meta.items():
        if cached_meta.get(key) != value:
            print(f"[WARN] Cache metadata mismatch for {cache_path}, ignoring stale cache.")
            return None

    print(f"[OK] Loaded dataset cache: {cache_path}")
    return (
        payload["A0"].to(device),
        payload["AL_clean"].to(device),
        payload["AL_noisy"].to(device),
    )


def build_dataset(n_samples, snr_db, cache_key="dataset", cache_dir=DATASET_CACHE_DIR, force_rebuild=False):
    """
    Build the training or test dataset:
      - A0: input A(0,t)
      - AL_clean: clean SSFM reference A(L,t)
      - AL_noisy: noisy observation A_obs(L,t) after adding AWGN

    Training strategy:
      - PDE / IC / clean-anchor terms mainly fit AL_clean to learn the physics and the clean target
      - the observation loss teaches the model about measurement noise without forcing it to fit noise texture
    """
    if not force_rebuild:
        cached = load_dataset_cache(n_samples, snr_db, cache_key=cache_key, cache_dir=cache_dir)
        if cached is not None:
            return cached

    cache_path = get_dataset_cache_path(n_samples, snr_db, cache_key=cache_key, cache_dir=cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE MISS] Building dataset from SSFM: {cache_path}")

    A0_all = torch.zeros(n_samples, N_t, dtype=torch.complex64, device=device)
    AL_clean_all = torch.zeros_like(A0_all)
    AL_noisy_all = torch.zeros_like(A0_all)

    for n in range(n_samples):
        # Generate the input waveform A(0,t).
        A0, _, _ = generate_qam_waveform()

        # Use SSFM to obtain the clean target A(L,t).
        AL_clean = ssfm_propagate(A0, L)

        # Always add noise to obtain the observation A_obs(L,t).
        AL_noisy = add_awgn(AL_clean, snr_db=float(snr_db))

        # Store this sample in the batch tensors.
        A0_all[n] = A0
        AL_clean_all[n] = AL_clean
        AL_noisy_all[n] = AL_noisy

    torch.save(
        {
            "meta": _dataset_cache_metadata(n_samples, snr_db),
            "A0": A0_all.detach().cpu(),
            "AL_clean": AL_clean_all.detach().cpu(),
            "AL_noisy": AL_noisy_all.detach().cpu(),
        },
        cache_path,
    )
    print(f"[OK] Saved dataset cache: {cache_path}")

    return A0_all, AL_clean_all, AL_noisy_all


def make_branch_input(A0_batch):
    """
    The DeepONet branch expects a real-valued vector input.

    Convert complex A0(t) into a concatenated [Re, Im] representation.

    Input:
      A0_batch: [B, N_t] complex

    Output:
      [B, 2 * N_t] float
    """
    return torch.cat([A0_batch.real, A0_batch.imag], dim=1)


# ============================================================
# 4) DeepONet architecture + Fourier trunk
# ============================================================

class ResidualBlock(nn.Module):
    """Simple two-layer MLP residual block: x -> x + f(x)."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        return x + h


class DeepONet(nn.Module):
    """
    DeepONet structure:
      - the branch network takes a full waveform A0(t) as input and outputs latent coefficients B
      - the trunk network takes coordinates (t, z) as input and outputs basis-function values T
      - the final output u(t,z), v(t,z) is the real/imaginary part of the complex inner product <B, T>
    """
    def __init__(self, n_sensors, p_dim, fourier_dim):
        super().__init__()

        # -------- Branch: input dimension 2 * N_t for concatenated [Re, Im] --------
        self.branch_in = nn.Linear(2 * n_sensors, branch_hidden)
        self.branch_blocks = nn.Sequential(
            ResidualBlock(branch_hidden),
            ResidualBlock(branch_hidden),
        )
        # Output 2 * p_dim: first p_dim for real coefficients, second p_dim for imaginary coefficients.
        self.branch_out = nn.Linear(branch_hidden, 2 * p_dim)

        # -------- Trunk: Fourier-feature input built from (t, z) --------
        self.fourier_dim = fourier_dim
        # Random projection matrix used to map (t, z) into Fourier-feature space.
        # register_buffer keeps it with the model without making it trainable.
        B_matrix = torch.randn(2, fourier_dim) * 3.0
        self.register_buffer("B", B_matrix)

        self.trunk_in = nn.Linear(2 * fourier_dim, trunk_hidden)
        self.trunk_blocks = nn.Sequential(
            ResidualBlock(trunk_hidden),
            ResidualBlock(trunk_hidden),
        )
        self.trunk_out = nn.Linear(trunk_hidden, 2 * p_dim)

        # Trainable output scaling can sometimes stabilise optimisation.
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, u_in, t, z):
        """
        Inputs:
          u_in: [B, 2 * N_t]   branch input for one waveform/function
          t,z:  [B, 1]         trunk coordinates

        Output:
          u,v:  [B, 1]         predicted complex field A(t,z) = u + jv
        """
        # ---------------- Branch forward ----------------
        bx = torch.relu(self.branch_in(u_in))
        bx = self.branch_blocks(bx)
        bx = self.branch_out(bx)  # [B, 2 * P_dim]

        # Split into complex coefficients B_r + j B_i.
        B_r, B_i = bx[:, :P_dim], bx[:, P_dim:]

        # ---------------- Trunk forward (Fourier features) ----------------
        # Normalise coordinates to keep their scale training-friendly.
        t_norm = t / T_max
        z_norm = z / L
        coords = torch.cat([t_norm, z_norm], dim=1)  # [B, 2]

        # Random projection followed by sine/cosine concatenation.
        proj = coords @ self.B  # [B, fourier_dim]
        fourier_features = torch.cat(
            [torch.sin(2 * np.pi * proj), torch.cos(2 * np.pi * proj)], dim=1
        )  # [B, 2 * fourier_dim]

        tx = torch.relu(self.trunk_in(fourier_features))
        tx = self.trunk_blocks(tx)
        tx = self.trunk_out(tx)  # [B, 2 * P_dim]

        T_r, T_i = tx[:, :P_dim], tx[:, P_dim:]

        # ---------------- Complex inner product ----------------
        # (B_r + jB_i) * (T_r + jT_i)
        # real = sum(B_r * T_r - B_i * T_i)
        # imag = sum(B_r * T_i + B_i * T_r)
        real_part = torch.sum(B_r * T_r - B_i * T_i, dim=1, keepdim=True)
        imag_part = torch.sum(B_r * T_i + B_i * T_r, dim=1, keepdim=True)

        return self.output_scale * real_part, self.output_scale * imag_part


def create_model():
    """Create the model, optimiser, and learning-rate scheduler."""
    model = DeepONet(N_t, P_dim, fourier_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    return model, optimizer, scheduler


# ============================================================
# 5) PDE residual in real/imaginary form
# ============================================================

def physics_loss(model, u_in_batch, n_samples, z_max=1.0, chunk=1024):
    """
    Randomly sample points (t, z) inside the domain and compute the mean-square NLSE residuals f_u and f_v.

    Chunking is used to avoid allocating too many autograd points at once.

    Inputs:
      - u_in_batch: [B_fns, 2 * N_t] where B_fns waveforms/functions form one batch
      - n_samples: total number of physics sampling points

    Output:
      - scalar loss
    """
    B_fns = u_in_batch.shape[0]  # Number of waveform functions in the current batch.
    total = 0.0
    count = 0

    for start in range(0, n_samples, chunk):
        m = min(chunk, n_samples - start)

        # For each physics point, randomly pick one waveform as the branch input.
        idx_fn = torch.randint(0, B_fns, (m,), device=device)
        u_in_pde = u_in_batch[idx_fn]  # [m, 2 * N_t]

        # Randomly sample (t, z).
        t_f = (torch.rand(m, 1, device=device) * 2 - 1) * T_max  # [-T_max, +T_max]
        z_f = torch.rand(m, 1, device=device) * z_max            # [0, z_max]

        # Gradients must be enabled to compute u_t, u_tt, u_z, and related terms.
        t_f.requires_grad_(True)
        z_f.requires_grad_(True)

        # Network output u(t,z), v(t,z).
        u, v = model(u_in_pde, t_f, z_f)
        ones = torch.ones_like(u)

        # First derivatives with respect to z.
        u_z = torch.autograd.grad(u, z_f, ones, create_graph=True)[0]
        v_z = torch.autograd.grad(v, z_f, ones, create_graph=True)[0]

        # First and second derivatives with respect to t.
        u_t  = torch.autograd.grad(u, t_f, ones, create_graph=True)[0]
        v_t  = torch.autograd.grad(v, t_f, ones, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_f, ones, create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t, t_f, ones, create_graph=True)[0]

        # Intensity |A|^2 = u^2 + v^2.
        intensity = u**2 + v**2

        # NLSE residual after splitting the complex equation into real and imaginary parts.
        f_u = u_z + 0.5 * alpha * u + 0.5 * beta2 * v_tt + gamma * intensity * v
        f_v = v_z + 0.5 * alpha * v - 0.5 * beta2 * u_tt - gamma * intensity * u

        # Mean squared residual.
        total = total + torch.mean(f_u**2 + f_v**2)
        count += 1

    return total / count


# ============================================================
# 6) Training: Phase 1 (PDE + IC) + Phase 2 (PDE + IC + anchors)
# ============================================================

def train_model(model, optimizer, scheduler, A0_train, AL_clean_train, AL_noisy_train):
    # ---------------- Phase 1: physics-first ----------------
    print("\n=== Phase 1: Physics-first training (PDE + IC) ===")

    # Start by making the model satisfy physics and the initial condition.
    w_ic = 10.0
    w_pde = 1.0

    for ep in range(1, epochs_pretrain + 1):
        model.train()
        optimizer.zero_grad()

        # (1) Sample a batch of waveform functions A0(t) from the training set.
        idx_fns = torch.randint(0, N_train, (batch_size_fns,), device=device)
        A0_batch = A0_train[idx_fns]             # [Bf, N_t] complex
        u_in_batch = make_branch_input(A0_batch) # [Bf, 2 * N_t] real

        # (2) Initial-condition loss: enforce A(t,0) = A0(t) at z = 0.
        idx_time_ic = torch.randint(0, N_t, (N_ic,), device=device)               # Which time samples to use.
        idx_fn_ic   = torch.randint(0, batch_size_fns, (N_ic,), device=device)    # Which waveform each point belongs to.

        t_ic = t_grid[idx_time_ic].view(-1, 1)     # [N_ic, 1]
        z_ic = torch.zeros_like(t_ic)              # z = 0

        # Ground truth comes from A0_batch itself.
        A0_ic = A0_batch[idx_fn_ic, idx_time_ic]   # [N_ic] complex
        u_true_ic = A0_ic.real.view(-1, 1)
        v_true_ic = A0_ic.imag.view(-1, 1)

        # Prediction uses the corresponding branch vector together with coordinates (t, 0).
        u_in_ic = u_in_batch[idx_fn_ic]            # [N_ic, 2 * N_t]
        u_pred_ic, v_pred_ic = model(u_in_ic, t_ic, z_ic)

        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2 +
                             (v_pred_ic - v_true_ic) ** 2)

        # (3) PDE loss: sample random points inside the domain and drive the residual to zero.
        loss_pde_raw = physics_loss(model, u_in_batch, N_pde, z_max=L)
        # Clamp large spikes to keep occasional exploding gradients from destabilising training.
        loss_pde = torch.clamp(loss_pde_raw, max=1e2)

        # (4) Total loss.
        loss = w_ic * loss_ic + w_pde * loss_pde

        # (5) Backpropagation, gradient clipping, and parameter update.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ep % 200 == 0 or ep == 1:
            print(f"Epoch {ep}/{epochs_pretrain} | "
                  f"Loss={loss.item():.4e} | IC={loss_ic.item():.4e} | PDE={loss_pde_raw.item():.4e}")

    # ---------------- Phase 2: physics + data anchors ----------------
    print("\n=== Phase 2: Physics-driven + sparse data anchors (z=L) ===")

    # Physics still leads, but now sparse endpoint data pulls the model toward the SSFM solution at z = L.
    w_ic    = 10.0
    w_pde   = 1.0
    w_clean = 2.0    # Fit the clean SSFM endpoint closely.
    w_obs   = 0.5    # Reflect noise statistics without chasing noise texture.

    for k in range(1, epochs_hybrid + 1):
        ep = epochs_pretrain + k
        model.train()
        optimizer.zero_grad()

        # (1) Sample batch_size_fns waveform functions.
        idx_fns = torch.randint(0, N_train, (batch_size_fns,), device=device)
        A0_batch  = A0_train[idx_fns]        # A(0,t)
        ALc_batch = AL_clean_train[idx_fns]  # A_clean(L,t)
        ALn_batch = AL_noisy_train[idx_fns]  # A_obs(L,t)

        # Rebuild the branch input here as well so the graph and sampled batch stay aligned.
        u_in_batch = make_branch_input(A0_batch)

        # (2) Initial-condition loss at z = 0.
        idx_time_ic = torch.randint(0, N_t, (N_ic,), device=device)
        idx_fn_ic   = torch.randint(0, batch_size_fns, (N_ic,), device=device)
        t_ic = t_grid[idx_time_ic].view(-1, 1)
        z_ic = torch.zeros_like(t_ic)

        A0_ic = A0_batch[idx_fn_ic, idx_time_ic]
        u_true_ic = A0_ic.real.view(-1, 1)
        v_true_ic = A0_ic.imag.view(-1, 1)

        u_in_ic = u_in_batch[idx_fn_ic]
        u_pred_ic, v_pred_ic = model(u_in_ic, t_ic, z_ic)
        loss_ic = torch.mean((u_pred_ic - u_true_ic) ** 2 + (v_pred_ic - v_true_ic) ** 2)

        # (3) PDE residual loss inside the domain.
        loss_pde_raw = physics_loss(model, u_in_batch, N_pde, z_max=L)
        loss_pde = torch.clamp(loss_pde_raw, max=1e2)

        # (4) Endpoint data anchor at z = L using randomly selected time samples.
        idx_time_data = torch.randint(0, N_t, (N_data,), device=device)
        idx_fn_data   = torch.randint(0, batch_size_fns, (N_data,), device=device)
        t_data = t_grid[idx_time_data].view(-1, 1)
        z_data = torch.full_like(t_data, L)  # z = L

        # Clean target.
        ALc = ALc_batch[idx_fn_data, idx_time_data]
        u_true_clean = ALc.real.view(-1, 1)
        v_true_clean = ALc.imag.view(-1, 1)

        # Noisy observation.
        ALn = ALn_batch[idx_fn_data, idx_time_data]
        u_obs = ALn.real.view(-1, 1)
        v_obs = ALn.imag.view(-1, 1)

        u_in_data = u_in_batch[idx_fn_data]
        u_pred_L, v_pred_L = model(u_in_data, t_data, z_data)

        # Clean anchor that pulls predictions toward the clean SSFM endpoint.
        loss_clean = torch.mean((u_pred_L - u_true_clean) ** 2 + (v_pred_L - v_true_clean) ** 2)

        # Observation likelihood, equivalent to (y - pred)^2 / sigma^2.
        # sigma^2 is derived from snr_db_train; no_grad keeps it out of the training graph.
        with torch.no_grad():
            obs_power = torch.mean(torch.abs(ALn) ** 2) + 1e-12
            snr_lin = 10 ** (snr_db_train / 10)
            sigma2 = obs_power / snr_lin
            sigma2_r = sigma2 / 2  # Real and imaginary parts each take half the variance.

        loss_obs = torch.mean((u_obs - u_pred_L) ** 2 + (v_obs - v_pred_L) ** 2) / (sigma2_r + 1e-12)

        # (5) Total loss.
        loss = w_ic * loss_ic + w_pde * loss_pde + w_clean * loss_clean + w_obs * loss_obs

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if k % 200 == 0 or k == 1:
            print(f"Epoch {ep}/{epochs_pretrain + epochs_hybrid} | "
                  f"Loss={loss.item():.4e} | IC={loss_ic.item():.4e} | "
                  f"PDE={loss_pde_raw.item():.4e} | CLEAN={loss_clean.item():.4e} | OBS={loss_obs.item():.4e}")


# ============================================================
# 7) Evaluation and plotting
# ============================================================

def evaluate_and_plot(model, A0_test, AL_clean_test, AL_noisy_test, title_suffix="Hybrid PINN DeepONet"):
    model.eval()
    output_dir = make_figure_output_dir(__file__)
    print(f"Saving evaluation figures to {output_dir}")

    # Pick one random test sample for visual inspection.
    idx = np.random.randint(0, len(A0_test))
    A0 = A0_test[idx]
    AL_clean = AL_clean_test[idx]
    AL_noisy = AL_noisy_test[idx]

    # Use DeepONet to predict the full waveform A_pred(L,t).
    with torch.no_grad():
        # repeat(N_t, 1) means the trunk evaluates every t point separately,
        # while the branch input stays fixed for the same waveform A0.
        u_in = make_branch_input(A0.unsqueeze(0)).repeat(N_t, 1)  # [N_t, 2 * N_t]
        t_plot = t_grid.view(-1, 1)                               # [N_t, 1]
        z_plot = torch.full_like(t_plot, L)                       # [N_t, 1]
        u_pred, v_pred = model(u_in, t_plot, z_plot)
        A_pred = (u_pred + 1j * v_pred).cpu().numpy().flatten()

    # Convert to numpy for plotting.
    t_np = t_grid.cpu().numpy()
    AL_clean_np = AL_clean.cpu().numpy()
    AL_noisy_np = AL_noisy.cpu().numpy()

    # --- Figure 1: amplitude comparison ---
    fig = plt.figure(figsize=(12, 4))
    plt.plot(t_np, np.abs(AL_noisy_np), label="SSFM+AWGN |A_obs(L)|", alpha=0.7)
    plt.plot(t_np, np.abs(AL_clean_np), label="SSFM clean |A_clean(L)|", alpha=0.7)
    plt.plot(t_np, np.abs(A_pred), '--', label="PINN pred (clean)")
    plt.xlabel("Time")
    plt.ylabel("|A|")
    plt.title(f"Amplitude at z = L ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, "amplitude_at_z_L")
    plt.show()

    # --- Figure 2: I/Q constellation plot ---
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(AL_noisy_np.real, AL_noisy_np.imag, s=10, alpha=0.4, label="SSFM+AWGN (obs)")
    plt.scatter(A_pred.real, A_pred.imag, s=10, alpha=0.7, label="PINN pred (clean)")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"Constellation at z = L ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    save_figure(fig, output_dir, "constellation_at_z_L")
    plt.show()

    # --- Figure 3: amplitude error ---
    amp_err_clean = np.abs(A_pred) - np.abs(AL_clean_np)
    amp_err_obs   = np.abs(A_pred) - np.abs(AL_noisy_np)

    fig = plt.figure(figsize=(12, 3))
    plt.plot(t_np, amp_err_clean, label="Error vs clean")
    plt.plot(t_np, amp_err_obs,   label="Error vs obs", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Amplitude Error at z = L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure(fig, output_dir, "amplitude_error_at_z_L")
    plt.show()


# ============================================================
# 8) Main entry point
#    (data generation -> training -> evaluation -> save)
# ============================================================

def main():
    print("Building training dataset with SSFM...")
    A0_train, ALc_train, ALn_train = build_dataset(
        N_train,
        snr_db_train,
        cache_key=TRAIN_DATASET_CACHE_KEY,
    )

    print("Building test dataset with SSFM...")
    A0_test,  ALc_test,  ALn_test  = build_dataset(
        N_test,
        snr_db_eval,
        cache_key=TEST_DATASET_CACHE_KEY,
    )

    model, optimizer, scheduler = create_model()

    train_model(model, optimizer, scheduler, A0_train, ALc_train, ALn_train)
    evaluate_and_plot(model, A0_test, ALc_test, ALn_test)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
