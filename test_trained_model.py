# test_trained_model.py (Demo-style evaluator with EVM overlay)
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# Import from your DEMO training script
# If your file name differs, change it here.
# ============================================================
from pinn_physics_model import (
    DeepONet, N_t, P_dim, fourier_dim,
    t_grid, L, T_max,
    alpha, beta2, gamma,
    num_symbols, sps, rrc_beta, rrc_span,
    generate_qam_waveform, ssfm_propagate,
    make_branch_input, add_awgn,
    MODEL_PATH, device,
)

# --- evaluation settings ---
snr_db_eval = 25.0   # ALWAYS add AWGN for observation realism
num_eval_samples = 50


def print_config():
    print("\n========== Demo-style Config ==========")
    print(f"Device      : {device}")
    print(f"Fiber params : alpha={alpha}, beta2={beta2}, gamma={gamma}")
    print(f"Link         : L={L}, T_max={T_max}, N_t={N_t}")
    print(f"QAM/RRC      : num_symbols={num_symbols}, sps={sps}, rrc_beta={rrc_beta}, rrc_span={rrc_span}")
    print(f"Eval AWGN    : SNR={snr_db_eval} dB (AL_noisy = AL_clean + AWGN)")
    print(f"MODEL_PATH   : {MODEL_PATH}")
    print("=======================================\n")


def load_trained_model(model_path: str):
    model = DeepONet(N_t, P_dim, fourier_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] Loaded model: {model_path}")
    return model


def evm_rms_pct(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    """RMS EVM(%) = 100 * sqrt( E|e|^2 / E|ref|^2 )"""
    num = np.mean(np.abs(pred - ref) ** 2)
    den = np.mean(np.abs(ref) ** 2)
    return float(100.0 * np.sqrt(num / (den + eps)))


def amp_mae_rmse(pred: np.ndarray, ref: np.ndarray):
    ap = np.abs(pred)
    ar = np.abs(ref)
    mae = float(np.mean(np.abs(ap - ar)))
    rmse = float(np.sqrt(np.mean((ap - ar) ** 2)))
    return mae, rmse


@torch.no_grad()
def predict_AL(model, A0: torch.Tensor) -> np.ndarray:
    """Predict A(L,t) (clean) from A0(t). Returns complex numpy [N_t]."""
    u_in = make_branch_input(A0.unsqueeze(0)).repeat(N_t, 1)  # [N_t, 2*N_t]
    t_plot = t_grid.view(-1, 1)                               # [N_t, 1]
    z_plot = torch.full_like(t_plot, L)                       # [N_t, 1]
    u_pred, v_pred = model(u_in, t_plot, z_plot)
    return (u_pred + 1j * v_pred).detach().cpu().numpy().reshape(-1).astype(np.complex128)


def overlay_text(ax, text: str):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )


def plot_demo_style(t_np, A0_np, AL_clean_np, AL_noisy_np, A_pred, title_prefix: str):
    # metrics
    evm_clean = evm_rms_pct(A_pred, AL_clean_np)
    evm_obs = evm_rms_pct(A_pred, AL_noisy_np)
    mae_c, rmse_c = amp_mae_rmse(A_pred, AL_clean_np)
    mae_o, rmse_o = amp_mae_rmse(A_pred, AL_noisy_np)

    info = (
        f"EVM(clean)={evm_clean:.2f}% | EVM(obs)={evm_obs:.2f}%\n"
        f"AmpRMSE(clean)={rmse_c:.3e} | AmpRMSE(obs)={rmse_o:.3e}"
    )

    # ---------------- Figure 1: Amplitude ----------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_np, np.abs(A0_np), ":", alpha=0.6, label="Input |A(0)|")
    ax.plot(t_np, np.abs(AL_noisy_np), alpha=0.7, label="SSFM+AWGN |A_obs(L)|")
    ax.plot(t_np, np.abs(AL_clean_np), alpha=0.7, label="SSFM clean |A_clean(L)|")
    ax.plot(t_np, np.abs(A_pred), "--", label="PINN pred (clean)")
    ax.set_xlabel("Time")
    ax.set_ylabel("|A|")
    ax.set_title(f"{title_prefix} | EVM(clean)={evm_clean:.2f}%  EVM(obs)={evm_obs:.2f}%")
    ax.grid(True)
    ax.legend()
    overlay_text(ax, info)
    fig.tight_layout()
    plt.show()

    # ---------------- Figure 2: Constellation ----------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(AL_noisy_np.real, AL_noisy_np.imag, s=10, alpha=0.35, label="SSFM+AWGN (obs)")
    ax.scatter(A_pred.real, A_pred.imag, s=10, alpha=0.7, label="PINN pred (clean)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"Constellation at z = L | EVM(clean)={evm_clean:.2f}%  EVM(obs)={evm_obs:.2f}%")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal", "box")
    overlay_text(ax, info)
    fig.tight_layout()
    plt.show()

    # ---------------- Figure 3: Amplitude Error ----------------
    amp_err_clean = np.abs(A_pred) - np.abs(AL_clean_np)
    amp_err_obs = np.abs(A_pred) - np.abs(AL_noisy_np)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t_np, amp_err_clean, label="Error vs clean")
    ax.plot(t_np, amp_err_obs, alpha=0.7, label="Error vs obs")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.set_title(f"Amplitude Error at z = L | EVM(clean)={evm_clean:.2f}%  EVM(obs)={evm_obs:.2f}%")
    ax.legend()
    ax.grid(True)
    overlay_text(ax, info)
    fig.tight_layout()
    plt.show()


def main():
    print_config()
    model = load_trained_model(MODEL_PATH)

    t_np = t_grid.cpu().numpy()

    # evaluate multiple random samples, pick best by EVM(clean)
    best = None
    best_evm = None

    evm_clean_list = []
    evm_obs_list = []

    for _ in range(num_eval_samples):
        A0, _, _ = generate_qam_waveform()
        AL_clean = ssfm_propagate(A0, L)
        AL_noisy = add_awgn(AL_clean, snr_db=snr_db_eval)  # ALWAYS noisy observation
        A_pred = predict_AL(model, A0)

        A0_np = A0.detach().cpu().numpy()
        AL_clean_np = AL_clean.detach().cpu().numpy()
        AL_noisy_np = AL_noisy.detach().cpu().numpy()

        evm_c = evm_rms_pct(A_pred, AL_clean_np)
        evm_o = evm_rms_pct(A_pred, AL_noisy_np)
        evm_clean_list.append(evm_c)
        evm_obs_list.append(evm_o)

        if (best_evm is None) or (evm_c < best_evm):
            best_evm = evm_c
            best = (A0_np, AL_clean_np, AL_noisy_np, A_pred)

    evm_clean_arr = np.array(evm_clean_list)
    evm_obs_arr = np.array(evm_obs_list)

    print("========== Demo-style Metrics ==========")
    print(f"Samples      : {num_eval_samples}")
    print(f"[vs CLEAN] EVM mean : {evm_clean_arr.mean():.2f}% | best : {best_evm:.2f}%")
    print(f"[vs  OBS ] EVM mean : {evm_obs_arr.mean():.2f}%")
    print("=======================================\n")

    # plot best sample
    A0_np, AL_clean_np, AL_noisy_np, A_pred = best
    title_prefix = "Amplitude at z = L (Best Test Sample, Demo-style)"
    plot_demo_style(t_np, A0_np, AL_clean_np, AL_noisy_np, A_pred, title_prefix)


if __name__ == "__main__":
    main()