# test_trained_model_csv.py (minimal)
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn_physics_model import (
    DeepONet, N_t, P_dim, fourier_dim,
    t_grid, L, device, MODEL_PATH,
    make_branch_input
)


def load_model(model_path: str):
    model = DeepONet(N_t, P_dim, fourier_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] Loaded model: {model_path}")
    return model


def load_csv(csv_path: str):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"CSV format error: expected >=5 columns, got {data.shape}")

    time = data[:, 0]
    tx = data[:, 1] + 1j * data[:, 2]
    rx = data[:, 3] + 1j * data[:, 4]
    return time, tx.astype(np.complex128), rx.astype(np.complex128)


def pick_window(tx: np.ndarray, rx: np.ndarray, trim_ratio: float = 0.1, mode: str = "center"):
    """
    Return one window (tx_win, rx_win) of length N_t.
    - trim_ratio: cut edges to reduce transient effects
    - mode: "center" | "start" | "end"
    """
    n = len(tx)
    k = int(n * trim_ratio)
    if n - 2 * k < N_t:
        k = max(0, (n - N_t) // 2)

    tx_t = tx[k:n - k]
    rx_t = rx[k:n - k]
    n2 = len(tx_t)

    if n2 < N_t:
        raise ValueError(f"After trim, length {n2} < N_t={N_t}")

    if mode == "start":
        s = 0
    elif mode == "end":
        s = n2 - N_t
    else:  # center
        s = (n2 - N_t) // 2

    return tx_t[s:s + N_t], rx_t[s:s + N_t], (k + s)


def normalize_by_tx_rms(tx_win: np.ndarray, rx_win: np.ndarray, eps: float = 1e-12):
    s = np.sqrt(np.mean(np.abs(tx_win) ** 2))
    s = max(float(s), eps)
    return tx_win / s, rx_win / s, s


def optimal_phase_align(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> complex:
    inner = np.vdot(pred, true)  # sum(conj(pred)*true)
    if np.abs(inner) < eps:
        return 1.0 + 0j
    return np.exp(1j * np.angle(inner))


@torch.no_grad()
def predict_window(model, tx_win: np.ndarray) -> np.ndarray:
    A0 = torch.tensor(tx_win, dtype=torch.complex64, device=device)  # [N_t]
    u_in = make_branch_input(A0.unsqueeze(0)).repeat(N_t, 1)         # [N_t, 2*N_t]
    t_plot = t_grid.view(-1, 1)                                      # [N_t, 1]
    z_plot = torch.full_like(t_plot, L)                              # [N_t, 1]
    u_pred, v_pred = model(u_in, t_plot, z_plot)
    return (u_pred + 1j * v_pred).detach().cpu().numpy().reshape(-1).astype(np.complex128)


def metrics(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12):
    # complex
    rmse = float(np.sqrt(np.mean(np.abs(pred - true) ** 2)))
    true_rms = float(np.sqrt(np.mean(np.abs(true) ** 2)))
    nrmse = rmse / max(true_rms, eps)
    evm_pct = 100.0 * nrmse

    # amplitude
    ap = np.abs(pred)
    at = np.abs(true)
    amp_rmse = float(np.sqrt(np.mean((ap - at) ** 2)))
    amp_mae = float(np.mean(np.abs(ap - at)))
    return {
        "c_rmse": rmse,
        "c_nrmse": nrmse,
        "evm_pct": evm_pct,
        "amp_rmse": amp_rmse,
        "amp_mae": amp_mae,
    }


def plot_compare(tx_win: np.ndarray, rx_true: np.ndarray, rx_pred: np.ndarray, title: str):
    x = np.arange(len(tx_win))
    plt.figure(figsize=(12, 4))
    plt.plot(
        x, np.abs(tx_win),
        color="gray", linestyle=":", alpha=0.6,
        label="Input |A(0)|"
    )
    plt.plot(
        x, np.abs(rx_true),
        color="tab:blue", linewidth=2,
        label="SSFM |A(L)|"
    )
    plt.plot(
        x, np.abs(rx_pred),
        color="tab:orange", linestyle="--", linewidth=2,
        label="PINN |A(L)|"
    )

    plt.xlabel("Sample index (window)")
    plt.ylabel("|A|")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(
    rx_true.real, rx_true.imag,
    s=12, alpha=0.35,
    color="tab:blue", label="SSFM"
    )
    plt.scatter(
        rx_pred.real, rx_pred.imag,
        s=12, alpha=0.7,
        color="tab:orange", label="PINN"
    )

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title(title + " (Constellation)")
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(x, np.abs(rx_pred) - np.abs(rx_true))
    plt.xlabel("Sample index (window)")
    plt.ylabel("|pred|-|true|")
    plt.title(title + " (Amplitude error)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser("Minimal CSV evaluator for trained DeepONet PINN")
    p.add_argument("--csv", type=str, default="Data_Output/waveform_data_trainmatch.csv")
    p.add_argument("--model", type=str, default=MODEL_PATH)
    p.add_argument("--trim", type=float, default=0.10, help="trim ratio from each side")
    p.add_argument("--window", type=str, default="center", choices=["center", "start", "end"])
    p.add_argument("--normalize", action="store_true", help="normalize window by Tx RMS")
    p.add_argument("--phase-align", action="store_true", help="optimal global phase alignment per window")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = load_model(args.model)
    _, tx, rx = load_csv(args.csv)

    tx_win, rx_true, start_idx = pick_window(tx, rx, trim_ratio=args.trim, mode=args.window)

    scale = 1.0
    if args.normalize:
        tx_win, rx_true, scale = normalize_by_tx_rms(tx_win, rx_true)

    rx_pred = predict_window(model, tx_win)

    phase_c = 1.0 + 0j
    if args.phase_align:
        phase_c = optimal_phase_align(rx_pred, rx_true)
        rx_pred = phase_c * rx_pred

    m = metrics(rx_pred, rx_true)

    print("\n===== Minimal CSV Eval Result =====")
    print(f"CSV       : {args.csv}")
    print(f"Model     : {args.model}")
    print(f"Window    : {args.window} (start index in original tx/rx = {start_idx})")
    print(f"Trim ratio: {args.trim}")
    print(f"Normalize : {args.normalize} (scale={scale:.4e})")
    print(f"PhaseAlign: {args.phase_align} (c={phase_c})")
    print("Metrics   :", m)

    title = f"{os.path.basename(args.csv)} | win={args.window} start={start_idx} | " \
            f"NRMSE={m['c_nrmse']:.3e}, EVM={m['evm_pct']:.2f}%"
    plot_compare(tx_win, rx_true, rx_pred, title)


if __name__ == "__main__":
    main()
