"""
Step 1: Freeze trunk into constant matrices for FINN-friendly deployment.

Usage:
    python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz

This script:
  1) Imports your existing pinn_physics_model.py to reuse DeepONet + constants (N_t, P_dim, T_max, L, t_grid, etc.).
  2) Loads the trained checkpoint into the original DeepONet model.
  3) Evaluates trunk over coords (t_grid, z=L) to get T_r(t), T_i(t).
  4) Builds constant matrices:
       M_real[t] = output_scale * concat(T_r(t), -T_i(t))   shape [N_t, 2P]
       M_imag[t] = output_scale * concat(T_i(t),  T_r(t))   shape [N_t, 2P]
  5) Saves matrices to .npz (float32).

After this, the FPGA model only needs:
  branch(u) -> b(2P), then 2 GEMVs: real=M_real@b, imag=M_imag@b
"""
# python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz
import argparse
import numpy as np
import torch

import pinn_physics_model as pm


def select_time_indices(n_t_out: int, mode: str) -> torch.Tensor:
    if n_t_out <= 0 or n_t_out > pm.N_t:
        raise ValueError(f"n_t_out must be in [1, {pm.N_t}], got {n_t_out}")

    if mode == "start":
        start = 0
    elif mode == "end":
        start = pm.N_t - n_t_out
    elif mode == "uniform":
        if pm.N_t % n_t_out == 0:
            step = pm.N_t // n_t_out
            return torch.arange(0, pm.N_t, step, dtype=torch.long)[:n_t_out]

        indices = torch.round(torch.linspace(0, pm.N_t - 1, steps=n_t_out)).long()
        if torch.unique_consecutive(indices).numel() != n_t_out:
            raise ValueError(
                f"Could not build {n_t_out} unique uniformly spaced indices from N_t={pm.N_t}"
            )
        return indices
    else:
        start = (pm.N_t - n_t_out) // 2

    return torch.arange(start, start + n_t_out, dtype=torch.long)


@torch.no_grad()
def main(ckpt: str, out: str, n_t_out: int, p_dim_out: int, time_slice: str):
    device = torch.device("cpu")  # trunk freezing is deterministic; CPU is fine
    model = pm.DeepONet(pm.N_t, pm.P_dim, pm.fourier_dim).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    if p_dim_out <= 0 or p_dim_out > pm.P_dim:
        raise ValueError(f"p_dim_out must be in [1, {pm.P_dim}], got {p_dim_out}")

    # coords: (t_grid, z=L)
    t = pm.t_grid.detach().to(device).view(-1, 1).float()
    z = torch.full_like(t, float(pm.L))

    # --- trunk forward (replicates pm.DeepONet forward trunk path) ---
    t_norm = t / float(pm.T_max)
    z_norm = z / float(pm.L)
    coords = torch.cat([t_norm, z_norm], dim=1)                       # [N_t,2]
    proj = coords @ model.B                                           # [N_t, fourier_dim]
    fourier = torch.cat([torch.sin(2*np.pi*proj), torch.cos(2*np.pi*proj)], dim=1)  # [N_t,2*fourier_dim]

    Tx = torch.relu(model.trunk_in(fourier))
    Tx = model.trunk_blocks(Tx)
    Tx = model.trunk_out(Tx)                                          # [N_t,2P]
    T_r, T_i = Tx[:, :pm.P_dim], Tx[:, pm.P_dim:]                     # [N_t,P]

    scale = float(model.output_scale.detach().cpu().item())

    M_real = torch.cat([T_r, -T_i], dim=1) * scale                     # [N_t,2P]
    M_imag = torch.cat([T_i,  T_r], dim=1) * scale                     # [N_t,2P]

    time_idx = select_time_indices(n_t_out, time_slice)
    M_real = torch.cat([M_real[time_idx, :p_dim_out], M_real[time_idx, pm.P_dim:pm.P_dim + p_dim_out]], dim=1)
    M_imag = torch.cat([M_imag[time_idx, :p_dim_out], M_imag[time_idx, pm.P_dim:pm.P_dim + p_dim_out]], dim=1)

    np.savez(
        out,
        M_real=M_real.cpu().numpy().astype(np.float32),
        M_imag=M_imag.cpu().numpy().astype(np.float32),
        meta=np.array([n_t_out, p_dim_out], dtype=np.int32),
        time_indices=time_idx.cpu().numpy().astype(np.int32),
    )
    print(f"[OK] Saved trunk matrices to: {out}")
    print(
        f"     M_real: {tuple(M_real.shape)}  M_imag: {tuple(M_imag.shape)}"
        f"  scale={scale}  n_t_out={n_t_out}  p_dim_out={p_dim_out}  time_slice={time_slice}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="trunk_matrices.npz")
    ap.add_argument("--n_t_out", type=int, default=pm.N_t)
    ap.add_argument("--p_dim_out", type=int, default=pm.P_dim)
    ap.add_argument(
        "--time_slice",
        type=str,
        default="center",
        choices=["center", "start", "end", "uniform"],
    )
    args = ap.parse_args()
    main(args.ckpt, args.out, args.n_t_out, args.p_dim_out, args.time_slice)
