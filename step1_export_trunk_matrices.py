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


@torch.no_grad()
def main(ckpt: str, out: str):
    device = torch.device("cpu")  # trunk freezing is deterministic; CPU is fine
    model = pm.DeepONet(pm.N_t, pm.P_dim, pm.fourier_dim).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

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

    np.savez(
        out,
        M_real=M_real.cpu().numpy().astype(np.float32),
        M_imag=M_imag.cpu().numpy().astype(np.float32),
        meta=np.array([pm.N_t, pm.P_dim], dtype=np.int32),
    )
    print(f"[OK] Saved trunk matrices to: {out}")
    print(f"     M_real: {tuple(M_real.shape)}  M_imag: {tuple(M_imag.shape)}  scale={scale}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="trunk_matrices.npz")
    args = ap.parse_args()
    main(args.ckpt, args.out)
