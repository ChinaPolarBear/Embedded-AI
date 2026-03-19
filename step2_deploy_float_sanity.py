"""
Step 2: Build a FINN-friendly "deploy float" model and sanity-check it.

Usage:
    python step2_deploy_float_sanity.py --ckpt hybrid_pinn_deeponet.pth --mat trunk_matrices.npz

What it does:
  - Loads your original trained DeepONet.
  - Exports trunk matrices (or loads them from step1 output).
  - Runs ONE sample through:
        (A) original DeepONet point-wise (your current way)
        (B) deploy model: branch(A0)->b, then M@b to produce full waveform
  - Prints max|diff| and (optional) EVM-like metric.

Important:
  - This sanity check uses the original branch (with GELU residual blocks),
    so outputs should match extremely closely (up to tiny numerical noise).

This confirms trunk-freezing math is correct before you do quantization/QAT.
"""
# python step2_deploy_float_sanity.py --ckpt hybrid_pinn_deeponet.pth --mat trunk_matrices.npz
import argparse
import numpy as np
import torch

import pinn_physics_model as pm


class DeepONetFrozenTrunkFloat(torch.nn.Module):
    """
    Float deploy model for validation:
      - Uses ORIGINAL trained branch submodules from pm.DeepONet (including GELU).
      - Replaces trunk + complex inner product by constant GEMV matrices.

    Input:
      u_in: [B, 2*N_t]
    Output:
      real: [B, N_t], imag: [B, N_t]
    """
    def __init__(self, branch_in, branch_blocks, branch_out, M_real, M_imag):
        super().__init__()
        self.branch_in = branch_in
        self.branch_blocks = branch_blocks
        self.branch_out = branch_out
        self.register_buffer("M_real", M_real)  # [N_t, 2P]
        self.register_buffer("M_imag", M_imag)  # [N_t, 2P]

    def forward(self, u_in):
        x = torch.relu(self.branch_in(u_in))
        x = self.branch_blocks(x)
        b = self.branch_out(x)                                  # [B,2P]
        real = b @ self.M_real.T                                # [B,N_t]
        imag = b @ self.M_imag.T                                # [B,N_t]
        return real, imag


@torch.no_grad()
def run_one(model_full, model_deploy, A0_complex):
    device = next(model_full.parameters()).device
    A0_complex = A0_complex.to(device)

    # (A) full model point-wise (B=N_t points)
    u_in = pm.make_branch_input(A0_complex.unsqueeze(0)).repeat(pm.N_t, 1)  # [N_t,2N]
    t = pm.t_grid.view(-1, 1).to(device)
    z = torch.full_like(t, float(pm.L))
    real_full, imag_full = model_full(u_in, t, z)                           # [N_t,1]
    real_full = real_full.view(1, pm.N_t)
    imag_full = imag_full.view(1, pm.N_t)

    # (B) deploy model in one shot (B=1 waveform)
    u_in2 = pm.make_branch_input(A0_complex.unsqueeze(0))                   # [1,2N]
    real_dep, imag_dep = model_deploy(u_in2)                                # [1,N_t]

    diff = torch.sqrt((real_full-real_dep)**2 + (imag_full-imag_dep)**2)
    max_err = diff.max().item()
    mse = diff.pow(2).mean().item()
    print(f"max|diff|={max_err:.6e},  mse={mse:.6e}")

    # a simple EVM-like: sqrt(E[|e|^2] / E[|ref|^2])
    ref_pow = (real_full**2 + imag_full**2).mean().item()
    evm = (mse / (ref_pow + 1e-12))**0.5
    print(f"relative_rms_error (EVM-like) = {evm:.6e}")


def main(ckpt: str, mat: str):
    device = torch.device("cpu")
    full = pm.DeepONet(pm.N_t, pm.P_dim, pm.fourier_dim).to(device)
    full.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    full.eval()

    data = np.load(mat)
    M_real = torch.from_numpy(data["M_real"]).to(device)
    M_imag = torch.from_numpy(data["M_imag"]).to(device)

    deploy = DeepONetFrozenTrunkFloat(full.branch_in, full.branch_blocks, full.branch_out, M_real, M_imag).to(device)
    deploy.eval()

    # build one waveform sample using your existing generator
    A0, _, _ = pm.generate_qam_waveform()
    run_one(full, deploy, A0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--mat", type=str, required=True)
    args = ap.parse_args()
    main(args.ckpt, args.mat)
