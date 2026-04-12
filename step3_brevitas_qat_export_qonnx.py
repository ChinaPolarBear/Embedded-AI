# Step 3
# pip install brevitas qonnx onnx onnxruntime onnxoptimizer
# python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices.npz --epochs 10 --qonnx_out deeponet_u250_int8_qonnx.onnx

import argparse
import numpy as np
import torch
import torch.nn as nn

import pinn_physics_model as pm

from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

from finn_qonnx_utils import ensure_brevitas_qonnx_export_support, print_initializer_shapes


class BranchMLPInt8(nn.Module):
    """
    FINN-friendly branch:
      signed input quant -> QuantLinear -> Uint8 QuantReLU -> QuantLinear -> Uint8 QuantReLU -> QuantLinear
    Input:  [B, 2*N_t]
    Output: [B, 2P]
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc1 = QuantLinear(
            in_dim,
            hidden,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True,
        )
        self.act1 = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc2 = QuantLinear(
            hidden,
            hidden,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True,
        )
        self.act2 = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=True,
        )
        self.fc3 = QuantLinear(
            hidden,
            out_dim,
            weight_quant=Int8WeightPerTensorFloat,
            bias=True,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepONetFinnDeployInt8(nn.Module):
    """
    Full deploy model for FINN:
      u_in -> BranchMLPInt8 -> b(2P) -> 2 fixed-weight QuantLinear layers -> packed [B, 2, N_t]
    """

    def __init__(self, M_real: torch.Tensor, M_imag: torch.Tensor, hidden: int):
        super().__init__()
        assert M_real.shape == M_imag.shape
        n_t, two_p = M_real.shape
        self.n_t = n_t
        self.two_p = two_p

        self.branch = BranchMLPInt8(in_dim=2 * pm.N_t, hidden=hidden, out_dim=two_p)

        self.real_fc = QuantLinear(
            two_p,
            n_t,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
        )
        self.imag_fc = QuantLinear(
            two_p,
            n_t,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
        )

        with torch.no_grad():
            self.real_fc.weight.copy_(M_real)
            self.imag_fc.weight.copy_(M_imag)

        for param in self.real_fc.parameters():
            param.requires_grad = False
        for param in self.imag_fc.parameters():
            param.requires_grad = False

    def forward(self, u_in):
        branch_out = self.branch(u_in)
        real = self.real_fc(branch_out)
        imag = self.imag_fc(branch_out)
        return torch.stack([real, imag], dim=1)


def crop_targets(AL_clean_batch: torch.Tensor, n_t_out: int, time_indices: torch.Tensor | None):
    if n_t_out == pm.N_t and time_indices is None:
        return AL_clean_batch

    if time_indices is not None:
        return AL_clean_batch[:, time_indices]

    start = (pm.N_t - n_t_out) // 2
    return AL_clean_batch[:, start:start + n_t_out]


def make_supervised_batch(
    A0_batch: torch.Tensor,
    AL_clean_batch: torch.Tensor,
    n_t_out: int,
    time_indices: torch.Tensor | None,
):
    """
    A0_batch, AL_clean_batch: complex [B, N_t]
    returns:
      u_in: [B, 2*N_t]
      y_real,y_imag: [B, n_t_out]
    """
    AL_clean_batch = crop_targets(AL_clean_batch, n_t_out, time_indices)
    u_in = pm.make_branch_input(A0_batch)
    y_real = AL_clean_batch.real
    y_imag = AL_clean_batch.imag
    return u_in.float(), y_real.float(), y_imag.float()


def export_model_to_qonnx(model: nn.Module, qonnx_out: str) -> None:
    ensure_brevitas_qonnx_export_support()

    export_model = model.cpu().eval()
    dummy = torch.zeros(1, 2 * pm.N_t, dtype=torch.float32)

    export_qonnx(
        export_model,
        args=dummy,
        export_path=qonnx_out,
        dynamo=False,
        opset_version=13,
        input_names=["u_in"],
        output_names=["y_out"],
    )

    print(f"[OK] Exported raw QONNX -> {qonnx_out}")
    print_initializer_shapes(qonnx_out)
    print("Next: run step4.1_export_QONNX_ready_model.py before ConvertQONNXtoFINN().")


def load_supervised_dataset(n_samples: int, snr_db: float, force_rebuild_cache: bool):
    if not force_rebuild_cache:
        train_cached = pm.load_dataset_cache(
            pm.N_train,
            pm.snr_db_train,
            cache_key=pm.TRAIN_DATASET_CACHE_KEY,
        )
        if train_cached is not None and n_samples <= train_cached[0].shape[0]:
            print(f"[OK] Reusing first {n_samples} samples from training cache for Step 3.")
            return tuple(t[:n_samples] for t in train_cached)

    return pm.build_dataset(
        n_samples=n_samples,
        snr_db=snr_db,
        cache_key=pm.QAT_DATASET_CACHE_KEY,
        force_rebuild=force_rebuild_cache,
    )


def main(
    mat: str,
    epochs: int,
    qonnx_out: str,
    lr: float,
    dataset_samples: int,
    force_rebuild_cache: bool,
    hidden: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(mat)
    M_real = torch.from_numpy(data["M_real"]).float().to(device)
    M_imag = torch.from_numpy(data["M_imag"]).float().to(device)
    time_indices = None
    if "time_indices" in data.files:
        time_indices = torch.from_numpy(data["time_indices"]).long().to(device)
    n_t_out = int(M_real.shape[0])

    model = DeepONetFinnDeployInt8(M_real=M_real, M_imag=M_imag, hidden=hidden).to(device)
    model.train()

    A0, AL_clean, _ = load_supervised_dataset(
        n_samples=dataset_samples,
        snr_db=pm.snr_db_train,
        force_rebuild_cache=force_rebuild_cache,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(A0.shape[0], device=device)
        total = 0.0
        for i in range(0, A0.shape[0], 32):
            idx = perm[i : i + 32]
            u_in, y_r, y_i = make_supervised_batch(A0[idx], AL_clean[idx], n_t_out, time_indices)
            pred = model(u_in)
            pred_r = pred[:, 0, :]
            pred_i = pred[:, 1, :]
            loss = ((pred_r - y_r) ** 2 + (pred_i - y_i) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"epoch {ep:03d}: loss={total * 32 / A0.shape[0]:.6e}")

    export_model_to_qonnx(model, qonnx_out=qonnx_out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--qonnx_out", type=str, default="deeponet_u250_int8_qonnx.onnx")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dataset_samples", type=int, default=256)
    ap.add_argument("--force_rebuild_cache", action="store_true")
    ap.add_argument("--hidden", type=int, default=pm.branch_hidden)
    args = ap.parse_args()
    main(
        args.mat,
        args.epochs,
        args.qonnx_out,
        args.lr,
        args.dataset_samples,
        args.force_rebuild_cache,
        args.hidden,
    )
