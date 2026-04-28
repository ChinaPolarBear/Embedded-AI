# Step 3
# pip install brevitas qonnx onnx onnxruntime onnxoptimizer

# python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices.npz --epochs 10 --input_bit_width 4 --weight_bit_width 4 --act_bit_width 4 --qonnx_out deeponet_u250_int4_qonnx.onnx

import argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

import pinn_physics_model as pm

from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat

from finn_qonnx_utils import ensure_brevitas_qonnx_export_support, print_initializer_shapes


@dataclass(frozen=True)
class QuantConfig:
    input_bit_width: int = 8
    weight_bit_width: int = 8
    act_bit_width: int = 8
    output_bit_width: int | None = None

    def describe(self) -> str:
        output_desc = "accumulator/unquantized" if self.output_bit_width is None else f"int{self.output_bit_width}"
        return (
            f"input=int{self.input_bit_width}, "
            f"weight=int{self.weight_bit_width}, "
            f"hidden_act=uint{self.act_bit_width}, "
            f"output={output_desc}"
        )


def _validate_bit_width(name: str, value: int | None, allow_none: bool = False) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} cannot be None")
    if value < 1 or value > 32:
        raise ValueError(f"{name} must be in [1, 32], got {value}")


class BranchMLPQuant(nn.Module):
    """
    FINN-friendly branch:
      signed input quant -> QuantLinear -> Uint8 QuantReLU -> QuantLinear -> Uint8 QuantReLU -> QuantLinear
    Input:  [B, 2*N_deploy]
    Output: [B, 2P]
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, quant_cfg: QuantConfig):
        super().__init__()
        self.input_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            bit_width=quant_cfg.input_bit_width,
            return_quant_tensor=True,
        )
        self.fc1 = QuantLinear(
            in_dim,
            hidden,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=quant_cfg.weight_bit_width,
            bias=True,
            return_quant_tensor=True,
        )
        self.act1 = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=quant_cfg.act_bit_width,
            return_quant_tensor=True,
        )
        self.fc2 = QuantLinear(
            hidden,
            hidden,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=quant_cfg.weight_bit_width,
            bias=True,
            return_quant_tensor=True,
        )
        self.act2 = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=quant_cfg.act_bit_width,
            return_quant_tensor=True,
        )
        self.fc3 = QuantLinear(
            hidden,
            out_dim,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=quant_cfg.weight_bit_width,
            bias=True,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepONetFinnDeployQuant(nn.Module):
    """
    Full deploy model for FINN:
      u_in -> BranchMLPQuant -> b(2P) -> 1 fixed-weight QuantLinear layer -> [B, 2*N_deploy]

    Output layout:
      out[:, :N_deploy]  = real(A(L,t_deploy))
      out[:, N_deploy:]  = imag(A(L,t_deploy))
    """

    def __init__(self, M_real: torch.Tensor, M_imag: torch.Tensor, hidden: int, quant_cfg: QuantConfig):
        super().__init__()
        assert M_real.shape == M_imag.shape
        n_t_deploy, two_p = M_real.shape
        self.n_t = n_t_deploy
        self.two_p = two_p
        self.quant_cfg = quant_cfg

        self.branch = BranchMLPQuant(
            in_dim=2 * n_t_deploy,
            hidden=hidden,
            out_dim=two_p,
            quant_cfg=quant_cfg,
        )

        self.out_fc = QuantLinear(
            two_p,
            2 * n_t_deploy,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=quant_cfg.weight_bit_width,
            bias=False,
        )
        self.output_quant = None
        if quant_cfg.output_bit_width is not None:
            self.output_quant = QuantIdentity(
                act_quant=Int8ActPerTensorFloat,
                bit_width=quant_cfg.output_bit_width,
                return_quant_tensor=False,
            )

        with torch.no_grad():
            self.out_fc.weight.copy_(torch.cat([M_real, M_imag], dim=0))

        for param in self.out_fc.parameters():
            param.requires_grad = False

    def forward(self, u_in):
        branch_out = self.branch(u_in)
        out = self.out_fc(branch_out)
        if self.output_quant is not None:
            out = self.output_quant(out)
        return out


def crop_complex_batch(batch: torch.Tensor, n_t_out: int, time_indices: torch.Tensor | None):
    if n_t_out == pm.N_t and time_indices is None:
        return batch

    if time_indices is not None:
        return batch[:, time_indices]

    start = (pm.N_t - n_t_out) // 2
    return batch[:, start:start + n_t_out]


def make_supervised_batch(
    A0_batch: torch.Tensor,
    AL_clean_batch: torch.Tensor,
    n_t_out: int,
    time_indices: torch.Tensor | None,
):
    """
    A0_batch, AL_clean_batch: complex [B, N_t]
    returns:
      u_in: [B, 2*n_t_out]
      y_out: [B, 2*n_t_out]
    """
    A0_batch = crop_complex_batch(A0_batch, n_t_out, time_indices)
    AL_clean_batch = crop_complex_batch(AL_clean_batch, n_t_out, time_indices)
    u_in = torch.cat([A0_batch.real, A0_batch.imag], dim=1)
    y_out = torch.cat([AL_clean_batch.real, AL_clean_batch.imag], dim=1)
    return u_in.float(), y_out.float()


def export_model_to_qonnx(model: nn.Module, qonnx_out: str) -> None:
    ensure_brevitas_qonnx_export_support()

    export_model = model.cpu().eval()
    dummy = torch.zeros(1, 2 * model.n_t, dtype=torch.float32)

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
    print("Next: run step4_convert_qonnx_to_finn.py with this raw QONNX as input.")


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
    input_bit_width: int,
    weight_bit_width: int,
    act_bit_width: int,
    output_bit_width: int | None,
):
    _validate_bit_width("input_bit_width", input_bit_width)
    _validate_bit_width("weight_bit_width", weight_bit_width)
    _validate_bit_width("act_bit_width", act_bit_width)
    _validate_bit_width("output_bit_width", output_bit_width, allow_none=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_cfg = QuantConfig(
        input_bit_width=input_bit_width,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        output_bit_width=output_bit_width,
    )

    data = np.load(mat)
    M_real = torch.from_numpy(data["M_real"]).float().to(device)
    M_imag = torch.from_numpy(data["M_imag"]).float().to(device)
    time_indices = None
    if "time_indices" in data.files:
        time_indices = torch.from_numpy(data["time_indices"]).long().to(device)
    n_t_out = int(M_real.shape[0])
    if time_indices is not None and int(time_indices.numel()) != n_t_out:
        raise ValueError(
            f"time_indices length {int(time_indices.numel())} does not match M_real rows {n_t_out}"
        )

    print(f"[INFO] Quantization config: {quant_cfg.describe()}")
    if output_bit_width is None:
        print("[INFO] Output tensor is left unquantized, so exported output dtype may stay wide (e.g. accumulator INT24).")
    else:
        print(f"[INFO] Final output is explicitly quantized to signed INT{output_bit_width}.")

    model = DeepONetFinnDeployQuant(
        M_real=M_real,
        M_imag=M_imag,
        hidden=hidden,
        quant_cfg=quant_cfg,
    ).to(device)
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
            u_in, y_out = make_supervised_batch(A0[idx], AL_clean[idx], n_t_out, time_indices)
            pred = model(u_in)
            loss = ((pred - y_out) ** 2).mean()

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
    ap.add_argument("--qonnx_out", type=str, default="deeponet_u250_int4_qonnx.onnx")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dataset_samples", type=int, default=256)
    ap.add_argument("--force_rebuild_cache", action="store_true")
    ap.add_argument("--hidden", type=int, default=pm.branch_hidden)
    ap.add_argument("--input_bit_width", type=int, default=8)
    ap.add_argument("--weight_bit_width", type=int, default=8)
    ap.add_argument("--act_bit_width", type=int, default=8)
    ap.add_argument(
        "--output_bit_width",
        type=int,
        default=None,
        help="Optional signed output bit width. Set this if you want the exported graph output dtype to be smaller.",
    )
    args = ap.parse_args()
    main(
        args.mat,
        args.epochs,
        args.qonnx_out,
        args.lr,
        args.dataset_samples,
        args.force_rebuild_cache,
        args.hidden,
        args.input_bit_width,
        args.weight_bit_width,
        args.act_bit_width,
        args.output_bit_width,
    )
