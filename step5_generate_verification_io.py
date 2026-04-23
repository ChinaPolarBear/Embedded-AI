"""
Step 5: Generate one board-side verification case using an SSFM reference.

Typical usage:
    python step5_generate_verification_io.py \
        --out_dir verification_io \
        --source random \
        --seed 123

This script:
  1) Selects one waveform sample from the existing project data path.
  2) Converts it into the deploy-model branch input `u_in`.
  3) Uses SSFM to obtain the clean reference output at z=L.
  4) Saves:
       - input.npy
       - ssfm_output.npy
       - ssfm_output_probe.npy
       - verification_case.npz

Notes:
  - `input.npy` is the compact float32 complex input expected by the exported probe model:
    first 128 values are real(A(0,t_probe)), last 128 values are imag(A(0,t_probe)).
  - `ssfm_output_probe.npy` matches the compact deploy output layout:
    first 128 values are real(A(L,t_probe)), last 128 values are imag(A(L,t_probe)).
  - If your board runtime later expects a quantized/raw format, keep this script as the
    source of truth for the waveform and add the board-specific conversion in the host code.
"""
# python step5_generate_verification_io.py --out_dir verification_io --source random --seed 123

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import pinn_physics_model as pm


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_sample(source: str, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")

    if source == "random":
        A0, _, _ = pm.generate_qam_waveform()
        AL_clean = pm.ssfm_propagate(A0, pm.L)
        return A0.detach().cpu(), AL_clean.detach().cpu(), "random_ssfm"

    if source == "train":
        cached = pm.load_dataset_cache(pm.N_train, pm.snr_db_train, cache_key=pm.TRAIN_DATASET_CACHE_KEY)
        if cached is not None:
            A0_all, AL_clean_all, _ = cached
            if sample_index >= A0_all.shape[0]:
                raise IndexError(
                    f"sample_index={sample_index} exceeds cached train dataset size {A0_all.shape[0]}"
                )
            return (
                A0_all[sample_index].detach().cpu(),
                AL_clean_all[sample_index].detach().cpu(),
                f"train_cache[{sample_index}]",
            )

        n_samples = sample_index + 1
        A0_all, AL_clean_all, _ = pm.build_dataset(
            n_samples=n_samples,
            snr_db=pm.snr_db_train,
            cache_key="verify_train_subset",
        )
        return (
            A0_all[sample_index].detach().cpu(),
            AL_clean_all[sample_index].detach().cpu(),
            f"train_subset[{sample_index}]",
        )

    if source == "test":
        cached = pm.load_dataset_cache(pm.N_test, pm.snr_db_eval, cache_key=pm.TEST_DATASET_CACHE_KEY)
        if cached is not None:
            A0_all, AL_clean_all, _ = cached
            if sample_index >= A0_all.shape[0]:
                raise IndexError(
                    f"sample_index={sample_index} exceeds cached test dataset size {A0_all.shape[0]}"
                )
            return (
                A0_all[sample_index].detach().cpu(),
                AL_clean_all[sample_index].detach().cpu(),
                f"test_cache[{sample_index}]",
            )

        n_samples = sample_index + 1
        A0_all, AL_clean_all, _ = pm.build_dataset(
            n_samples=n_samples,
            snr_db=pm.snr_db_eval,
            cache_key="verify_test_subset",
        )
        return (
            A0_all[sample_index].detach().cpu(),
            AL_clean_all[sample_index].detach().cpu(),
            f"test_subset[{sample_index}]",
        )

    raise ValueError(f"Unsupported source: {source}")


def _complex_to_two_channel(x: torch.Tensor) -> np.ndarray:
    x_np = x.detach().cpu().numpy()
    return np.stack([x_np.real.astype(np.float32), x_np.imag.astype(np.float32)], axis=0)


def _probe_indices() -> torch.Tensor:
    return torch.arange(0, pm.N_t, 2)[: pm.N_t // 2].long()


def main(out_dir: str, source: str, sample_index: int, seed: int | None) -> None:
    _set_seed(seed)

    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    A0, AL_clean, source_desc = _select_sample(source, sample_index)
    probe_indices = _probe_indices()

    with torch.no_grad():
        A0_probe = A0[probe_indices]
        u_in_t = torch.cat([A0_probe.real, A0_probe.imag], dim=0).unsqueeze(0).float()

    u_in = u_in_t.detach().cpu().numpy().astype(np.float32)
    ssfm_output = _complex_to_two_channel(AL_clean)[None, :, :]
    AL_probe = AL_clean[probe_indices]
    ssfm_output_probe = np.concatenate(
        [
            AL_probe.real.detach().cpu().numpy().astype(np.float32),
            AL_probe.imag.detach().cpu().numpy().astype(np.float32),
        ]
    )[None, :]
    t_grid = pm.t_grid.detach().cpu().numpy().astype(np.float32)
    probe_t_grid = t_grid[probe_indices.detach().cpu().numpy()]

    np.save(out_dir_path / "input.npy", u_in)
    np.save(out_dir_path / "ssfm_output.npy", ssfm_output)
    np.save(out_dir_path / "ssfm_output_probe.npy", ssfm_output_probe)
    np.savez(
        out_dir_path / "verification_case.npz",
        source=np.array([source_desc]),
        sample_index=np.array([sample_index], dtype=np.int32),
        seed=np.array([-1 if seed is None else seed], dtype=np.int64),
        input_format=np.array(["compact_real128_imag128_float32"]),
        output_format=np.array(["two_channel_real_imag_float32"]),
        deploy_output_format=np.array(["compact_real128_imag128_float32"]),
        t_grid=t_grid,
        deploy_t_grid=probe_t_grid,
        propagation_distance=np.array([pm.L], dtype=np.float32),
        u_in=u_in,
        ssfm_output=ssfm_output,
        ssfm_output_probe=ssfm_output_probe,
        expected_output=ssfm_output_probe,
        probe_indices=probe_indices.detach().cpu().numpy().astype(np.int32),
        A0_real=A0.real.detach().cpu().numpy().astype(np.float32),
        A0_imag=A0.imag.detach().cpu().numpy().astype(np.float32),
        AL_clean_real=AL_clean.real.detach().cpu().numpy().astype(np.float32),
        AL_clean_imag=AL_clean.imag.detach().cpu().numpy().astype(np.float32),
    )

    print(f"[OK] Generated verification tensors in: {out_dir_path}")
    print(f"     source        : {source_desc}")
    print(f"     seed          : {seed if seed is not None else 'none'}")
    print(f"     input.npy     : shape={tuple(u_in.shape)} dtype={u_in.dtype}")
    print(f"     ssfm_output   : shape={tuple(ssfm_output.shape)} dtype={ssfm_output.dtype}")
    print(f"     ssfm_probe    : shape={tuple(ssfm_output_probe.shape)} dtype={ssfm_output_probe.dtype}")
    print("     note          : input.npy is float32 deploy input; board-specific raw quantization")
    print("                     should be added later in the board host/runtime layer.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="verification_io")
    ap.add_argument("--source", type=str, choices=["train", "test", "random"], default="random")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Only affects --source random.",
    )
    args = ap.parse_args()
    main(
        out_dir=args.out_dir,
        source=args.source,
        sample_index=args.sample_index,
        seed=args.seed,
    )
