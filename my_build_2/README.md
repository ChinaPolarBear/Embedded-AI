# my_build_2

Second FINN build workspace for the higher-accuracy deployment variant.

Use this folder when you want to run a second `build_dataflow` job in parallel
with the main `my_build` directory on the lab computer.

Current variant target:

- `shape = (1,256)`
- `hidden = 64`
- `latent = 64` via `p_dim_out = 32`
- `epochs = 150`
- `dataset_samples = 2048`
- `input / weight = 4-bit`, `hidden activations = 8-bit`

Current status:

- `deeponet_u250_int4_qonnx.onnx` and `trunk_matrices.npz` already match the
  latent-64 variant
- `model.onnx`, the completed dataflow build, and the latest verification bundle
  now correspond to the latent-64 variant
- latest dequantized board + PyTorch comparison figures are under:
  `my_build_2/figures/dequantized/20260526_160038`
- see `variant_notes.txt` for the current batch metrics and timing summary

This folder now includes:

- `deeponet_u250_int4_qonnx.onnx`
  Raw QONNX export for Step 4.
- `step4_convert_qonnx_to_finn.py`
  Self-contained Step 4 script.
- `step5_generate_verification.py`
  Generate `verification/input.bin` and local reference tensors.
- `step6_compare_deploy_output.py`
  Plot and compare decoded board output against the SSFM reference.
- `../dequantize_board_output.py`
  Dequantize raw integer board output into `output_dequant.npy` using the scale inferred from the raw QONNX graph.
- `../run_finn_xrt_timed_1x256.cpp`
  Timing-enabled XRT host template for board-side runs. It prints latency stats and writes `board_inference_timing.txt`.
- `plot_output_utils.py`
  Shared plotting helper used by Step 6 and `pinn_physics_model.py`.
- `pinn_physics_model.py`
  Local dependency used by Step 5 for SSFM/reference generation.
- `trunk_matrices.npz`
  Local default trunk matrix file for Step 5.
- `dataflow_build_config.json`
  Independent build config for this variant.
- `parameter_tuning_workflow.txt`
  End-to-end handoff notes for teammates who will only modify `my_build_2`
  parameters without retraining a new `.pth` checkpoint.

Previous successful lighter version is archived under:

- `archive_hidden32_latent32_success/`
- `archive_shape64_hidden64_latent64_20260508/`

Suggested workflow:

1. Copy this whole folder to `~/finn/my_build_2`
2. Run Step 4 inside that folder to create `*_finn.onnx`
3. Rename the Step 4 output to `model.onnx`
4. Run:

```bash
cd ~/finn
./run-docker.sh build_dataflow /home/xband/finn/my_build_2
```

Local verification commands inside `my_build_2`:

```powershell
python step5_generate_verification.py --out_dir verification --source random --seed 123
python ..\dequantize_board_output.py --model deeponet_u250_int4_qonnx.onnx --cases_dir verification --input_name output_raw_int16_1x256.npy
python step6_compare_deploy_output.py --cases_dir verification --actual_name output.npy
python step6_compare_deploy_output.py --cases_dir verification --actual_name output_dequant.npy
python step6_compare_deploy_output.py --cases_dir verification --actual_name output_dequant.npy --include_pytorch
```

With `--include_pytorch`, Step 6 also runs the original PyTorch model on the
same verification cases, saves additional PyTorch-vs-SSFM figures, and writes
`pytorch_inference_timing.json` for timing comparison against the board run.

This folder is intentionally separate from `my_build` so both variants can be
built without overwriting each other's `output_dir`, logs, verification data,
or figures.
