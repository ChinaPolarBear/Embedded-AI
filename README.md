# Embedded-AI
# pinn_physics_codex

Physics-driven DeepONet PINN for optical-fibre NLSE learning, SSFM data generation, evaluation, and a FINN-oriented FPGA deployment flow targeting Xilinx Alveo U250.

## Overview

This project trains a DeepONet-style model to predict the complex output waveform `A(L, t)` from an input waveform `A(0, t)`. The workflow combines:

- SSFM-generated clean labels
- AWGN-corrupted observations
- physics loss, initial-condition loss, and endpoint anchor loss
- a deployment path that freezes the trunk and exports a FINN-friendly quantized branch

The current repository includes:

- model training and checkpoint export
- SSFM waveform CSV generation
- trained-model evaluation scripts
- trunk freezing for deployment
- float deploy sanity checking
- Brevitas-based QONNX export
- QONNX cleanup / validation before FINN conversion
- verification I/O generation for FINN / board-side checking
- deployed-output comparison against saved reference tensors

## Main Files

### Training and model definition

- `pinn_physics_model.py`
  Main physics-driven DeepONet training script.

- `pinn_physics_model_demo.py`
  Demo-style variant of the same overall model and training idea.

### Data generation and evaluation

- `SSFM_Data_Generation.py`
  Generates waveform CSV files with the same waveform generator and SSFM logic used by training.

- `test_trained_model.py`
  Evaluates the trained model on randomly generated waveforms and plots results.

- `test_trained_model_csv.py`
  Evaluates the trained model from CSV waveform data.

- `test_GPU.py`
  Prints CUDA availability and PyTorch version information.

### Deployment pipeline

- `step1_export_trunk_matrices.py`
  Freezes the trained trunk into `M_real` and `M_imag` and saves them into `trunk_matrices.npz`.

- `step2_deploy_float_sanity.py`
  Verifies that the frozen-trunk deploy model matches the original float model before quantization.

- `step3_brevitas_qat_export_qonnx.py`
  Builds the FINN-oriented int4 deploy model with Brevitas, optionally runs a short QAT pass, and exports a true QONNX model.

- `step4.1_export_QONNX_ready_model.py`
  Runs cleanup, shape inference, datatype inference, and node validation to produce a FINN-ready QONNX model.

- `step4.2_convert_qonnx_to_finn.py`
  Runs the same preparation flow and then calls `ConvertQONNXtoFINN()` to generate FINN-ONNX.

- `step5_generate_verification_io.py`
  Generates one verification case for board-side checking: `input.npy` for the deploy branch input, `ssfm_output.npy` as the clean SSFM reference, and `verification_case.npz` with the full sample metadata.

- `step6_compare_deploy_output.py`
  Compares decoded runtime / board output against the saved SSFM reference, reports error metrics, and saves comparison figures.

- `finn_qonnx_utils.py`
  Shared helpers for QONNX export compatibility, graph cleanup, shape/datatype inference, and `Gemm` / `MatMul` validation.

## Generated Artifacts

- `hybrid_pinn_deeponet.pth`
  Trained checkpoint.

- `trunk_matrices.npz`
  Frozen trunk matrices exported by Step 1.

- `deeponet_u250_int4_qonnx.onnx`
  Raw QONNX export from Step 3. This is the input file for `step4.2_convert_qonnx_to_finn.py`.

- `deeponet_u250_int4_qonnx_ready.onnx`
  Cleaned and validated QONNX export from Step 4.1, or the `_ready` intermediate written by Step 4.2 before FINN conversion.

- `verification_io/input.npy`
  Deploy-model input tensor that can be copied into the FINN build directory or used for board-side testing.

- `verification_io/ssfm_output.npy`
  Clean SSFM reference output paired with `input.npy`.

- `verification_io/verification_case.npz`
  Full verification bundle containing the waveform, branch input, SSFM reference output, and metadata such as `t_grid`.

- `deeponet_u250_int4_qonnx_finn.onnx`
  FINN-ONNX model produced by Step 4.2 by default.

- `dataset_cache/`
  Cached training / test / Step 3 supervised datasets generated from SSFM to avoid rebuilding the same waveforms every run.

## Dependencies

The current `requirements.txt` covers the local Python-side workflow:

- `numpy`
- `matplotlib`
- `torch`
- `brevitas`
- `qonnx`
- `onnx`
- `onnxoptimizer`
- `onnxruntime`

Notes:

- Install PyTorch separately if you need a specific CUDA build.
- `onnxoptimizer` is recommended for Brevitas QONNX export.
- `step4.2_convert_qonnx_to_finn.py` additionally requires `finn`, which is not part of the local `requirements.txt`.

## Installation

### Local Python environment

```bash
pip install -r requirements.txt
python test_GPU.py
```

### FINN environment

`step4.2_convert_qonnx_to_finn.py` and later FINN flows should be run inside a FINN-supported Linux environment, typically the official FINN Docker setup.

Recommended practical setup:

- Windows host + WSL2 + Docker Desktop, then run FINN inside Docker
- or native Linux + Docker

## Recommended End-to-End Flow

### 1. Train the original model

```bash
python pinn_physics_model.py
```

The training script now caches the generated train/test datasets under `dataset_cache/` and reuses them on later runs if the waveform and SSFM settings still match.

### 2. Optional evaluation

```bash
python test_trained_model.py
python test_trained_model_csv.py --csv Data_Output/waveform_data_trainmatch.csv --phase-align
```

### 3. Freeze the trunk

```bash
python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz
```

Bring-up recommendation for a much smaller deployment model:

```bash
python step1_export_trunk_matrices.py \
  --ckpt hybrid_pinn_deeponet.pth \
  --out trunk_matrices_bringup.npz \
  --n_t_out 64 \
  --p_dim_out 16 \
  --time_slice center
```

### 4. Float deploy sanity check

```bash
python step2_deploy_float_sanity.py --ckpt hybrid_pinn_deeponet.pth --mat trunk_matrices.npz
```

### 5. Export raw QONNX

```bash
python step3_brevitas_qat_export_qonnx.py \
  --mat trunk_matrices.npz \
  --epochs 10 \
  --input_bit_width 4 \
  --weight_bit_width 4 \
  --act_bit_width 4 \
  --output_bit_width 4 \
  --qonnx_out deeponet_u250_int4_qonnx.onnx
```

For FINN bring-up, prefer a much smaller model first:

```bash
python step3_brevitas_qat_export_qonnx.py \
  --mat trunk_matrices_bringup.npz \
  --epochs 1 \
  --dataset_samples 32 \
  --hidden 32 \
  --input_bit_width 4 \
  --weight_bit_width 4 \
  --act_bit_width 4 \
  --output_bit_width 4 \
  --qonnx_out deeponet_u250_bringup_qonnx.onnx
```

If you only want to validate the FINN tool flow and not the model quality yet, `--epochs 0` is also acceptable.

Step 3 now tries to reuse the training cache first and only rebuilds a separate supervised dataset if that cache is missing. You can force a refresh with:

```bash
python step3_brevitas_qat_export_qonnx.py \
  --mat trunk_matrices.npz \
  --input_bit_width 4 \
  --weight_bit_width 4 \
  --act_bit_width 4 \
  --output_bit_width 4 \
  --force_rebuild_cache
```

What Step 3 now does:

- uses Brevitas `export_qonnx(...)` instead of plain `torch.onnx.export(...)`
- keeps quantization as QONNX `Quant` nodes instead of expanding it into many standard ONNX ops
- uses signed int4 input quantization
- uses unsigned int4 quantized `ReLU` activations
- can explicitly quantize the final output tensor to signed int4

### 6. Prepare FINN-ready QONNX

```bash
python step4.1_export_QONNX_ready_model.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```

This produces `deeponet_u250_int4_qonnx_ready.onnx` by default and performs:

- cleanup
- shape inference
- datatype inference
- `Gemm` / `MatMul` weight checks
- initializer / constant-like tensor checks

### 7. Convert QONNX to FINN-ONNX

Run this inside the FINN environment:

```bash
python step4.2_convert_qonnx_to_finn.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```

By default this script:

- takes the raw QONNX from Step 3 as input
- re-runs QONNX preparation
- saves a `_ready.onnx` intermediate
- calls `ConvertQONNXtoFINN()`
- saves the FINN-ONNX result

Important distinction:

- `deeponet_u250_int4_qonnx.onnx` is the raw Step 3 export and is the normal input to Step 4.2
- `deeponet_u250_int4_qonnx_ready.onnx` is the prepared intermediate generated by Step 4.1 or internally by Step 4.2
- run Step 4.1 first if you want to inspect or debug the prepared graph before FINN conversion

### Step 4.2 detailed usage

Default command:

```bash
python step4.2_convert_qonnx_to_finn.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```

This will usually produce:

- `deeponet_u250_int4_qonnx_ready.onnx`
- `deeponet_u250_int4_qonnx_finn.onnx`

If you want to control the filenames explicitly:

```bash
python step4.2_convert_qonnx_to_finn.py \
  --qonnx_in deeponet_u250_int4_qonnx.onnx \
  --qonnx_ready_out deeponet_u250_int4_ready_manual.onnx \
  --finn_out deeponet_u250_int4_finn_manual.onnx
```

What the script does before FINN conversion:

- loads the raw QONNX model from Step 3
- runs cleanup
- runs shape inference
- runs datatype inference
- checks `Gemm` and `MatMul` weight tensors
- verifies that linear weights are initializer-backed or QONNX constant-like tensors
- saves the prepared intermediate model

Recommended workflow while debugging:

1. Run `step4.1_export_QONNX_ready_model.py` first and inspect the printed graph summary.
2. Run `step4.2_convert_qonnx_to_finn.py`.
3. If FINN fails, inspect the `_ready.onnx` intermediate before trying `build_dataflow`.

Important notes:

- `step4.2_convert_qonnx_to_finn.py` should be run inside the FINN Linux / Docker environment.
- The current local Python environment is suitable for Steps 1 to 4.1, but not for the actual FINN conversion unless `finn` is installed there.
- The final FINN-ONNX file is the model you should feed into `build_dataflow`.

### 8. Run FINN `build_dataflow`

After Step 4.2 succeeds, the next stage is FINN dataflow compilation.

Recommended input to `build_dataflow`:

- `deeponet_u250_int4_qonnx_finn.onnx`

FINN simple dataflow mode expects a dedicated build directory containing:

- `model.onnx`
- `dataflow_build_config.json`
- optionally `folding_config.json`
- optionally `specialize_layers_config.json`

Example directory layout:

```text
my_build/
  model.onnx
  dataflow_build_config.json
  folding_config.json                 # optional
  specialize_layers_config.json       # optional
```

Copy the FINN-ONNX model and rename it exactly to `model.onnx`:

```bash
cd /home/xband/finn/qonnx_models_new
cp deeponet_u250_int4_qonnx_finn.onnx /home/xband/finn/my_build/model.onnx

```

#### Minimal terminal workflow

运行的时候确保是在finn/my_build的路径下运行，尤其是修改json文件内容的时候
Inside the FINN environment, the practical workflow used in this project is:

Before running `run-docker.sh`, export the Xilinx-related environment variables in the same shell on the host side:

```bash
export FINN_XILINX_PATH=/tools/Xilinx22_Full
export FINN_XILINX_VERSION=2022.2
export HLS_PATH=/tools/Xilinx22_Full/Vitis_HLS/2022.2
export VIVADO_PATH=/tools/Xilinx22_Full/Vivado/2022.2
export VITIS_PATH=/tools/Xilinx22_Full/Vitis/2022.2
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
#export NUM_DEFAULT_WORKERS=1 (add it if the system is shutted down)
```

Then check them:

```bash
echo $FINN_XILINX_PATH
echo $FINN_XILINX_VERSION
echo $HLS_PATH
echo $VIVADO_PATH
echo $VITIS_PATH
echo $PLATFORM_REPO_PATHS
ls $HLS_PATH
find /tools/Xilinx22_Full -name "*.xpfm" | head
```

Notes:

- `PLATFORM_REPO_PATHS` must point to the directory that contains the Vitis platform files for your Alveo target
- if `/tools/Xilinx22_Full/platforms` does not exist on your machine, use `find /tools/Xilinx22_Full -name "*.xpfm"` first and then set `PLATFORM_REPO_PATHS` to the directory that contains the relevant U250 platform files
- if you do not have separate `Vitis` or `Vivado` directories in your installation, adjust `VITIS_PATH` and `VIVADO_PATH` to match your actual tool installation layout
- the exports must be done before calling `./run-docker.sh build_dataflow ...`

```bash
cd /home/xband/finn
./run-docker.sh build_dataflow /home/xband/finn/my_build
```

#### How to edit `dataflow_build_config.json` in terminal

Open the config file inside `my_build`:

```bash
cd /home/xband/finn/my_build
nano dataflow_build_config.json
```

Paste the JSON content into `nano`, then save and exit:

- press `Ctrl+O` to write the file
- press `Enter` to confirm the filename
- press `Ctrl+X` to exit

After saving, you can verify the contents:

```bash
cat dataflow_build_config.json
```

Then run the build from the FINN root directory:

```json
{
  "output_dir": "output_u250_bitfile",
  "synth_clk_period_ns": 5.0,
  "fpga_part": "xcu250-figd2104-2L-e",
  "shell_flow_type": "vitis_alveo",
  "generate_outputs": [
    "estimate_reports",
    "stitched_ip",
    "out_of_context_synth",
    "bitfile",
    "pynq_driver",
    "deployment_package"
  ],
  "save_intermediate_models": true
}
```

```bash
cd /home/xband/finn
./run-docker.sh build_dataflow /home/xband/finn/my_build
```

Important:

- `dataflow_build_config.json` must contain exactly one valid JSON object
- do not place two `{ ... }` blocks in the same file
- standard JSON does not allow comments
- if you change pass strategy later, overwrite the file contents instead of appending another JSON block

For this project and FINN setup, using `"fpga_part": "xcu250-figd2104-2L-e"` is more reliable than using `"board": "U250"` in the build config. Some FINN environments fail to resolve the board name during `step_specialize_layers`.

If `vitis_hls` is not found, export the tool variables again in the same shell before rerunning:

```bash
export FINN_XILINX_PATH=/tools/Xilinx22_Full
export FINN_XILINX_VERSION=2022.2
export HLS_PATH=/tools/Xilinx22_Full/Vitis_HLS/2022.2
export VIVADO_PATH=/tools/Xilinx22_Full/Vivado/2022.2
export VITIS_PATH=/tools/Xilinx22_Full/Vitis/2022.2
export PLATFORM_REPO_PATHS=/tools/Xilinx22_Full/platforms
cd /home/xband/finn
./run-docker.sh build_dataflow /home/xband/finn/my_build
```

#### Outputs to inspect after `build_dataflow`

FINN will place outputs under the `output_dir` specified in the JSON config. The most useful ones are:

- `build_dataflow.log`
- `time_per_step.json`
- `final_hw_config.json`
- `intermediate_models/`
- `report/estimate_network_performance.json`
- `report/estimate_layer_resources.json`
- `report/ooc_synth_and_timing.json`
- `report/rtlsim_performance.json`
- `deploy/` for deployment packaging

Important:

- if you are not using FINN's own deploy-model `verify_steps`, do not include `verify_steps`
- also remove `verify_input_npy` and `verify_expected_output_npy` from `dataflow_build_config.json`
- otherwise the build can fail simply because those files do not exist yet

### 9. Generate verification input/output

This is the missing step if you want a board-side validation case that starts from one known waveform, exports the matching deploy input, and keeps an SSFM reference for later comparison.

This step only needs the normal local Python environment. It does not execute the exported ONNX model and does not require `qonnx`.

```bash
python step5_generate_verification_io.py \
  --out_dir verification_io \
  --source random \
  --seed 123
```

This will generate:

- `verification_io/input.npy`
- `verification_io/ssfm_output.npy`
- `verification_io/verification_case.npz`

Recommended practical use:

- feed `input.npy` to the board/runtime path as the deploy-model branch input
- keep `ssfm_output.npy` and `verification_case.npz` on the host side as the clean reference bundle
- if the board host later requires a quantized/raw input representation, add that board-specific conversion in the host/runtime layer rather than changing the reference case generation

### 10. Compare board/runtime output against the reference

After running the model on hardware or through the final runtime stack, save the decoded returned tensor as `board_output.npy` and compare it with:

```bash
python step6_compare_deploy_output.py \
  --case verification_io/verification_case.npz \
  --actual board_output.npy
```

The script reports:

- max absolute error
- mean absolute error
- RMSE
- relative RMSE
- complex-domain RMSE
- amplitude-domain MAE / RMSE
- pass/fail against configurable `atol` / `rtol`
- and it also saves amplitude, constellation, and error-summary figures

Important:

- `board_output.npy` must already be decoded into `[B,2,N_t]`, `[B,N_t,2]`, `[2,N_t]`, `[N_t,2]`, or complex waveform form
- raw board-specific integer dumps such as `(1, 256) int32` are not self-describing enough for this script and must be decoded first by the board host code

This is the point where you can honestly say the deployed FPGA path has been functionally checked against the SSFM reference, rather than only synthesized and packaged.

#### Practical warning for this project

This DeepONet-style graph is less standard than FINN's small end-to-end tutorial networks. Because of that, the simple `build_dataflow` path may still need extra tuning or a custom build script even after Step 4.2 succeeds.

That is an informed expectation based on the topology difference from common FINN examples, not a confirmed failure.

## Notes on Step 4.1 vs Step 4.2

- `step4.1_export_QONNX_ready_model.py` is useful when you want to inspect the cleaned graph before entering FINN.
- `step4.2_convert_qonnx_to_finn.py` already includes the same preparation pass, so Step 4.1 is not strictly required.
- In practice, it is still a good idea to run Step 4.1 first when debugging a graph-conversion issue.

## Current Status of the Export Flow

The updated flow is designed to avoid the earlier problems where:

- quantization was expanded into standard ONNX nodes such as `Clip`, `Where`, `Abs`, `Div`, and `Mul`
- `GemmToMatMul()` failed because weight tensor shapes were not inferred correctly
- downstream FINN passes saw non-constant-looking linear weights

The new QONNX preparation step explicitly validates:

- `Gemm` weight tensor rank
- bias initializer presence
- whether `Gemm` / `MatMul` weights are initializer-backed or QONNX constant-like tensors

## Quick Commands

```bash
python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz
python step2_deploy_float_sanity.py --ckpt hybrid_pinn_deeponet.pth --mat trunk_matrices.npz
python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices.npz --epochs 10 --input_bit_width 4 --weight_bit_width 4 --act_bit_width 4 --output_bit_width 4 --qonnx_out deeponet_u250_int4_qonnx.onnx
python step4.1_export_QONNX_ready_model.py --qonnx_in deeponet_u250_int4_qonnx.onnx
# Step 4.2 normally still takes the raw Step 3 QONNX as input and writes a _ready intermediate itself.
python step4.2_convert_qonnx_to_finn.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```
