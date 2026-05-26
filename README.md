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

### Data generation and evaluation

- `SSFM_Data_Generation.py`
  Generates waveform CSV files with the same waveform generator and SSFM logic used by training.

- `test_trained_model.py`
  Evaluates the trained model on randomly generated waveforms, plots results, and prints/saves PyTorch inference timing.

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

- `step4_convert_qonnx_to_finn.py`
  Runs QONNX cleanup / validation internally and then calls `ConvertQONNXtoFINN()` to generate FINN-ONNX.
  The script is self-contained, so for lab-side Step 4 you only need this file and the raw QONNX model.

- `step5_generate_verification.py`
  Generates one or more verification cases for board-side checking: `input.npy` for the deploy branch input, `ssfm_output.npy` as the clean SSFM reference, and `verification_case.npz` with the full sample metadata.

- `step6_compare_deploy_output.py`
  Compares decoded runtime / board output against the saved SSFM reference, reports error metrics, and saves comparison figures.

- `dequantize_board_output.py`
  Decodes raw integer board output such as `output.bin` or `output_raw_int16_*.npy`, infers the output scale from the raw QONNX graph, and writes `output_dequant.npy`.

- `run_finn_xrt_timed_1x256.cpp`
  Timing-enabled XRT host program template for `(1,256)` FINN deployments. It prints board-side end-to-end latency and writes `board_inference_timing.txt`.

- `finn_qonnx_utils.py`
  Shared helpers used by local export / debugging utilities. Step 4 itself is now self-contained and does not need this file on the lab computer.

## Generated Artifacts

- `hybrid_pinn_deeponet.pth`
  Trained checkpoint.

- `trunk_matrices.npz`
  Frozen trunk matrices exported by Step 1.

- `deeponet_u250_int4_qonnx.onnx`
  Raw QONNX export from Step 3. This is the input file for `step4_convert_qonnx_to_finn.py`.

- `verification/input.npy`
  Deploy-model input tensor that can be copied into the FINN build directory or used for board-side testing.

- `verification/input.bin`
  Board-ready raw input generated from the same sample as `verification/input.npy`.

- `verification/ssfm_output.npy`
  Clean SSFM reference output paired with `input.npy`.

- `verification/ssfm_output_probe.npy`
  Compact complex SSFM reference with shape `[1,64]`: first 32 values are real, last 32 values are imag.

- `verification/verification_case.npz`
  Full verification bundle containing the waveform, branch input, SSFM reference output, and metadata such as `t_grid`.

- `deeponet_u250_int4_qonnx_finn.onnx`
  FINN-ONNX model produced by Step 4 by default.

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
- `step4_convert_qonnx_to_finn.py` additionally requires `finn`, which is not part of the local `requirements.txt`.

## Installation

### Local Python environment

```powershell
pip install -r requirements.txt
python test_GPU.py
```

### FINN environment

`step4_convert_qonnx_to_finn.py` and later FINN flows should be run inside a FINN-supported Linux environment, typically the official FINN Docker setup.

Recommended practical setup:

- Windows host + WSL2 + Docker Desktop, then run FINN inside Docker
- or native Linux + Docker

## Recommended End-to-End Flow

### 1. Train the original model

```powershell
python pinn_physics_model.py
```

The training script now caches the generated train/test datasets under `dataset_cache/` and reuses them on later runs if the waveform and SSFM settings still match.

### 2. Optional evaluation

```powershell
python test_trained_model.py
python test_trained_model_csv.py --csv Data_Output/waveform_data_trainmatch.csv --phase-align
```

`test_trained_model.py` now also prints PyTorch inference latency for one full
waveform prediction `A(0,t) -> A(L,t)` and saves the numbers into
`pytorch_inference_timing.json` inside its figure output directory.

If you also want Step 6 to run the original PyTorch model on the same
verification case(s), save PyTorch-only comparison figures, and benchmark the
PyTorch prediction time on the exact same probe points, add:

```powershell
--include_pytorch
```

### 3. Freeze the trunk

```powershell
python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz
```

Next recommended mainline experiment for a `(1,256)` compact-complex interface:

```powershell
python step1_export_trunk_matrices.py --ckpt hybrid_pinn_deeponet.pth --out trunk_matrices.npz --n_t_out 128 --p_dim_out 16 --time_slice uniform
```

### 4. Float deploy sanity check

```powershell
python step2_deploy_float_sanity.py --ckpt hybrid_pinn_deeponet.pth --mat trunk_matrices.npz
```

### 5. Export raw QONNX

Recommended mainline local export for the next `(1,256)` experiment:

```powershell
python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices.npz --epochs 150 --dataset_samples 2048 --lr 2e-4 --hidden 64 --input_bit_width 4 --weight_bit_width 4 --act_bit_width 4 --qonnx_out deeponet_u250_int4_qonnx.onnx
```

Why this combination:

- `n_t_out = 128` gives the desired `(1,256)` board-side waveform view
- `p_dim_out = 16` keeps `latent = 32`, which is still the safest known latent size from the successful build path
- `hidden = 64` keeps the successful mainline branch width instead of adding another routing risk
- `epochs = 150` and `dataset_samples = 2048` raise the training budget substantially without changing the deploy structure beyond the larger probe shape
- final output is left unquantized on purpose, so Step 6 can compare a cleaner float-domain output after dequantization

For FINN bring-up, prefer a much smaller model first:

```powershell
python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices_bringup.npz --epochs 1 --dataset_samples 32 --hidden 32 --input_bit_width 4 --weight_bit_width 4 --act_bit_width 4 --output_bit_width 4 --qonnx_out deeponet_u250_bringup_qonnx.onnx
```

If you only want to validate the FINN tool flow and not the model quality yet, `--epochs 0` is also acceptable.

Step 3 now tries to reuse the training cache first and only rebuilds a separate supervised dataset if that cache is missing. You can force a refresh with:

```powershell
python step3_brevitas_qat_export_qonnx.py --mat trunk_matrices.npz --input_bit_width 4 --weight_bit_width 4 --act_bit_width 4 --output_bit_width 4 --force_rebuild_cache
```

What Step 3 now does:

- uses Brevitas `export_qonnx(...)` instead of plain `torch.onnx.export(...)`
- keeps quantization as QONNX `Quant` nodes instead of expanding it into many standard ONNX ops
- uses signed int4 input quantization
- uses unsigned int4 quantized `ReLU` activations
- can explicitly quantize the final output tensor to signed int4
- uses a board-probe deploy interface with input `[B,64]`: first 32 values are real(A(0,t_probe)), last 32 values are imag(A(0,t_probe))
- exports one `[B,64]` tensor: first 32 values are real(A(L,t_probe)), last 32 values are imag(A(L,t_probe))

### 6. Convert QONNX to FINN-ONNX (Step 4)

Run this inside the FINN environment:

```bash
python step4_convert_qonnx_to_finn.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```

By default this script:

- takes the raw QONNX from Step 3 as input
- re-runs QONNX preparation
- calls `ConvertQONNXtoFINN()`
- saves the FINN-ONNX result

Important distinction:

- `deeponet_u250_int4_qonnx.onnx` is the raw Step 3 export and is the normal input to Step 4
- Step 4 still runs the prepared-graph cleanup and validation internally, but it no longer writes a separate `_ready.onnx` file

### Step 4 detailed usage

Default command:

```bash
python step4_convert_qonnx_to_finn.py --qonnx_in deeponet_u250_int4_qonnx.onnx
```

This will usually produce:

- `deeponet_u250_int4_qonnx_finn.onnx`

If you want to control the filenames explicitly:

```bash
python step4_convert_qonnx_to_finn.py \
  --qonnx_in deeponet_u250_int4_qonnx.onnx \
  --finn_out deeponet_u250_int4_finn_manual.onnx
```

What the script does before FINN conversion:

- loads the raw QONNX model from Step 3
- runs cleanup
- runs shape inference
- runs datatype inference
- checks `Gemm` and `MatMul` weight tensors
- verifies that linear weights are initializer-backed or QONNX constant-like tensors

Recommended workflow while debugging:

1. Run `step4_convert_qonnx_to_finn.py`.
2. Inspect the printed Raw / Prepared / FINN-ONNX graph summaries.
3. If FINN fails, use the printed Prepared graph summary before trying `build_dataflow`.

Important notes:

- `step4_convert_qonnx_to_finn.py` should be run inside the FINN Linux / Docker environment.
- For lab-side Step 4, copy only `deeponet_u250_int4_qonnx.onnx` and `step4_convert_qonnx_to_finn.py`.
- The current local Python environment is suitable for Steps 1 to 3, but not for the actual FINN conversion unless `finn` is installed there.
- The final FINN-ONNX file is the model you should feed into `build_dataflow`.

### 7. Run FINN `build_dataflow`

After Step 4 succeeds, the next stage is FINN dataflow compilation.

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

#### Minimal terminal workflow


Inside the FINN environment, the practical workflow used in this project is:

```bash
cd /home/xband/finn/my_build
```

Before running `run-docker.sh`, export the Xilinx-related environment variables in the same shell on the host side:

```bash
export FINN_XILINX_PATH=/opt/Xilinx
export FINN_XILINX_VERSION=2022.2
export VIVADO_PATH=/opt/Xilinx/Vivado/2022.2
export VITIS_PATH=/opt/Xilinx/Vitis/2022.2
export HLS_PATH=/opt/Xilinx/Vitis_HLS/2022.2
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
#export NUM_DEFAULT_WORKERS=1 (add it if the system is shutted down)

source /opt/Xilinx/Vivado/2022.2/settings64.sh
source /opt/Xilinx/Vitis/2022.2/settings64.sh
source /opt/Xilinx/Vitis_HLS/2022.2/settings64.sh
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
find /opt/Xilinx -name "*.xpfm" | head
```

Notes:

- `PLATFORM_REPO_PATHS` must point to the directory that contains the Vitis platform files for your Alveo target
- if `/opt/xilinx/platforms` does not exist on your machine, use `find /opt/Xilinx -name "*.xpfm"` first and then set `PLATFORM_REPO_PATHS` to the directory that contains the relevant U250 platform files
- after exporting the variables, also run the three `source .../settings64.sh` commands in the same shell so Vivado, Vitis, and Vitis HLS are fully loaded
- the exports must be done before calling `./run-docker.sh build_dataflow ...`

```bash
cd ~/finn

mkdir -p ~/finn_tmp

unset FINN_BUILD_DIR
unset FINN_HOST_BUILD_DIR

export FINN_BUILD_DIR=/home/xband/finn_tmp
export FINN_HOST_BUILD_DIR=/home/xband/finn_tmp

echo $FINN_BUILD_DIR
echo $FINN_HOST_BUILD_DIR

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
  "synth_clk_period_ns": 10.0,
  "fpga_part": "xcu250-figd2104-2L-e",
  "shell_flow_type": "vitis_alveo",
  "vitis_platform": "xilinx_u250_gen3x16_xdma_4_1_202210_1",
  "generate_outputs": [
    "estimate_reports",
    "stitched_ip",
    "rtlsim_performance",
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
mkdir -p /home/xband/finn_tmp
unset FINN_BUILD_DIR
unset FINN_HOST_BUILD_DIR
export FINN_BUILD_DIR=/home/xband/finn_tmp
export FINN_HOST_BUILD_DIR=/home/xband/finn_tmp
echo $FINN_BUILD_DIR
echo $FINN_HOST_BUILD_DIR
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
export FINN_XILINX_PATH=/opt/Xilinx
export FINN_XILINX_VERSION=2022.2
export VIVADO_PATH=/opt/Xilinx/Vivado/2022.2
export VITIS_PATH=/opt/Xilinx/Vitis/2022.2
export HLS_PATH=/opt/Xilinx/Vitis_HLS/2022.2
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
source /opt/Xilinx/Vivado/2022.2/settings64.sh
source /opt/Xilinx/Vitis/2022.2/settings64.sh
source /opt/Xilinx/Vitis_HLS/2022.2/settings64.sh
cd /home/xband/finn
mkdir -p /home/xband/finn_tmp
unset FINN_BUILD_DIR
unset FINN_HOST_BUILD_DIR
export FINN_BUILD_DIR=/home/xband/finn_tmp
export FINN_HOST_BUILD_DIR=/home/xband/finn_tmp
echo $FINN_BUILD_DIR
echo $FINN_HOST_BUILD_DIR
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

### 8. Generate verification input/output

This is the missing step if you want a board-side validation case that starts from one known waveform, exports the matching deploy input, and keeps an SSFM reference for later comparison.

This step only needs the normal local Python environment. It does not execute the exported ONNX model and does not require `qonnx`.

```powershell
python step5_generate_verification.py --out_dir verification --source random --seed 123
```

This will generate:

- `verification/input.npy`
- `verification/input_int4.npy`
- `verification/input.bin`
- `verification/ssfm_output.npy`
- `verification/ssfm_output_probe.npy`
- `verification/verification_case.npz`

If you want multiple independent cases while keeping the folder clean:

```powershell
python step5_generate_verification.py --out_dir verification --source random --seed 123 --num_cases 5
```

That mode creates:

- `verification/case_000/`
- `verification/case_001/`
- `verification/case_002/`
- ...
- `verification/cases_manifest.json`

Each `case_xxx` subdirectory contains its own `input.bin`, `input.npy`, `input_int4.npy`, `ssfm_output.npy`, `ssfm_output_probe.npy`, and `verification_case.npz`.

Recommended practical use:

- copy `verification/input.bin` to the board/runtime path as the deploy-model input
- keep `verification/input.npy`, `verification/ssfm_output.npy`, and `verification/verification_case.npz` on the host side as the clean reference bundle
- `verification/input.bin` and `verification_case.npz` now always refer to the same single sample

### 9. Compare board/runtime output against the reference

After running the model on hardware or through the final runtime stack, save the decoded returned tensor as `board_output.npy` and compare it with:

```powershell
python step6_compare_deploy_output.py --case verification/verification_case.npz --actual board_output.npy
```

If you generated multiple verification cases with `step5_generate_verification.py --num_cases N`, and you place one decoded runtime output file into each `verification/case_xxx/` directory, you can compare all of them in one pass:

```powershell
python step6_compare_deploy_output.py --cases_dir verification --actual_name output.npy
```

If the returned board output is still a raw integer dump such as `output.bin` or `output_raw_int16_*.npy`, dequantize it first:

```powershell
python dequantize_board_output.py --model deeponet_u250_int4_qonnx.onnx --cases_dir verification --input_name output.bin
python step6_compare_deploy_output.py --cases_dir verification --actual_name output_dequant.npy
```

To generate the board-vs-SSFM figures and, on the same cases, additional
PyTorch-vs-SSFM figures plus `pytorch_inference_timing.json`, run:

```powershell
python step6_compare_deploy_output.py --cases_dir verification --actual_name output_dequant.npy --include_pytorch
```

That batch mode writes:

- one figure subdirectory per case
- `batch_summary.json`
- `batch_summary.csv`

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

- `board_output.npy` must already be decoded/dequantized into `[B,64]` float for the current compact-complex probe, or into one of the older complex forms `[B,2,N_t]`, `[B,N_t,2]`, flattened `[B,2*N_t]`, `[2,N_t]`, `[N_t,2]`, flattened `[2*N_t]`, or complex waveform form
- for the current board-probe deploy output, use `[1,64]`: `output[0,0:32]` is real(A(L,t_probe)) and `output[0,32:64]` is imag(A(L,t_probe))
- raw board-specific integer dumps such as `(1, 64) int16` or `(1, 64) int32` are not self-describing enough for this script and must be decoded/dequantized first, for example with `dequantize_board_output.py`

This is the point where you can honestly say the deployed FPGA path has been functionally checked against the SSFM reference, rather than only synthesized and packaged.

#### Practical warning for this project

This DeepONet-style graph is less standard than FINN's small end-to-end tutorial networks. Because of that, the simple `build_dataflow` path may still need extra tuning or a custom build script even after Step 4 succeeds.

That is an informed expectation based on the topology difference from common FINN examples, not a confirmed failure.

## Current Status of the Export Flow

The updated flow is designed to avoid the earlier problems where:

- quantization was expanded into standard ONNX nodes such as `Clip`, `Where`, `Abs`, `Div`, and `Mul`
- `GemmToMatMul()` failed because weight tensor shapes were not inferred correctly
- downstream FINN passes saw non-constant-looking linear weights

The new QONNX preparation step explicitly validates:

- `Gemm` weight tensor rank
- bias initializer presence
- whether `Gemm` / `MatMul` weights are initializer-backed or QONNX constant-like tensors

