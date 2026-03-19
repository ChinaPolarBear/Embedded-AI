# python step4.1_export_QONNX_ready_model.py --qonnx_in deeponet_u250_int8_qonnx.onnx

import argparse

from finn_qonnx_utils import derive_output_path, prepare_qonnx_for_finn


def main(qonnx_in: str, qonnx_ready_out: str | None):
    if qonnx_ready_out is None:
        qonnx_ready_out = derive_output_path(qonnx_in, "_ready")

    prepare_qonnx_for_finn(
        in_path=qonnx_in,
        out_path=qonnx_ready_out,
        verbose=True,
    )
    print(f"\n[OK] Saved FINN-ready QONNX -> {qonnx_ready_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qonnx_in", type=str, default="deeponet_u250_int8_qonnx.onnx")
    ap.add_argument("--qonnx_ready_out", type=str, default=None)
    args = ap.parse_args()
    main(args.qonnx_in, args.qonnx_ready_out)
