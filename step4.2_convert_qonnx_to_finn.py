import argparse

from finn_qonnx_utils import convert_qonnx_to_finn, derive_output_path


def main(qonnx_in: str, qonnx_ready_out: str | None, finn_out: str | None):
    if qonnx_ready_out is None:
        qonnx_ready_out = derive_output_path(qonnx_in, "_ready")
    if finn_out is None:
        finn_out = derive_output_path(qonnx_in, "_finn")

    convert_qonnx_to_finn(
        in_path=qonnx_in,
        out_path=finn_out,
        prepared_qonnx_path=qonnx_ready_out,
        verbose=True,
    )
    print(f"\n[OK] Saved FINN-ONNX -> {finn_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qonnx_in", type=str, default="deeponet_u250_int8_qonnx.onnx")
    ap.add_argument("--qonnx_ready_out", type=str, default=None)
    ap.add_argument("--finn_out", type=str, default=None)
    args = ap.parse_args()
    main(args.qonnx_in, args.qonnx_ready_out, args.finn_out)
