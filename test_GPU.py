import sys

import torch


def main():
    print(f"python_executable: {sys.executable}")
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_version: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("selected_device: cpu")
        return

    device = torch.device("cuda")
    print(f"selected_device: {device}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")

    x = torch.randn(1024, 1024, device=device)
    y = torch.randn(1024, 1024, device=device)
    z = x @ y
    print(f"tensor_device: {z.device}")
    print(f"tensor_mean: {z.mean().item():.6f}")


if __name__ == "__main__":
    main()
