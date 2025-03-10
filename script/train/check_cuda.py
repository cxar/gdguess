#!/usr/bin/env python3
import subprocess

import torch


def check_cuda():
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("number of cuda devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"device {i}:", torch.cuda.get_device_name(i))
            print("memory allocated (mb):", torch.cuda.memory_allocated(i) / 1024**2)
            print("memory reserved (mb):", torch.cuda.memory_reserved(i) / 1024**2)
    else:
        print("no cuda devices detected; using cpu")


def check_nvidia_smi():
    print("\nchecking nvidia-smi:")
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print(output)
    except Exception as e:
        print("nvidia-smi not available:", e)


if __name__ == "__main__":
    check_cuda()
    check_nvidia_smi()
