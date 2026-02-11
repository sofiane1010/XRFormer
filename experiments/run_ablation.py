"""
Script for running experiments (pretraining) with different configurations.

"""

import subprocess
from datetime import datetime

# Define training configurations
configs = [
    # ----------------------------- impact of tokenizer mode -----------------------------
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 6,
    #     "mode": "ViT",
    #     "dim": 128,
    #     "num_patches": 128,
    #     "peak_prediction": True,
    #     "n_heads": 8,
    # },
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 6,
    #     "mode": "ViT",
    #     "dim": 128,
    #     "num_patches": 128,
    #     "peak_prediction": False,
    #     "n_heads": 8,
    # },
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 8,
    #     "mode": "ViT",
    #     "dim": 256,
    #     "num_patches": 128,
    #     "peak_prediction": False,
    #     "n_heads": 16,
    # },
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 8,
    #     "mode": "ViT",
    #     "dim": 256,
    #     "num_patches": 128,
    #     "peak_prediction": True,
    #     "n_heads": 16,
    # },
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 10,
    #     "mode": "ViT",
    #     "dim": 512,
    #     "num_patches": 128,
    #     "peak_prediction": False,
    #     "n_heads": 32,
    # },
    # {
    #     "tokenizer_mode": "multiscale_conv",
    #     "depth": 10,
    #     "mode": "ViT",
    #     "dim": 512,
    #     "num_patches": 128,
    #     "peak_prediction": True,
    #     "n_heads": 32,
    # },
]

# Loop over each configuration

for cfg in configs:
    size = "B" if cfg["depth"] == 6 else "L" if cfg["depth"] == 8 else "H"
    task = "MSM+PPP" if cfg["peak_prediction"] else "MSM"
    log_name = f"logs/XRFormer-{size}_{task}.txt"
    cmd = [
        "python3",
        "scripts/pretrain_ssl.py",
        "--tokenizer_mode",
        cfg["tokenizer_mode"],
        "--depth",
        str(cfg["depth"]),
        "--mode",
        cfg["mode"],
        "--dim",
        str(cfg["dim"]),
        "--num_patches",
        str(cfg["num_patches"]),
        "--n_heads",
        str(cfg["n_heads"]),
    ]
    if cfg["peak_prediction"]:
        cmd.append("--peak_prediction")

    print(f"[{datetime.now()}] Running: {' '.join(cmd)} | Logging to {log_name}")

    with open(log_name, "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"[{datetime.now()}] Finished: {log_name}")
