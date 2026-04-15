# ----------------- Imports ---------------------------

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import numpy as np
import argparse
import os

from xrf.models import CNNClassifier1D, XRFClassifier
from xrf.dataset import SpectraDataset
from xrf.utils import (
    train_downstream,
    val_downstream,
    get_optimal_thresholds,
    test_downstream,
    downstream_metrics,
)

parser = argparse.ArgumentParser(description="Train downstream models")
parser.add_argument(
    "--downstream_task",
    type=str,
    default="identification",
    help="'identification' / 'unmixing'",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=5,
    help="Number of runs for averaging results (default: 5)",
)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--patience", type=int, default=60)
parser.add_argument("--num_patches", type=int, default=128)
parser.add_argument("--size", type=str, default="B", help="Model size: 'B' / 'L'")
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--peak_prediction", action="store_true")
parser.add_argument(
    "--model",
    type=str,
    default="XRFormer",
    help="Model type: 'XRFormer', 'ViT', 'SF', 'SF_no-CAF', 'CNN'",
)
parser.add_argument("--dropout", type=float, default=0.1)

args = parser.parse_args()

os.makedirs(f"models/downstream/{args.downstream_task}", exist_ok=True)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.makedirs(f"models/downstream/{args.downstream_task}", exist_ok=True)

# ----------------- print model information ---------------------------

pretrained_str = (
    "" if not args.pretrained else "(MSM)" if not args.peak_prediction else "(MSM+PPP)"
)
size_str = "" if not args.model == "XRFormer" else f"-{args.size}"
print("######################################################")
print("######################################################")
print(f"Model: {args.model}{size_str} {pretrained_str}")
print(f"Downstream task: {args.downstream_task}")
print(f"Number of runs: {args.num_runs}")
print("######################################################")
print("######################################################")


# ----------------- Constants ---------------------------

DEVICE = torch.device(args.device)
NEAR_BANDS = None
if not args.model == "CNN":
    if args.size == "B":
        DEPTH = 6
        DIM = 128
        N_HEADS = 8
    elif args.size == "L":
        DEPTH = 8
        DIM = 256
        N_HEADS = 16
    else:
        raise ValueError("Invalid model size. Choose from 'B' or 'L'.")
    if "SF" in args.model:
        TOKENIZER_MODE = "GSE"
        MODE = "ViT" if "no-CAF" in args.model else "CAF"
        NEAR_BANDS = 7
    elif args.model == "ViT":
        TOKENIZER_MODE = "linear"
        MODE = "ViT"
    elif args.model == "XRFormer":
        TOKENIZER_MODE = "multiscale_conv"
        MODE = "ViT"
    else:
        raise ValueError(
            "Invalid model type. Choose from 'XRFormer', 'ViT', 'SpectralFormer', 'CNN'."
        )


# ----------------- Data ---------------------------

dataset = SpectraDataset(
    "data/checker_v5_mixed_xrf_spectra.npz",
    mode=args.downstream_task,
    near_band=NEAR_BANDS,
    peak_prediction=args.peak_prediction,
)
train_size, test_size, val_size = (
    int(len(dataset) * 0.6),
    int(len(dataset) * 0.2),
    int(len(dataset) * 0.2 + 1),
)
train_dataset, test_dataset, val_dataset = random_split(
    dataset,
    [train_size, test_size, val_size],
    generator=torch.Generator().manual_seed(args.seed),
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


# ----------------- Model ---------------------------

if args.downstream_task == "identification":
    aa, ha, f1_score = [], [], []
elif args.downstream_task == "unmixing":
    r_rmses, a_rmses, sam_scores = [], [], []
else:
    raise ValueError(
        "Invalid downstream task. Choose from 'identification' or 'unmixing'."
    )


for i in range(args.num_runs):
    if args.model == "CNN":
        model = CNNClassifier1D(
            n_classes=dataset.components.shape[-1],
            dropout=args.dropout,
            downstream_task=args.downstream_task,
        ).to(DEVICE)
    else:
        model = XRFClassifier(
            spectral_bands=dataset.spectra.shape[-1],
            num_patches=args.num_patches,
            dim=DIM,
            heads=N_HEADS,
            dim_head=DIM // N_HEADS,
            depth=DEPTH,
            mlp_dim=DIM * 4,
            dropout=args.dropout,
            emb_dropout=args.dropout,
            mode=MODE,
            tokenizer_mode=TOKENIZER_MODE,
            downstream_task=args.downstream_task,
            n_classes=dataset.components.shape[-1],
            peak_prediction=args.peak_prediction,
            near_bands=NEAR_BANDS,
        ).to(DEVICE)

        if args.pretrained:
            if args.model == "XRFormer":
                task = "MSM+PPP" if args.peak_prediction else "MSM"
                model.vit.load_state_dict(
                    torch.load(f"models/pretrained/XRFormer-{args.size}_{task}.pth")
                )
            else:
                raise ValueError("Pretraining only available for XRFormer model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=20,
        min_lr=1e-7,
    )

    # ----------------- Training ---------------------------

    if args.model == "XRFormer":
        task = ""
        if args.pretrained:
            task = "_MSM+PPP" if args.peak_prediction else "_MSM"
        best_model_path = f"models/downstream/{args.downstream_task}/{args.model}-{args.size}{task}_v5.pth"
    else:
        best_model_path = (
            f"models/downstream/{args.downstream_task}/{args.model}_v5.pth"
        )

    BEST_VAL_LOSS = float("inf")
    patience = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0

        train_loss = train_downstream(
            model, train_loader, DEVICE, args.downstream_task, optimizer
        )
        val_loss = val_downstream(model, val_loader, DEVICE, args.downstream_task)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < BEST_VAL_LOSS:
            patience = 0
            BEST_VAL_LOSS = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated.")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    # -------------------------- Testing --------------------------

    print(f"Reloading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))

    thresholds = None
    if args.downstream_task == "identification":
        print("Calculating optimal thresholds on validation set...")
        thresholds = get_optimal_thresholds(model, val_loader, DEVICE)

    preds, labels = test_downstream(
        model, test_loader, DEVICE, args.downstream_task, thresholds
    )
    endmembers = dataset.endmembers if args.downstream_task == "unmixing" else None
    metrics = downstream_metrics(preds, labels, args.downstream_task, endmembers)

    print("-------------------------------------------")
    print(f"------------- Run num: {i+1} -------------")
    print("-------------------------------------------")
    if args.downstream_task == "identification":
        print(f"Mean per-class f1: {metrics['f1_score']:.4f}")
        print(f"overall accuracy: {metrics['absolute_accuracy']:.4f}")
        print(f"hamming accuracy: {metrics['hamming_accuracy']:.4f}")
        aa.append(metrics["absolute_accuracy"])
        ha.append(metrics["hamming_accuracy"])
        f1_score.append(metrics["f1_score"])

    elif args.downstream_task == "unmixing":
        print(f"Abundance RMSE: {metrics['abundance_rmse']:.5f}")
        print(f"Reconstruction RMSE: {metrics['reconstruction_rmse']:.5f}")
        print(f"SAM (radians): {metrics['sam_score']:.5f}")
        a_rmses.append(metrics["abundance_rmse"])
        r_rmses.append(metrics["reconstruction_rmse"])
        sam_scores.append(metrics["sam_score"])

print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")

if args.downstream_task == "identification":
    aa = np.array(aa)
    ha = np.array(ha)
    f1_score = np.array(f1_score)
    print(f"aa : {aa.mean()} +- {aa.std()}")
    print(f"ha : {ha.mean()} +- {ha.std()}")
    print(f"f1: {f1_score.mean()} +- {f1_score.std()}")

elif args.downstream_task == "unmixing":
    a_rmses = np.array(a_rmses)
    r_rmses = np.array(r_rmses)
    sam_scores = np.array(sam_scores)
    print(f"a_rmse : {a_rmses.mean()} +- {a_rmses.std()}")
    print(f"r_rmse : {r_rmses.mean()} +- {r_rmses.std()}")
    print(f"SAM: {sam_scores.mean()} +- {sam_scores.std()}")

###
