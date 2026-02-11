import torch
from torch.utils.data import DataLoader, random_split
import argparse

from xrf import ViT, SpectraDataset, pretrain

parser = argparse.ArgumentParser(description="pretrain XRFormer with config")
parser.add_argument("--size", type=str, default="B", help="Model size: 'B' / 'L'")
parser.add_argument("--mode", type=str, default="ViT")  # 'ViT' or 'CAF'
parser.add_argument("--tokenizer_mode", type=str, default="multiscale_conv")
parser.add_argument("--num_patches", type=int, default=128)
parser.add_argument("--peak_prediction", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=400)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device(args.device)

# ----------------- print model information ---------------------------

pretrining_str = "(MSM)" if not args.peak_prediction else "(MSM+PPP)"
print("######################################################")
print("######################################################")
print(f"Model: XRFormer-{args.size}")
print(f"Pretraining task: {pretrining_str}")
print("######################################################")
print("######################################################")

# ----------------- Dataset -------------------------

dataset = SpectraDataset(
    "data/infraart_mixtures.npz", peak_prediction=args.peak_prediction
)
train_size, test_size, val_size = (
    int(len(dataset) * 0.9),
    int(len(dataset) * 0.05),
    int(len(dataset) * 0.05 + 2),
)
train_dataset, test_dataset, val_dataset = random_split(
    dataset,
    [train_size, test_size, val_size],
    generator=torch.Generator().manual_seed(args.seed),
)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
)

# ----------------- Model ---------------------------

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

model = ViT(
    spectral_bands=dataset.spectra.shape[-1],
    num_patches=args.num_patches,
    dim=DIM,
    heads=N_HEADS,
    dim_head=DIM // N_HEADS,
    depth=DEPTH,
    mlp_dim=DIM * 4,
    dropout=0.1,
    emb_dropout=0.1,
    ratio=0.15,
    mode=args.mode,
    tokenizer_mode=args.tokenizer_mode,
    peak_prediction=args.peak_prediction,
).to(DEVICE)

# ----------------- Optimizer -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.6,
    patience=10,
    cooldown=1,
    min_lr=1e-6,
)

# ----------------- Training Loop -------------------
BEST_VAL_LOSS = float("inf")
patience = 0
lambda_peak_pred = 5e-4
print("Training...")
for epoch in range(args.num_epochs):
    model.train()
    total_train_loss = 0.0

    avg_train_loss = pretrain(
        model,
        train_loader,
        DEVICE,
        optimizer,
        peak_prediction=args.peak_prediction,
        lambda_peak_pred=lambda_peak_pred,
    )

    print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.8f}")

    size = "B" if args.depth == 6 else "L" if args.depth == 8 else "H"
    task = "MSM+PPP" if args.peak_prediction else "MSM"

    torch.save(
        model.state_dict(),
        f"models/pretrained/XRFormer-{size}_{task}.pth",
    )
    print("Best model updated.")
