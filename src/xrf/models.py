import torch
import torch.nn as nn
from einops import repeat
from xrf.layers import Tokenizer, MaskedSpectrum, Transformer


class ViT(nn.Module):
    def __init__(
        self,
        spectral_bands=2048,
        num_patches=32,
        dim=256,
        heads=8,
        dim_head=32,
        depth=4,
        mlp_dim=512,
        ratio=0.5,
        dropout=0.0,
        emb_dropout=0.0,
        mode="ViT",
        tokenizer_mode="linear",
        peak_prediction=False,
        near_bands=None,
    ):
        super().__init__()

        self.num_patches = spectral_bands if tokenizer_mode == "GSE" else num_patches
        self.patch_size = (
            near_bands if tokenizer_mode == "GSE" else spectral_bands // num_patches
        )

        self.masked_spectrum = MaskedSpectrum(dim, ratio=ratio)
        self.tokenizer = Tokenizer(
            tokenizer_mode, dim, self.patch_size, self.num_patches
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, self.num_patches + 1, mode
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, self.patch_size)
        )
        self.peak_prediction = peak_prediction
        if self.peak_prediction:
            self.peak_prediction_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, self.num_patches)
            )

    def forward(self, x, downstream=False):
        # Tokenize: [B, L] -> [B, N, D]
        x = self.tokenizer(x)
        b = x.shape[0]

        # Apply masking
        if not downstream:
            x, mask = self.masked_spectrum(x)  # x_masked [B, N, D], mask = [B, N]

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        x = x + self.pos_embedding  # [B, N+1, D]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)  #  [B, N+1, D]

        if self.peak_prediction:
            peak_predictions = self.peak_prediction_head(x[:, 0])  # [B, 1, 1]

        if downstream:
            return x[:, 0]

        x = self.mlp_head(x)  # [B, N+1, P]

        if self.peak_prediction:
            return peak_predictions, x[:, 1:], mask

        return x[:, 1:], mask


class XRFClassifier(nn.Module):
    def __init__(
        self,
        spectral_bands=2048,
        num_patches=64,
        dim=128,
        heads=8,
        dim_head=128 // 8,
        depth=6,
        mlp_dim=128 * 2,
        dropout=0.1,
        emb_dropout=0.1,
        mode="ViT",
        tokenizer_mode="linear",
        downstream_task="identification",
        n_classes=22,
        peak_prediction=False,
        near_bands=None,
    ):
        super().__init__()

        self.vit = ViT(
            spectral_bands=spectral_bands,
            num_patches=num_patches,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            depth=depth,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=mode,
            tokenizer_mode=tokenizer_mode,
            peak_prediction=peak_prediction,
            near_bands=near_bands,
        )

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes),
            nn.Softmax(dim=1) if downstream_task == "unmixing" else nn.Identity(),
        )

    def forward(self, x):
        x = self.vit(x, downstream=True)
        return self.classifier_head(x)


class CNNClassifier1D(nn.Module):
    def __init__(self, n_classes=22, dropout=0.25, downstream_task="identification"):
        super().__init__()
        self.downstream_task = downstream_task
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, 5),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(64),
        )  # [B, 1, 2048] => [B, 64, 1022]
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(64),
        )  # [B, 64, 1022] => [B, 64, 510]
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(64),
        )  # [B, 64, 510] => [B, 64, 254]
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(64),
        )  # [B, 64, 254] => [B, 64, 126]
        self.block5 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(128, 128, 3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(16),
        )  # [B, 64, 126] => [B, 128, 16]
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.decision_head = nn.Sequential(
            nn.Linear(128 * 16, n_classes),
            nn.Softmax(dim=1) if downstream_task == "unmixing" else nn.Identity(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # reshape x : [B, L] => [B, 1, L]
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        x = self.pool(self.block4(x))
        x = self.block5(x)  # pool included in the block

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.decision_head(x)
        return x
