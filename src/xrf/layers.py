import torch
import torch.nn as nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout))
                        ),
                    ]
                )
            )

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel, num_channel, [1, 2], 1, 0))

    def forward(self, x):
        if self.mode == "ViT":
            for attn, ff in self.layers:
                x = attn(x)
                x = ff(x)
        elif self.mode == "CAF":
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat(
                            [x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3
                        )
                    ).squeeze(3)
                x = attn(x)
                x = ff(x)
                nl += 1
        return x


class MaskedSpectrum(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))  # learnable [MASK] token
        self.ratio = ratio

    def forward(self, x):
        """
        x: Tensor of shape [B, N, D] (batch, tokens, dim)
        Returns:
            x_masked: masked tensor
            mask: boolean array of masked token positions
        """
        B, N, _ = x.shape
        device = x.device

        # Step 1: select which tokens to corrupt
        mask = torch.rand(B, N, device=device) < self.ratio  # bool mask

        # Total number of tokens to corrupt
        total = mask.sum().item()
        if total == 0:
            return x.clone(), mask  # no masking needed

        # Flatten mask indices
        idx_b, idx_n = torch.where(mask)
        num_mask = int(0.8 * total)
        num_rand = int(0.1 * total)

        perm = torch.randperm(total, device=device)
        mask_idx = perm[:num_mask]
        rand_idx = perm[num_mask : num_mask + num_rand]

        x_masked = x.clone()

        # Apply [MASK] token
        x_masked[idx_b[mask_idx], idx_n[mask_idx]] = self.mask_token

        # Replace with random tokens from other batch positions
        if num_rand > 0:
            rand_b = torch.randint(0, B, (num_rand,), device=device)
            rand_n = torch.randint(0, N, (num_rand,), device=device)
            random_tokens = x[rand_b, rand_n]  # shape (num_rand, D)
            x_masked[idx_b[rand_idx], idx_n[rand_idx]] = random_tokens

        return x_masked, mask


# Tokenization + Token embedding
class Tokenizer(nn.Module):
    def __init__(self, mode, dim, patch_size, num_patches):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.bn = nn.BatchNorm1d(dim)
        if self.mode == "linear" or self.mode == "GSE":
            self.tokenizer = nn.Linear(self.patch_size, dim)
            self.ln = nn.LayerNorm(dim)

        elif self.mode == "vanilla_conv":
            self.tokenizer = nn.Conv1d(1, dim, self.patch_size, stride=self.patch_size)
        elif self.mode == "multiscale_conv":
            self.tokenizer = nn.Sequential(
                # -------- Block 1 (high resolution) --------
                nn.Conv1d(1, dim // 4, kernel_size=3, padding=1),  # [B, D/4, L]
                nn.GELU(),
                nn.BatchNorm1d(dim // 4),
                # Downsample
                nn.Conv1d(
                    dim // 4, dim // 4, kernel_size=3, stride=2, padding=1
                ),  # [B, D/4, L/2]
                nn.GELU(),
                nn.BatchNorm1d(dim // 4),
                # -------- Block 2 (mid resolution) --------
                nn.Conv1d(
                    dim // 4, dim // 2, kernel_size=3, padding=1
                ),  # [B, D/2, L/2]
                nn.GELU(),
                nn.BatchNorm1d(dim // 2),
                nn.Conv1d(
                    dim // 2, dim // 2, kernel_size=3, padding=1
                ),  # [B, D/2, L/2]
                nn.GELU(),
                nn.BatchNorm1d(dim // 2),
                # Downsample
                nn.Conv1d(
                    dim // 2, dim // 2, kernel_size=3, stride=2, padding=1
                ),  # [B, D/2, L/4]
                nn.GELU(),
                nn.BatchNorm1d(dim // 2),
                # -------- Block 3 (low resolution) --------
                nn.Conv1d(dim // 2, dim, kernel_size=3, padding=1),  # [B, D, L/4]
                nn.GELU(),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),  # [B, D, L/4]
                nn.GELU(),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),  # [B, D, L/4]
                nn.GELU(),
                nn.BatchNorm1d(dim),
                # -------- Token projection --------
                nn.AdaptiveMaxPool1d(self.num_patches),  # [B, D, N]
            )

    def preprocess(self, x):
        if self.mode == "GSE":
            x = rearrange(x, "b p n -> b n p")  # [B, N, P]
        elif self.mode == "linear":
            x = rearrange(x, "b (n p) -> b n p", p=self.patch_size)  # [B, N, P]
        elif "conv" in self.mode:
            x = x.unsqueeze(1)  # [B, 1, L]
        return x

    def forward(self, x):
        # patchify for linear tokenizer, unsqueeze for vanilla conv tokenizer
        x = self.preprocess(x)
        if self.mode == "linear" or self.mode == "GSE":
            x = self.tokenizer(x)  # [B, N, D]
            x = rearrange(x, "b n d -> b d n")  # [B, D, N]
            x = self.bn(x)
            x = rearrange(x, "b d n -> b n d")  # [B, N, D]
        elif self.mode == "vanilla_conv":
            x = self.bn(self.tokenizer(x))  # [B, D, N]
            x = rearrange(x, "b d n -> b n d")  # [B, N, D]
        elif self.mode == "multiscale_conv":
            x = self.tokenizer(x)
            x = rearrange(x, "b d n -> b n d")  # [B, N, D]
        return x
