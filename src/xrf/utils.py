import torch
import torch.nn.functional as F
import numpy as np
from torcheval.metrics.functional import binary_f1_score, multilabel_accuracy

from xrf.data_utils import load_pcsv5_spectra


def _downstream_loss(pred, target, downstream_task):
    if downstream_task == "identification":
        return F.binary_cross_entropy_with_logits(pred, target)
    elif downstream_task == "unmixing":
        return F.l1_loss(pred, target)


def train_downstream(model, train_loader, device, downstream_task, optimizer):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:

        spectra, targets = batch
        spectra = spectra.to(device)
        targets = targets.to(device)
        pred = model(spectra)
        loss = _downstream_loss(pred, targets, downstream_task)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / (len(train_loader))
    return avg_train_loss


@torch.no_grad()
def val_downstream(model, val_loader, device, downstream_task):
    model.eval()
    total_loss = 0.0
    for batch in val_loader:

        spectra, targets = batch
        spectra = spectra.to(device)
        targets = targets.to(device)
        pred = model(spectra)
        loss = _downstream_loss(pred, targets, downstream_task)
        total_loss += loss.item()

    return total_loss / len(val_loader)


@torch.no_grad()
def get_optimal_thresholds(model, val_loader, device, num_steps=20):
    model.eval()
    all_probs = []
    all_targets = []

    for batch in val_loader:
        spectra, targets = batch
        spectra = spectra.to(device)
        logits = model(spectra)
        probs = torch.sigmoid(logits).cpu()  # move to CPU
        all_probs.append(probs)
        all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs, dim=0)  # [B, N_CLASSES] on CPU
    all_targets = torch.cat(all_targets, dim=0)  # [B, N_CLASSES] on CPU
    ths = np.linspace(0.1, 0.95, num_steps)
    thresholds = [0.5] * all_probs.shape[-1]
    f1s = []

    for i in range(all_probs.shape[-1]):
        class_f1 = 0
        best_th = 0.5
        targets_i = all_targets[:, i]
        probs_i = all_probs[:, i]

        for t in ths:
            pred_bin = (probs_i > t).float()
            current_f1 = binary_f1_score(pred_bin, targets_i)
            if current_f1 > class_f1:
                class_f1 = current_f1
                best_th = t

        thresholds[i] = best_th
        f1s.append(class_f1)

    return thresholds


@torch.no_grad()
def test_downstream(model, test_loader, device, downstream_task, thresholds=None):
    model.eval()
    all_predictions, all_targets = [], []

    for batch in test_loader:
        spectra, targets = batch
        spectra = spectra.to(device)
        preds = model(spectra)
        if downstream_task == "identification":
            preds = torch.sigmoid(preds)
            preds = (preds > torch.tensor(thresholds).to(device)).int()
        all_predictions.append(preds.cpu())
        all_targets.append(targets)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_predictions, all_targets


def downstream_metrics(preds, targets, downstream_task, endmembers=None):
    if downstream_task == "identification":
        per_class_f1 = []
        for i in range(preds.shape[1]):
            per_class_f1.append(binary_f1_score(preds[:, i], targets[:, i]).item())

        return {
            "f1_score": np.array(per_class_f1).mean(),
            "absolute_accuracy": multilabel_accuracy(preds, targets).item(),
            "hamming_accuracy": multilabel_accuracy(
                preds, targets, criteria="hamming"
            ).item(),
        }
    elif downstream_task == "unmixing":

        # stack of endmembers, shape [n_classes, n_channels]
        E = torch.stack(endmembers).float().to(preds.device)

        X_pred = torch.matmul(preds, E)
        X_true = torch.matmul(
            targets, E
        )  # Reference reconstruction based on GT abundances

        a_mse = F.mse_loss(preds, targets)
        a_rmse = torch.sqrt(a_mse)

        r_mse = F.mse_loss(X_pred, X_true)
        r_rmse = torch.sqrt(r_mse)

        cos_sim = F.cosine_similarity(X_pred, X_true, dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        sam_per_sample = torch.acos(cos_sim)
        sam_score = torch.mean(sam_per_sample)

        return {
            "abundance_rmse": a_rmse.item(),
            "reconstruction_rmse": r_rmse.item(),
            "sam_score": sam_score.item(),
        }


# validation for pretraining loss = reconstruction + lambda * peak_prediction_loss
@torch.no_grad()
def val_pretrain(model, val_loader, device, lambda_peak_prediction=0.2):
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        if model.peak_prediction:
            spectra, peak_predictions = batch
            spectra, peak_predictions = spectra.to(device), peak_predictions.to(device)
            peak_mask = peak_predictions
            pred_peak_predictions, recon, mask = model(spectra)
            peak_prediction_loss = F.binary_cross_entropy_with_logits(
                pred_peak_predictions, peak_mask
            )
        else:
            spectra = batch
            spectra = spectra.to(device)
            recon, mask = model(spectra)
        original = spectra.view(spectra.size(0), model.num_patches, model.patch_size)
        recon_loss = F.mse_loss(recon[mask], original[mask])
        loss = recon_loss
        if model.peak_prediction:
            loss = loss + lambda_peak_prediction * peak_prediction_loss
        total_loss += loss.item()
    return total_loss / len(val_loader)


def pretrain(
    model, train_loader, device, optimizer, peak_prediction=False, lambda_peak_pred=0.2
):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        if peak_prediction:
            spectra, peaks = batch
            spectra, peak_mask = spectra.to(device), peaks.to(device)
            peak_preds, recon, mask = model(spectra)
            rand = np.random.rand()

            if rand > 0.8:
                peak_pred_loss = F.binary_cross_entropy_with_logits(
                    peak_preds, peak_mask
                )
                total_train_loss += lambda_peak_pred * peak_pred_loss.item()
                loss = lambda_peak_pred * peak_pred_loss
            else:
                original = spectra.view(
                    spectra.size(0), model.num_patches, model.patch_size
                )
                recon_loss = F.mse_loss(recon[mask], original[mask])
                total_train_loss += recon_loss.item()
                loss = recon_loss
        else:
            spectra = batch
            spectra = spectra.to(device)
            recon, mask = model(spectra)

            original = spectra.view(
                spectra.size(0), model.num_patches, model.patch_size
            )
            recon_loss = F.mse_loss(recon[mask], original[mask])
            total_train_loss += recon_loss.item()
            loss = recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_train_loss / (len(train_loader))


def gain_neighborhood_band(x_train, band_patch, patch=1):
    x_train = x_train.reshape((x_train.shape[0], 1, 1, x_train.shape[1]))
    band = x_train.shape[-1]
    nn = band_patch // 2
    pp = (patch * patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch * patch, band)
    x_train_band = np.zeros(
        (x_train.shape[0], patch * patch * band_patch, band), dtype=np.float32
    )

    x_train_band[:, nn * patch * patch : (nn + 1) * patch * patch, :] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:, i * patch * patch : (i + 1) * patch * patch, : i + 1] = (
                x_train_reshape[:, :, band - i - 1 :]
            )
            x_train_band[:, i * patch * patch : (i + 1) * patch * patch, i + 1 :] = (
                x_train_reshape[:, :, : band - i - 1]
            )
        else:
            x_train_band[:, i : (i + 1), : (nn - i)] = x_train_reshape[
                :, 0:1, (band - nn + i) :
            ]
            x_train_band[:, i : (i + 1), (nn - i) :] = x_train_reshape[
                :, 0:1, : (band - nn + i)
            ]

    for i in range(nn):
        if pp > 0:
            x_train_band[
                :,
                (nn + i + 1) * patch * patch : (nn + i + 2) * patch * patch,
                : band - i - 1,
            ] = x_train_reshape[:, :, i + 1 :]
            x_train_band[
                :,
                (nn + i + 1) * patch * patch : (nn + i + 2) * patch * patch,
                band - i - 1 :,
            ] = x_train_reshape[:, :, : i + 1]
        else:
            x_train_band[:, (nn + 1 + i) : (nn + 2 + i), (band - i - 1) :] = (
                x_train_reshape[:, 0:1, : (i + 1)]
            )
            x_train_band[:, (nn + 1 + i) : (nn + 2 + i), : (band - i - 1)] = (
                x_train_reshape[:, 0:1, (i + 1) :]
            )
    return x_train_band
