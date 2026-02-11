# Version of your package
__version__ = "0.1.0"

# Import key classes to expose them at the top level
from .dataset import SpectraDataset
from .models import XRFClassifier, CNNClassifier1D, ViT
from .utils import (
    train_downstream,
    val_downstream,
    test_downstream,
    get_optimal_thresholds,
    downstream_metrics,
    val_pretrain,
    gain_neighborhood_band,
    pretrain,
)
from .data_utils import (
    load_infraart_spectra,
    load_pcsv5_spectra,
    create_mixture_dataset,
    minmaxnormalize,
)


# Optional: Define what gets imported with "from xrf import *"
__all__ = [
    "SpectraDataset",
    "XRFClassifier",
    "CNNClassifier1D",
    "ViT",
    "train_downstream",
    "val_downstream",
    "test_downstream",
    "get_optimal_thresholds",
    "downstream_metrics",
    "load_infraart_spectra",
    "load_pcsv5_spectra",
    "create_mixture_dataset",
    "minmaxnormalize",
    "val_pretrain",
    "gain_neighborhood_band",
    "pretrain",
]
