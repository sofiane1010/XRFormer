###
import numpy as np
import argparse

from xrf.data_utils import (
    load_infraart_spectra,
    load_pcsv5_spectra,
    create_mixture_dataset,
)

parser = argparse.ArgumentParser(description="Dataset generation config")
parser.add_argument("--dataset", type=str, default="infraart")  # 'infraart' or 'PCSv5'
parser.add_argument("--size", type=int, default=2_000_000)  # 'infraart' or 'PCSv5'
parser.add_argument("--seed", type=int, default=42)  # seed for reproducibility
parser.add_argument("--photon_scale", type=int, default=1e4)  # scale for Poisson noise
args = parser.parse_args()

###


###
# create dataset for downstream tasks: identification, unmixing

if args.dataset == "PCSv5":
    checker_spectra, energy, pigment_names = load_pcsv5_spectra()

    # create mixture dataset
    mixed_spectra, components = create_mixture_dataset(
        checker_spectra,
        n_mixtures=args.size,
        photon_scale=args.photon_scale,
        seed=args.seed,
    )

    # downsample spectra and energy by a factor of 4 (2048 channels -> 512 channels)
    mixed_spectra = mixed_spectra[:, ::4]
    energy = energy[::4]

    # save dataset
    np.savez(
        "data/checker_v5_mixed_xrf_spectra.npz",
        endmembers=checker_spectra,
        spectra=mixed_spectra,
        components=components,
        energy=energy,
        pigment_names=pigment_names,
    )
    print("\nChecker dataset created with shape:", mixed_spectra.shape)
elif args.dataset == "infraart":
    infraart_spectra, energy = load_infraart_spectra()

    # create mixture dataset
    mixed_spectra, components, num_peaks = create_mixture_dataset(
        infraart_spectra,
        n_mixtures=args.size,
        photon_scale=args.photon_scale,
        seed=args.seed,
        detect_peaks=True,
    )

    # downsample spectra and energy by a factor of 4 (2048 channels -> 512 channels)
    infraart_spectra, energy = infraart_spectra[:, ::4], energy[::4]

    # save dataset
    np.savez(
        "data/infraart_mixtures.npz",
        endmembers=infraart_spectra,
        spectra=mixed_spectra,
        components=components,
        energy=energy,
        num_peaks=num_peaks,
    )
    print("\nInfraart dataset created with shape:", mixed_spectra.shape)
else:
    raise ValueError("Invalid dataset. Choose from 'infraart' or 'PCSv5'.")
