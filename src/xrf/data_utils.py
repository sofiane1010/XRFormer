import os
import numpy as np
import random
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def read_xrf_spectra_from_mca(folder_path: str):
    """
    Read XRF spectra from .mca files in the specified folder and return calibrated data.

    Parameters:
        folder_path (str): Path to folder containing .mca files

    Returns:
        tuple: (spectra, energies, pigment_name) where spectra is a list of intensity arrays and energies is the energy values
    """
    # Get all .mca files in folder
    mca_files = [f for f in os.listdir(folder_path) if f.endswith(".mca")]
    if not mca_files:
        raise ValueError(f"No .mca files found in {folder_path}")

    spectra = []
    energies = None  # Will store the energy values once we have them
    pigment_names = [file.split(".")[0] for file in mca_files]
    for file in mca_files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        # Initialize variables
        cal_params = None
        spectrum_data = []
        reading_data = False

        # Parse the file
        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith("<<CALIBRATION>>"):
                # Next two lines contain calibration points
                if i + 3 < len(lines):
                    cal_line1 = lines[i + 2].strip().split()
                    cal_line2 = lines[i + 3].strip().split()
                    # Convert calibration points to floats
                    cal_params = [
                        float(cal_line1[0]),
                        float(cal_line1[1]),  # First point (channel, energy)
                        float(cal_line2[0]),
                        float(cal_line2[1]),  # Second point (channel, energy)
                    ]

            elif line.startswith("<<DATA>>"):
                reading_data = True
                continue

            elif line.startswith("<<END>>"):
                reading_data = False

            elif reading_data and line:  # Only process non-empty lines
                spectrum_data.append(float(line))

        if cal_params is None:
            raise ValueError(f"No calibration parameters found in {file}")

        # Convert spectrum data to numpy array
        spectrum_data = np.array(spectrum_data)

        # Create channel numbers array
        channels = np.arange(len(spectrum_data))

        # Calculate linear calibration parameters
        # Using the two calibration points to create a linear fit
        x1, y1 = cal_params[0], cal_params[1]  # First point (channel, energy)
        x2, y2 = cal_params[2], cal_params[3]  # Second point (channel, energy)

        # Calculate slope and intercept
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Apply calibration to convert channels to energy (keV)
        if energies is None:  # Only calculate energies once
            energies = slope * channels + intercept

        spectra.append(spectrum_data)

    return np.array(spectra), energies, pigment_names


def read_csv_spectrum(file_path: str, return_energy: bool = False):
    """
    Read XRF spectra from .csv files in the specified folder and return calibrated data.

    Parameters:
        file_path (str): Path to .csv file

    Returns:
        tuple: (spectra, energy) where spectra is a list of intensity arrays and energies is the energy values
    """
    evpc = None
    spectrum_start_index = None

    # extract energy scale and header size
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("eV per channel"):
                evpc = float(line.split(",")[-1].strip()) * 0.001
            elif line.startswith("Channel#"):
                spectrum_start_index = i + 1
                break

    # load spectrum
    spectrum = np.loadtxt(
        file_path,
        delimiter=",",
        skiprows=spectrum_start_index,
        converters={1: lambda s: float(s.decode("utf-8").strip('"'))},
        dtype=np.float32,
    )[:, 1]
    energy = np.arange(len(spectrum), dtype=np.float32) * evpc
    if return_energy:
        return spectrum, energy
    return spectrum


def minmaxnormalize(spectra: np.ndarray):
    """
    Min-max normalize the spectra.
    """
    return (spectra - np.min(spectra)) / (np.max(spectra) - np.min(spectra))


def load_infraart_spectra(
    folder_path: str = "datasets/infraart",
):
    """
    Load infraart dataset spectra from .csv files

    Parameters:
        folder_path (str): path to folder containing .csv files
    Returns:
        spectra: numpy array of shape (n_spectra, n_channels)
        energy: numpy array of shape (n_channels,)
    """
    csv_files = [f for f in os.listdir(folder_path)]
    spectra, energy = None, None
    for i, file in enumerate(csv_files):
        file_path = os.path.join(folder_path, file)
        if i == 0:
            spectra, energy = read_csv_spectrum(file_path, return_energy=True)
        else:
            spectrum = read_csv_spectrum(file_path)
            spectra = np.vstack([spectra, spectrum])
    return minmaxnormalize(spectra), energy


def load_pcsv5_spectra():
    _, energy = load_infraart_spectra()
    checker_spectra, old_energy, pigment_names = read_xrf_spectra_from_mca(
        "datasets/XRF_pigments_checker_v5"
    )

    # realign checker spectra with infraart spectra
    checker_spectra = realign_spectra(checker_spectra, old_energy, energy)
    checker_spectra = minmaxnormalize(checker_spectra)

    # take a subset of checker spectra with aparent peaks (20% threshold for paper results)
    valid_pigments = checker_spectra.max(axis=1) > 0.2
    checker_spectra = checker_spectra[valid_pigments]
    pigment_names = [p for i, p in enumerate(pigment_names) if valid_pigments[i]]
    return checker_spectra, energy, pigment_names


def create_mixture_dataset(
    base_spectra: np.ndarray,
    n_mixtures: int = 1_500_000,
    min_components: int = 2,
    max_components: int = 3,
    seed: int = 42,
    detect_peaks: bool = False,
    photon_scale: int = 1e4,
    n_tokens: int = 128,
):
    # Set seed
    np.random.seed(seed)
    random.seed(seed)

    # Initialize variables
    n_spectra, n_channels = base_spectra.shape
    mixed_spectra = np.zeros((n_spectra + n_mixtures, n_channels), dtype=np.float32)
    mixed_spectra[:n_spectra] = base_spectra
    components = np.zeros((n_spectra + n_mixtures, n_spectra), dtype=np.float32)
    components[:n_spectra, :] = np.eye(n_spectra, dtype=np.float32)
    if detect_peaks:
        peak_heatmap = np.zeros((n_spectra + n_mixtures, n_tokens), dtype=np.float32)
        peak_heatmap[:n_spectra] = detect_xrf_peaks(base_spectra, height=0.05)

    # Create mixtures
    for i in range(n_mixtures):

        print(f"\rProgress: iteration {i+1} of {n_mixtures}", end="", flush=True)

        # Randomly select number of components in the mixture
        n_comp = random.randint(min_components, max_components)

        # Randomly select components from the base spectra
        indices = sorted(random.sample(range(n_spectra), n_comp))

        # Randomly weight the components
        weights = np.random.dirichlet(np.ones(n_comp, dtype=np.int8))

        # Ensure no component has a weight less than 0.05 to avoid extremely sparse mixtures
        while np.any(weights < 0.05):
            weights = np.random.dirichlet(np.ones(n_comp, dtype=np.int8))

        # Mix the components
        mix = np.zeros(n_channels, dtype=np.float32)
        for j, idx in enumerate(indices):
            mix += weights[j] * base_spectra[idx]

        # Introduce itensity variations
        mix *= np.random.uniform(0.5, 2)

        # Add Poisson noise (most representative for photon counts)
        lam = np.clip(mix, 0, None) * photon_scale
        noisy_counts = np.random.poisson(lam)
        mix = noisy_counts.astype(np.float32) / photon_scale

        # Add the mixture to the dataset
        mixed_spectra[n_spectra + i] = mix
        components[n_spectra + i, indices] = weights
        if detect_peaks:
            peak_heatmap[n_spectra + i] = detect_xrf_peaks(
                mix.reshape(1, -1), height=0.05, n_tokens=n_tokens
            )
    if detect_peaks:
        return mixed_spectra, components, peak_heatmap
    return mixed_spectra, components


def detect_xrf_peaks(
    spectra: np.ndarray,
    width: int = 0,
    height: float = None,
    prominence: float = None,
    n_tokens: int = 128,
):
    """
    Detect peaks in XRF spectra and return a binary mask indicating the presence of peaks in each token.

    Parameters:
        spectra (np.ndarray): shape (n_spectra, n_channels)
        width (int): minimum width of peaks in channels
        height (float): minimum height of peaks (normalized intensity)
        prominence (float): minimum prominence of peaks (normalized intensity)
        n_tokens (int): number of tokens to divide the spectrum into

    Returns:
        np.ndarray: binary mask of shape (n_spectra, n_tokens)
    """
    B, L = spectra.shape
    patch_size = L // n_tokens
    peak_mask = np.zeros((B, n_tokens), dtype=np.float32)

    for i, spectrum in enumerate(spectra):
        peaks, _ = find_peaks(
            spectrum,
            width=width,
            height=height,
            prominence=prominence,
        )
        peak_mask[i, peaks // patch_size] = 1.0

    return peak_mask


def realign_spectra(
    spectra: np.ndarray,
    energy_old: np.ndarray,
    energy_target: np.ndarray,
    kind: str = "nearest",
):
    """
    Realign multiple spectra to a new energy axis.

    Parameters:
        spectra (np.ndarray): shape (n_spectra, n_channels)
        energy_old (np.ndarray): shape (n_channels,), original energy positions
        energy_target (np.ndarray): shape (n_channels,), target energy positions
        kind (str): interpolation method ('linear', 'cubic', etc.)

    Returns:
        np.ndarray: shape (n_spectra, n_channels), realigned spectra
    """
    n_spectra, n_channels = spectra.shape
    assert energy_old.shape[0] == n_channels
    assert energy_target.shape[0] == n_channels

    aligned_spectra = np.empty_like(spectra)
    for i in range(n_spectra):
        interpolator = interp1d(
            energy_old,
            spectra[i],
            kind=kind,
            bounds_error=False,
            fill_value="extrapolate",
        )
        aligned_spectra[i] = interpolator(energy_target)

    return aligned_spectra
