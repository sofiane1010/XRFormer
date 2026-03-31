# XRFormer: Multiscale Tokenization for XRF Representation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper **"XRFormer: Multiscale Tokenization for XRF Representation Learning"**.

**Authors:** Sofiane Daimellah¹, Sylvie Le Hégarat-Mascle¹, and Clotilde Boust²
<br>*¹Université Paris-Saclay, ²Centre de Recherche et de Restauration des Musées de France (C2RMF)*

## 📂 Repository Structure

```text
XRFormer-Multiscale-Tokenization-for-XRF-Representation-Learning/
├── data/                    # Processed datasets (.npz files)
├── datasets/                # Raw source files (InfraArt/PCSv5)
├── models/                  # Checkpoints and logs
├── scripts/                 # Executable scripts
│   ├── generate_data.py     # Synthetic mixture generation
│   ├── pretrain_ssl.py      # Self-supervised pretraining (MSM + PPP)
│   └── train_downstream.py  # Downstream tasks (Identification/Unmixing)
├── src/                     # Source code package
│   └── xrf/                 # Core library (Models, Layers, Tokenizers)
├── environment.yml          # Conda environment definition
├── requirements.txt         # Pip requirements
└── README.md
```

## 🚀 Installation


### 1. Clone the repository

```bash
git clone https://github.com/sofiane1010/XRFormer-Multiscale-Tokenization-for-XRF-Representation-Learning.git
cd XRFormer-Multiscale-Tokenization-for-XRF-Representation-Learning
```

### 2. Install dependencies
We provide a unified environment file for Conda users (recommended) and a requirements file for Pip users.

Using Conda :
```bash
conda env create -f environment.yml
conda activate xrformer
```
or using pip
```bash
pip install -r requirements.txt
```

### 📊 Data preparation
All models are trained on synthetic mixtures that are generated from the infraart and PCSv5 datasets of pure samples located in the ```datasets/``` folder. 

To generate the infraart dataset for SSL pretraining:
```bash
python scripts/generate_data.py --dataset infraart --size 2000000
```
To generate the PCSv5 dataset for downstream training:
```bash 
python scripts/generate_data.py --dataset PCSv5 --size 2000
```


### Training and Evaluation

#### 1. SSL pretraining
Because pretraining takes time, we include the pretraining weights for both **base (B)** and **large (L)** XRFormer models **(MSM and MSM+PPP)** in ```models/pretrained```.

Still, the pretraining script is provided for experiments:

XRFormer-B (MSM)
```bash
python scripts/pretrain_ssl.py --size B
```
XRFormer-B (MSM+PPP)

```bash
python scripts/pretrain_ssl.py --size B --peak_prediction
```
XRFormer-L (MSM)

```bash
python scripts/pretrain_ssl.py --size L
```
XRFormer-L (MSM+PPP)

```bash
python scripts/pretrain_ssl.py --size L --peak_prediction
```

#### 2. Downstream tasks
Results reported in the paper are obtained by averaging 5 runs. Therefore, ```--num-runs 5``` is the default value, althrough it can be changed.

**A. For pigment identification**:

XRFormer-B: 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task identification
```
XRFormer-B (MSM): 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task identification --pretrained
```
XRFormer-B (MSM+PPP): 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task identification --pretrained --peak_prediction
```
XRFormer-L: 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task identification
```
XRFormer-L (MSM): 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task identification --pretrained
```
XRFormer-L (MSM+PPP): 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task identification --pretrained --peak_prediction
```
ViT:
```bash
python scripts/train_downstream.py --model ViT --downstream_task identification
```
SpectralFormer:
```bash
python scripts/train_downstream.py --model SF --downstream_task identification
```
SpectralFormer (no CAF):
```bash
python scripts/train_downstream.py --model SF_no-CAF --downstream_task identification
```

**B. For pigment unmixing**:

XRFormer-B: 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task unmixing
```
XRFormer-B (MSM): 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task unmixing --pretrained
```
XRFormer-B (MSM+PPP): 
```bash
python scripts/train_downstream.py --model XRFormer --size B --downstream_task unmixing --pretrained --peak_prediction
```
XRFormer-L: 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task unmixing
```
XRFormer-L (MSM): 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task unmixing --pretrained
```
XRFormer-L (MSM+PPP): 
```bash
python scripts/train_downstream.py --model XRFormer --size L --downstream_task unmixing --pretrained --peak_prediction
```
ViT:
```bash
python scripts/train_downstream.py --model ViT --downstream_task unmixing
```
SpectralFormer:
```bash
python scripts/train_downstream.py --model SF --downstream_task unmixing
```
SpectralFormer (no CAF):
```bash
python scripts/train_downstream.py --model SF_no-CAF --downstream_task unmixing
```
## 📈 Results

| Task | Metric | ViT | SpectralFormer | XRFormer (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Identification** | **F1-Score** | 90.48% | 89.44% | **93.85%** (w/ MSM) |
| | **Accuracy (AA)** | 64.80% | 59.85% | **75.69%** (w/ MSM) |
| **Unmixing** | **Abundance RMSE** | 0.0533 | 0.0454 | **0.0440** (w/ MSM+PPP) |
| | **SAM (rad)** | 0.1662 | 0.1561 | **0.1546** (w/ MSM+PPP) |

See Tables 2 & 4 in the paper for full comparisons.

## 🔗 Citation

If you use this code or dataset in your research, please cite our paper:
```
@inproceedings{daimellah2024xrformer,
  title={XRFormer: Multiscale Tokenization for XRF Representation Learning},
  author={Daimellah, Sofiane and Le H{\'e}garat-Mascle, Sylvie and Boust, Clotilde},
  booktitle={Proceedings of the International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```

## 📧 Contact

I would love to discuss the project or answer any questions! 

* **Issues:** If you find a bug or have a feature request, please [open an issue](https://github.com/sofiane1010/XRFormer-Multiscale-Tokenization-for-XRF-Representation-Learning/issues) here.
* **Email:** For research collaborations or questions, feel free to reach out to me directly:

**Sofiane Daimellah** *Université Paris-Saclay* ✉️ `sofiane.daimellah@universite-paris-saclay.fr`

