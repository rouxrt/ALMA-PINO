# ALMA-PINO: Physics-Informed Neural Operators for Interferometric Datacube Restoration
 
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img width="1264" height="399" alt="image" src="https://github.com/user-attachments/assets/955b9abb-fedc-404b-9901-6d7d32f64f4e" />

**ALMA-PINO** (*Physics-Informed Neural Operators for ALMA*) is a Scientific Machine Learning (SciML) framework designed for the reconstruction of three-dimensional (spatial and spectral) interferometric astronomical datacubes from the **ALMA (Atacama Large Millimeter/submillimeter Array)** observatory.
 
This framework leverages the discretization-invariant properties of Fourier Neural Operators (FNOs) to learn the inverse mapping from dirty interferometric observations to reconstructed sky brightness distributions.

---
 
## Scientific Motivation
 
Radio interferometric imaging is an ill-posed inverse problem. Traditional approaches such as CLEAN reconstruct the sky brightness distribution through iterative deconvolution procedures that can become computationally expensive for large spectral cubes.
 
Recent advances in Scientific Machine Learning suggest that Neural Operators may learn families of inverse operators directly from data while preserving physical consistency. This project investigates whether Physics-Informed Neural Operators can provide accurate and computationally efficient deconvolution of ALMA datacubes.
 
---

## Interferometric Imaging as a Fredholm Integral Equation

The reconstruction of an astronomical sky brightness distribution from an interferometric observation can be formulated as an inverse problem governed by a Fredholm integral equation of the first kind.

For an ideal linear imaging system, the observed dirty image (or dirty datacube channel) is related to the true sky brightness distribution through:

$$ I_{\mathrm{dirty}}=\int_{\Omega} K(\mathbf{x},\mathbf{y}) I_{\mathrm{true}}(\mathbf{y}) \, d\mathbf{y} + n(\mathbf{x}) $$

where:


* $I_{\mathrm{true}}$ is the unknown sky brightness distribution,
* $I_{\mathrm{dirty}}$ is the observed dirty image,
* $K(\mathbf{x},\mathbf{y})$ is the instrumental response kernel, also known as Point Spread Function (PSF)
* $n(\mathbf{x})$ represents observational noise.

For radio interferometric imaging, the kernel corresponds to the synthesized beam (dirty beam), which makes the forward model equivalent to a convolution operator:

$$ I_{\mathrm{dirty}}=I_{\mathrm{true}} * \mathrm{PSF} + n. $$

In Fourier space this relation becomes

$$ \mathcal{F}(I_{\mathrm{dirty}}) = \mathcal{F}({I}_{\mathrm{true}}) \cdot \mathcal{F}({\mathrm{PSF}}) + \mathcal{F}({n}). $$

Recovering $I_{\mathrm{true}}$ from $I_{\mathrm{dirty}}$ is therefore an ill-posed inverse problem because the forward operator is compact and information is partially lost by incomplete spatial-frequency sampling.

Traditional methods such as CLEAN iteratively approximate the inverse solution. In this work, we instead seek to learn the inverse operator $[\mathcal{G}^{-1} : I_{\mathrm{dirty}} \mapsto I_{\mathrm{true}} ]$ using Physics-Informed Neural Operators (PINO).

The Fourier Neural Operator is particularly suitable for this task because it is designed to learn mappings between infinite-dimensional function spaces rather than between finite-dimensional vectors. Consequently, the network learns an approximation of the inverse Fredholm operator itself, potentially enabling inference across varying spatial discretizations and improving generalization beyond the training grid.

---
 
## Mathematical Framework & Architecture
 
The scientific core of the project lies in its hybrid **Physics-Informed Loss Function**, structured in a dimensionless space (normalized against the dynamic peak of each batch $c = \max(I_{gt})$ ) to ensure universal numerical stability and independence from source magnitude.
 
The total loss is defined as:
 
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{    \text{data}} + \lambda_{    \text{phys}}\mathcal{L}_{   \text{physics}} + \lambda_{ \text{spec}}\mathcal{L}_{\text{spectral}} $$
 
### 1. Data-Fidelity Loss ($\mathcal{L}_{   \text{data}}$)
Optimized to handle the high sparsity of the radioastronomical cosmic background and prevent structural artifacts or visual hallucinations:

$$ \mathcal{L}_{ \text{data}} =  \alpha \cdot (1 -   \text{SSIM}(I_{ \text{pred}}, I_{   \text{gt}})) + (1 -  \alpha) \cdot  \text{MAE}(I_{  \text{pred}}, I_{   \text{gt}}) $$

*Recommended setup:* $\alpha = 0.03$ to give statistical dominance to the MAE (L1) metric, forcing the background to remain strictly at absolute zero.
 
### 2. Forward Physics Loss ($\mathcal{L}_{ \text{physics}}$)
Exploits the convolution theorem and the linearity of the Fourier Transform to constrain the prediction to respect the convolution equation of the interferometric instrument:

$$ I_{\text{dirty-pred}} = \mathcal{F}^{-1}\left[\mathcal{F}(I_{\text{pred}}) \cdot \mathcal{F}(\text{PSF})\right] $$

$$ \mathcal{L}_{\text{physics}} = \text{MSE}(I_{\text{dirty-pred}}, I_{\text{dirty}}) $$
 
### 3. Spectral Continuity Loss ($\mathcal{L}_{ \text{spectral}}$)
A spectral-gradient consistency loss computed along the frequency axis that forces the neural operator layers to preserve the physical continuity of the cosmic gas spectral emission lines:

$$ \mathcal{L}_{\text{spectral}} = \frac{1}{N} \sum_{z} | (I_{\text{pred}, z+1} - I_{\text{pred}, z}) - (I_{\text{gt}, z+1} - I_{\text{gt}, z})|^2 $$
 
 
---
 
## Installation & Requirements
 
The framework is developed to run natively in isolated virtual environments on Windows and Linux operating systems.
 
```bash
# Clone the repository
git clone https://github.com/rouxrt/ALMA-PINO.git
cd ALMA-PINO
 
# Create a native Python virtual environment
python -m venv venv
 
# Activate the virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
 
# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
 
---
##  Command-Line Interface (CLI) & Configuration

The current pipeline is designed to generate and train on **synthetic interferometric mock datacubes**. This allows for a controlled environment to rigorously benchmark the physics-informed loss components against known ground truths before transitioning to real ALMA Science Archive data. 

The framework is highly modular and can be dynamically configured using the following `argparse` flags:

###  Mock Dataset Generation
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--num_samples` | `int` | `200` | Total number of mock datacubes to generate for the dataset. |
| `--channels` | `int` | `16` | Number of frequency slices (Z-axis) representing the spectral domain. |
| `--img_size` | `int` | `32` | Spatial resolution (X, Y) of the datacubes. |
| `--extended_source` | `flag` | `False` | If active, forces the generator to simulate extended galactic structures rather than point sources. |

###  FNO Architecture
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--modes` | `int` | `8` | Number of Fourier frequencies (modes) to retain. Governs the low-pass filtering capacity of the operator. |
| `--width` | `int` | `32` | Latent dimension (channel width) of the neural operator. |
| `--fourier_layers`| `int` | `4` | Total number of sequential Fourier layers in the network. |

###  Training & Optimization
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `20` | Total number of training epochs scheduled. |
| `--batch_size` | `int` | `8` | Number of datacubes processed per batch. |
| `--learning_rate` | `float`| `0.005` | Initial learning rate for the Adam optimizer (modulated by the Cosine Annealing scheduler). |

###  Physics-Informed Loss Balancing
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--lambda_data` | `float`| `1.0` | Multiplier for the primary data-fidelity loss component. |
| `--lambda_phys` | `float`| `0.5` | Multiplier for the Forward Physics Loss (MSE in Fourier space). |
| `--lambda_spec` | `float`| `0.1` | Multiplier for the 1D Total Variation Spectral Continuity Loss. |
| `--alpha` | `float`| `0.03` | Balancing factor within the data loss. Lower values (e.g., `0.03`) heavily favor MAE (L1) over MS-SSIM to preserve sparse background dynamics. |
## Usage
 
### Training Loop
 
```bash
python train.py   --epochs 300   --batch_size 4   --width 32   --modes 16   --lambda_phys 10.0   --lambda_spec 10.0   --alpha 0.03
```

---
 
## Repository Structure
 
```text
├── dataset/
│   └── mock_dataset.py
├── models/
│   ├── basics.py
│   ├── fno.py           # Fourier Neural Operator (2D/3D) architecture definition
│   ├── losses.py        # Physics-Informed Loss (PILoss) implementation
│   └── utils.py        
├── visualize/
│   └── plot.py      # Monitoring functions for loss curves and spectral profiles
├── train.py             # Main script for training and curriculum scheduling
├── requirements.txt     # Deterministic project dependencies
└── README.md            # Technical documentation of the framework
```
 
---

 
## Future Work
- [ ] Native extension of the neural operator to the third dimension (**FNO3D**) with spectral convolution kernels to simulate the 3D diffraction PSF of radio telescopes.
- [ ] Zero-shot testing and validation on real datacubes from the ALMA *Science Archive* (Restoration of real substellar and high-redshift galactic sources).
- [ ] Integration of kinematic constraints based on the differential calculus of higher-order astronomical moments (Moment 1 for local velocity and Moment 2 for velocity dispersion).
 
---
 
