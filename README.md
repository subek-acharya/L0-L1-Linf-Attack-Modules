# L0 White-Box Adversarial Attack

A comprehensive PyTorch implementation for evaluating the robustness of deep learning models using L0-norm constrained white-box adversarial attacks with sparse perturbations.

## Overview

This project implements three variants of L0-norm white-box adversarial attacks designed to evaluate model robustness under sparse pixel perturbations. The framework includes a binary search algorithm to automatically find the optimal sparsity level (minimum number of pixels to perturb) required to successfully fool a classifier.

**Attack Type:** White-Box (full access to model parameters, architecture, and gradients)

## What is a White-Box Attack?

A white-box adversarial attack assumes the attacker has **complete knowledge** of the target model:
- Model architecture and parameters
- Access to model gradients (via backpropagation)
- Knowledge of loss functions and decision boundaries
- Full transparency of the model's internal workings

This is the strongest threat model and provides an upper bound on model vulnerability.

## Attack Methods

### 1. **L0_PGD (Basic L0 Sparse Attack)**
The fundamental L0 attack that constrains perturbations to a fixed number of pixels (sparsity k). At each iteration:
- Limits perturbations to k pixels (L0 constraint)
- Updates only those k pixels using PGD-style optimization with box constraints [0,1]

Parameters:
- `sparsity (k)`: Number of pixels to perturb

### 2. **L0_Linf_PGD (L0 + L∞ Hybrid Attack)**
Combines L0 sparsity constraints with L∞ bounded perturbations. Features:
- Limits perturbations to k pixels (L0 constraint)
- Constrains each perturbation to ε maximum magnitude (L∞ constraint) within  box constraints [0,1]

Parameters:
- `sparsity (k)`: Number of pixels to perturb
- `epsilon (ε)`: Maximum perturbation magnitude per pixel

### 3. **L0_Sigma_PGD (L0 with Adaptive Scaling)**
Advanced variant using dynamic perturbation scaling with σ parameter:
- Limits perturbations to k pixels (L0 constraint)
- Computes sigma-map from neighboring pixel variance to identify visually noisy regions
- Adaptively scales perturbations based on local image properties (targets high-variance areas)
- Uses κ (kappa) to control maximum perturbation magnitude
- Perturbations remain within box constraints [0,1]

**Use Case**: Most effective for finding minimal adversarial perturbations.

**Parameters**:
- `sparsity (k)`: Number of pixels to perturb
- `kappa (κ)`: Scaling factor for perturbation intensity

## Project Structure

```bash
L0_AttackWrapper/
│
├── model_architecture/           # Model implementations
│   └── ResNet.py                # ResNet-20 architecture
│
├── checkpoint/                   # Trained model weights
│   └── Trained_model.pth
│
├── data/                         # Dataset storage
│   └── Dataset.pth
│
├── l0_attack/                    # Attack implementations
│   ├── init.py
│   ├── L0_PGD.py                # Basic L0 attack
│   ├── L0_Linf_PGD.py           # L0 + L∞ hybrid attack
│   ├── L0_Sigma_PGD.py          # Adaptive scaling attack
│   └── L0_Utils.py              # Attack utility functions
│
├── main.py                       # Main execution script
├── binary_search.py             # Optimal sparsity finder
├── utils.py                      # Helper functions and data loaders
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

### Installation

1. Clone the repository:
```bash
git clone https://github.com/subek-acharya/L0-WhiteBox-Adversarial-Attack.git
cd L0-WhiteBox-Adversarial-Attack
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Attacks

Edit `main.py` to select which attack to run:

#### 1. L0_PGD Attack
```bash
# In main.py, uncomment:
print("L0_PGD: ")
l0_attack.L0_PGD_AttackWrapper(model, device, correctLoader, n_restarts, 
                                num_steps, step_size, sparsity, random_start)
```

#### 2. L0_Linf_PGD Attack
```bash
# In main.py, uncomment:
print("L0_Linf: ")
l0_attack.L0_Linf_PGD_AttackWrapper(model, device, correctLoader, n_restarts, 
                                     num_steps, step_size, sparsity, epsilon, random_start)
```

#### 3. L0_Sigma_PGD Attack
```bash
# In main.py, uncomment:
print("L0_Sigma_PGD_AttackWrapper: ")
l0_attack.L0_Sigma_PGD_AttackWrapper(model, device, correctLoader, n_restarts, 
                                      num_steps, step_size, sparsity, kappa, random_start)
```
                                
```bash
# Then run
python main.py
```

### Attack Parameters
```bash
n_restarts = 1        # Number of random restarts
num_steps = 300       # PGD iterations
step_size = 20        # Step size for gradient updates
sparsity = 2000       # Number of pixels to perturb (k)
epsilon = 1           # L∞ bound (for L0_Linf only)
kappa = 8             # Scaling factor (for L0_Sigma only)
random_start = True   # Use random initialization
```

### Viewing Per-Class Robust Accuracy
```bash
#To analyze model robustness for each class individually, uncomment the following line in any attack file:
#In L0_PGD.py, L0_Linf_PGD.py, or L0_Sigma_PGD.py:

# At the end of the attack wrapper function, uncomment:
utils.print_per_class_robust_accuracy(all_labels, pgd_adv_acc)
```

## Configuring Loss Functions

The framework supports two loss functions. Switch between them in `utils.py`:
```bash
# In utils.py, in the get_predictions_and_gradients() function:

with torch.enable_grad():
    output = model(x)

    # Uncomment this line to use Cross Entropy Loss
    loss = nn.CrossEntropyLoss()(output, y)

    # Uncomment this line to use DLR Loss (Recommended for stronger attacks)
    loss = dlr_loss(output, y).mean()
```

## Finding Optimal Sparsity with Binary Search

The binary search algorithm automatically finds the minimum number of pixels needed to successfully attack the model.

### Step 1: Configure Binary Search Parameters
```bash
# In `main.py`, set the search parameters:

# Binary Search Parameters
tau = 0.0        # Target robust accuracy (0% = complete attack success)
k_min = 1        # Minimum sparsity to test
k_max = 2000     # Maximum sparsity to test
tolerance = 10   # Stop when range < tolerance
```
### Step 2: Select Attack Type

```bash
# Uncomment the desired attack in main.py:

# Choose one:
attack_name = 'L0_PGD_AttackWrapper'
# attack_name = 'L0_Linf_PGD_AttackWrapper'
# attack_name = 'L0_Sigma_PGD_AttackWrapper'
```

### Step 3: Run Binary Search

```bash
# Uncomment to run biinary search to find the optimal sparsity value
binary_search.binary_search_optimal_k(model, device, correctLoader, n_restarts, num_steps, step_size, epsilon, kappa, random_start, tau, k_min, k_max, tolerance, attack_name)
```

## References

- [1] Croce, F., & Hein, M. (2019). [Sparse and Imperceivable Adversarial Attacks](https://arxiv.org/abs/1909.05040). *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*.