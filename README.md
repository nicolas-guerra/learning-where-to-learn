# Code for the paper [**Learning Where to Learn: Training Distribution Selection for Provable OOD Performance**](https://arxiv.org/abs/2505.21626)

This repository contains code to reproduce experiments from the paper. It includes:

- **Bilevel kernel-based function approximation experiments**
- **Dirichlet-to-Neumann (NtD) and Darcy flow examples**

Both folders are organized independently, but share a unified objective of exploring optimal training distributions for out-of-distribution (OOD) generalization.

---

## Repository Structure

```
./
├── function_approximation/         # Kernel-based function approximation experiments
│   ├── driver.py
│   ├── plot.py
│   ├── Project.yml
│   └── ...
│
├── operator_learning/              # NtD and Darcy flow examples
│   ├── NtDExample.py
│   ├── DarcyFlowGPU.py
│   ├── PlotResults.ipynb
│   ├── requirements.txt
│   └── ...
```

---

## Installation

### For Bilevel Experiments

```bash
conda env create -f function_approximation/Project.yml
conda activate bilevel
```

### For NtD/Darcy Flow Experiments

```bash
pip install -r operator_learning/requirements.txt
```

> Note: Both environments assume GPU support if available.

---

## Usage

### Bilevel Function Approximation

1. **Run experiments**:

   ```bash
   cd function_approximation
   python -u driver.py
   ```

   *Outputs:* `errors.npy` and other intermediate files.

2. **Plot results**:

   ```bash
   python plot.py
   ```

---

### NtD and Darcy Flow Experiments

1. **Run NtD example**:

   ```bash
   cd operator_learning
   python NtDExample.py
   ```

2. **Run Darcy flow example**:

   ```bash
   python DarcyFlowGPU.py
   ```

3. **Plot results**:

   ```bash
   jupyter notebook PlotResults.ipynb
   ```

   - Make sure `NtD_results.pkl` and `DarcyFlow_results.pkl` are in the directory.

---

## Notes & Recommendations

- Hyperparameters (e.g., sample sizes, training iterations) can be modified within each script.
- Ensure all required dependencies are installed before executing code.
- For plotting, a LaTeX distribution may be required.
- The Github for the AMINO architecture used in Figure 1 of [our paper](https://arxiv.org/pdf/2505.21626) can be found [here](https://github.com/nicolas-guerra/amino).

---
