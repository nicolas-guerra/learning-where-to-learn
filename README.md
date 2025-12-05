# Code for the paper [**Learning Where to Learn: Training Data Distribution Optimization for Scientific Machine Learning**](https://arxiv.org/abs/2505.21626)

This repository contains code to reproduce experiments from the paper. It includes:
- **Bilevel kernel-based function approximation experiments**
- **Burgers equation experiments with active learning comparisons**
- **Dirichlet-to-Neumann (NtD) and Darcy flow examples**
- **Radiative transport equation (RTE) experiments**

All folders are organized independently, but share a unified objective of exploring optimal training distributions for out-of-distribution (OOD) accuracy.

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
├── operator_learning/              # Operator learning experiments
│   ├── burgers/                    # Burgers equation with AMA and active learning
│   │   ├── AMA_script.py
│   │   ├── QBC_script.py
│   │   ├── activelearning.py
│   │   ├── burgers.py
│   │   ├── run_all.sh
│   │   └── ...
│   │
│   ├── ntd_darcyflow/              # NtD and Darcy flow examples
│   │   ├── NtDExample.py
│   │   ├── DarcyFlowGPU.py
│   │   └── ...
│   │
│   └── rte/                        # Radiative transport equation experiments
│       ├── particle_script.py
│       ├── rte.py
│       ├── run_all.sh
│       └── ...
```

---

## Installation

### For Bilevel Experiments
```bash
conda env create -f function_approximation/Project.yml
conda activate bilevel
```

### For Operator Learning Experiments
Each subfolder has its own `requirements.txt`:
```bash
# For Burgers experiments
pip install -r operator_learning/burgers/requirements.txt

# For NtD/Darcy Flow experiments
pip install -r operator_learning/ntd_darcyflow/requirements.txt

# For RTE experiments
pip install -r operator_learning/rte/requirements.txt
```

> Note: All environments assume GPU support if available.

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
   python plot_compare.py
```

---

### Burgers Equation Experiments
1. **Run all experiments**:
```bash
   cd operator_learning/burgers
   bash run_all.sh
```
   Or run individual scripts:
```bash
   python AMA_script.py      # AMA optimization
   python QBC_script.py      # Query-by-Committee active learning
```

2. **Plot results**:
```bash
   jupyter notebook PlotResults.ipynb
```

---

### NtD and Darcy Flow Experiments
1. **Run NtD example**:
```bash
   cd operator_learning/ntd_darcyflow
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

---

### Radiative Transport Equation (RTE) Experiments
1. **Run all experiments**:
```bash
   cd operator_learning/rte
   bash run_all.sh
```
   Or run the main script:
```bash
   python particle_script.py
```

2. **Plot results**:
```bash
   jupyter notebook PlotResults.ipynb
```

---

## Notes & Recommendations
- Hyperparameters (e.g., sample sizes, training iterations) can be modified within each script.
- Ensure all required dependencies are installed before executing code.
- For plotting, a LaTeX distribution may be required.
- The Github for the AMINO architecture used in Figure 1 of [our paper](https://arxiv.org/pdf/2505.21626) can be found [here](https://github.com/nicolas-guerra/amino).

---
