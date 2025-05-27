# Supplementary Code for the Paper: "Learning Where to Learn: Training Distribution Selection for Provable OOD Performance"

This repository contains scripts to reproduce the main experimental results for the bilevel minimization kernel-based function approximation experiments. The code defaults to running on GPU, if one is available. 

## Installation and Requirements

The command
```bash
conda env create -f Project.yml
```
creates an environment called ``bilevel``. [PyTorch](https://pytorch.org/) will be installed in this step.

Activate the environment with
```bash
conda activate bilevel
```
and deactivate with
```bash
conda deactivate
```

Additional dependencies:

* a latex distribution (for plotting)

## Usage

Users may edit config files for each function they wish to approximate, which then gets imported by the **driver.py** file.

1. **Reproduce experiments**

   ```bash
   python -u driver.py
   ```

   * Outputs: `errors.npy` containing model performance metrics and intermediate data, as well as other data files.

2. **Plotting results**

   ```bash
   python plot.py
   ```

## Notes & Recommendations

* Hyperparameters (e.g., number of training iterations, sample sizes) can be adjusted within each script.
* Ensure that all required Python packages are installed in your environment before running the scripts.

---
