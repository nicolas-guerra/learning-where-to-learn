# NtD and Darcy Flow Supplementary Code

This repository contains scripts and notebook to reproduce the main experimental results for the Dirichlet-to-Neumann (NtD) example and the Darcy flow forward problem, along with a plotting notebook. 

## Files

* **NtDExample.py**: Python script to run multiple independent runs of the Alternating Minimization Algorithm on the NtD example and save results to `NtD_results.pkl`.
* **NtDGPU.py**: Python script to run multiple independent runs of the Alternating Minimization Algorithm on the NtD example (using GPU if available), and save results to `NtD_results.pkl`.
* **DarcyFlowGPU.py**: Python script to run multiple independent runs of the Alternating Minimization Algorithm on the Darcy flow forward problem (using GPU if available), and save results to `DarcyFlow_results.pkl`.
* **PlotResults.ipynb**: Jupyter notebook that loads `NtD_results.pkl` and `DarcyFlow_results.pkl`, generates the figures presented in the paper, and saves or displays them.

## Requirements

All required Python packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Required packages:

* torch==2.5.1
* numpy==1.25.0
* scipy==1.11.1
* pandas==2.0.3
* matplotlib==3.5.1
* deepxde==1.14.0

Additional dependencies:

* pickle (standard library)
* os, math (standard library)

## Usage

1. **NtD example**

   ```bash
   python NtDExample.py
   ```

   * Outputs: `NtD_results.pkl` containing model performance metrics and intermediate data.
   
2. **NtD GPU**

   ```bash
   python NtDGPU.py
   ```

   * Outputs: `NtD_results.pkl` containing model performance metrics and intermediate data.

3. **Darcy flow example**

   ```bash
   python DarcyFlowGPU.py
   ```

   * Outputs: `DarcyFlow_results.pkl` containing model performance metrics and intermediate data.

4. **Plotting results**

   ```bash
   jupyter notebook PlotResults.ipynb
   ```

   * The notebook expects `NtD_results.pkl` and `DarcyFlow_results.pkl` to be in the same directory.
   * Run all cells to reproduce all figures.

## Directory Structure

```
./
├── NtDExample.py         # Script for NtD example
├── NtDGPU.py             # Script for NtD example with GPU
├── DarcyFlowGPU.py       # Script for Darcy flow example
├── PlotResults.ipynb     # Notebook for plotting saved results
├── NtD_results.pkl       # Output from NtDExample.py/NtDGPU.py
└── DarcyFlow_results.pkl # Output from DarcyFlowGPU.py
```

## Notes & Recommendations

* Hyperparameters (e.g., number of training iterations, sample sizes) can be adjusted within each script.
* Ensure that all required Python packages are installed in your environment before running the scripts.

---
