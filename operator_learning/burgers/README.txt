========================================
AMA FOR BURGERS
========================================

ENVIRONMENT SETUP
-----------------
1. Create environment:
   python -m venv ama_env

2. Activate the environment:
   source ama_env/bin/activate

3. Install packages:
   pip install -r requirements.txt

4. Verify GPU availability:
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"


DESCRIPTION OF MAIN SCRIPTS
---------------------------
The repository includes four core scripts for running experiments:

1. AMA_script.py  
   - Computes the optimal parameters for the training distribution using our AMA procedure.

2. QBC_script.py  
   - Runs 10 trials of Query-by-Committee active learning.  
   - Saves the resulting Error-vs-N data as CSV files.

3. feature_script.py  
   - Runs 10 trials of feature-diversity–based active learning.  
   - Saves the resulting Error-vs-N data as CSV files.

4. dist_script.py  
   - Runs 10 trials of training on N samples drawn from the optimized training distribution.  
   - Saves the resulting Error-vs-N data as CSV files.


RUNNING EXPERIMENTS
-------------------
Each script can be executed directly. Example:

   python AMA_script.py
   python QBC_script.py
   python feature_script.py
   python dist_script.py

Output CSV files will be written to automatically generated folders corresponding to each experiment.


OUTPUT FILES
------------
Each experiment produces CSV files containing:
   - Error vs. sample size N
   - Trial-wise performance data

Directories are created automatically based on experiment type and parameter choices.


ANALYSIS NOTEBOOKS
------------------
The repository includes one notebook for visualization and analysis:

1. PlotResults.ipynb  
   - Aggregates and plots Error-vs-N across all methods  
   - Compares AMA, QBC, feature diversity, and training-from-optimal-distribution  
   - Shows example input and output pairs for the operator being learned  
   - Useful for understanding the functional mapping and qualitative behavior


TYPICAL WORKFLOW
----------------
1. Set up the environment using the requirements.txt file.
2. Run AMA_script.py to obtain optimal training distribution parameters.
3. Run QBC_script.py, feature_script.py, and dist_script.py to generate benchmark data.
4. Use PlotResults.ipynb to compare methods and visualize OOD performance.


RUNNING MULTIPLE EXPERIMENTS ON SEPARATE GPUS
----------------------------------------------
A convenience script `run_all.sh` is provided to launch `feature_script.py`,  
`QBC_script.py`, and `dist_script.py` simultaneously on different GPUs.  
It also measures and prints the total wall-clock runtime.

Usage:
   chmod +x run_all.sh
   ./run_all.sh


REQUIREMENTS
------------
- CUDA-capable GPU (tested with CUDA 11.8)
- Python 3.10+ recommended
- All dependencies listed in requirements.txt


NOTES
-----
- Each set of 10 trials may take several hours depending on method, N, and model size.
- GPU memory usage increases with N; monitor using `nvidia-smi`.
- Ensure sufficient disk space for storing trial outputs.