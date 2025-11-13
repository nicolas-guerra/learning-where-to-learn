========================================
PARTICLE-BASED AMA FOR RTE
========================================

ENVIRONMENT SETUP
-----------------
1. Create environment:
   python -m venv rte

2. Activate the environment:
   source rte/bin/activate

3. Install packages:
   pip install -r requirements.txt 

4. Verify GPU availability:
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"


RUNNING THE PARTICLE OPTIMIZATION SCRIPT
-----------------------------------------
The main script particle_script.py accepts two command-line arguments:
- --ep_index: Epsilon index (integer, e.g. -3, -1, 3. Note that epsilon=1/(2^ep_index) in RTE)
- --N: Number of particles (integer, e.g., 12, 51, 102, 204, 408)

Example commands:
   python particle_script.py --ep_index 3 --N 50
   python particle_script.py --ep_index 1 --N 102

To run multiple experiments on 4 GPUs:
   chmod +x run_all.sh
   ./run_all.sh 


OUTPUT FILES
------------
Results are saved in directories named:
   training_log_eps{1/2^ep_index}N{N}/

Each directory contains CSV files with training logs for each trial.


ANALYSIS NOTEBOOKS
------------------
Three Jupyter notebooks are provided for analyzing results:

1. PlotResults.ipynb
   - Plots relative OOD errors from all trials
   - Compares performance across different epsilon and N values
   - Generates confidence intervals

2. PlotTime.ipynb
   - Visualizes computation time for each iteration
   - Shows scaling behavior with different N values
   
3. RTEInputOutput.ipynb
   - Given scattering coefficient, visualizes spatial-domain density 

4. QuickAnalysis.ipynb
   - Computes error statistics and running time

TYPICAL WORKFLOW
----------------
1. Set up environment using the requirements.txt file
2. Run experiments for desired ep_index and N combinations using run_all.sh
3. Use PlotResults.ipynb to analyze OOD error performance
4. Use PlotTime.ipynb to analyze computational efficiency
5. Use QuickAnalysis.ipynb to output error statistics and runtime


REQUIREMENTS
------------
- CUDA-capable GPU (tested with CUDA 11.8)
- Python 3.11
- All dependencies listed in requirements.txt


NOTES
-----
- Each full run with 10 trials takes several hours depending on N
- GPU memory usage scales with N; monitor with nvidia-smi
- Ensure sufficient disk space for output directories