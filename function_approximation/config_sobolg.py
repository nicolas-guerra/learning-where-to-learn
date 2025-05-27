# Config
FLOAT64_FLAG = False
d = 2
sigma = 1e0
J = 10
Q_sigma = 1e-0
Q_gamma = 1e-0
N_final = 250*1+1000*0                     # number of training data per gradient descent iteration
N_initial = 250*1+1000*0                     # number of training data per gradient descent iteration
N_test_total = 5000         # number of total samples per test distribution
N_val = 500                 # number of test samples per test distribution used in adjoint solve
my_mean = 1/2*1
steps = 50*1+5000*0
lr_initial = 1e-2
eps_initial = 1e-3
eps_final = 1e-7
num_loops = 10              # number of MC trials
