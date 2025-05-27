import numpy as np
import os
from util import plt


plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=18)
plt.rcParams['lines.linewidth'] = 3.5
msz = 14
handlelength = 5.0     # 2.75
borderpad = 0.25     # 0.15

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]



# USER INPUT
FLAG_save_plots = True
FLAG_WIDE = False
n_std = 2
plot_tol = 1e-6
save_pref = "f1_d5_eval"

# Derived
plot_folder = "./results/" + save_pref + "/"
os.makedirs(plot_folder, exist_ok=True)

# Load
x_fevals = np.load(plot_folder + "x_fevals.npy")
sample_size_list = np.load(plot_folder + "sample_size_list.npy")
errors_loops = np.load(plot_folder + "errors.npy")
errors_seq_loops = np.load(plot_folder + "errors_seq_loops.npy")
errors_static_loops = np.load(plot_folder + "errors_static_loops.npy")

def get_stats(ar):
    out = np.zeros((*ar.shape[-(ar.ndim - 1):], 2))
    out[..., 0] = np.mean(ar, axis=0)
    out[..., 1] = np.std(ar, axis=0)
    return out

errors = get_stats(errors_loops)
errors_seq = get_stats(errors_seq_loops)
errors_static = get_stats(errors_static_loops)


# Legend
legs = [r"Optimized", r"Fixed"]

# Colors
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan
# clist = ['C0', 'C1'] # blue, orange
    
if FLAG_WIDE:
    plt.rcParams['figure.figsize'] = [6.0, 4.0]     # [6.0, 4.0]
else:
    plt.rcParams['figure.figsize'] = [6.0, 6.0]     # [6.0, 4.0]


# Plot 1: long iter with shaded 2 std bands
iter_var = np.arange(1, len(errors) + 1)
twosigma = n_std*errors[...,1]
lb = np.maximum(errors[...,0] - twosigma, plot_tol)
ub = errors[...,0] + twosigma

plt.figure(0)
my_ls = '-'
plt.semilogy(iter_var, errors[...,0], my_ls, markersize=msz, color=color_list[0])
plt.fill_between(iter_var, lb, ub, facecolor=color_list[0], alpha=0.125)
plt.xlabel(r'Iteration')
if errors.shape[0] > 99:
    plt.xlim(-5, errors.shape[0] + 5)
plt.ylabel(r'Relative OOD Error')
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'loss' + '.pdf', format='pdf')
plt.show()


# Plot 2: Err vs Sample size
plt.figure(1)
for i, error_array in enumerate([errors_seq, errors_static]):
    twosigma = n_std*error_array[...,1]
    lb = np.maximum(error_array[...,0] - twosigma, plot_tol)
    ub = error_array[...,0] + twosigma

    plt.loglog(sample_size_list, error_array[...,0], ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
    
    plt.fill_between(sample_size_list, lb, ub, facecolor=color_list[i], alpha=0.125)
    
    # plt.errorbar(sample_size_list, error_array[...,0], yerr=[lb, ub], fmt='none', capsize=1, color=color_list[i])
    
plt.xlabel(r'Sample Size')
# plt.ylabel(r'Relative OOD Error')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'samplesize' + '.pdf', format='pdf')
plt.show()


# Plot 3: Err vs Func evals (corrected for distribution optimization)
x_fevals = x_fevals[1:,-1]
tup1 = (sample_size_list + x_fevals[-1], errors_seq)
tup2 = (sample_size_list, errors_static)

plt.figure(2)
for i, (xplot, error_array) in enumerate((tup1, tup2)):
    twosigma = n_std*error_array[...,1]
    lb = np.maximum(error_array[...,0] - twosigma, plot_tol)
    ub = error_array[...,0] + twosigma

    plt.semilogy(xplot, error_array[...,0], ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
    
    plt.fill_between(xplot, lb, ub, facecolor=color_list[i], alpha=0.125)

plt.xlabel(r'Total Function Evaluations')
# plt.ylabel(r'Relative OOD Error')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'funcevals' + '.pdf', format='pdf')
plt.show()
