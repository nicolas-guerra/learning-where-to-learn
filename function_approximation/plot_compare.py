import numpy as np
import os
from util import plt


plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=15)
plt.rcParams['lines.linewidth'] = 3.5
msz = 14
handlelength = 3.0     # 2.75
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
FLAG_WIDE = True
n_std = 2
plot_tol = 1e-6
ONE_TEST = True
save_pref = "sobol_compare_iclr"

# Derived
plot_folder = "./results/" + save_pref + "/"
os.makedirs(plot_folder, exist_ok=True)

# Load
x_fevals = np.load(plot_folder + "x_fevals.npy")
sample_size_list = np.load(plot_folder + "sample_size_list.npy")
errors_loops = np.load(plot_folder + "errors.npy")
errors_seq_loops = np.load(plot_folder + "errors_seq_loops.npy")
errors_static_loops = np.load(plot_folder + "errors_static_loops.npy")
errors_bary_loops = np.load(plot_folder + "errors_bary_loops.npy")
errors_gmm_loops = np.load(plot_folder + "errors_gmm_loops.npy")
errors_unif_loops = np.load(plot_folder + "errors_unif_loops.npy")

def get_stats(ar):
    out = np.zeros((*ar.shape[-(ar.ndim - 1):], 2))
    out[..., 0] = np.mean(ar, axis=0)
    out[..., 1] = np.std(ar, axis=0)
    return out

errors = get_stats(errors_loops)
errors_seq = get_stats(errors_seq_loops)
errors_static = get_stats(errors_static_loops)
errors_bary = get_stats(errors_bary_loops)
errors_gmm = get_stats(errors_gmm_loops)
errors_unif = get_stats(errors_unif_loops)


# Legend
if not ONE_TEST:
    legs = [r"Optimized", r"Normal", r"Barycenter", r"Mixture", r"Uniform"]
else:
    legs = [r"Optimized", r"Normal", r"Test", r"Uniform"]

# Colors
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan
    
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
plt.semilogy(iter_var, errors[...,0,0], my_ls, markersize=msz, color=color_list[0], label=r"Seen")
plt.fill_between(iter_var, lb[...,0], ub[...,0], facecolor=color_list[0], alpha=0.125)
plt.semilogy(iter_var, errors[...,1,0], markersize=msz, color=color_list[1], label=r"Unseen", ls='--')
plt.fill_between(iter_var, lb[..., 1], ub[...,1], facecolor=color_list[1], alpha=0.125)
plt.xlabel(r'Iteration')
if errors.shape[0] > 99:
    plt.xlim(-5, errors.shape[0] + 5)
plt.ylabel(r'Relative OOD Error')
plt.legend(framealpha=1, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'loss' + '.pdf', format='pdf')
plt.show()


# Plot 2: Err vs Sample size
plt.figure(1)

if not ONE_TEST:
    plot2_tup = [errors_seq, errors_static, errors_bary, errors_gmm, errors_unif]
else:
    plot2_tup = [errors_seq, errors_static, errors_gmm, errors_unif]
    
for i, error_array in enumerate(plot2_tup):
    twosigma = n_std*error_array[...,1]
    lb = np.maximum(error_array[...,0] - twosigma, plot_tol)
    ub = error_array[...,0] + twosigma

    plt.loglog(sample_size_list, error_array[...,0], ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz, label=legs[i])
    
    plt.fill_between(sample_size_list, lb, ub, facecolor=color_list[i], alpha=0.125)
    
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
tup3 = (sample_size_list, errors_bary)
tup4 = (sample_size_list, errors_gmm)
tup5 = (sample_size_list, errors_unif)

if not ONE_TEST:
    plot3_tup = (tup1, tup2, tup3, tup4, tup5)
else:
    plot3_tup = (tup1, tup2, tup4, tup5)

plt.figure(2)
for i, (xplot, error_array) in enumerate(plot3_tup):
    twosigma = n_std*error_array[...,1]
    lb = np.maximum(error_array[...,0] - twosigma, plot_tol)
    ub = error_array[...,0] + twosigma

    plt.semilogy(xplot, error_array[...,0], ls=style_list[i], color=color_list[i], marker=marker_list[i], markersize=msz)
    
    plt.fill_between(xplot, lb, ub, facecolor=color_list[i], alpha=0.125)

plt.xlabel(r'Total Function Evaluations')
plt.grid(True, which="both")
plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'funcevals' + '.pdf', format='pdf')
plt.show()


# Plot 4: covariance visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable

cov_emp = np.load(plot_folder + "cov_emp.npy")

if not ONE_TEST:
    matrices = [cov_emp[0,...], cov_emp[2,...], cov_emp[1,...]]
    titles = [r"Optimized", r"Barycenter", r"Mixture"]
else:
    matrices = [cov_emp[0,...], cov_emp[1,...]]
    titles = [r"Optimized", r"Test Covariance"]

fig, axes = plt.subplots(1, len(titles), figsize=(12, 4))

for ax, mat, title in zip(axes, matrices, titles):
    im = ax.imshow(mat, cmap='viridis', origin='upper')
    ax.set_title(title)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])    

# create a dedicated axis for the colorbar to the right of the last axes
divider = make_axes_locatable(axes[-1])
cax = divider.append_axes("right", size="5%", pad=0.12)
fig.colorbar(im, cax=cax)#, label='Value')

plt.tight_layout()
if FLAG_save_plots:
    plt.savefig(plot_folder + 'cov_compare' + '.pdf', format='pdf')
plt.show()
