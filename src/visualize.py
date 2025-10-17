#------------------------ Header Comment ----------------------#
#Visualize battery data: draw capacity curve, SOH curve, EOL histogram of "all batteries",

# src/visualize.py — Show ALL batteries equally (start-jump fix applied)
import os
import numpy as np
import matplotlib.pyplot as plt

#This helper function applies a simple median filter with edge padding.
#It is used only for visualization, to make curves look smoother
#by suppressing isolated spikes or small oscillations without altering
#the true underlying trend.
#Parameters：
# y : np.ndarray   Smoothed series (such as capacities or SOH series)
#Returns :
#win : int   Size of the sliding window. Typically an odd number (3, 5, 7...).
def _smooth_display(y, win=5):
    if len(y) < win or win <= 1:
        return y
    pad = win // 2
    yy = np.pad(y, (pad, pad), mode='edge')
    return np.array([np.median(yy[i:i+win]) for i in range(len(y))], dtype=float)

#Validate inputs (length/finite) and draw a line with a standard style.
#Return True if actually drawn, otherwise False.
#Parameters：
#ax : Matplotlib axes object (target plotting area)
#x : Horizontal axis data
#y : Vertical axis data
#label : Curve label, used for legend display.
#Returns :
#bool
#True  → Successfully drawn;
#False → The data is invalid or not drawn (skipped).
def _plot(ax, x, y, label):
    if len(x) < 2 or len(y) < 2:
        return False
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return False
    ax.plot(x, y, label=label, alpha=0.9, linewidth=1.4)
    return True

# Group by file, sort by cycle, apply a small median smoothing for readability,
#and apply the start-jump fix (skip the very first point)
#Parameters：
# df : pd.DataFrame  Contains at least three columns: ['file', 'cycle', 'capacity']
#Returns :
# save_path : str | None
# If a path is provided, the graph is saved to that path; otherwise, it is simply generated in memory and then closed.
def plot_capacity_curve(df, save_path=None):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    drawn = False
    for name, g in df.groupby('file'):
        g = g.sort_values('cycle')
        x = g['cycle'].values
        y = _smooth_display(g['capacity'].values, win=5)
        # Start-jump fix: skip very first point to avoid edge-connection artifact
        if len(x) > 1:
            x = x[1:]; y = y[1:]
        if _plot(ax, x, y, name):
            drawn = True
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Battery Capacity Fade (All Curves)')
    ax.grid(True)
    if drawn:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.text(0.5, 0.5, 'No valid series', ha='center', va='center', transform=ax.transAxes)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

#Similar to capacity plotting but using 'soh' values, clipped to [0, 1] for
#display only. A light smoothing is applied and the start-jump fix is used.
#Parameters：
# df : pd.DataFrame  Contains at least three columns: ['file', 'cycle', 'capacity']
#Returns :
# save_path : str | None
# If a path is provided, the graph is saved to that path; otherwise, it is simply generated in memory and then closed.
def plot_soh_curve(df, save_path=None):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    drawn = False
    for name, g in df.groupby('file'):
        g = g.sort_values('cycle')
        x = g['cycle'].values
        y = np.clip(g['soh'].values, 0.0, 1.0)# clip only for display
        y = _smooth_display(y, win=5)
        # Start-jump fix
        if len(x) > 1:
            x = x[1:]; y = y[1:]
        if _plot(ax, x, y, name):
            drawn = True
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('SOH')
    ax.set_title('Battery SOH vs Cycle (All Curves)')
    ax.grid(True)
    ax.set_ylim(0.6, 1.3)
    if drawn:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.text(0.5, 0.5, 'No valid series', ha='center', va='center', transform=ax.transAxes)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

#For each battery, find the first cycle where SOH <= 0.7 and collect those
#cycle counts to form a distribution. If none detected, show a placeholder text.
#Parameters：
# df : pd.DataFrame  Contains at least three columns: ['file', 'cycle', 'capacity']
#Returns :
# save_path : str | None
# If a path is provided, the graph is saved to that path; otherwise, it is simply generated in memory and then closed.
def plot_eol_hist(df, save_path=None):
    eols = []
    for name, g in df.groupby('file'):
        g = g.sort_values('cycle')
        soh = np.clip(g['soh'].values, 0.0, 1.0)
        idx = np.where(soh <= 0.7)[0]
        if idx.size > 0:
            eols.append(int(g['cycle'].values[idx[0]]))
    fig = plt.figure(figsize=(8, 5))
    if len(eols) >= 1:
        import math
        bins = min(20, max(3, int(math.sqrt(len(eols)))))
        plt.hist(eols, bins=bins)
        plt.xlabel('Cycles to EOL (SOH<=0.7)')
        plt.ylabel('Count')
        plt.title('Distribution of Estimated Cycles to EOL')
    else:
        plt.text(0.5, 0.5, 'No EOL detected (SOH<=0.7)',
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Distribution of Estimated Cycles to EOL')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

#For each battery file, export two standalone PNGs:
#  - capacity vs cycle
#  - SOH vs cycle
#Each uses the same light smoothing and start-jump fix as the aggregated plots.
#Parameters：
# df : pd.DataFrame  Contains at least three columns: ['file', 'cycle', 'capacity']
#Returns :
# save_path : str | None
# If a path is provided, the graph is saved to that path; otherwise, it is simply generated in memory and then closed.
def plot_per_battery(df, out_dir='outputs/plots_per_file'):
    os.makedirs(out_dir, exist_ok=True)
    for name, g in df.groupby('file'):
        g = g.sort_values('cycle')
        # capacity
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        x = g['cycle'].values
        y = _smooth_display(g['capacity'].values, win=5)
        if len(x) > 1:
            x = x[1:]; y = y[1:]
        ax.plot(x, y)
        ax.grid(True)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Capacity (Ah)')
        ax.set_title(f'{name} - Capacity')
        plt.savefig(os.path.join(out_dir, f'{name}_capacity.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
        # SOH
        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        x = g['cycle'].values
        y = np.clip(g['soh'].values, 0.0, 1.0)
        y = _smooth_display(y, win=5)
        if len(x) > 1:
            x = x[1:]; y = y[1:]
        ax.plot(x, y)
        ax.grid(True)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('SOH')
        ax.set_ylim(0.6, 1.3)
        ax.set_title(f'{name} - SOH')
        plt.savefig(os.path.join(out_dir, f'{name}_soh.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
