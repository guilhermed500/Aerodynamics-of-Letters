"""
Plots the aerodynamic forces over time and summary statistics for two letters.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import AerodynamicsOfLetters as aero

def plot_forces_history(History, Letters, Output_Dir):
    name1, name2 = Letters[0], Letters[1]
    steps = History["step"]


    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(steps, History[f"{name1}_Cd"], label=f"{name1} Cd", color="black")
    axes[0].plot(steps, History[f"{name2}_Cd"], label=f"{name2} Cd", color="red")
    axes[0].set_ylabel("Drag Coefficient (Cd)")
    axes[0].set_title(f"Drag Coefficient Comparison: {name1} vs {name2}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, History[f"{name1}_Cl"], label=f"{name1} Cl", color="black")
    axes[1].plot(steps, History[f"{name2}_Cl"], label=f"{name2} Cl", color="red")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Lift Coefficient (Cl)")
    axes[1].set_title(f"Lift Coefficient Comparison: {name1} vs {name2}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, History[f"{name1}_downforce"], label=f"{name1} Downforce", color="black")
    axes[2].plot(steps, History[f"{name2}_downforce"], label=f"{name2} Downforce", color="red")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Downforce")
    axes[2].set_title(f"Downforce Comparison: {name1} vs {name2}")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    timeseries_path = os.path.join(Output_Dir, "forces_timeseries.png")
    fig.savefig(timeseries_path, dpi=300)
    plt.close(fig)
    return timeseries_path

def plot_summary(History, Letters, Output_Dir):
    name1, name2 = Letters[0], Letters[1]
    steps = History["step"]
    half=len(steps)//2

    def average(lst):
        return float(np.mean(History[lst][half:]))

    avg_Cd1 = [average(f"{name1}_Cd"), average(f"{name2}_Cd")]
    avg_Cl1 = [average(f"{name1}_Cl"), average(f"{name2}_Cl")]
    avg_down1 = [average(f"{name1}_downforce"), average(f"{name2}_downforce")]

    fig,ax=plt.subplots(figsize=(8,5))

    x=np.arange(2)
    width=0.35

    ax.bar(x - width/2, [avg_Cd1[0], avg_Cd1[1]], width, label="Drag (Cd)", color="black")
    ax.bar(x + width/2, [avg_Cl1[0], avg_Cl1[1]], width, label="Lift (Cl)", color="red")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{name1}", f"{name2}"])
    ax.set_ylabel("Average Coefficient")
    ax.set_title("Average Coefficients Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    for i,v in enumerate([avg_Cd1[0], avg_Cl1[0]]):
        ax.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    
    for i,v in enumerate([avg_Cd1[1], avg_Cl1[1]]):
        ax.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    summary_path = os.path.join(Output_Dir, "forces_summary.png")
    fig.savefig(summary_path, dpi=300)
    plt.close(fig)
    return summary_path, avg_Cd1, avg_Cl1, avg_down1

def generate_comparison_charts(History, Letters, Output_Dir):

    name1, name2 = Letters[0], Letters[1]

    timeseries_path = plot_forces_history(History, Letters, Output_Dir)
    summary_path, avg_Cd1, avg_Cl1, avg_down1 = plot_summary(History, Letters, Output_Dir)

    print(f"Comparison charts saved:")
    print(f"  {timeseries_path}  (Cd/Cl/downforce over time)")
    print(f"  {summary_path}  (steady-state average bar chart)")
    print(f"Steady-state averages (last half of run):")
    print(f"  {name1}: Cd={avg_Cd1[0]:.4f}  Cl={avg_Cl1[0]:.4f}  Downforce={avg_down1[0]:.4f}")
    print(f"  {name2}: Cd={avg_Cd1[1]:.4f}  Cl={avg_Cl1[1]:.4f}  Downforce={avg_down1[1]:.4f}")

    return timeseries_path, summary_path