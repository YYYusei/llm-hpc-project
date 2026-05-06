"""
figures/plot_role_swap_bar.py

Generates Figure 4.X — Role-Swap prompt-variant results.

Usage:
    cd llm-hpc-project
    mkdir -p figures
    python figures/plot_role_swap_bar.py

Output:
    figures/role_swap_bar.png
    figures/role_swap_bar.pdf   (preferred for LaTeX inclusion)

Data source: canonical pf-taxonomy changes from results/pf_summary/summary_table.md
All three variants share identical S1=GPT-4o, S2=GPT-5.2, temperature=0, source code.
Only the Stage-2 prompt template varies.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Canonical data (pf taxonomy, tol=0.3) ----
variants = ["Original\n(validate / correct)", "V1 Neutral\n(no validation)", "V3 Biased\n(agree / confirm)"]
changes  = [5, 4, 1]
totals   = 9  # programs per variant

# ---- Style (publication-quality, matches University of Leeds thesis convention) ----
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=150)

# Colours: a subtle gradient from a "skeptical" prompt (strongest) to "agreeing" prompt (weakest).
# Using a single ramp (purples) to avoid implying categorical colour-coding.
bar_colors = ["#3C3489", "#7F77DD", "#CECBF6"]

bars = ax.bar(variants, changes, color=bar_colors, edgecolor="#26215C",
              linewidth=0.8, width=0.55)

# Add value labels on top
for bar, v in zip(bars, changes):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.15,
            f"{v}/{totals}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

# Axis setup
ax.set_ylabel("S2 primary-category changes (out of 9 programs)")
ax.set_ylim(0, totals + 0.7)
ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)

# Annotate the two mechanisms
# Mechanism A: Original -> V1 (weak effect, -1 program)
ax.annotate("", xy=(1, 4.3), xytext=(0, 5.3),
            arrowprops=dict(arrowstyle="->", color="#5F5E5A", lw=0.9,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(0.5, 5.9, "weak effect: −1", ha="center", va="bottom",
        fontsize=9, style="italic", color="#5F5E5A")

# Mechanism B: V1 -> V3 (strong effect, -3 programs; or Original -> V3 is -4)
ax.annotate("", xy=(2, 1.3), xytext=(1, 4.3),
            arrowprops=dict(arrowstyle="->", color="#993556", lw=0.9,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(1.5, 3.3, "strong effect: −3", ha="center", va="bottom",
        fontsize=9, style="italic", color="#993556")

# Sub-caption in the figure (not in the LaTeX caption — this documents what varies)
ax.text(0.5, -0.28, "S1 = GPT-4o, S2 = GPT-5.2, temperature = 0, identical inputs",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#444441")

plt.tight_layout()

# Save both PNG (for preview) and PDF (for LaTeX inclusion)
plt.savefig("figures/role_swap_bar.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/role_swap_bar.pdf",           bbox_inches="tight")

print("Saved:")
print("  figures/role_swap_bar.png")
print("  figures/role_swap_bar.pdf")
