import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  


def add_landcover_pc1(pred_df, features, scale_label, make_plot=True):
    # Select landcover features
    X = pred_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_

    print(f"\n=== {scale_label} PCA Explained Variance ===")
    for i, var in enumerate(explained_var):
        print(f"PC{i+1}: {var:.3f}")

    # attach PC1 scores
    pred_df[f"landcover_PC1_{scale_label}"] = X_pca[:, 0]

    # make loadings plot
    if make_plot and len(features) >= 2:
        plot_pca_loadings(pca, features, scale_label)

    return pred_df, explained_var

def plot_pca_loadings(pca, features, scale_label, out_dir="figs"):
    
    os.makedirs(out_dir, exist_ok=True)

    loadings = pca.components_.T
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, var in enumerate(features):
        x, y = float(loadings[i, 0]), float(loadings[i, 1])
        ax.arrow(0, 0, x, y, head_width=0.05, length_includes_head=True)
        ax.text(x * 1.10, y * 1.10, var, fontsize=10)

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(f"PC1 ({pc1_var:.2f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.2f}%)")
    ax.set_title(f"Landcover PCA Loadings ({scale_label})")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    out_path = os.path.join(out_dir, f"pca_loadings_{scale_label}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")  # <-- no tight_layout
    plt.close(fig)

    print(f"[saved] {out_path}")
    
if __name__ == "__main__":

    features = [
        "p_forest_total",
        "p_dev_total",
        "p_ag_total",
        "p_wetlands_total",
    ]

    pred_100 = pd.read_csv("output/full_predictors_100m.csv")
    pred_1km = pd.read_csv("output/full_predictors_1km.csv")
    pred_5km = pd.read_csv("output/full_predictors_5000m.csv")

    datasets = {
        "100m": pred_100,
        "1km": pred_1km,
        "5km": pred_5km,
    }

    for scale, df in datasets.items():
        datasets[scale], _ = add_landcover_pc1(df, features, scale, make_plot=True)

    # Now each df has PC1 attached
    pred_100 = datasets["100m"]
    pred_1km = datasets["1km"]
    pred_5km = datasets["5km"]

    pred_100.to_csv("full_predictors_100m_with_pc1.csv", index=False)
    pred_1km.to_csv("full_predictors_1km_with_pc1.csv", index=False)
    pred_5km.to_csv("full_predictors_5000m_with_pc1.csv", index=False)