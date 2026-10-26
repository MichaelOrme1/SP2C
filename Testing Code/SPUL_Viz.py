




import torch
import numpy as np
import pandas as pd
import umap
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress, pearsonr
from sklearn.decomposition import PCA
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

matplotlib.use('Agg')
sns.set_theme(style="whitegrid", font_scale=1.0)

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "BlackSheep-Llama3.2-3B"
BASE_PATH = Path("BlackSheep-Llama3.2-3B")

PALETTE = sns.color_palette("tab10", 12)

category_keywords = {

    "S1":  ["violence", "murder", "assault", "physical harm"], 

    "S2":  ["fraud", "scam", "money laundering", "theft"],

    "S3":  ["sexual assault", "harassment", "non-consensual"],

    "S4":  ["child abuse", "pedophilia", "underage", "exploitation"],

    "S5":  ["medical advice", "legal counsel", "financial strategy"], 

    "S6":  ["home address", "social security", "private phone number"],

    "S7":  ["copyright infringement", "piracy", "torrent", "plagiarism"],

    "S8":  ["explosive", "chemical weapon", "terrorism", "bomb-making"],

    "S9":  ["hate speech", "nigger", "misogyny", "white supremacy","slur"],

    "S10": ["suicide", "self-harm", "cutting", "anorexia","noose", "suicide","cut myself","kill myself"],

    "S11": ["pornography", "NSFW", "erotica", "explicit content"],

    "S12": ["Fuck", "Shit","Asshole","Bitch","Cunt"], 

}

# ==========================================
# 2. DATA EXTRACTION
# ==========================================
def run_pipeline():
    print(f"--- Loading Model: {MODEL_ID} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    embed_layer = model.get_input_embeddings()

    all_data = []

    print("--- Vectorising Keywords ---")
    for cat_id, words in category_keywords.items():
        if not words:
            continue
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
            vec = embed_layer(inputs["input_ids"]).detach().float()
            avg_vec = vec.mean(dim=1).squeeze().cpu().numpy()
            all_data.append({
                "vector": avg_vec, "label": f"{cat_id}_Keyword",
                "category": cat_id, "type": "Landmark",
                "percentage": 0.0, "run_type": "Keyword",
            })

    print(f"--- Crawling Directory: {BASE_PATH} ---")
    folder_pattern = re.compile(r"(?:cumulative_)?(\d+)pct(?:_(\d+))?")

    for folder in BASE_PATH.iterdir():
        match = folder_pattern.match(folder.name)
        if folder.is_dir() and match:
            slice_val = int(match.group(1))
            run_suffix = match.group(2)
            run_idx = int(run_suffix) if run_suffix else 10
            calc_pct = run_idx * (slice_val / 100) * 0.1

            for pt_file in folder.glob("soft_prompt_S*.pt"):
                cat_match = re.search(r"S(\d+)", pt_file.name)
                if cat_match:
                    cat_id = f"S{cat_match.group(1)}"
                    tensor = torch.load(pt_file, map_location="cpu").detach().float()
                    if tensor.ndim == 2:
                        avg_vec = tensor.mean(dim=0)
                    elif tensor.ndim == 1:
                        avg_vec = tensor
                    else:
                        avg_vec = tensor.view(-1, tensor.shape[-1]).mean(dim=0)

                    all_data.append({
                        "vector": avg_vec.numpy(),
                        "label": f"{cat_id}_{calc_pct:.3f}",
                        "category": cat_id,
                        "type": "SPUL",
                        "percentage": calc_pct,
                        "run_type": "cumulative" if not run_suffix else f"run_{run_suffix}",
                    })

    df = pd.DataFrame(all_data)

    # ==========================================
    # 3. SAFETY SCORES
    # ==========================================
    print("\n--- Calculating Safety Scores ---")
    stats = []
    bad_centroids = {}

    for cat_id in category_keywords.keys():
        cat_df = df[df["category"] == cat_id]
        landmarks = cat_df[cat_df["type"] == "Landmark"]["vector"].values
        if len(landmarks) == 0:
            continue
        bad_centroid = np.mean(np.stack(landmarks), axis=0)
        bad_centroids[cat_id] = bad_centroid

        for pct in sorted(df["percentage"].unique()):
            if pct == 0:
                continue
            vecs = cat_df[(cat_df["type"] == "SPUL") & (cat_df["percentage"] == pct)]["vector"].values
            if len(vecs) == 0:
                continue
            distances = [np.linalg.norm(v - bad_centroid) for v in vecs]
            
            # cosine similarity to bad centroid (1 = identical direction, -1 = opposite)
            cos_sims = []
            for v in vecs:
                norm_v = np.linalg.norm(v)
                norm_c = np.linalg.norm(bad_centroid)
                if norm_v > 0 and norm_c > 0:
                    cos_sims.append(np.dot(v, bad_centroid) / (norm_v * norm_c))
            
            stats.append({
                "Category": cat_id,
                "Percentage": pct,
                "Safety_Score": np.mean(distances),
                "Safety_Score_Std": np.std(distances),
                "Cosine_Sim_To_Bad": np.mean(cos_sims) if cos_sims else np.nan,
                "N_Runs": len(distances),
            })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("safety_scores.csv", index=False)

    return df, stats_df, bad_centroids


# ==========================================
# 4. ANALYSIS HELPERS
# ==========================================
def compute_analysis(stats_df):
    rows = []
    for cat, grp in stats_df.groupby("Category"):
        grp = grp.sort_values("Percentage")
        x = grp["Percentage"].values
        y = grp["Safety_Score"].values

        final = y[-1]
        initial = y[0]
        gain = final - initial
        slope, _, r, _, _ = linregress(x, y)

        # plateau onset
        plateau_pct = None
        for i in range(len(y)):
            if np.all(y[i:] >= 0.95 * final):
                plateau_pct = x[i]
                break

        idx_10 = np.argmin(np.abs(x - 0.1))
        early_eff = y[idx_10] / final if final > 0 else 0
        diffs = np.abs(np.diff(y))
        oscillation = np.mean(diffs) / final if final > 0 else 0

        # rate of change in two halves
        mid = len(x) // 2
        slope_early, *_ = linregress(x[:mid], y[:mid])
        slope_late, *_ = linregress(x[mid:], y[mid:])

        rows.append({
            "Category": cat,
            "Initial_Score": round(initial, 2),
            "Final_Score": round(final, 2),
            "Total_Gain": round(gain, 2),
            "Slope": round(slope, 2),
            "R_squared": round(r**2, 3),
            "Plateau_Onset": round(plateau_pct, 3) if plateau_pct is not None else None,
            "Early_Efficiency": round(early_eff, 3),
            "Oscillation_Index": round(oscillation, 4),
            "Slope_Early_Half": round(slope_early, 2),
            "Slope_Late_Half": round(slope_late, 2),
        })

    return pd.DataFrame(rows).sort_values("Final_Score", ascending=False)


# ==========================================
# 5. PER-CATEGORY PLOTS (fixed labels)
# ==========================================
def plot_per_category(stats_df, analysis_df):
    cats = sorted(stats_df["Category"].unique(), key=lambda s: int(s[1:]))
    for i, cat in enumerate(cats):
        grp = stats_df[stats_df["Category"] == cat].sort_values("Percentage")
        x = grp["Percentage"].values
        y = grp["Safety_Score"].values
        std = grp["Safety_Score_Std"].values
        info = analysis_df[analysis_df["Category"] == cat].iloc[0]
        colour = PALETTE[i % len(PALETTE)]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.fill_between(x, y - std, y + std, alpha=0.15, color=colour,
                        label="±1 SD (multi-run)")
        ax.plot(x, y, "o-", color=colour, markersize=4, linewidth=1.5,
                label="Mean safety score")

        if len(y) >= 5:
            smoothed = uniform_filter1d(y, size=5)
            ax.plot(x, smoothed, "--", color=colour, linewidth=2, alpha=0.7,
                    label="Smoothed trend")

        # ---- FIXED: plateau annotation placed above the line, never behind box ----
        po = info["Plateau_Onset"]
        ymin, ymax = ax.get_ylim()
        if po is not None:
            ax.axvline(po, color="grey", linestyle=":", linewidth=1.2, zorder=1)
            # place label at top of chart, well clear of the stats box
            ax.text(po - 0.02, ymax * 0.97,
                    f"Plateau @ {po:.2f}",
                    fontsize=8, color="dimgrey", va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey",
                              alpha=0.85, linewidth=0.6))

        # ---- stats box: top-left corner so it never overlaps mid-range data ----
        textstr = (
            f"Final score:  {info['Final_Score']:.2f}\n"
            f"Gain:         +{info['Total_Gain']:.2f}\n"
            f"Slope:        {info['Slope']:.2f} /unit\n"
            f"R²:           {info['R_squared']:.3f}\n"
            f"Early eff.:   {info['Early_Efficiency']:.2%}\n"
            f"Oscillation:  {info['Oscillation_Index']:.4f}"
        )
        props = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9,
                     edgecolor=colour, linewidth=0.8)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="left",
                bbox=props, family="monospace", zorder=10)

        ax.set_xlabel("Training progress  (0 = untrained, 1 = fully unlearned)",
                      fontsize=9)
        ax.set_ylabel("Safety score  (Euclidean distance from bad anchor)", fontsize=9)
        ax.set_title(f"Category {cat} — unlearning trajectory", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(-0.02, 1.05)

        plt.tight_layout()
        fname = f"spul_category_{cat}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")


# ==========================================
# 6. OVERVIEW PLOTS
# ==========================================
def plot_overview(stats_df, analysis_df):
    cats = sorted(stats_df["Category"].unique(), key=lambda s: int(s[1:]))

    # 6a. All trajectories
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(cats):
        grp = stats_df[stats_df["Category"] == cat].sort_values("Percentage")
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(grp["Percentage"], grp["Safety_Score"], "-", color=colour,
                linewidth=1.4, alpha=0.85, label=cat)
        last = grp.iloc[-1]
        ax.scatter([last["Percentage"]], [last["Safety_Score"]], s=40,
                   color=colour, zorder=5)
    ax.set_xlabel("Training progress", fontsize=10)
    ax.set_ylabel("Safety score", fontsize=10)
    ax.set_title("All categories — safety score trajectories", fontsize=13,
                 fontweight="bold")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("spul_overview_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 6b. Final score bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ranked = analysis_df.sort_values("Final_Score", ascending=True)
    colours = [PALETTE[cats.index(c) % len(PALETTE)] for c in ranked["Category"]]
    bars = ax.barh(ranked["Category"], ranked["Final_Score"], color=colours,
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, ranked["Final_Score"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Final safety score at training progress = 1.0", fontsize=9)
    ax.set_title("Final safety scores ranked by category", fontsize=12,
                 fontweight="bold")
    ax.set_xlim(0, ranked["Final_Score"].max() * 1.12)
    plt.tight_layout()
    plt.savefig("spul_final_scores_ranked.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 6c. Analysis scatter
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    pairs = [
        ("Slope", "Slope (score gain per unit progress)", axes[0], "Slope vs final score"),
        ("Oscillation_Index", "Oscillation index (noisiness)", axes[1], "Stability vs final score"),
        ("Early_Efficiency", "Early efficiency (score @ 10% / final)", axes[2],
         "Early efficiency vs final score"),
    ]
    for col, xlabel, ax, title in pairs:
        for _, row in analysis_df.iterrows():
            c = PALETTE[cats.index(row["Category"]) % len(PALETTE)]
            ax.scatter(row[col], row["Final_Score"], color=c, s=60, zorder=5)
            ax.text(row[col], row["Final_Score"] + 0.04, row["Category"],
                    fontsize=7, color=c, ha="center")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Final safety score", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
    plt.suptitle("Category analysis — derived metrics", fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("spul_analysis_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 6d. Checkpoint heatmap
    checkpoints = [0.1, 0.25, 0.5, 0.75, 1.0]
    hmap_rows = []
    for cat in cats:
        grp = stats_df[stats_df["Category"] == cat].sort_values("Percentage")
        row = {"Category": cat}
        for cp in checkpoints:
            idx = np.argmin(np.abs(grp["Percentage"].values - cp))
            row[f"{int(cp*100)}%"] = round(grp.iloc[idx]["Safety_Score"], 2)
        hmap_rows.append(row)
    hmap_df = pd.DataFrame(hmap_rows).set_index("Category")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(hmap_df, annot=True, fmt=".2f", cmap="YlGn",
                linewidths=0.4, ax=ax, cbar_kws={"label": "Safety score"})
    ax.set_title("Safety scores at key training checkpoints", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Training progress checkpoint", fontsize=9)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("spul_checkpoint_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("  Saved overview plots.")


# ==========================================
# 7. NEW: MECHANISTIC VISUALIZATIONS
# ==========================================

def plot_direction_of_movement(df, bad_centroids):
    """
    For each category, project the soft prompt vectors onto the axis defined by:
      bad_centroid → mean_final_prompt
    This shows *how aligned* the training trajectory is with the desired direction.
    A perfectly directed optimiser would show a straight line; deviations show
    off-axis movement.
    """
    print("--- Plotting direction-of-movement (projection onto unlearning axis) ---")
    cats = sorted(bad_centroids.keys(), key=lambda s: int(s[1:]))
    ncols = 4
    nrows = int(np.ceil(len(cats) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for idx, cat in enumerate(cats):
        ax = axes[idx]
        colour = PALETTE[idx % len(PALETTE)]
        cat_df = df[(df["category"] == cat) & (df["type"] == "SPUL")].copy()
        if cat_df.empty:
            ax.set_visible(False)
            continue

        bad_c = bad_centroids[cat]

        # unlearning axis: from bad centroid to the mean of the fully-unlearned prompts
        final_vecs = cat_df[cat_df["percentage"] >= 0.99]["vector"].values
        if len(final_vecs) == 0:
            ax.set_visible(False)
            continue
        good_c = np.mean(np.stack(final_vecs), axis=0)

        axis_vec = good_c - bad_c
        axis_norm = axis_vec / (np.linalg.norm(axis_vec) + 1e-9)

        projections = []
        perp_magnitudes = []
        percentages = []
        for _, row in cat_df.iterrows():
            v = row["vector"] - bad_c        # centre on bad anchor
            proj = np.dot(v, axis_norm)      # component along unlearning axis
            perp = np.linalg.norm(v - proj * axis_norm)  # off-axis magnitude
            projections.append(proj)
            perp_magnitudes.append(perp)
            percentages.append(row["percentage"])

        sc = ax.scatter(projections, perp_magnitudes, c=percentages,
                        cmap="viridis", s=25, alpha=0.75, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="Progress")

        # ideal trajectory: straight line along axis (perp = 0)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

        # annotate start and end clusters
        early = cat_df[cat_df["percentage"] <= 0.05]
        if not early.empty:
            ep = [np.dot(v - bad_c, axis_norm) for v in early["vector"].values]
            eperp = [np.linalg.norm((v - bad_c) - np.dot(v - bad_c, axis_norm) * axis_norm)
                     for v in early["vector"].values]
            ax.scatter(ep, eperp, s=50, color="navy", zorder=6, marker="^",
                       label="Early (≤5%)")

        late = cat_df[cat_df["percentage"] >= 0.95]
        if not late.empty:
            lp = [np.dot(v - bad_c, axis_norm) for v in late["vector"].values]
            lperp = [np.linalg.norm((v - bad_c) - np.dot(v - bad_c, axis_norm) * axis_norm)
                     for v in late["vector"].values]
            ax.scatter(lp, lperp, s=50, color="gold", zorder=6, marker="^",
                       label="Late (≥95%)")

        ax.set_xlabel("Projection onto unlearning axis", fontsize=7)
        ax.set_ylabel("Off-axis magnitude", fontsize=7)
        ax.set_title(f"Category {cat}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper left")

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Direction of movement: projection onto the unlearning axis\n"
        "x-axis = progress toward 'good' embedding; y-axis = off-axis drift\n"
        "Ideal unlearning = points move right along x with low y",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("spul_direction_of_movement.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_direction_of_movement.png")


def plot_cosine_similarity(stats_df):
    """
    Plot cosine similarity between the soft prompt and the bad anchor over training.
    As unlearning succeeds, cosine similarity should decrease (prompts point away
    from the bad direction).  This is a cleaner normalised metric than raw distance.
    """
    print("--- Plotting cosine similarity to bad anchor ---")
    cats = sorted(stats_df["Category"].unique(), key=lambda s: int(s[1:]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(cats):
        grp = stats_df[stats_df["Category"] == cat].sort_values("Percentage")
        if "Cosine_Sim_To_Bad" not in grp.columns:
            continue
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(grp["Percentage"], grp["Cosine_Sim_To_Bad"], "-",
                color=colour, linewidth=1.4, alpha=0.85, label=cat)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.4,
               label="Orthogonal (cosine=0)")
    ax.set_xlabel("Training progress", fontsize=10)
    ax.set_ylabel("Cosine similarity to bad anchor", fontsize=10)
    ax.set_title(
        "Cosine similarity to bad anchor over training\n"
        "Decreasing → soft prompt is pointing away from harmful direction",
        fontsize=12, fontweight="bold"
    )
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("spul_cosine_similarity.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_cosine_similarity.png")


def plot_pca_trajectories(df, bad_centroids):
    """
    For each category, run PCA on the soft prompt vectors + bad anchor and plot
    the trajectory through the first 2 PCs.  This shows the geometric path of
    unlearning in a low-dimensional space without UMAP's non-linear distortion,
    making it easier to see if the path is smooth or erratic.
    """
    print("--- Plotting PCA trajectories ---")
    cats = sorted(bad_centroids.keys(), key=lambda s: int(s[1:]))
    ncols = 4
    nrows = int(np.ceil(len(cats) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8))
    axes = axes.flatten()

    for idx, cat in enumerate(cats):
        ax = axes[idx]
        colour = PALETTE[idx % len(PALETTE)]
        cat_df = df[(df["category"] == cat) & (df["type"] == "SPUL")].sort_values(
            "percentage")
        bad_c = bad_centroids[cat]
        if cat_df.empty:
            ax.set_visible(False)
            continue

        # stack all vectors + bad centroid
        all_vecs = np.vstack([bad_c[np.newaxis, :],
                              np.stack(cat_df["vector"].values)])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_vecs)

        bad_2d = coords[0]
        prompt_coords = coords[1:]
        percentages = cat_df["percentage"].values

        # draw trajectory as ordered line segments coloured by progress
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis
        for k in range(len(prompt_coords) - 1):
            ax.plot([prompt_coords[k, 0], prompt_coords[k+1, 0]],
                    [prompt_coords[k, 1], prompt_coords[k+1, 1]],
                    color=cmap(norm(percentages[k])), linewidth=1.5, alpha=0.8)

        sc = ax.scatter(prompt_coords[:, 0], prompt_coords[:, 1],
                        c=percentages, cmap="viridis", s=20, zorder=5,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, shrink=0.75, label="Progress")

        # bad anchor
        ax.scatter(bad_2d[0], bad_2d[1], marker="X", s=150, color="#E24B4A",
                   zorder=10, label="Bad anchor")

        # start and end markers
        ax.scatter(prompt_coords[0, 0], prompt_coords[0, 1], marker="o",
                   s=80, color="navy", zorder=10, label="Start")
        ax.scatter(prompt_coords[-1, 0], prompt_coords[-1, 1], marker="*",
                   s=150, color="gold", edgecolors="darkorange", zorder=10,
                   label="End")

        var_exp = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} var)", fontsize=7)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} var)", fontsize=7)
        ax.set_title(f"Category {cat}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="best", markerscale=0.8)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "PCA trajectories of soft prompt vectors through training\n"
        "Linear projection — colour = training progress (purple→yellow)\n"
        "Smooth arcs away from the red ✕ indicate clean directional unlearning",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("spul_pca_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_pca_trajectories.png")


def plot_rate_of_change(stats_df, analysis_df):
    """
    Plot the instantaneous rate of change (first derivative of safety score
    w.r.t. training progress).  Shows *when* the most unlearning happens —
    early, late, or steadily throughout.
    """
    print("--- Plotting rate of change ---")
    cats = sorted(stats_df["Category"].unique(), key=lambda s: int(s[1:]))

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cat in enumerate(cats):
        ax = axes[idx]
        colour = PALETTE[idx % len(PALETTE)]
        grp = stats_df[stats_df["Category"] == cat].sort_values("Percentage")
        x = grp["Percentage"].values
        y = grp["Safety_Score"].values

        # central differences for derivative
        dx = np.diff(x)
        dy = np.diff(y)
        deriv = dy / dx
        mid_x = (x[:-1] + x[1:]) / 2

        ax.bar(mid_x, deriv, width=np.minimum(dx * 0.8, 0.04),
               color=[colour if d >= 0 else "#E24B4A" for d in deriv],
               alpha=0.75, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

        # smoothed derivative
        if len(deriv) >= 5:
            smooth_d = uniform_filter1d(deriv, size=4)
            ax.plot(mid_x, smooth_d, "-", color="black", linewidth=1.5,
                    alpha=0.6, label="Smoothed")

        ax.set_xlabel("Training progress", fontsize=7)
        ax.set_ylabel("ΔSafety / ΔProgress", fontsize=7)
        ax.set_title(f"Category {cat}", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

        # annotate peak
        peak_idx = np.argmax(deriv)
        ax.annotate(f"peak\n@ {mid_x[peak_idx]:.2f}",
                    xy=(mid_x[peak_idx], deriv[peak_idx]),
                    xytext=(mid_x[peak_idx] + 0.05, deriv[peak_idx] * 0.9),
                    fontsize=6, color="dimgrey",
                    arrowprops=dict(arrowstyle="->", color="dimgrey", lw=0.7))

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Rate of change of safety score during training\n"
        "Positive bars = safety improving; red bars = regression steps\n"
        "Peak location shows when most unlearning happens",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("spul_rate_of_change.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_rate_of_change.png")


def plot_cross_category_interference(df, bad_centroids):
    """
    For every category pair (A, B): compute the distance from each B soft prompt
    to A's bad centroid.  If unlearning category B also moves the prompt far from
    A's bad centroid, the categories share representational structure.
    Shown as a heatmap of cross-category distances at training progress = 1.0.
    """
    print("--- Plotting cross-category interference heatmap ---")
    cats = sorted(bad_centroids.keys(), key=lambda s: int(s[1:]))

    matrix = np.zeros((len(cats), len(cats)))
    for i, cat_a in enumerate(cats):
        bad_c = bad_centroids[cat_a]
        for j, cat_b in enumerate(cats):
            vecs = df[
                (df["category"] == cat_b) &
                (df["type"] == "SPUL") &
                (df["percentage"] >= 0.99)
            ]["vector"].values
            if len(vecs) == 0:
                matrix[i, j] = np.nan
            else:
                dists = [np.linalg.norm(v - bad_c) for v in vecs]
                matrix[i, j] = np.mean(dists)

    df_mat = pd.DataFrame(matrix, index=cats, columns=cats)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(cats), dtype=bool)  # mask diagonal for visual clarity
    sns.heatmap(df_mat, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, mask=mask,
                cbar_kws={"label": "Distance from row-category bad anchor"})

    # overlay diagonal separately in grey
    for k in range(len(cats)):
        ax.add_patch(mpatches.Rectangle((k, k), 1, 1, fill=True,
                     color="lightgrey", zorder=2))
        ax.text(k + 0.5, k + 0.5, f"{matrix[k,k]:.1f}",
                ha="center", va="center", fontsize=8, color="dimgrey")

    ax.set_xlabel("Soft prompt trained for category (column)", fontsize=10)
    ax.set_ylabel("Distance measured from bad anchor of category (row)", fontsize=10)
    ax.set_title(
        "Cross-category interference at full training (progress = 1.0)\n"
        "High off-diagonal = unlearning category B also moves away from A's bad anchor\n"
        "Diagonal = own-category distance (same as final safety score)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("spul_cross_category_interference.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_cross_category_interference.png")


def plot_vector_norm_over_training(df):
    """
    Track the L2 norm of the soft prompt vectors over training.
    If the norm grows, the model is amplifying the prompt magnitude.
    If it stabilises, the direction changes but magnitude is controlled.
    Helps diagnose whether unlearning is driven by direction change vs magnitude growth.
    """
    print("--- Plotting vector norm over training ---")
    cats = sorted(df["category"].unique(), key=lambda s: int(s[1:]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cat in enumerate(cats):
        grp = df[(df["category"] == cat) & (df["type"] == "SPUL")].copy()
        grp = grp.sort_values("percentage")
        norms_by_pct = grp.groupby("percentage")["vector"].apply(
            lambda vs: np.mean([np.linalg.norm(v) for v in vs])
        ).reset_index()
        norms_by_pct.columns = ["percentage", "mean_norm"]
        colour = PALETTE[i % len(PALETTE)]
        ax.plot(norms_by_pct["percentage"], norms_by_pct["mean_norm"],
                "-", color=colour, linewidth=1.4, alpha=0.85, label=cat)

    ax.set_xlabel("Training progress", fontsize=10)
    ax.set_ylabel("Mean L2 norm of soft prompt vector", fontsize=10)
    ax.set_title(
        "Soft prompt vector magnitude over training\n"
        "Rising norm → magnitude-driven separation; flat norm → direction-driven",
        fontsize=12, fontweight="bold"
    )
    ax.legend(ncol=4, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig("spul_vector_norms.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_vector_norms.png")


def plot_early_late_slope_comparison(analysis_df):
    """
    Bar chart comparing the slope in the first vs second half of training
    for each category.  Reveals whether unlearning is front-loaded, back-loaded,
    or evenly distributed.
    """
    print("--- Plotting early vs late slope comparison ---")
    cats_sorted = analysis_df.sort_values("Final_Score", ascending=False)["Category"].tolist()
    x = np.arange(len(cats_sorted))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    early_slopes = [analysis_df[analysis_df["Category"] == c]["Slope_Early_Half"].values[0]
                    for c in cats_sorted]
    late_slopes = [analysis_df[analysis_df["Category"] == c]["Slope_Late_Half"].values[0]
                   for c in cats_sorted]

    bars1 = ax.bar(x - w/2, early_slopes, w, label="First half (0→0.5)",
                   color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w/2, late_slopes, w, label="Second half (0.5→1.0)",
                   color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Slope (ΔSafety / ΔProgress)", fontsize=10)
    ax.set_title(
        "Rate of unlearning: first half vs second half of training\n"
        "Categories where early > late front-load their gains",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("spul_early_late_slope.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_early_late_slope.png")


# ==========================================
# 8. UMAP (improved)
# ==========================================
def plot_umap(df):
    cats = sorted(df["category"].unique(), key=lambda s: int(s[1:]))
    ncols = 4
    nrows = int(np.ceil(len(cats) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for idx, cat in enumerate(cats):
        ax = axes[idx]
        colour = PALETTE[idx % len(PALETTE)]
        cat_df = df[df["category"] == cat]

        spul = cat_df[cat_df["type"] == "SPUL"].copy()
        land = cat_df[cat_df["type"] == "Landmark"]

        if not spul.empty:
            sc = ax.scatter(spul["x"], spul["y"], c=spul["percentage"],
                            cmap="viridis", s=30, alpha=0.7, vmin=0, vmax=1,
                            linewidths=0)
            plt.colorbar(sc, ax=ax, shrink=0.7, label="Progress")

        if not land.empty:
            ax.scatter(land["x"], land["y"], marker="X", s=160,
                       color="#E24B4A", zorder=10, label="Bad anchor")

        final = spul[spul["percentage"] >= 0.99]
        for _, row in final.iterrows():
            ax.annotate("1.0", (row["x"], row["y"]), fontsize=7.5,
                        fontweight="bold", xytext=(3, 3),
                        textcoords="offset points", color="black")

        ax.set_title(f"Category {cat}", fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.tick_params(labelsize=6)

        legend_elements = [
            Line2D([0], [0], marker="X", color="w", markerfacecolor="#E24B4A",
                   markersize=8, label="Bad anchor"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#440154",
                   markersize=5, label="0.0 (start)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FDE725",
                   markersize=5, label="1.0 (full)"),
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc="lower right",
                  framealpha=0.8)

    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "BlackSheep-Llama3.2-3B: SPUL embedding trajectories\n"
        "(colour = training progress; red ✕ = bad anchor)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig("spul_umap_improved.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved spul_umap_improved.png")


# ==========================================
# 9. SUMMARY
# ==========================================
def print_summary(analysis_df):
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(analysis_df.to_string(index=False))

    best = analysis_df.iloc[0]
    worst = analysis_df.iloc[-1]
    fastest = analysis_df.sort_values("Slope", ascending=False).iloc[0]
    most_stable = analysis_df.sort_values("Oscillation_Index").iloc[0]
    earliest = analysis_df.dropna(subset=["Plateau_Onset"]).sort_values(
        "Plateau_Onset").iloc[0]
    most_frontloaded = analysis_df.sort_values("Early_Efficiency",
                                               ascending=False).iloc[0]

    print(f"\n  Highest final score:    {best['Category']}  ({best['Final_Score']:.2f})")
    print(f"  Lowest final score:     {worst['Category']}  ({worst['Final_Score']:.2f})")
    print(f"  Fastest rising:         {fastest['Category']}  (slope={fastest['Slope']:.2f})")
    print(f"  Most stable:            {most_stable['Category']}  (osc={most_stable['Oscillation_Index']:.4f})")
    print(f"  Earliest plateau:       {earliest['Category']}  (onset @ {earliest['Plateau_Onset']:.2f})")
    print(f"  Most front-loaded:      {most_frontloaded['Category']}  (early eff={most_frontloaded['Early_Efficiency']:.2%})")
    print("="*70)

    analysis_df.to_csv("spul_analysis_summary.csv", index=False)
    print("  Full table saved to spul_analysis_summary.csv")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    final_df, stats_df, bad_centroids = run_pipeline()

    print("\n--- Computing derived analysis ---")
    analysis_df = compute_analysis(stats_df)
    print_summary(analysis_df)

    print("\n--- Generating per-category plots (fixed labels) ---")
    plot_per_category(stats_df, analysis_df)

    print("\n--- Generating overview plots ---")
    plot_overview(stats_df, analysis_df)

    print("\n--- Generating mechanistic visualizations ---")
    plot_direction_of_movement(final_df, bad_centroids)
    plot_cosine_similarity(stats_df)
    plot_pca_trajectories(final_df, bad_centroids)
    plot_rate_of_change(stats_df, analysis_df)
    plot_cross_category_interference(final_df, bad_centroids)
    plot_vector_norm_over_training(final_df)
    plot_early_late_slope_comparison(analysis_df)

    print("\n--- Generating improved UMAP ---")
    plot_umap(final_df)

    print("\nDone. Output files:")
    print("  Per-category:  spul_category_S*.png")
    print("  Overview:      spul_overview_trajectories.png")
    print("                 spul_final_scores_ranked.png")
    print("                 spul_analysis_scatter.png")
    print("                 spul_checkpoint_heatmap.png")
    print("  Mechanistic:   spul_direction_of_movement.png")
    print("                 spul_cosine_similarity.png")
    print("                 spul_pca_trajectories.png")
    print("                 spul_rate_of_change.png")
    print("                 spul_cross_category_interference.png")
    print("                 spul_vector_norms.png")
    print("                 spul_early_late_slope.png")
    print("  UMAP:          spul_umap_improved.png")
    print("  Data:          safety_scores.csv")
    print("                 spul_analysis_summary.csv")