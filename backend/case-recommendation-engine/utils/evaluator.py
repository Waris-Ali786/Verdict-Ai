import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────
# Legum AI dark theme palette
# ──────────────────────────────────────────────────
DARK_BG   = "#0D1117"
CARD_BG   = "#161B22"
BORDER    = "#30363D"
MUTED     = "#8B949E"
TEXT      = "#E6EDF3"

COLORS = {
    "teal":   "#1D9E75",
    "purple": "#7F77DD",
    "amber":  "#EF9F27",
    "red":    "#E24B4A",
    "blue":   "#378ADD",
    "green":  "#639922",
    "pink":   "#D4537E",
}
PALETTE = list(COLORS.values())

RISK_COLORS = {
    "high":   COLORS["red"],
    "medium": COLORS["amber"],
    "low":    COLORS["teal"],
}

TYPE_COLORS = {
    "criminal":       COLORS["red"],
    "civil":          COLORS["blue"],
    "family":         COLORS["pink"],
    "service":        COLORS["amber"],
    "tax":            COLORS["green"],
    "corporate":      COLORS["purple"],
    "constitutional": COLORS["teal"],
}


def _apply_dark_theme():
    """Apply consistent dark theme to all matplotlib plots."""
    matplotlib.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.5,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  CARD_BG,
        "legend.edgecolor":  BORDER,
        "legend.labelcolor": TEXT,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _save_or_show(fig, path=None):
    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"  ✓ Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────
# Chart 1 — Dataset Overview
# ──────────────────────────────────────────────────

def plot_dataset_overview(save_path=None):
    """
    4-panel overview of the 1,414 real SC Pakistan cases.
    Panels: case types, risk levels, outcomes, year distribution.
    """
    _apply_dark_theme()

    from data.hf_dataset_loader import HFDatasetLoader
    df = HFDatasetLoader().load()

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Supreme Court of Pakistan — 1,414 Real Judgments\n"
        "HuggingFace: Ibtehaj10/supreme-court-of-pak-judgments",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Case type bar chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    tc = df["case_type"].value_counts()
    colors_t = [TYPE_COLORS.get(t, MUTED) for t in tc.index]
    bars = ax1.barh(tc.index, tc.values, color=colors_t, edgecolor="none", height=0.6)
    ax1.set_title("Case Types", fontweight="bold")
    ax1.set_xlabel("Count")
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, tc.values):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9, color=MUTED)

    # ── Risk level pie ──
    ax2 = fig.add_subplot(gs[0, 1])
    rc = df["risk_level"].value_counts()
    wedge_colors = [RISK_COLORS.get(r, MUTED) for r in rc.index]
    wedges, texts, pcts = ax2.pie(
        rc.values, labels=rc.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops={"linewidth": 1.5, "edgecolor": DARK_BG},
    )
    for t in texts + pcts:
        t.set_color(TEXT)
        t.set_fontsize(10)
    ax2.set_title("Risk Levels", fontweight="bold")

    # ── Outcomes bar ──
    ax3 = fig.add_subplot(gs[0, 2])
    oc = df["outcome"].value_counts()
    bars3 = ax3.bar(oc.index, oc.values, color=PALETTE[:len(oc)],
                    edgecolor="none", width=0.6)
    ax3.set_title("Case Outcomes", fontweight="bold")
    ax3.set_ylabel("Count")
    ax3.grid(axis="y", alpha=0.3)
    ax3.tick_params(axis="x", rotation=25)
    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1,
                 str(int(h)), ha="center", fontsize=9, color=MUTED)

    # ── Year histogram ──
    ax4 = fig.add_subplot(gs[1, 0])
    df["year"].hist(ax=ax4, bins=25, color=COLORS["teal"], edgecolor=DARK_BG, alpha=0.85)
    ax4.set_title("Cases by Year", fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Count")
    ax4.grid(axis="y", alpha=0.3)

    # ── Text length distribution ──
    ax5 = fig.add_subplot(gs[1, 1])
    lengths = df["full_text"].str.len()
    ax5.hist(lengths, bins=30, color=COLORS["purple"], edgecolor=DARK_BG, alpha=0.85)
    ax5.set_title("Judgment Text Length", fontweight="bold")
    ax5.set_xlabel("Characters")
    ax5.set_ylabel("Count")
    ax5.axvline(lengths.mean(), color=COLORS["amber"], linewidth=1.5,
                linestyle="--", label=f"Mean: {lengths.mean():.0f}")
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    # ── Case type × Risk heatmap ──
    ax6 = fig.add_subplot(gs[1, 2])
    pivot = pd.crosstab(df["case_type"], df["risk_level"])
    for col in ["high", "medium", "low"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["high", "medium", "low"]]
    im = ax6.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax6.set_xticks(range(len(pivot.columns)))
    ax6.set_xticklabels(pivot.columns, color=TEXT)
    ax6.set_yticks(range(len(pivot.index)))
    ax6.set_yticklabels(pivot.index, color=TEXT, fontsize=8)
    ax6.set_title("Case Type × Risk Level", fontweight="bold")
    plt.colorbar(im, ax=ax6, label="Count")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax6.text(j, i, str(pivot.values[i, j]),
                     ha="center", va="center", fontsize=8,
                     color="black" if pivot.values[i, j] > pivot.values.max() * 0.5 else TEXT)

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 2 — TF-IDF Term Importance
# ──────────────────────────────────────────────────

def plot_tfidf_terms(query: str, save_path=None):
    """Horizontal bar chart of top TF-IDF terms for a query."""
    _apply_dark_theme()
    from models.tfidf_retriever import TFIDFRetriever

    retriever = TFIDFRetriever(top_k=5)
    retriever.fit()
    terms = retriever.get_top_terms(query, n=15)

    if not terms:
        print("  No TF-IDF terms found.")
        return

    words  = [t[0] for t in terms][::-1]
    scores = [t[1] for t in terms][::-1]
    bar_colors = [COLORS["teal"] if i == len(scores) - 1 else COLORS["purple"]
                  for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("TF-IDF Term Importance\n(Supreme Court of Pakistan Corpus — 30,000 term vocabulary)",
                 fontsize=12, fontweight="bold", color=TEXT)

    bars = ax.barh(words, scores, color=bar_colors, edgecolor="none", height=0.65)
    ax.set_xlabel("TF-IDF Score")
    ax.set_title(f'Query: "{query[:70]}"', fontsize=9, color=MUTED)
    ax.grid(axis="x", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8, color=MUTED)

    legend_elements = [
        mpatches.Patch(color=COLORS["teal"],   label="Highest weight term"),
        mpatches.Patch(color=COLORS["purple"], label="Other important terms"),
    ]
    ax.legend(handles=legend_elements)
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 3 — TF-IDF vs BERT Retrieval Comparison
# ──────────────────────────────────────────────────

def plot_retrieval_comparison(query: str, save_path=None):
    """Side-by-side grouped bar chart: TF-IDF vs BERT similarity scores."""
    _apply_dark_theme()
    from models.tfidf_retriever import TFIDFRetriever
    from models.bert_embedder import BERTEmbedder

    print("  Fitting TF-IDF...")
    retriever = TFIDFRetriever(top_k=6)
    retriever.fit()
    tfidf_cases = retriever.retrieve_similar_cases(query)

    print("  Building BERT index (using HF embeddings)...")
    bert = BERTEmbedder(top_k=6, use_hf_embeddings=True)
    bert.build_index()
    bert_cases = bert.retrieve_similar_cases(query)

    # Build score maps
    tfidf_map = {c["citation"]: c["similarity_score"] for c in tfidf_cases}
    bert_map  = {c["citation"]: c["similarity_score"] for c in bert_cases}
    all_cits  = list({c["citation"] for c in tfidf_cases + bert_cases})[:8]

    # Truncate labels
    labels = [c[:22] + "…" if len(c) > 22 else c for c in all_cits]
    tf_scores   = [tfidf_map.get(c, 0) for c in all_cits]
    bert_scores = [bert_map.get(c, 0) for c in all_cits]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Retrieval Comparison: TF-IDF (keyword) vs Sentence-BERT (semantic)\n"
        "Real Supreme Court of Pakistan Judgments",
        fontsize=12, fontweight="bold", color=TEXT,
    )

    b1 = ax.bar(x - w / 2, tf_scores,   w, label="TF-IDF",
                color=COLORS["teal"],   edgecolor="none", alpha=0.85)
    b2 = ax.bar(x + w / 2, bert_scores, w, label="Sentence-BERT (1024-dim)",
                color=COLORS["purple"], edgecolor="none", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Similarity Score (%)")
    ax.set_ylim(0, max(max(tf_scores, default=0), max(bert_scores, default=0)) + 12)
    ax.set_title(f'Query: "{query[:80]}"', fontsize=9, color=MUTED)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, color=MUTED)

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 4 — Risk Prediction Probabilities
# ──────────────────────────────────────────────────

def plot_risk_prediction(case_text: str, save_path=None):
    """Dual horizontal bar charts: risk probabilities + outcome probabilities."""
    _apply_dark_theme()
    from models.risk_predictor import RiskPredictor

    print("  Training risk predictor on 1,414 real cases...")
    pred = RiskPredictor()
    pred.train()
    result = pred.predict(case_text)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Risk & Outcome Prediction — Random Forest + Gradient Boosting\n"
        "Trained on 1,414 Supreme Court of Pakistan Judgments",
        fontsize=12, fontweight="bold", color=TEXT,
    )

    # Risk probabilities
    risk_probs = result.get("risk_probabilities", {})
    if risk_probs:
        labels_r = list(risk_probs.keys())
        vals_r   = list(risk_probs.values())
        colors_r = [RISK_COLORS.get(l, MUTED) for l in labels_r]
        bars_r   = axes[0].barh(labels_r, vals_r, color=colors_r, edgecolor="none", height=0.5)
        axes[0].set_title(
            f"Risk Level: {result['risk_level'].upper()} ({result['risk_confidence_pct']})",
            fontweight="bold",
        )
        axes[0].set_xlabel("Probability")
        axes[0].set_xlim(0, 1.18)
        axes[0].grid(axis="x", alpha=0.3)
        for bar, val in zip(bars_r, vals_r):
            axes[0].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1%}", va="center", fontsize=11, color=TEXT)

    # Outcome probabilities
    out_probs = result.get("outcome_probabilities", {})
    if out_probs:
        labels_o = list(out_probs.keys())
        vals_o   = list(out_probs.values())
        colors_o = PALETTE[:len(labels_o)]
        bars_o   = axes[1].barh(labels_o, vals_o, color=colors_o, edgecolor="none", height=0.5)
        axes[1].set_title(
            f"Likely Outcome: {result['likely_outcome'].upper()} ({result['outcome_confidence_pct']})",
            fontweight="bold",
        )
        axes[1].set_xlabel("Probability")
        axes[1].set_xlim(0, 1.18)
        axes[1].grid(axis="x", alpha=0.3)
        for bar, val in zip(bars_o, vals_o):
            axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1%}", va="center", fontsize=11, color=TEXT)

    # Risk factors annotation
    factors_text = "\n".join(f"• {f}" for f in result.get("risk_factors", []))
    fig.text(0.5, -0.06, f"Risk Factors:\n{factors_text}",
             ha="center", fontsize=9, color=MUTED,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG, edgecolor=BORDER))

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 5 — BERT Embedding 2D PCA (all 1,414 cases)
# ──────────────────────────────────────────────────

def plot_embeddings_2d(save_path=None, n_cases=None):
    """
    Project 1024-dim BERT embeddings to 2D using PCA.
    All 1,414 real SC Pakistan cases coloured by case type.
    """
    _apply_dark_theme()
    from data.hf_dataset_loader import HFDatasetLoader
    from sklearn.decomposition import PCA

    loader = HFDatasetLoader()
    df     = loader.load()
    emb    = loader.get_embeddings_matrix()   # (1414, 1024)

    if n_cases:
        df  = df.iloc[:n_cases].reset_index(drop=True)
        emb = emb[:n_cases]

    print(f"  Running PCA on {len(df)} cases × 1024 dims...")
    pca    = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.suptitle(
        f"BERT Embeddings — 2D PCA Projection\n"
        f"{len(df)} Supreme Court of Pakistan judgments "
        f"(mxbai-embed-large-v1, 1024-dim)",
        fontsize=13, fontweight="bold", color=TEXT,
    )

    case_types = df["case_type"].unique()
    for ct in case_types:
        mask = df["case_type"] == ct
        idx  = np.where(mask)[0]
        ax.scatter(
            emb_2d[idx, 0], emb_2d[idx, 1],
            c=TYPE_COLORS.get(ct, MUTED),
            label=f"{ct} ({mask.sum()})",
            s=18, alpha=0.7, edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance explained)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance explained)")
    ax.legend(
        title="Case Type", title_fontsize=9,
        markerscale=2, framealpha=0.8,
        loc="upper right",
    )
    ax.grid(alpha=0.15)
    ax.set_title(
        "Cases with similar legal content cluster together in 1024-dim semantic space",
        fontsize=9, color=MUTED,
    )
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 6 — BiLSTM Training Curves
# ──────────────────────────────────────────────────

def plot_training_curves(history, save_path=None):
    """
    Plot accuracy and loss curves from Keras BiLSTM training history.
    Call after BiLSTMClassifier.train().
    """
    _apply_dark_theme()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "BiLSTM Training Curves — 1,414 Real Supreme Court Judgments\n"
        "(30k vocab, 512 seq len, 7 case types)",
        fontsize=12, fontweight="bold", color=TEXT,
    )

    # Accuracy
    ax = axes[0]
    ax.plot(history.history["accuracy"], color=COLORS["teal"],
            linewidth=2.5, label="Train accuracy")
    if "val_accuracy" in history.history:
        ax.plot(history.history["val_accuracy"], color=COLORS["purple"],
                linewidth=2.5, linestyle="--", label="Val accuracy")
    ax.set_title("Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    best_val = max(history.history.get("val_accuracy", [0]))
    ax.axhline(best_val, color=COLORS["amber"], linewidth=1,
               linestyle=":", alpha=0.7, label=f"Best val: {best_val:.2%}")
    ax.legend()

    # Loss
    ax2 = axes[1]
    ax2.plot(history.history["loss"], color=COLORS["red"],
             linewidth=2.5, label="Train loss")
    if "val_loss" in history.history:
        ax2.plot(history.history["val_loss"], color=COLORS["amber"],
                 linewidth=2.5, linestyle="--", label="Val loss")
    ax2.set_title("Loss", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Categorical Cross-Entropy Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────
# Chart 7 — Confusion Matrix for Risk Predictor
# ──────────────────────────────────────────────────

def plot_confusion_matrix(save_path=None, n_samples=300):
    """
    Confusion matrix for the risk predictor on real SC data.
    """
    _apply_dark_theme()
    from models.risk_predictor import RiskPredictor
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from data.hf_dataset_loader import HFDatasetLoader

    df = HFDatasetLoader().load()
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)

    print(f"  Training risk predictor on full dataset...")
    pred = RiskPredictor()
    pred.train()

    print(f"  Predicting {len(sample_df)} samples...")
    y_true, y_pred = [], []
    for _, row in sample_df.iterrows():
        p = pred.predict(row["full_text"])
        y_true.append(row["risk_level"])
        y_pred.append(p["risk_level"])

    labels = ["low", "medium", "high"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"Risk Predictor — Confusion Matrix\n"
        f"{n_samples} SC Pakistan cases  |  Accuracy: {acc:.1%}",
        fontsize=12, fontweight="bold", color=TEXT,
    )
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="YlGn", colorbar=True)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.set_title("Predicted → True Risk Level", color=MUTED, fontsize=9)

    _save_or_show(fig, save_path)
