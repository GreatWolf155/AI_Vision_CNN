"""Charts in the site's blue-purple theme.

Interactive Plotly figures for the Streamlit site, and static PNGs (matplotlib) for the REST API.
Both draw the same data the same way.
"""

import io

import numpy as np

from . import config as c

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]


# ---------- Plotly (Streamlit) ----------

def _layout(fig, height, **kwargs):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=c.INK, size=13),
        hoverlabel=dict(bgcolor="white", bordercolor=c.GRID, font=dict(family=FONT, color=c.INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=c.GRID, linecolor=c.GRID, zeroline=False, tickfont=dict(color=c.INK_MUTED))
    fig.update_yaxes(gridcolor=c.GRID, linecolor=c.GRID, zeroline=False, tickfont=dict(color=c.INK_MUTED))
    return fig


def probability_chart(prediction, spec, compact=False):
    # One bar per class, most likely first; the winner is highlighted and the threshold is marked
    import plotly.graph_objects as go

    ranked = prediction.probabilities
    labels = [spec.label(name) for name, _ in ranked][::-1]  # Plotly draws bottom-up
    values = [p * 100 for _, p in ranked][::-1]
    top_color = c.MUTED_BAR if prediction.is_unknown else c.SERIES[0]
    colors = [c.MUTED_BAR] * (len(values) - 1) + [top_color]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=c.INK),
        hovertemplate="%{y}: <b>%{x:.2f}%</b><extra></extra>",
    ))
    threshold = prediction.threshold * 100
    fig.add_vline(x=threshold, line=dict(color=c.PURPLE, width=1.5))
    fig.add_annotation(
        x=threshold, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
        text=f"{threshold:.0f}% threshold", font=dict(size=11, color=c.PURPLE),
    )
    bar_height = 30 if compact else 36
    _layout(fig, height=70 + bar_height * len(values), bargap=0.25, showlegend=False)
    fig.update_xaxes(range=[0, 112], ticksuffix="%", dtick=25)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=c.INK, size=13))
    return fig


def confusion_chart(stats, compact=False):
    # Rows are the real class, columns the predicted class; every cell is a number of test images.
    # The color is the cell's share of its row, so small classes (e.g. 236 eggplants next to
    # 5,350 apples) still show a dark diagonal when they are classified well.
    import plotly.graph_objects as go

    matrix = np.array(stats["confusion_matrix"], dtype=float)
    counts = matrix.astype(int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = matrix / np.maximum(row_totals, 1)
    labels = stats["labels"]
    text = [[f"{counts[i, j]:,}" for j in range(len(labels))] for i in range(len(labels))]
    custom = np.dstack([counts, np.repeat(row_totals.astype(int), len(labels), axis=1)])

    scale = [[k / (len(c.BLUE_RAMP) - 1), color] for k, color in enumerate(c.BLUE_RAMP)]
    fig = go.Figure(go.Heatmap(
        z=shares, x=labels, y=labels, colorscale=scale, zmin=0, zmax=1, showscale=False,
        text=text, texttemplate="%{text}", textfont=dict(size=11 if compact else 13),
        customdata=custom, xgap=2, ygap=2,
        hovertemplate="Real: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>"
                      "<b>%{customdata[0]:,}</b> of %{customdata[1]:,} images<extra></extra>",
    ))
    side = 340 if compact else 120 + 62 * len(labels)
    _layout(fig, height=max(side, 340))
    fig.update_xaxes(title="Predicted class", side="bottom", gridcolor="rgba(0,0,0,0)",
                     tickangle=-35 if compact or len(labels) > 5 else 0)
    fig.update_yaxes(title="Real class", autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return fig


def distribution_chart(stats, compact=False):
    # How many training and test images there are of every class
    import plotly.graph_objects as go

    data = stats["dataset"]
    labels = stats["labels"]
    fig = go.Figure()
    for (key, name), color in zip([("train_counts", "Training images"), ("test_counts", "Test images")], c.SERIES):
        values = [data[key].get(cls, 0) for cls in stats["classes"]]
        fig.add_bar(name=name, x=labels, y=values, marker=dict(color=color, cornerradius=4),
                    hovertemplate=f"%{{x}}<br>{name}: <b>%{{y:,}}</b><extra></extra>")
    _layout(fig, height=280 if compact else 340, barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", tickangle=-30 if compact and len(labels) > 4 else 0)
    fig.update_yaxes(separatethousands=True)
    return fig


def threshold_chart(stats, threshold, compact=False):
    # What the "Unknown" threshold does: raising it makes the model answer fewer test images,
    # but the answers it still gives are more often right
    import plotly.graph_objects as go

    curve = [p for p in stats["confidence"]["threshold_curve"] if p["accuracy"] is not None]
    x = [p["threshold"] for p in curve]
    series = [
        ("accuracy", "Accuracy of the answers", c.SERIES[0]),
        ("answered_fraction", "Share of images answered", c.SERIES[1]),
    ]
    fig = go.Figure()
    for key, name, color in series:
        fig.add_scatter(
            x=x, y=[p[key] for p in curve], name=name, mode="lines",
            line=dict(color=color, width=2.5, shape="spline", smoothing=0.4),
            customdata=[p["answered"] for p in curve],
            hovertemplate=f"{name}: <b>%{{y:.3f}}</b>" + (" (%{customdata:,} images)" if key == "answered_fraction" else "")
                          + "<extra></extra>",
        )
        # Direct label at the end of each line
        fig.add_annotation(x=x[-1], y=curve[-1][key], text=f" {name.split(' ')[0] if key == 'accuracy' else 'Answered'}",
                           xanchor="left", showarrow=False, font=dict(size=11, color=c.INK_MUTED))

    # The current threshold, with the two values it gives
    nearest = min(curve, key=lambda p: abs(p["threshold"] - threshold))
    fig.add_vline(x=threshold, line=dict(color=c.PURPLE, width=1.5))
    fig.add_annotation(x=threshold, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"Your threshold: {threshold:.2f}", font=dict(size=11, color=c.PURPLE))
    for key, _name, color in series:
        fig.add_scatter(x=[nearest["threshold"]], y=[nearest[key]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(size=10, color=color, line=dict(color="white", width=2)))

    low = min(min(p["accuracy"], p["answered_fraction"]) for p in curve)
    _layout(fig, height=300 if compact else 360, hovermode="x unified")
    fig.update_xaxes(title="Threshold (minimum confidence needed to answer)", range=[0, max(x) + 0.08],
                     dtick=0.1, tickformat=".1f", showspikes=True, spikemode="across", spikethickness=1,
                     spikedash="solid", spikecolor=c.GRID)
    fig.update_yaxes(range=[max(0, low - 0.03), 1.005], tickformat=".2f")
    return fig


# ---------- Static PNGs (REST API) ----------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "text.color": c.INK, "axes.labelcolor": c.INK, "axes.edgecolor": c.GRID,
        "xtick.color": c.INK_MUTED, "ytick.color": c.INK_MUTED,
        "axes.grid": True, "grid.color": c.GRID, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": c.SURFACE, "axes.facecolor": c.SURFACE,
    })
    return plt


def _png(fig, plt):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _plain(label):
    # matplotlib's fonts have no color emoji, so drop the emoji from the labels
    return label.split(" ", 1)[1] if label and not label[0].isascii() and " " in label else label


def png_per_class(stats):
    plt = _plt()
    labels = [_plain(r["label"]) for r in stats["per_class"]]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4))
    for k, ((key, name), color) in enumerate(zip(METRICS, c.SERIES)):
        ax.bar(x + (k - 1) * width, [r[key] * 100 for r in stats["per_class"]], width * 0.92, label=name, color=color)
    ax.set_xticks(x, labels, rotation=30 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(f"{stats['model_name']} - per-class metrics", loc="left", pad=28, fontweight="bold")
    return _png(fig, plt)


def png_confusion(stats):
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    matrix = np.array(stats["confusion_matrix"], dtype=float)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    labels = [_plain(label) for label in stats["labels"]]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.0 * n + 1.8))
    cmap = LinearSegmentedColormap.from_list("blue", c.BLUE_RAMP)
    im = ax.imshow(shares * 100, cmap=cmap, vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{shares[i, j]:.0%}\n{int(matrix[i, j]):,}", ha="center", va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.45 else c.INK)
    ax.set_xticks(range(n), labels, rotation=35, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Real class")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%d%%").outline.set_visible(False)
    ax.set_title(f"{stats['model_name']} - confusion matrix (% of each real class)", loc="left", fontweight="bold")
    return _png(fig, plt)


def png_distribution(stats):
    plt = _plt()
    labels = [_plain(label) for label in stats["labels"]]
    x = np.arange(len(labels))
    width = 0.38
    data = stats["dataset"]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels)), 4))
    for k, ((key, name), color) in enumerate(zip([("train_counts", "Training images"), ("test_counts", "Test images")], c.SERIES)):
        ax.bar(x + (k - 0.5) * width, [data[key].get(cls, 0) for cls in stats["classes"]], width * 0.92, label=name, color=color)
    ax.set_xticks(x, labels, rotation=30 if len(labels) > 5 else 0, ha="right" if len(labels) > 5 else "center")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(f"{stats['model_name']} - images per class", loc="left", pad=28, fontweight="bold")
    return _png(fig, plt)


def png_accuracy_overview(all_stats):
    # Test accuracy of every model (one series, so one color)
    plt = _plt()
    names = [s["model_name"] for s in all_stats]
    values = [s["accuracy"] * 100 for s in all_stats]
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(names) + 1.2))
    bars = ax.barh(names[::-1], values[::-1], color=c.SERIES[0], height=0.55)
    for bar, v in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", color=c.INK)
    ax.set_xlim(0, 110)
    ax.grid(axis="y", visible=False)
    ax.set_title("Test accuracy by model", loc="left", fontweight="bold")
    return _png(fig, plt)


PNG_GRAPHS = {
    "per_class_metrics": png_per_class,
    "confusion_matrix": png_confusion,
    "class_distribution": png_distribution,
}
