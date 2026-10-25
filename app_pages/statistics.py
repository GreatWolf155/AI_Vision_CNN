"""Model statistics page: how well each model does on its test images."""

import pandas as pd
import streamlit as st

from app_pages import common
from vision import charts, inference

mobile = common.is_mobile()

common.hero("Model statistics", "How accurate each model is on test images it never saw during training.", icon="📊")
spec = common.model_menu("Model")
common.sidebar_footer()
s = common.get_stats(spec.id)
if s is None:
    st.warning(
        f"There are no statistics for the {spec.title} model yet. They are computed from the test images, "
        "which are not in the GitHub repo - run the site once on the computer that has the `data` folder, "
        f"then commit `models/{spec.name}.stats.json`.",
        icon="📭",
    )
    st.stop()
chart_config = {"displayModeBar": False}

# ---------- Headline numbers ----------
tiles = [
    ("Test accuracy", f"{s['accuracy']:.3f}", "Share of test images classified correctly (1 = all of them)"),
    ("Macro F1-score", f"{s['macro_avg']['f1']:.3f}", "Average F1 over the classes - every class counts the same"),
    ("Test images", f"{s['test_images']:,}", "Images the model never saw while training"),
    ("Parameters", f"{s['model']['total_params'] / 1e6:.1f} M", "Numbers the network learned"),
]
# A grid instead of st.columns, so phones get two tiles per row rather than one
st.markdown(
    '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="tile-label">{label}</div><div class="tile-value">{value}</div>'
        f'<div class="tile-help">{help_text}</div></div>'
        for label, value, help_text in tiles
    ) + "</div>",
    unsafe_allow_html=True,
)

conf = s["confidence"]
answered = round((1 - conf["share_unknown"]) * s["test_images"])
st.info(
    f"With the **Unknown** rule (confidence below {conf['threshold']:.2f}), the model answers **{answered:,}** of the "
    f"{s['test_images']:,} test images, with an accuracy of **{conf['accuracy_when_confident']:.3f}** on those.",
    icon="🎯",
)

# ---------- Details ----------
metrics_tab, matrix_tab, confidence_tab, data_tab, model_tab = st.tabs(
    ["📋 Metrics", "🔢 Confusion matrix", "🎯 Confidence", "🗂️ Dataset", "🧠 Architecture"]
)
number = st.column_config.NumberColumn

with metrics_tab:
    st.markdown(
        "All values go from **0** (worst) to **1** (best).  \n"
        "**Precision** – when the model says *X*, how often it's right.  \n"
        "**Recall** – of all the real *X* images, how many it found.  \n"
        "**F1-score** – one number that balances the two."
    )
    st.markdown("**Overall**")
    overall = pd.DataFrame([
        ("Accuracy", s["accuracy"], "Correct answers out of all test images"),
        ("Precision", s["macro_avg"]["precision"], "Average over the classes"),
        ("Recall", s["macro_avg"]["recall"], "Average over the classes"),
        ("F1-score", s["macro_avg"]["f1"], "Average over the classes"),
    ], columns=["Metric", "Value", "Notes"])
    st.dataframe(overall, hide_index=True, width="stretch",
                 column_config={"Value": number("Value", format="%.3f")})

    st.markdown("**Per class**")
    table = pd.DataFrame(s["per_class"])[["label", "precision", "recall", "f1", "support"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "label": "Class",
            "precision": number("Precision", format="%.3f"),
            "recall": number("Recall", format="%.3f"),
            "f1": number("F1-score", format="%.3f"),
            "support": number("Test images", format="localized"),
        },
    )

with matrix_tab:
    st.markdown("Each cell is a number of test images: rows are the **real** class, columns the **predicted** class. "
                "The diagonal holds the correct answers. The darker a cell, the bigger its part of that row.")
    st.plotly_chart(charts.confusion_chart(s, compact=mobile), width="stretch", config=chart_config)
    if s["top_confusions"]:
        st.markdown("**Most common mistakes**")
        totals = s["dataset"]["test_counts"]
        for mistake in s["top_confusions"][:5]:
            actual = spec.label(mistake["actual"])
            st.markdown(
                f"- {actual} → predicted as {spec.label(mistake['predicted'])}: "
                f"**{mistake['count']:,}** times (out of {totals[mistake['actual']]:,} {actual} images)"
            )

with confidence_tab:
    threshold = st.session_state.get("threshold", 60) / 100
    right, wrong = conf["mean_when_correct"], conf["mean_when_wrong"]
    st.markdown(
        "Every answer comes with a **confidence** from 0 to 1. Answers below the threshold are shown as **Unknown**. "
        "Raising the threshold means fewer answers, but the ones given are more often right."
    )
    st.markdown(
        '<div class="tiles">'
        f'<div class="tile"><div class="tile-label">Average confidence when right</div><div class="tile-value">{right:.3f}</div></div>'
        + (f'<div class="tile"><div class="tile-label">Average confidence when wrong</div><div class="tile-value">{wrong:.3f}</div></div>'
           if wrong is not None else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(charts.threshold_chart(s, threshold, compact=mobile), width="stretch", config=chart_config)
    rows = [p for p in conf["threshold_curve"] if round(p["threshold"], 2) in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)]
    st.dataframe(
        pd.DataFrame(rows)[["threshold", "answered", "accuracy"]], hide_index=True, width="stretch",
        column_config={
            "threshold": number("Threshold", format="%.2f"),
            "answered": number(f"Images answered (of {s['test_images']:,})", format="localized"),
            "accuracy": number("Accuracy of the answers", format="%.3f"),
        },
    )
    st.caption("You can change your threshold on the Classify page.")

with data_tab:
    st.markdown("How many images of each class the model learned from, and was tested on.")
    st.plotly_chart(charts.distribution_chart(s, compact=mobile), width="stretch", config=chart_config)
    counts = pd.DataFrame({
        "Class": s["labels"],
        "Training images": [s["dataset"]["train_counts"].get(c, 0) for c in s["classes"]],
        "Test images": [s["dataset"]["test_counts"].get(c, 0) for c in s["classes"]],
    })
    st.dataframe(counts, hide_index=True, width="stretch")

with model_tab:
    m = s["model"]
    channels = "grayscale" if m["color_mode"] == "grayscale" else "color (RGB)"
    st.markdown(
        f'<span class="chip">Input {m["image_size"][1]}×{m["image_size"][0]} {channels}</span>'
        f'<span class="chip">{len(m["layers"])} layers</span>'
        f'<span class="chip">{m["total_params"]:,} parameters</span>'
        f'<span class="chip">{m["file_size_mb"]} MB file</span>'
        f'<span class="chip">Saved {m["saved_at"]}</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    layers = pd.DataFrame(m["layers"])
    layers["output_shape"] = layers["output_shape"].apply(lambda shape: " × ".join(map(str, shape)) if shape else "")
    st.dataframe(
        layers, hide_index=True, width="stretch",
        column_config={
            "name": "Layer", "type": "Type", "output_shape": "Output shape",
            "params": st.column_config.NumberColumn("Parameters", format="localized"),
        },
    )

# ---------- All models side by side ----------
st.divider()
st.markdown("#### All models at a glance")
summary = []
for other in inference.available_models():
    o = common.get_stats(other.id)
    if o is None:
        continue
    summary.append({
        "Model": f"{other.icon} {other.title}",
        "Accuracy": o["accuracy"],
        "Macro F1": o["macro_avg"]["f1"],
        "Classes": len(o["classes"]),
        "Test images": o["test_images"],
    })
st.dataframe(
    pd.DataFrame(summary), hide_index=True, width="stretch",
    column_config={
        "Accuracy": number("Accuracy", format="%.3f"),
        "Macro F1": number("Macro F1", format="%.3f"),
        "Test images": number("Test images", format="localized"),
    },
)
st.caption(f"Statistics computed {s['computed_at']} · also available from the REST API at "
           f"[/api/models/{spec.id}/stats](/api/models/{spec.id}/stats) · "
           f"[report with graphs](/api/models/{spec.id}/stats/report)")
