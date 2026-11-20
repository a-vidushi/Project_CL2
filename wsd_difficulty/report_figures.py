"""Static figures for the markdown report, drawn from the CSVs in results/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# Ordinal blue steps for low / mid / high entropy
TERCILE_COLOURS = {"low": "#86b6ef", "mid": "#2a78d6", "high": "#104281"}
NEUTRAL = "#9a9890"

MODEL_LABELS = {"mfs": "MFS", "lesk": "Simplified Lesk", "naive_bayes": "Naive Bayes",
                "logistic_regression": "Logistic regression", "decision_list": "Decision list"}


def style_axes(axes, title, x_label=None, y_label=None):
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=10)
    if x_label:
        axes.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=9)
    if y_label:
        axes.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def new_figure(width=7.0, height=3.6):
    figure, axes = plt.subplots(figsize=(width, height), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    return figure, axes


def save(figure, name):
    path = config.RESULTS_DIR / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def grouped_bars(axes, groups, series_names, values, lowers, uppers, colours, labels=None):
    """Thin bars with 2px gaps between neighbours and confidence-interval whiskers."""
    width = 0.8 / len(series_names)
    positions = np.arange(len(groups))
    for index, name in enumerate(series_names):
        offsets = positions - 0.4 + width * (index + 0.5)
        axes.bar(offsets, values[index], width=width * 0.92, color=colours[index], label=labels[index] if labels else name,
                 edgecolor=SURFACE, linewidth=1)
        axes.errorbar(offsets, values[index], yerr=[np.array(values[index]) - lowers[index],
                                                    np.array(uppers[index]) - values[index]],
                      fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=0)
    axes.set_xticks(positions, groups)
    axes.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)


def accuracy_overview(accuracy):
    rows = [("mfs", "none"), ("lesk", "none"), ("decision_list", "all"), ("naive_bayes", "all"),
            ("logistic_regression", "all"), ("logistic_regression", "all+cohesion"),
            ("logistic_regression", "all+discourse"), ("logistic_regression", "all+discourse_oracle")]
    cv = accuracy[accuracy["evaluation"] == "semcor_cv"].set_index(["model", "feature_set"])
    rows = [row for row in rows if row in cv.index]
    labels = [MODEL_LABELS[model] + ("" if feature_set in ("none", "all") else f" + {feature_set[4:].replace('_', ' ')}")
              for model, feature_set in rows]
    values = [cv.loc[row, "accuracy"] for row in rows]
    lowers = [cv.loc[row, "accuracy_lower"] for row in rows]
    uppers = [cv.loc[row, "accuracy_upper"] for row in rows]

    figure, axes = new_figure(7.0, 3.8)
    positions = np.arange(len(rows))[::-1]
    axes.hlines(positions, 0.4, values, color=GRID, linewidth=1)
    axes.errorbar(values, positions, xerr=[np.array(values) - lowers, np.array(uppers) - values],
                  fmt="o", color=SERIES[0], markersize=6, elinewidth=1.2, ecolor=SERIES[0])
    for position, value in zip(positions, values):
        axes.text(value + 0.008, position, f"{value:.1%}", va="center", fontsize=8, color=TEXT_PRIMARY)
    axes.set_yticks(positions, labels)
    axes.set_xlim(0.4, max(uppers) + 0.05)
    axes.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style_axes(axes, "Held-out accuracy on SemCor (10-fold, by document)", x_label="Accuracy, 95% bootstrap CI")
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    return save(figure, "accuracy_overview.png")


def entropy_scatter(word_accuracy, word_table):
    data = word_accuracy[(word_accuracy["model"] == "logistic_regression") & (word_accuracy["feature_set"] == "all")]
    data = data.merge(word_table[["target_word", "sense_entropy", "mfs_probability"]], on="target_word")
    figure, axes = new_figure(7.0, 3.6)
    scatter = axes.scatter(data["sense_entropy"], data["accuracy"], c=data["mfs_probability"], cmap="Blues",
                           vmin=0, vmax=1, s=14, edgecolors=SURFACE, linewidths=0.5)
    bar = figure.colorbar(scatter, ax=axes, pad=0.01)
    bar.set_label("MFS probability (skew)", color=TEXT_SECONDARY, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=0)
    bar.outline.set_visible(False)
    axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style_axes(axes, "Per-word accuracy falls with sense entropy, but entropy tracks skew",
               x_label="Sense entropy H(S|w), bits", y_label="Logistic regression accuracy")
    return save(figure, "entropy_vs_accuracy.png")


def lift_by_tercile(lift):
    data = lift[(lift["feature_set"] == "all") & (lift["pos"] == "ALL")]
    models = ["naive_bayes", "logistic_regression", "decision_list"]
    terciles = ["low", "mid", "high"]
    figure, axes = new_figure()
    values, lowers, uppers = [], [], []
    for model in models:
        rows = data[data["model"] == model].set_index("entropy_tercile").reindex(terciles)
        values.append(rows["mean_context_lift"].to_numpy())
        lowers.append(rows["lower"].to_numpy())
        uppers.append(rows["upper"].to_numpy())
    grouped_bars(axes, [f"{t} entropy" for t in terciles], models, values, lowers, uppers, SERIES,
                 [MODEL_LABELS[m] for m in models])
    axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axes(axes, "Context helps most where senses are least skewed", y_label="Accuracy minus MFS (mean over words)")
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=3, loc="upper left")
    return save(figure, "context_lift_by_tercile.png")


def information_gain_by_context(gain):
    data = gain[(gain["model"] == "logistic_regression") & (gain["pos"] == "ALL")]
    feature_sets = ["window", "collocations", "pos", "dependencies", "all"]
    terciles = ["low", "mid", "high"]
    figure, axes = new_figure()
    values, lowers, uppers = [], [], []
    for tercile in terciles:
        rows = data[data["entropy_tercile"] == tercile].set_index("feature_set").reindex(feature_sets)
        values.append(rows["mean_information_gain"].to_numpy())
        lowers.append(rows["lower"].to_numpy())
        uppers.append(rows["upper"].to_numpy())
    grouped_bars(axes, feature_sets, terciles, values, lowers, uppers, [TERCILE_COLOURS[t] for t in terciles],
                 [f"{t} entropy" for t in terciles])
    style_axes(axes, "Bits of sense uncertainty removed by each context type",
               y_label="H(S|w) − H(S|w,C), bits (held out)")
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=3, loc="upper left")
    return save(figure, "information_gain_by_context.png")


def one_sense_per_discourse(consistency):
    data = consistency[(consistency["entropy_tercile"] == "ALL") & (consistency["pos"] != "ALL")]
    data = data.set_index("pos").reindex(["NOUN", "VERB", "ADJ", "ADV"]).dropna(how="all")
    figure, axes = new_figure(7.0, 3.4)
    positions = np.arange(len(data))
    axes.bar(positions - 0.2, data["chance_agreement"], width=0.38, color=NEUTRAL, label="Chance (Σ p²)",
             edgecolor=SURFACE, linewidth=1)
    axes.bar(positions + 0.2, data["pairwise_agreement"], width=0.38, color=SERIES[0],
             label="Same document", edgecolor=SURFACE, linewidth=1)
    for position, value in zip(positions, data["pairwise_agreement"]):
        axes.text(position + 0.2, value + 0.01, f"{value:.0%}", ha="center", fontsize=8, color=TEXT_PRIMARY)
    axes.set_xticks(positions, data.index)
    axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axes(axes, "Two mentions in one document share a sense far more often than chance",
               y_label="Pairwise sense agreement")
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")
    return save(figure, "one_sense_per_discourse.png")


def discourse_gain(gain):
    data = gain[(gain["model"] == "logistic_regression") & (gain["pos"] == "ALL")]
    order = ["all+previous_mention", "all+document_senses", "all+lexical_chains", "all+rst",
             "all+cohesion", "all+discourse", "all+discourse_oracle"]
    order = [feature_set for feature_set in order if feature_set in set(data["feature_set"])]
    terciles = ["low", "mid", "high"]
    figure, axes = new_figure(7.4, 3.8)
    values, lowers, uppers = [], [], []
    for tercile in terciles:
        rows = data[data["entropy_tercile"] == tercile].set_index("feature_set").reindex(order)
        values.append(rows["mean_discourse_gain"].to_numpy())
        lowers.append(rows["lower"].to_numpy())
        uppers.append(rows["upper"].to_numpy())
    grouped_bars(axes, [feature_set[4:].replace("_", "\n") for feature_set in order], terciles, values, lowers, uppers,
                 [TERCILE_COLOURS[t] for t in terciles], [f"{t} entropy" for t in terciles])
    axes.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    style_axes(axes, "Accuracy gained by adding discourse features (logistic regression)",
               y_label="Accuracy minus sentence-level model")
    axes.tick_params(axis="x", labelsize=7)
    axes.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncols=3, loc="upper left")
    return save(figure, "discourse_gain_by_tercile.png")


def draw_all():
    results = config.RESULTS_DIR
    word_table = pd.read_csv(results / "word_statistics.csv")
    paths = [
        accuracy_overview(pd.read_csv(results / "accuracy_overall.csv")),
        entropy_scatter(pd.read_csv(results / "accuracy_by_word.csv"), word_table),
        lift_by_tercile(pd.read_csv(results / "context_lift_by_tercile_and_pos.csv")),
        information_gain_by_context(pd.read_csv(results / "rq2_information_gain_by_tercile_and_pos.csv")),
        one_sense_per_discourse(pd.read_csv(results / "discourse_one_sense_per_discourse.csv")),
    ]
    discourse_file = results / "rq4_discourse_gain_by_tercile_and_pos.csv"
    if discourse_file.exists():
        paths.append(discourse_gain(pd.read_csv(discourse_file)))
    for path in paths:
        print(f"  drew {path.name}")


if __name__ == "__main__":
    draw_all()
