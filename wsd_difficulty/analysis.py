"""Result tables for RQ1 (entropy vs. accuracy) and RQ2 (information from context)."""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from . import statistics
from .experiments import CROSS_VALIDATION
from .information import information_gain_table
from .models import CONTEXT_FREE_MODELS

DIFFICULTY_PREDICTORS = (
    "sense_entropy", "normalised_entropy", "attested_senses", "wordnet_senses",
    "mean_sense_information_content", "frequency_weighted_relatedness_z",
)
CONTROLS = ("log_frequency", "mfs_probability")


def word_accuracy_table(predictions, evaluation=CROSS_VALIDATION):
    subset = predictions[predictions["evaluation"] == evaluation]
    accuracy = (subset.groupby(["target_word", "model", "feature_set"])["correct"]
                .agg(accuracy="mean", instances="size").reset_index())
    mfs_accuracy = (accuracy[accuracy["model"] == "mfs"][["target_word", "accuracy"]]
                    .rename(columns={"accuracy": "mfs_accuracy"}))
    accuracy = accuracy.merge(mfs_accuracy, on="target_word", how="left")
    accuracy["context_lift"] = accuracy["accuracy"] - accuracy["mfs_accuracy"]
    return accuracy


def overall_accuracy_table(predictions):
    rows = []
    for (evaluation, model, feature_set), group in predictions.groupby(["evaluation", "model", "feature_set"]):
        mean, lower, upper = statistics.bootstrap_mean_interval(group["correct"].astype(float))
        rows.append({"evaluation": evaluation, "model": model, "feature_set": feature_set,
                     "instances": len(group), "target_words": group["target_word"].nunique(),
                     "accuracy": mean, "accuracy_lower": lower, "accuracy_upper": upper})
    return pd.DataFrame(rows)


def main_configurations(word_accuracy, feature_set="all"):
    """One row per model: context-free models as they are, the others with the given feature set."""
    is_context_free = word_accuracy["model"].isin(CONTEXT_FREE_MODELS)
    return word_accuracy[is_context_free | (word_accuracy["feature_set"] == feature_set)]


def stratified_means(table, value_column, group_columns):
    """Bootstrap mean of a word-level value per stratum, plus an 'ALL' POS row per remaining grouping."""
    rows = []
    with_all_pos = pd.concat([table, table.assign(pos="ALL")])
    for keys, group in with_all_pos.groupby(group_columns, observed=True):
        mean, lower, upper = statistics.bootstrap_mean_interval(group[value_column])
        rows.append({**dict(zip(group_columns, keys)), "target_words": len(group),
                     f"mean_{value_column}": mean, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def context_lift_by_stratum(word_accuracy, word_table):
    merged = word_accuracy[word_accuracy["model"] != "mfs"].merge(word_table, on="target_word")
    return stratified_means(merged, "context_lift", ["model", "feature_set", "entropy_tercile", "pos"])


def difficulty_correlations(word_accuracy, word_table):
    """RQ1: Spearman and partial Spearman (controlling frequency and skew) of difficulty measures with accuracy."""
    merged = main_configurations(word_accuracy).merge(word_table, on="target_word")
    rows = []
    for (model, feature_set), group in merged.groupby(["model", "feature_set"]):
        outcomes = ["accuracy"] if model == "mfs" else ["accuracy", "context_lift"]
        for outcome in outcomes:
            for predictor in DIFFICULTY_PREDICTORS:
                if predictor not in group or group[predictor].isna().all():
                    continue
                rho, p_value = statistics.spearman(group[predictor], group[outcome])
                partial_rho, partial_p_value, words = statistics.partial_spearman(group, predictor, outcome, CONTROLS)
                rows.append({"model": model, "feature_set": feature_set, "outcome": outcome, "predictor": predictor,
                             "target_words": words, "spearman": rho, "spearman_p": p_value,
                             "partial_spearman": partial_rho, "partial_spearman_p": partial_p_value})
    return pd.DataFrame(rows)


def entropy_beyond_baseline_regressions(word_accuracy, word_table):
    """RQ1: does sense entropy explain accuracy once MFS probability and frequency are in the model?"""
    merged = main_configurations(word_accuracy).merge(word_table, on="target_word")
    rows = []
    for (model, feature_set), group in merged.groupby(["model", "feature_set"]):
        baseline = smf.ols("accuracy ~ mfs_probability + log_frequency", data=group).fit()
        with_entropy = smf.ols("accuracy ~ mfs_probability + log_frequency + sense_entropy", data=group).fit()
        robust = with_entropy.get_robustcov_results(cov_type="HC3")
        entropy_position = with_entropy.model.exog_names.index("sense_entropy")
        comparison = anova_lm(baseline, with_entropy)
        rows.append({
            "model": model, "feature_set": feature_set, "target_words": len(group),
            "r_squared_baseline": baseline.rsquared, "r_squared_with_entropy": with_entropy.rsquared,
            "r_squared_gain": with_entropy.rsquared - baseline.rsquared,
            "entropy_coefficient": with_entropy.params["sense_entropy"],
            "entropy_p_robust": robust.pvalues[entropy_position],
            "nested_f_test_p": comparison["Pr(>F)"].iloc[1],
        })
    return pd.DataFrame(rows)


def collinearity_table(word_table):
    columns = ["sense_entropy", "mfs_probability", "log_frequency", "attested_senses",
               "mean_sense_information_content", "frequency_weighted_relatedness_z"]
    return statistics.variance_inflation_factors(word_table, columns)


def information_gain_by_stratum(predictions, word_table, word_accuracy):
    """RQ2: bits of sense uncertainty removed by each context type, and whether that turns into accuracy."""
    tables, correlation_rows = [], []
    for model_name in ("naive_bayes", "logistic_regression"):
        gain = information_gain_table(predictions, model_name)
        if gain.empty:
            continue
        gain = gain.merge(word_table, on="target_word")
        tables.append(stratified_means(gain, "information_gain", ["model", "feature_set", "entropy_tercile", "pos"]))

        with_lift = gain.merge(word_accuracy[["target_word", "model", "feature_set", "context_lift"]],
                               on=["target_word", "model", "feature_set"])
        for feature_set, group in with_lift.groupby("feature_set"):
            rho, p_value = statistics.spearman(group["information_gain"], group["context_lift"])
            correlation_rows.append({"model": model_name, "feature_set": feature_set, "target_words": len(group),
                                     "spearman_gain_vs_lift": rho, "p_value": p_value})
    stratified = pd.concat(tables) if tables else pd.DataFrame()
    return stratified, pd.DataFrame(correlation_rows)


def naive_bayes_versus_logistic_regression(predictions):
    """McNemar's test on paired predictions and calibration of each model's confidence."""
    rows = []
    pair = predictions[predictions["model"].isin(["naive_bayes", "logistic_regression"])]
    for (evaluation, feature_set), group in pair.groupby(["evaluation", "feature_set"]):
        wide = group.pivot_table(index="instance_id", columns="model",
                                 values=["correct", "predicted_probability"], aggfunc="first").dropna()
        if wide.empty:
            continue
        test = statistics.mcnemar_test(wide[("correct", "naive_bayes")], wide[("correct", "logistic_regression")])
        row = {"evaluation": evaluation, "feature_set": feature_set, "paired_instances": len(wide),
               "mcnemar_only_naive_bayes_correct": test["only_first_correct"],
               "mcnemar_only_logistic_regression_correct": test["only_second_correct"],
               "mcnemar_p": test["p_value"]}
        for model in ("naive_bayes", "logistic_regression"):
            correct = wide[("correct", model)].astype(float)
            confidence = wide[("predicted_probability", model)].astype(float)
            row[f"{model}_accuracy"] = correct.mean()
            row[f"{model}_mean_confidence"] = confidence.mean()
            row[f"{model}_expected_calibration_error"] = statistics.expected_calibration_error(confidence, correct)
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_relatedness_summary(confusions):
    """RQ3: paired comparison of relatedness(gold, predicted) with relatedness(gold, other wrong senses)."""
    rows = []
    with_all_pos = pd.concat([confusions, confusions.assign(pos="ALL")])
    for error_type, subset in (("all_errors", with_all_pos),
                               ("errors_not_choosing_mfs", with_all_pos[~with_all_pos["predicted_is_mfs"]])):
        for (measure, pos), group in subset.groupby(["relatedness_measure", "pos"]):
            difference = group["relatedness_to_predicted"] - group["mean_relatedness_to_alternatives"]
            test = statistics.wilcoxon_signed_rank(difference)
            rows.append({
                "error_type": error_type, "relatedness_measure": measure, "pos": pos, "errors": len(group),
                "mean_relatedness_to_predicted": group["relatedness_to_predicted"].mean(),
                "mean_relatedness_to_alternatives": group["mean_relatedness_to_alternatives"].mean(),
                "mean_difference": difference.mean(),
                "wilcoxon_p": test,
                "share_predicted_most_related": group["predicted_is_most_related"].mean(),
                "chance_share": group["chance_most_related"].mean(),
            })
    return pd.DataFrame(rows)


def external_accuracy_by_tercile(predictions, word_table):
    external = predictions[predictions["evaluation"] != CROSS_VALIDATION].merge(
        word_table[["target_word", "entropy_tercile"]], on="target_word")
    return (external.groupby(["evaluation", "model", "feature_set", "entropy_tercile"], observed=True)["correct"]
            .agg(accuracy="mean", instances="size").reset_index())
