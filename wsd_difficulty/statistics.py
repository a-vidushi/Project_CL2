import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def spearman(x, y):
    result = stats.spearmanr(x, y, nan_policy="omit")
    return result.statistic, result.pvalue


def partial_spearman(data, predictor, outcome, controls):
    """Spearman correlation of predictor and outcome after regressing the controls out of both ranks."""
    columns = [predictor, outcome, *controls]
    complete = data[columns].dropna()
    ranks = complete.rank()
    control_matrix = add_constant(ranks[list(controls)].to_numpy())

    def residuals(column):
        values = ranks[column].to_numpy()
        coefficients, *_ = np.linalg.lstsq(control_matrix, values, rcond=None)
        return values - control_matrix @ coefficients

    correlation, _ = stats.pearsonr(residuals(predictor), residuals(outcome))
    # Degrees of freedom lose one per control variable
    degrees_of_freedom = len(complete) - 2 - len(controls)
    t_statistic = correlation * np.sqrt(degrees_of_freedom / max(1e-12, 1 - correlation ** 2))
    p_value = 2 * stats.t.sf(abs(t_statistic), degrees_of_freedom)
    return correlation, p_value, len(complete)


def variance_inflation_factors(data, columns):
    matrix = add_constant(data[list(columns)].dropna().to_numpy(), has_constant="add")
    if len(matrix) <= len(columns) + 1:
        return pd.DataFrame({"variable": list(columns), "variance_inflation_factor": np.nan})
    return pd.DataFrame({
        "variable": list(columns),
        "variance_inflation_factor": [variance_inflation_factor(matrix, i + 1) for i in range(len(columns))],
    })


def bootstrap_mean_interval(values, resamples=2000, confidence=0.95, seed=0):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    random_generator = np.random.default_rng(seed)
    means = random_generator.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    return values.mean(), np.quantile(means, tail), np.quantile(means, 1 - tail)


def mcnemar_test(first_correct, second_correct):
    first_correct, second_correct = np.asarray(first_correct, bool), np.asarray(second_correct, bool)
    table = [
        [np.sum(first_correct & second_correct), np.sum(first_correct & ~second_correct)],
        [np.sum(~first_correct & second_correct), np.sum(~first_correct & ~second_correct)],
    ]
    disagreements = table[0][1] + table[1][0]
    result = mcnemar(table, exact=disagreements < 25)
    return {"only_first_correct": table[0][1], "only_second_correct": table[1][0],
            "statistic": result.statistic, "p_value": result.pvalue}


def wilcoxon_signed_rank(differences):
    differences = np.asarray(differences, float)
    differences = differences[~np.isnan(differences) & (differences != 0)]
    if len(differences) < 10:
        return np.nan
    return stats.wilcoxon(differences).pvalue


def expected_calibration_error(confidence, correct, number_of_bins=10):
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    bin_edges = np.linspace(0, 1, number_of_bins + 1)
    bin_of_prediction = np.clip(np.digitize(confidence, bin_edges[1:-1]), 0, number_of_bins - 1)
    error = 0.0
    for bin_number in range(number_of_bins):
        in_bin = bin_of_prediction == bin_number
        if in_bin.any():
            error += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return error
