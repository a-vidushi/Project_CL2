"""Logistic regression of instance correctness on word, context and discourse predictors.

Instances of the same word are not independent, so standard errors are clustered by lemma.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

WORD_LEVEL = ["sense_entropy", "log_frequency", "frequency_weighted_relatedness_z"]
SKEW = ["mfs_probability"]
CONTEXT = ["log_sentence_length"]
DISCOURSE = ["has_previous_mention", "log_previous_gap", "log_mentions_in_document",
             "log_max_chain_support", "gold_sense_strongest_in_chain"]
RST = ["rst_depth", "rst_nucleus_share"]

SPECIFICATIONS = [
    ("logistic_regression", "all", "word_and_context", WORD_LEVEL + CONTEXT),
    ("logistic_regression", "all", "with_skew", WORD_LEVEL + SKEW + CONTEXT),
    ("logistic_regression", "all", "with_discourse", WORD_LEVEL + CONTEXT + DISCOURSE + RST),
    ("logistic_regression", "all+discourse", "with_discourse", WORD_LEVEL + CONTEXT + DISCOURSE + RST),
]


def build_instance_data(predictions, instances, word_table, model, feature_set):
    rows = predictions[(predictions["evaluation"] == "semcor_cv") & (predictions["model"] == model)
                       & (predictions["feature_set"] == feature_set)]
    data = (rows[["instance_id", "target_word", "document_id", "correct"]]
            .merge(instances.drop(columns=["target_word", "document_id"]), on="instance_id")
            .merge(word_table, on="target_word"))
    data["log_sentence_length"] = np.log(data["sentence_length"])
    data["log_previous_gap"] = np.log1p(data["previous_mention_gap"].fillna(0))
    data["log_mentions_in_document"] = np.log(data["mentions_in_document"])
    data["log_max_chain_support"] = np.log1p(data["max_chain_support"])
    for column in ("has_previous_mention", "gold_sense_strongest_in_chain", "correct"):
        data[column] = data[column].astype(float)
    return data


def standardise(data, predictors):
    """Continuous predictors to mean 0, SD 1 so coefficients are log-odds per SD; binary ones stay 0/1."""
    design = pd.DataFrame(index=data.index)
    for predictor in predictors:
        values = data[predictor].astype(float)
        is_binary = set(values.dropna().unique()) <= {0.0, 1.0}
        design[predictor] = values if is_binary else (values - values.mean()) / values.std()
    return design


def fit_clustered_logistic_regression(data, predictors):
    design = sm.add_constant(standardise(data, predictors), has_constant="add")
    lemma_groups = pd.factorize(data["target_word"])[0]
    result = sm.GLM(data["correct"], design, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": lemma_groups})
    return pd.DataFrame({"term": ["intercept", *predictors], "estimate": result.params.to_numpy(),
                         "std_error": result.bse.to_numpy(), "p_value": result.pvalues.to_numpy()})


def fit_correctness_models(predictions, instances, word_table):
    tables = []
    for model, feature_set, specification, predictors in SPECIFICATIONS:
        if not ((predictions["model"] == model) & (predictions["feature_set"] == feature_set)).any():
            continue
        data = build_instance_data(predictions, instances, word_table, model, feature_set)
        # RST columns only exist once trees are parsed
        available = [p for p in predictors if p in data.columns and data[p].notna().all() and data[p].std() > 0]
        print(f"  correctness regression: {model} / {feature_set} / {specification} on {len(data)} instances")

        coefficients = fit_clustered_logistic_regression(data, available)
        coefficients.insert(0, "specification", specification)
        coefficients.insert(0, "feature_set", feature_set)
        coefficients.insert(0, "model", model)
        tables.append(coefficients)

    table = pd.concat(tables, ignore_index=True)
    table["odds_ratio"] = np.exp(table["estimate"])
    table["z"] = table["estimate"] / table["std_error"]
    return table
