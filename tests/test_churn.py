"""Regressionsschutz für Datenqualität, getrenntes Training und Inferenz."""

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.churn import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_churn_pipeline,
    load_churn,
    train_churn_model,
)


def test_loader_keeps_missing_values_for_training_only_imputation():
    data = load_churn()
    assert data["TotalCharges"].isna().sum() == 11
    assert set(data["Churn"].unique()) == {0, 1}


@pytest.mark.parametrize("change", ["missing_column", "bad_target", "duplicate_id"])
def test_loader_rejects_invalid_data(tmp_path, change):
    import pandas as pd

    from src.churn import CHURN_CSV

    data = pd.read_csv(CHURN_CSV).head(20)
    if change == "missing_column":
        data = data.drop(columns="Contract")
    elif change == "bad_target":
        data.loc[0, "Churn"] = "Unknown"
    else:
        data.loc[1, "customerID"] = data.loc[0, "customerID"]
    path = tmp_path / "invalid.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_churn(path)


def test_pipeline_does_not_learn_test_median_or_test_categories():
    data = load_churn().head(20)
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    train, test = features.iloc[:16].copy(), features.iloc[16:].copy()
    train["TotalCharges"] = [10.0, np.nan] * 8
    train["Contract"] = "Training only"
    test["TotalCharges"] = 999999.0
    test["Contract"] = "Test only"
    model = build_churn_pipeline().fit(train, data["Churn"].iloc[:16])
    preprocess = model.named_steps["preprocess"]
    assert preprocess.named_transformers_["numeric"].statistics_[-1] == 10.0
    encoder = preprocess.named_transformers_["categorical"].named_steps["encoder"]
    assert "Test only" not in encoder.categories_[CATEGORICAL_FEATURES.index("Contract")]
    with pytest.warns(UserWarning, match="unknown categories"):
        probabilities = model.predict_proba(test)
    assert np.isfinite(probabilities).all()
    assert preprocess.named_transformers_["numeric"].statistics_[-1] == 10.0


def test_training_evaluates_unseen_customers_with_aligned_outputs():
    data = load_churn()
    result = train_churn_model(data)
    train_indices, test_indices = train_test_split(
        data.index,
        test_size=0.2,
        random_state=42,
        stratify=data["Churn"],
    )
    assert set(train_indices).isdisjoint(result["ids"].index)
    assert result["ids"].index.tolist() == list(test_indices)
    assert result["proba"].index.equals(result["y_test"].index)
    assert result["proba"].between(0, 1).all()
    assert result["cm"].sum() == len(test_indices)
    assert "customerID" not in result["model"].feature_names_in_
    assert "Churn" not in result["model"].feature_names_in_
    assert result["metrics"]["Accuracy"] > result["baseline_accuracy"]
    learned_medians = (
        result["model"].named_steps["preprocess"].named_transformers_["numeric"].statistics_
    )
    np.testing.assert_allclose(
        learned_medians, data.loc[train_indices, NUMERIC_FEATURES].median().to_numpy()
    )
