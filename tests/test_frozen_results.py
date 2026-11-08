from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_frozen_evidence_files_exist():
    required = [
        ROOT / "configs" / "g9_frozen.yaml",
        ROOT / "docs" / "g9_freeze.md",
        ROOT / "docs" / "g9_lockbox_results.md",
        RESULTS / "g9_lockbox" / "joinquant_summary.csv",
        RESULTS / "g9_lockbox" / "joinquant_master_2023_2025.png",
        RESULTS / "g9_lockbox" / "joinquant_random_forest_2023_2025.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    assert not missing, f"Missing frozen evidence: {missing}"


def test_model_comparison_contains_eight_models():
    table = pd.read_csv(RESULTS / "g8_main_rankic_summary.csv")
    expected = {
        "ols",
        "ridge",
        "lightgbm",
        "xgboost",
        "random_forest",
        "mlp",
        "transformer",
        "master",
    }
    assert set(table["model"]) == expected
    assert table["rankic_mean"].notna().all()
    assert table["n_days"].eq(606).all()


def test_joinquant_lockbox_is_fixed_and_complete():
    table = pd.read_csv(RESULTS / "g9_lockbox" / "joinquant_summary.csv").set_index("model")
    assert set(table.index) == {"master", "random_forest"}
    assert table["start_date"].eq("2023-01-01").all()
    assert table["end_date"].eq("2025-12-31").all()
    assert table["initial_capital"].eq(1_000_000).all()
    assert table["portfolio_rule"].str.contains("Top 30", regex=False).all()
    assert table.loc["master", "cumulative_return"] == 0.2445
    assert table.loc["random_forest", "cumulative_return"] == 0.5161
