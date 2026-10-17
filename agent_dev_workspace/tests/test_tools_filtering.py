import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools

def test_spearman_filter_tool_known_answer_6():
    # Instantiate session state once for loading dataset and running the tool under test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using CountVectorizer on session.dataframe
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    session.feature_matrix = vec.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vec.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].values)

    # Get student tools from tools_filtering.py
    tools = make_tools(session)
    spearman_filter_tool = next(t for t in tools if t.name == "spearman_filter_tool")

    # Execute spearman_filter_tool for catB target class
    result = spearman_filter_tool.invoke({
        "target_class": "catB"
    })

    # Known Answer 6 expected Spearman r values for target_class="catB":
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    expected_correlations = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }

    actual_correlations = result["correlations"]

    for term, expected_r in expected_correlations.items():
        assert term in actual_correlations, f"Expected term '{term}' in correlations output."
        assert actual_correlations[term] == pytest.approx(expected_r, abs=1e-3), \
            f"Expected Spearman r for {term} to be approx {expected_r}, got {actual_correlations[term]}"
