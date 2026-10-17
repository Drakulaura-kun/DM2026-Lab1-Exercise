import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools

def test_exploration_tools_known_answers_16_and_19():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM on session.dataframe
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    session.feature_matrix = vec.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vec.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].values)

    # Get student tools from tools_exploration.py
    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # -------------------------------------------------------------
    # Known Answer 16: cosine_similarity_tool
    # -------------------------------------------------------------
    # Pair 0 vs 1 (both catA, near-identical text): 0.9847
    sim_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert sim_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 0 vs 3 (catA vs catB, share only "always"): 0.0909
    sim_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert sim_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    # Pair 3 vs 4 (both catB, near-identical text): 0.9847
    sim_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert sim_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    # Pair 0 vs 7 (catA vs empty text row): 0.0
    sim_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert sim_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)

    # -------------------------------------------------------------
    # Known Answer 19: feature_correlation_matrix_tool
    # -------------------------------------------------------------
    res_corr = feature_correlation_matrix_tool.invoke({})
    
    assert res_corr["top_terms"] == ["alpha", "gamma", "delta", "beta", "always"]
    
    corr_matrix = np.array(res_corr["correlation_matrix"])
    
    # Expected correlation matrix rounded to 4 decimals:
    # alpha vs beta: 0.8885
    # gamma vs delta: 0.8307
    # alpha vs delta: -0.6882
    terms = res_corr["top_terms"]
    idx_alpha = terms.index("alpha")
    idx_beta = terms.index("beta")
    idx_gamma = terms.index("gamma")
    idx_delta = terms.index("delta")

    assert corr_matrix[idx_alpha, idx_beta] == pytest.approx(0.8885, abs=1e-3)
    assert corr_matrix[idx_gamma, idx_delta] == pytest.approx(0.8307, abs=1e-3)
    assert corr_matrix[idx_alpha, idx_delta] == pytest.approx(-0.6882, abs=1e-3)
    
    # Verify plotting requirement on session.pending_figure
    assert session.pending_figure is not None, "feature_correlation_matrix_tool must set session.pending_figure"
