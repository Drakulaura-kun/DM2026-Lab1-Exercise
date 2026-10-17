import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = build_dtm_tool.invoke({})
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]


def test_term_frequency_tool_known_answer_14():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    res = term_frequency_tool.invoke({})
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert res["frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Slice corner n_terms=2, n_documents=3
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_slice["sliced_matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert res_slice["terms"] == ["alpha", "always"]
    assert res_slice["n_documents"] == 3
    assert session.pending_figure is not None
