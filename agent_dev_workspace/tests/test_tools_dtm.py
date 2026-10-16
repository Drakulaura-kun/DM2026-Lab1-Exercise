import pytest
from pathlib import Path
import importlib.util
from agent_pipeline.session_state import SessionState

# Locate premade_tools relative to current file's workspace directory
workspace_dir = Path(__file__).resolve().parents[1]
premade_path = workspace_dir / "premade_tools" / "load_dataset_tool.py"

spec_premade = importlib.util.spec_from_file_location("load_dataset_tool_module", premade_path)
premade_mod = importlib.util.module_from_spec(spec_premade)
spec_premade.loader.exec_module(premade_mod)

from tools_dtm import make_tools


def test_build_dtm_tool():
    session = SessionState()

    # Load fixture dataset into session
    premade_tools = premade_mod.make_tools(session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Load and execute student's build_dtm_tool
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    result = build_dtm_tool.invoke({})

    # Check known answer #4 specifications
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-2)

    # Check vocabulary contents and vectorizer artifact
    assert sorted(session.feature_names) == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.artifacts.get("count_vectorizer") is not None
