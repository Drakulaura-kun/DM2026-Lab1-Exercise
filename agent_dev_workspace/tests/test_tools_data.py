import importlib.util
from pathlib import Path
from agent_pipeline.session_state import SessionState

# Locate premade_tools relative to workspace root
workspace_dir = Path(__file__).resolve().parents[1]
premade_path = workspace_dir / "premade_tools" / "load_dataset_tool.py"

spec_premade = importlib.util.spec_from_file_location("load_dataset_tool_module", premade_path)
premade_mod = importlib.util.module_from_spec(spec_premade)
spec_premade.loader.exec_module(premade_mod)

from tools_data import make_tools


def test_tokenize_tool():
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

    # Load and execute tokenize_tool
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Known answer #10 token lists
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert result["n_documents"] == 8
    assert result["total_tokens"] == 22
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["tokens"] == expected_tokens
