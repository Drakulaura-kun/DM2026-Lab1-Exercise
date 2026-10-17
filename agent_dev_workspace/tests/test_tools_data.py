import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"directory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = inspect_data_tool.invoke({"n_rows": 3})
    assert res["total_rows"] == 8
    assert "text" in res["columns"]
    assert len(res["sample_rows"]) == 3


def test_check_missing_tool_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = check_missing_tool.invoke({})
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1


def test_check_duplicates_tool_known_answer_3():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check["duplicate_count"] == 1
    assert res_check["remaining_rows"] == 8

    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["remaining_rows"] == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["n_samples"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["total_rows"] == 4


def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = describe_data_tool.invoke({})
    overall = res["overall_stats"]

    # Known answer 18 overall stats assertions
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert overall["min"] == 0
    assert pytest.approx(overall["25%"], abs=1e-3) == 17.75
    assert pytest.approx(overall["50%"], abs=1e-3) == 20.5
    assert pytest.approx(overall["75%"], abs=1e-3) == 25.25
    assert overall["max"] == 30

    # Category breakdown assertions
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert pytest.approx(catA["mean"], abs=1e-3) == 17.25
    assert pytest.approx(catA["std"], abs=1e-3) == 12.5

    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert pytest.approx(catB["mean"], abs=1e-3) == 22.5
    assert pytest.approx(catB["std"], abs=1e-3) == 5.7446

    assert session.pending_figure is not None


def test_tokenize_tool_known_answer_10():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res = tokenize_tool.invoke({})
    assert res["n_documents"] == 8
    # Fixture doc 0: 5, doc 1: 4, doc 2: 3, doc 3: 5, doc 4: 4, doc 5: 3, doc 6: 3, doc 7: 0 -> total 27 tokens
    assert res["total_tokens"] == 27
    tokens = session.artifacts["tokens"]
    assert tokens[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert tokens[7] == []
