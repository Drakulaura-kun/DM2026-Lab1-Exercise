import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_known_answers_17_and_20():
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

    # Get student mine_patterns_tool
    pattern_tools = make_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Test Known Answer 17: filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catA["patterns_found"] == 1
    assert res_catA["patterns"][0]["pattern"] == "alpha"
    assert res_catA["patterns"][0]["support"] == 3

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catB["patterns_found"] == 1
    assert res_catB["patterns"][0]["pattern"] == "gamma"
    assert res_catB["patterns"][0]["support"] == 4

    # Test Known Answer 17 edge case: filtering_method="variance" keeps 0 terms
    res_zero = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_zero["patterns"] == []
    assert res_zero.get("note") == "no terms survived filtering"

    # Test Known Answer 20: algorithm="topk" (k=1)
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })

    assert res_topk["patterns_found"] == 1
    assert res_topk["patterns"][0]["pattern"] == "alpha"
    assert res_topk["patterns"][0]["support"] == 3

    # Test Known Answer 20: algorithm="maxfpgrowth" returns 0 patterns on this data
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })

    assert res_max["patterns"] == []
