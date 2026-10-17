import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def test_reduce_dimensions_tool_known_answers_7_8_9():
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

    # Get student reduce_dimensions_tool
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # -------------------------------------------------------------
    # Known Answer 7: PCA (n_components=2)
    # -------------------------------------------------------------
    res_pca = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    expected_pca_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    expected_pca_evr = [0.7532, 0.1965]

    actual_pca_coords = np.array(res_pca["coordinates"])
    actual_pca_evr = np.array(res_pca["explained_variance_ratio"])

    # PCA can have deterministic sign flips across environments; assert absolute values or aligned coords
    assert np.allclose(np.abs(actual_pca_coords), np.abs(expected_pca_coords), atol=1e-3)
    assert np.allclose(actual_pca_evr, expected_pca_evr, atol=1e-3)
    assert session.pending_figure is not None, "reduce_dimensions_tool must set session.pending_figure"

    # Reset pending_figure for next run
    session.pending_figure = None

    # -------------------------------------------------------------
    # Known Answer 8: t-SNE (perplexity=2, random_state=42)
    # -------------------------------------------------------------
    res_tsne = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    assert res_tsne["n_samples"] == 8
    assert len(res_tsne["coordinates"]) == 8
    assert session.pending_figure is not None, "reduce_dimensions_tool must set session.pending_figure"

    session.pending_figure = None

    # -------------------------------------------------------------
    # Known Answer 9: UMAP (n_neighbors=3, random_state=42)
    # -------------------------------------------------------------
    res_umap = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    assert res_umap["n_samples"] == 8
    assert len(res_umap["coordinates"]) == 8
    assert session.pending_figure is not None, "reduce_dimensions_tool must set session.pending_figure"
