from typing import Literal, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None
    ) -> dict:
        """
        Reduces the dimensionality of session.feature_matrix using PCA, t-SNE, or UMAP,
        and plots the 2D reduced coordinates colored by category on session.pending_figure.

        Args:
            method: Dimensionality reduction algorithm ("pca", "tsne", or "umap"). Defaults to "pca".
            n_components: Number of target dimensions (default is 2).
            perplexity: Perplexity hyperparameter for t-SNE (default is 30.0).
            n_neighbors: Number of neighbors hyperparameter for UMAP (default is 15).
            random_state: Optional random seed for reproducible outputs.

        Returns:
            A summary dictionary with result_id, method, reduced coordinates, and status.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}

        dense_dtm = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        n_samples = dense_dtm.shape[0]

        if n_samples == 0:
            return {"status": "error", "message": "Document-term matrix is empty."}

        # Select and execute reduction method
        explained_variance_ratio = None
        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            # Adjust perplexity if sample count is small
            effective_perp = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(n_components=n_components, perplexity=effective_perp, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
        elif method == "umap":
            # Adjust n_neighbors if sample count is small
            effective_neighbors = min(n_neighbors, max(2, n_samples - 1))
            reducer = umap.UMAP(n_components=n_components, n_neighbors=effective_neighbors, random_state=random_state)
            coords = reducer.fit_transform(dense_dtm)
        else:
            return {"status": "error", "message": f"Unsupported reduction method '{method}'."}

        # Store reduced coordinates in artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords

        # Create scatter plot of 2D reduced coordinates
        categories = session.labels if session.labels is not None else ["Unlabeled"] * n_samples
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=categories,
            style=categories,
            s=100,
            ax=ax
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # MANDATORY: Set plot on session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("reduction")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": n_samples,
            "explained_variance_ratio": explained_variance_ratio,
            "coordinates": coords.tolist()
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={"method": method, "n_components": n_components, "perplexity": perplexity, "n_neighbors": n_neighbors, "random_state": random_state},
            summary=summary
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """
        One-hot encodes the category labels in session.dataframe and stores the binarized matrix.

        Note: As noted in the spec, this tool has no fixed contract/known answer;
        we design it to produce a clean one-hot encoded DataFrame stored in session.artifacts["binarized_labels"].

        Returns:
            A summary dictionary containing categories, document count, and status.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "session.dataframe is missing or empty."}

        if "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "'category_name' column missing in session.dataframe."}

        # Perform one-hot encoding on category_name
        one_hot_df = pd.get_dummies(session.dataframe["category_name"], prefix="cat", dtype=int)
        
        session.artifacts["binarized_labels"] = one_hot_df

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": one_hot_df.columns.tolist(),
            "n_documents": len(one_hot_df),
            "one_hot_matrix": one_hot_df.to_dict(orient="records")
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
