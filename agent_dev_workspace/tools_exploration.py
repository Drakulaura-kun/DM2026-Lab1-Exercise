from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> dict:
        """
        Computes the cosine similarity between two documents in session.feature_matrix using raw counts.

        Args:
            doc_index_1: 0-based row index of the first document.
            doc_index_2: 0-based row index of the second document.

        Returns:
            A summary dictionary containing doc indices, calculated similarity, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}

        dense_dtm = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        n_docs = dense_dtm.shape[0]

        if not (0 <= doc_index_1 < n_docs and 0 <= doc_index_2 < n_docs):
            return {
                "status": "error",
                "message": f"Document indices out of bounds (0 to {n_docs - 1})."
            }

        vec1 = dense_dtm[doc_index_1].reshape(1, -1)
        vec2 = dense_dtm[doc_index_2].reshape(1, -1)

        sim_matrix = cosine_similarity(vec1, vec2)
        similarity = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(similarity, 4)
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """
        Computes the Pearson correlation matrix over the top 20 terms by variance in session.feature_matrix,
        plots the correlation matrix as an annotated heatmap, and sets session.pending_figure.

        Returns:
            A summary dictionary containing top feature names, correlation matrix, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}

        dense_dtm = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        variances = np.var(dense_dtm, axis=0)

        # Select top 20 terms by variance (or all terms if fewer than 20 exist)
        top_k = min(20, len(session.feature_names))
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_terms = [session.feature_names[i] for i in top_indices]
        top_matrix = dense_dtm[:, top_indices]

        # Pearson correlation matrix (np.corrcoef treats rows as variables, so transpose)
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)

        # Handle zero-variance NaNs safely
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Build Seaborn correlation heatmap and set session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()

        # Set plot on session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": np.round(corr_matrix, 4).tolist()
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
