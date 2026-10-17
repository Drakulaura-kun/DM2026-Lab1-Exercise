from typing import List, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """
        Builds the global Document-Term Matrix (DTM) from document text in session.dataframe.

        Fits a CountVectorizer on session.dataframe["text"], stores the sparse matrix in
        session.feature_matrix, vocabulary in session.feature_names, and the fitted vectorizer
        in session.artifacts["count_vectorizer"]. Also calculates matrix sparsity stats.

        Args:
            max_features: Maximum number of terms to keep in vocabulary, ranked by term frequency.
            min_df: Minimum document frequency for terms (count or proportion).
            max_df: Maximum document frequency for terms (count or proportion).
            ngram_range: Lower and upper boundary of n-gram sizes [min_n, max_n]. Default is [1, 1] (unigrams).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded or 'text' column missing."}

        # Convert list to tuple for CountVectorizer compatibility
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        texts = session.dataframe["text"].fillna("")
        matrix = vectorizer.fit_transform(texts)

        # Update session state
        session.feature_matrix = matrix
        session.feature_names = list(vectorizer.get_feature_names_out())
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Sparsity calculations matching Master contract
        n_rows, n_cols = matrix.shape
        non_zero = int(matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("build_dtm"),
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in session.feature_matrix.

        Returns:
            Dict containing result_id, frequencies mapping each term to its total count.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "DTM has not been built yet."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        frequencies = {term: int(count) for term, count in zip(session.feature_names, counts)}

        summary = {
            "result_id": session.next_result_id("term_frequency"),
            "frequencies": frequencies,
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """
        Generates a heatmap visualization of a raw positional slice of session.feature_matrix.

        Args:
            n_terms: Number of terms (columns) to include from the start. Default is 20.
            n_documents: Number of documents (rows) to include from the start. Default is 20.

        Returns:
            Dict containing result_id, sliced_matrix, terms, and n_documents.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "DTM has not been built yet."}

        max_docs, max_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        dense_slice = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        sliced_terms = session.feature_names[:actual_terms]

        # Generate plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(dense_slice, annot=True, fmt="d", xticklabels=sliced_terms, ax=ax, cmap="Blues")
        ax.set_title(f"DTM Heatmap (First {actual_docs} Docs, First {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "sliced_matrix": dense_slice.tolist(),
            "terms": sliced_terms,
            "n_documents": actual_docs,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
