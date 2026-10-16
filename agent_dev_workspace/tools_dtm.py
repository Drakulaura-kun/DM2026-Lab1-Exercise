from typing import List, Optional, Dict, Any, Union
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

    return [build_dtm_tool]
