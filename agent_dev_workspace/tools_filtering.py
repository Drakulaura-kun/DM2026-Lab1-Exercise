import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        top_k: int = 0
    ) -> dict:
        """
        Computes the Spearman rank correlation coefficient between each term's count 
        in session.feature_matrix and a target binary class indicator vector.
        
        Monotonic relationships are evaluated rank-wise rather than strictly linear.
        
        Args:
            target_class: The label/category name to evaluate against (one-vs-rest).
            threshold: Minimum absolute Spearman correlation required to keep a term. Defaults to 0.0.
            top_k: Optional maximum number of terms to return after filtering/sorting. Defaults to 0.

        Returns:
            A dictionary containing a summary of calculated Spearman correlation values,
            kept terms, and status.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found. Run build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document labels found."}

        # One-vs-rest binary target vector (1 if label matches target_class, 0 otherwise)
        y_binary = (np.array(session.labels) == target_class).astype(int)

        dense_dtm = session.feature_matrix.toarray()
        n_terms = dense_dtm.shape[1]

        results = []
        for i in range(n_terms):
            term = session.feature_names[i]
            x = dense_dtm[:, i]

            # Handle edge case where term count or target binary vector has zero variance
            if np.std(x) == 0 or np.std(y_binary) == 0:
                corr = 0.0
            else:
                rho, _ = spearmanr(x, y_binary)
                corr = 0.0 if np.isnan(rho) else float(rho)

            results.append({
                "term": term,
                "spearman_r": corr,
                "abs_spearman_r": abs(corr)
            })

        # Create full_report DataFrame sorted by spearman_r in descending order
        # Column 1: term name, Column 2: spearman_r (for visualize_result_tool compatibility)
        report_df = pd.DataFrame(results)[["term", "spearman_r", "abs_spearman_r"]]
        
        # Apply threshold filtering on absolute Spearman correlation
        filtered_df = report_df[report_df["abs_spearman_r"] >= threshold].copy()
        filtered_df.sort_values(by="spearman_r", ascending=False, inplace=True)

        if top_k > 0:
            filtered_df = filtered_df.head(top_k)

        kept_terms = filtered_df["term"].tolist()

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "top_k": top_k,
            "total_terms": n_terms,
            "kept_terms_count": len(kept_terms),
            "kept_terms": kept_terms,
            "correlations": dict(zip(report_df["term"], report_df["spearman_r"]))
        }

        # Store full report with term as first column and spearman_r as second column
        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold, "top_k": top_k},
            summary=summary,
            full_report=filtered_df[["term", "spearman_r"]]
        )

        return summary

    return [spearman_filter_tool]
