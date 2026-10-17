import os
from typing import Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports for transactional database conversion and algorithms
from PAMI.extras.convert import DF2DB
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth"
    ) -> dict:
        """
        Mines frequent itemsets/patterns from a specific category's documents.
        
        Builds a fresh category-local DTM using CountVectorizer(stop_words="english"),
        filters terms according to the requested filtering_method (variance/term_frequency: keep 5th-95th percentile;
        tfidf: keep top 80%/at or above 20th percentile), converts surviving terms to a transactional DB,
        and mines patterns using PAMI algorithms (fpgrowth, topk, maxfpgrowth).

        Args:
            category_name: The target category/label to filter and mine.
            filtering_method: Vocabulary filtering method ("variance", "tfidf", or "term_frequency").
            min_sup: Minimum support count required for "fpgrowth" or "maxfpgrowth".
            k: Number of highest-support patterns to keep when algorithm="topk".
            algorithm: Mining algorithm to run ("fpgrowth", "topk", or "maxfpgrowth"). Defaults to "fpgrowth".

        Returns:
            A dictionary summary containing the mined patterns, item counts, and status.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "session.dataframe is missing or empty."}

        # Filter dataframe to target category_name
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        # Step 1: Category-local DTM with English stopwords removed
        texts = cat_df["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        if dtm.shape[1] == 0:
            summary = {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            return summary

        dense_dtm = dtm.toarray()

        # Step 2: Vocabulary filtering based on filtering_method
        if filtering_method == "variance":
            scores = dense_dtm.var(axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            keep_mask = (scores >= low_p) & (scores <= high_p)
        elif filtering_method == "term_frequency":
            scores = dense_dtm.sum(axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            keep_mask = (scores >= low_p) & (scores <= high_p)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm).toarray()
            scores = tfidf.mean(axis=0)
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20
        else:
            return {"status": "error", "message": f"Unsupported filtering_method '{filtering_method}'."}

        kept_indices = np.where(keep_mask)[0]
        
        # Step 3: Handle case where zero terms survive filtering
        if len(kept_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        kept_terms = [feature_names[i] for i in kept_indices]
        filtered_counts = dense_dtm[:, kept_indices]

        # Convert to boolean binary DataFrame for DF2DB transactional conversion
        binary_df = pd.DataFrame(filtered_counts >= 1, columns=kept_terms)

        # Temporary transactional DB file
        db_file = f"_temp_db_{category_name}_{filtering_method}.txt"
        conv = DF2DB.DF2DB(binary_df, "dense")
        conv.convert2TransactionalDatabase(db_file, ">=", 1)

        patterns = []
        try:
            # Step 4: Run requested PAMI pattern mining algorithm
            if algorithm in ["fpgrowth", "maxfpgrowth"]:
                if min_sup is None:
                    return {"status": "error", "message": f"min_sup parameter is required for algorithm '{algorithm}'."}
                
                if algorithm == "fpgrowth":
                    obj = FPGrowth.FPGrowth(db_file, min_sup, "\t")
                else:
                    obj = MaxFPGrowth.MaxFPGrowth(db_file, min_sup, "\t")
                
                obj.startMine()
                mined_df = obj.getPatternsAsDataFrame()
                
                if mined_df is not None and not mined_df.empty:
                    for _, row in mined_df.iterrows():
                        patterns.append({
                            "pattern": row["Patterns"],
                            "support": int(row["Support"])
                        })

            elif algorithm == "topk":
                if k is None:
                    return {"status": "error", "message": "k parameter is required for algorithm 'topk'."}
                
                obj = FAE.FAE(db_file, k, "\t")
                obj.startMine()
                mined_df = obj.getPatternsAsDataFrame()
                
                if mined_df is not None and not mined_df.empty:
                    for _, row in mined_df.iterrows():
                        patterns.append({
                            "pattern": row["Patterns"],
                            "support": int(row["Support"])
                        })

        finally:
            if os.path.exists(db_file):
                os.remove(db_file)

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
            "patterns_found": len(patterns),
            "patterns": patterns
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns
        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary
        )

        return summary

    return [mine_patterns_tool]
