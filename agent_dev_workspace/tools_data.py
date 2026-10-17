from typing import Any, Dict, Optional
import os
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Ensure standard punctuation / tokenization data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def list_files_tool(directory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in the workspace/directory path.

        Helps discover available files or dataset paths before loading them.

        Args:
            directory: Directory path to scan relative to working environment. Default is '.'.

        Returns:
            Dict containing result_id, directory, files list, and directories list.
        """
        if not os.path.exists(directory):
            return {"status": "error", "message": f"Directory '{directory}' does not exist."}

        files = []
        directories = []

        try:
            entries = os.listdir(directory)
            for entry in entries:
                full_path = os.path.join(directory, entry)
                if os.path.isdir(full_path):
                    directories.append(entry)
                else:
                    files.append(entry)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "directory": directory,
            "files": sorted(files),
            "directories": sorted(directories),
        }

        session.store_result("list_files_tool", {"directory": directory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> Dict[str, Any]:
        """Inspects the first few rows and column information of session.dataframe.

        Args:
            n_rows: Number of rows to inspect. Default is 5.

        Returns:
            Dict containing result_id, total_rows, columns list, and sample rows.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        df = session.dataframe
        sample = df.head(n_rows).to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "columns": list(df.columns),
            "sample_rows": sample,
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe for missing or empty text values.

        Identifies rows with null/None values or empty string text in session.dataframe["text"].

        Returns:
            Dict containing result_id, total_rows, and missing_count.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataframe loaded or missing 'text' column."}

        df = session.dataframe
        missing_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_count = int(missing_mask.sum())

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "missing_count": missing_count,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate document text in session.dataframe and optionally drops them.

        When drop=True, drops ALL copies of duplicated rows using keep=False to match
        Master pipeline requirements.

        Args:
            drop: If True, drops duplicated rows in-place from session.dataframe. Default is False.

        Returns:
            Dict containing result_id, duplicate_count, and remaining_rows.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataframe loaded or missing 'text' column."}

        df = session.dataframe
        # Count extra duplicate rows (default keep='first' behavior for counting duplicate instances)
        dup_mask = df.duplicated(subset=["text"], keep="first")
        duplicate_count = int(dup_mask.sum())

        if drop and duplicate_count > 0:
            # Drop ALL instances of duplicates matching keep=False contract
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        remaining_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "remaining_rows": remaining_rows,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples the working DataFrame (session.dataframe).

        Args:
            n: Number of samples to draw. Default is 5.
            random_state: Seed for reproducible random sampling.

        Returns:
            Dict containing result_id, n_samples, sampled_indices, and total_rows.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe
        sample_df = df.sample(n=min(n, len(df)), random_state=random_state)
        sampled_indices = sample_df.index.tolist()

        # Mutate working dataframe to the sampled subset
        session.dataframe = sample_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_samples": len(sample_df),
            "sampled_indices": sampled_indices,
            "total_rows": len(session.dataframe),
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary stats on document text length and generates a box plot by category.

        Computes text length in characters (df['text'].apply(len)) and standard describe() statistics
        overall and per category_name. Also sets a box plot on session.pending_figure.

        Returns:
            Dict containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataframe loaded or missing 'text' column."}

        df = session.dataframe
        # Compute text_length in character count
        df["text_length"] = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Overall summary stats
        overall_stats = df["text_length"].describe().to_dict()

        # Per category summary stats
        per_category_stats = {}
        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if category_col:
            for cat, group in df.groupby(category_col):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Generate box plot by category on pending_figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_col:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": {k: round(v, 4) for k, v in overall_stats.items()},
            "per_category_stats": {
                cat: {k: round(v, 4) for k, v in stats.items()}
                for cat, stats in per_category_stats.items()
            },
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize the text column in session.dataframe into unigrams.

        Uses nltk.word_tokenize on lowercased text for each document.
        Stores token lists in session.dataframe['unigrams'] and session.artifacts['tokens'].

        Returns:
            Dict containing result_id, n_documents, total_tokens, and avg_tokens_per_doc.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or missing 'text' column in session."}

        # Tokenize lowercased text for each document
        tokens_list = []
        for text in session.dataframe["text"]:
            if isinstance(text, str) and text.strip():
                tokens = nltk.word_tokenize(text.lower())
            else:
                tokens = []
            tokens_list.append(tokens)

        # Store in session state and artifacts
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list

        total_tokens = sum(len(t) for t in tokens_list)
        n_docs = len(tokens_list)
        avg_tokens = total_tokens / n_docs if n_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4),
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
    ]
