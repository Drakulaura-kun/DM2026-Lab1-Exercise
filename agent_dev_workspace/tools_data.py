from typing import Any, Dict
import nltk
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

    return [tokenize_tool]
