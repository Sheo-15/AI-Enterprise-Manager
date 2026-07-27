"""
Shared function tools for the AI Enterprise Knowledge Manager.

These are plain Python functions decorated with @function_tool so the
Agents SDK can expose them to any agent that lists them in `tools=[...]`.
No external APIs are required — everything reads/writes from the local
`mock_docs/` folder and an in-memory curated-knowledge store, which keeps
the whole capstone runnable without API keys beyond the OpenAI one.
"""

from __future__ import annotations

import os
import glob
import json
from datetime import datetime, timezone

from agents import function_tool

DOCS_DIR = os.path.join(os.path.dirname(__file__), "mock_docs")
CURATED_STORE_PATH = os.path.join(os.path.dirname(__file__), "curated_knowledge.json")


def _load_all_docs() -> dict[str, str]:
    """Internal helper: load every mock doc into memory as {filename: text}."""
    docs = {}
    for path in glob.glob(os.path.join(DOCS_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search all company documents (SOPs, policies, meeting notes) for a
    keyword or phrase and return matching filenames with a short snippet
    of surrounding context for each hit.

    Args:
        query: The keyword or phrase to search for (case-insensitive).
    """
    docs = _load_all_docs()
    query_lower = query.lower()
    results = []
    for filename, text in docs.items():
        lower_text = text.lower()
        idx = lower_text.find(query_lower)
        if idx != -1:
            start = max(0, idx - 80)
            end = min(len(text), idx + len(query) + 80)
            snippet = text[start:end].replace("\n", " ").strip()
            results.append(f"[{filename}] ...{snippet}...")
    if not results:
        return f"No documents matched '{query}'."
    return "\n".join(results[:10])


@function_tool
def read_document(filename: str) -> str:
    """Read and return the full text of a specific document by filename.

    Args:
        filename: Exact filename as returned by search_knowledge_base
            (e.g. 'policy_data_security.md').
    """
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        available = ", ".join(sorted(os.listdir(DOCS_DIR)))
        return f"File '{filename}' not found. Available files: {available}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@function_tool
def list_documents() -> str:
    """List every document currently available in the knowledge base,
    with its document type inferred from the filename prefix
    (sop_, policy_, meeting_notes_)."""
    docs = sorted(os.listdir(DOCS_DIR))
    if not docs:
        return "The knowledge base is empty."
    return "\n".join(f"- {d}" for d in docs)


@function_tool
def get_policy_by_id(document_id: str) -> str:
    """Look up a policy or SOP by its Document ID (e.g. 'POLICY-IT-007' or
    'SOP-FIN-014') rather than by filename, and return its full text.

    Args:
        document_id: The Document ID exactly as it appears in a doc header,
            e.g. 'POLICY-HR-021'.
    """
    docs = _load_all_docs()
    target = document_id.upper()

    # First pass: match only the document's own header line
    # ("Document ID: X"), so a doc that merely *references* another
    # doc's ID doesn't get returned instead of the real source.
    for filename, text in docs.items():
        for line in text.splitlines()[:5]:
            if line.upper().startswith("DOCUMENT ID:") and target in line.upper():
                return f"[{filename}]\n{text}"

    # Fallback: any document that mentions the ID somewhere in its body.
    for filename, text in docs.items():
        if target in text.upper():
            return f"[{filename}] (mentions {document_id} but is not its source document)\n{text}"

    return f"No document found with ID '{document_id}'."


@function_tool
def log_meeting_note(title: str, summary: str, action_items: str) -> str:
    """Save a new meeting note to the knowledge base so future queries can
    retrieve decisions and action items from it. Creates a new markdown
    file under mock_docs/.

    Args:
        title: Short title for the meeting (used in the filename and header).
        summary: A summary of what was discussed and decided.
        action_items: Action items from the meeting, one per line.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
    filename = f"meeting_notes_{date_str}_{safe_title}.md"
    path = os.path.join(DOCS_DIR, filename)
    content = (
        f"# Meeting Notes: {title}\n"
        f"Date: {date_str.replace('_', '-')}\n"
        f"Type: Internal — Meeting Memory (auto-logged)\n\n"
        f"## Summary\n{summary}\n\n"
        f"## Action Items\n{action_items}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Meeting note saved as '{filename}'."


@function_tool
def curate_knowledge_entry(topic: str, insight: str, source_documents: str) -> str:
    """Record a curated, human-approved knowledge entry — a distilled fact
    or recommendation the Knowledge Curator has decided is worth
    surfacing to employees going forward. Requires human approval before
    being called (the calling agent must have confirmed with a human
    first; see Curator agent instructions).

    Args:
        topic: Short topic label, e.g. 'Bulk export data handling'.
        insight: The distilled, curated insight or recommendation.
        source_documents: Comma-separated filenames or Document IDs this
            insight was derived from.
    """
    entry = {
        "topic": topic,
        "insight": insight,
        "source_documents": source_documents,
        "curated_at": datetime.now(timezone.utc).isoformat(),
    }
    store = []
    if os.path.exists(CURATED_STORE_PATH):
        with open(CURATED_STORE_PATH, "r", encoding="utf-8") as f:
            store = json.load(f)
    store.append(entry)
    with open(CURATED_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    return f"Curated entry on '{topic}' saved."
