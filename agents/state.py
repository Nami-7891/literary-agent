from typing import Annotated, Any, Dict, List, TypedDict


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer: merge per-section dicts written by nodes (incl. parallel branches)."""
    return {**(left or {}), **(right or {})}


def merge_unique(left: List[str], right: List[str]) -> List[str]:
    """Reducer: union of section names, de-duplicated and ordered."""
    return sorted(set(left or []) | set(right or []))


class AgentState(TypedDict, total=False):
    chapter_text: str
    timeline_events: str
    extracted_characters: str
    map_data: Dict[str, Any]
    final_dossier: str

    # Per-section bookkeeping. Analyzer and Cartographer run in parallel, so
    # these need reducers or LangGraph raises InvalidUpdateError.
    retry_counts: Annotated[Dict[str, int], merge_dicts]      # attempts made per section
    feedback: Annotated[Dict[str, str], merge_dicts]          # auditor issues fed into retries
    proceed: Annotated[Dict[str, bool], merge_dicts]          # verifier verdict used for routing
    unverified_sections: Annotated[List[str], merge_unique]   # sections that moved on unverified
    llm_errors: Annotated[Dict[str, Any], merge_dicts]        # last LLM call error per section (None = ok)