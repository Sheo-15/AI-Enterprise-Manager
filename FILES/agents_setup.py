"""
Agent definitions for the AI Enterprise Knowledge Manager.

Architecture (6 agents):

    Knowledge Search Agent (triage / entry point)
        |-- handoff --> Document Reader Agent
        |-- handoff --> Policy Expert Agent
        |-- handoff --> Meeting Memory Agent
        |-- handoff --> Recommendation Agent
                              |
                              v
                    Knowledge Curator Agent
                    (human-approval gated, called
                     directly by main.py, not via
                     an LLM handoff)

Note: module is named agents_setup.py (not agents.py) so it doesn't
shadow the installed `agents` package during local imports.
"""

from __future__ import annotations

from pydantic import BaseModel
from agents import Agent

from tools import (
    search_knowledge_base,
    read_document,
    list_documents,
    get_policy_by_id,
    log_meeting_note,
)


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------

class RecommendationOutput(BaseModel):
    """Structured output produced by the Recommendation Agent."""
    question: str
    recommendation: str
    confidence: str  # "high" | "medium" | "low"
    source_documents: list[str]
    needs_curator_review: bool


class CuratedEntryProposal(BaseModel):
    """A single proposed knowledge-base entry awaiting human approval."""
    topic: str
    insight: str
    source_documents: list[str]


class CuratorProposalOutput(BaseModel):
    """Structured output produced by the Knowledge Curator Agent."""
    proposals: list[CuratedEntryProposal]


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

document_reader_agent = Agent(
    name="Document Reader",
    handoff_description="Reads and quotes from a specific document when the user knows (or can be told) which file/ID they need in detail.",
    instructions=(
        "You read individual company documents in full and answer questions "
        "about their exact content. Use list_documents to see what's "
        "available and read_document to pull the full text of a specific "
        "file. Quote section headers and procedure steps precisely. If the "
        "user's question is really about company-wide policy interpretation "
        "rather than one document's content, say so plainly."
    ),
    tools=[list_documents, read_document],
)

policy_expert_agent = Agent(
    name="Policy Expert",
    handoff_description="Answers questions about company policy, compliance, and SOPs, and can interpret how policies apply to a situation.",
    instructions=(
        "You are the company's policy expert. Use get_policy_by_id to pull "
        "policies/SOPs by their Document ID, and search_knowledge_base when "
        "the user doesn't know the ID. Explain how a policy applies to the "
        "user's specific situation, and flag when a situation touches more "
        "than one policy (e.g. remote work + expense reimbursement) so the "
        "user knows to check both."
    ),
    tools=[search_knowledge_base, get_policy_by_id],
)

meeting_memory_agent = Agent(
    name="Meeting Memory",
    handoff_description="Recalls past meeting decisions/action items, and logs new meeting notes into the knowledge base.",
    instructions=(
        "You are responsible for the company's meeting memory. Use "
        "search_knowledge_base to find past decisions and action items from "
        "meeting notes. When the user gives you a summary of a new meeting "
        "to record, use log_meeting_note to save it — always ask for (or "
        "infer from what's given) a title, summary, and action items before "
        "logging."
    ),
    tools=[search_knowledge_base, log_meeting_note, read_document],
)

recommendation_agent = Agent(
    name="Recommendation Agent",
    handoff_description="Synthesizes across multiple documents to produce a structured, sourced recommendation for cross-cutting questions.",
    instructions=(
        "You answer cross-cutting questions that may require pulling "
        "context from multiple documents (policies, SOPs, and meeting "
        "notes). Use search_knowledge_base and read_document as needed, "
        "then produce a structured recommendation. Set needs_curator_review "
        "to true if the topic is likely to recur (i.e. worth permanently "
        "curating for future employees), false for one-off questions."
    ),
    tools=[search_knowledge_base, read_document, get_policy_by_id],
    output_type=RecommendationOutput,
)

knowledge_curator_agent = Agent(
    name="Knowledge Curator",
    handoff_description="Reviews recurring questions/recommendations and proposes distilled, permanent knowledge-base entries for human approval.",
    instructions=(
        "You review a recommendation that was flagged as recurring and "
        "propose one or more concise, reusable knowledge entries that "
        "capture the durable insight (not the one-off specifics). Do NOT "
        "save anything yourself — you only produce proposals. A human will "
        "approve or reject each proposal before it is saved by the calling "
        "program."
    ),
    output_type=CuratorProposalOutput,
)


# ---------------------------------------------------------------------------
# Triage / entry-point agent
# ---------------------------------------------------------------------------

knowledge_search_agent = Agent(
    name="Knowledge Search",
    instructions=(
        "You are the front door to the company's Enterprise Knowledge "
        "Manager. Do a quick search yourself with search_knowledge_base "
        "to understand what's relevant, then route the user: \n"
        "- Hand off to Document Reader for 'what exactly does document X "
        "say' questions.\n"
        "- Hand off to Policy Expert for 'am I allowed to...' / compliance "
        "/ policy-interpretation questions.\n"
        "- Hand off to Meeting Memory for 'what did we decide about...' or "
        "requests to log a new meeting.\n"
        "- Hand off to Recommendation Agent for open-ended questions that "
        "need synthesis across multiple documents.\n"
        "If the user's question is simple and your own search already "
        "answers it, just answer directly without handing off."
    ),
    tools=[search_knowledge_base],
    handoffs=[
        document_reader_agent,
        policy_expert_agent,
        meeting_memory_agent,
        recommendation_agent,
    ],
)
