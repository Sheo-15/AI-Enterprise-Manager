"""
AI Enterprise Knowledge Manager — main entry point.

Run interactively:
    python main.py

Or single-shot:
    python main.py "What's our policy on remote work eligibility?"

Requires OPENAI_API_KEY to be set in the environment (or a .env file —
see .env.example).
"""

from __future__ import annotations

import os
import sys
import logging
import asyncio

from agents import Runner, SQLiteSession
from dotenv import load_dotenv

from agents_setup import (
    knowledge_search_agent,
    recommendation_agent,
    knowledge_curator_agent,
    RecommendationOutput,
)
from tools import curate_knowledge_entry

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("knowledge_manager.log"),
    ],
)
logger = logging.getLogger("knowledge_manager")

SESSION_DB_PATH = "conversations.db"


async def run_query(user_input: str, session: SQLiteSession) -> None:
    """Run one user query through the agent system, with error handling
    and a human-approval gate for anything the Recommendation Agent flags
    as worth permanently curating."""
    try:
        result = await Runner.run(
            knowledge_search_agent,
            user_input,
            session=session,
        )
    except Exception:
        logger.exception("Agent run failed for input: %r", user_input)
        print(
            "Sorry — something went wrong answering that. It's been logged "
            "to knowledge_manager.log."
        )
        return

    print(f"\n{result.final_output}\n")

    # If the Recommendation Agent produced a structured output flagged for
    # curator review, run the curator -> human approval -> save flow.
    final = result.final_output
    if isinstance(final, RecommendationOutput) and final.needs_curator_review:
        await handle_curator_review(final, session)


async def handle_curator_review(
    recommendation: RecommendationOutput, session: SQLiteSession
) -> None:
    """Runs the Knowledge Curator agent on a flagged recommendation, then
    gates each proposed entry behind explicit human (CLI) approval before
    it is persisted — this is the human-approval-in-the-loop step."""
    logger.info("Recommendation flagged for curator review: %s", recommendation.question)
    try:
        curator_result = await Runner.run(
            knowledge_curator_agent,
            (
                f"A recurring question was answered and flagged for curation.\n"
                f"Question: {recommendation.question}\n"
                f"Recommendation: {recommendation.recommendation}\n"
                f"Sources: {', '.join(recommendation.source_documents)}\n"
                f"Propose 1-3 durable knowledge entries worth saving."
            ),
            session=session,
        )
    except Exception:
        logger.exception("Curator agent run failed")
        return

    proposals = curator_result.final_output.proposals
    if not proposals:
        return

    print("--- Knowledge Curator has proposals for the permanent knowledge base ---")
    for i, proposal in enumerate(proposals, start=1):
        print(f"\n[{i}] Topic: {proposal.topic}")
        print(f"    Insight: {proposal.insight}")
        print(f"    Sources: {', '.join(proposal.source_documents)}")
        answer = input("    Approve saving this entry? (y/n): ").strip().lower()
        if answer == "y":
            outcome = curate_knowledge_entry(
                topic=proposal.topic,
                insight=proposal.insight,
                source_documents=", ".join(proposal.source_documents),
            )
            print(f"    -> {outcome}")
            logger.info("Curated entry approved and saved: %s", proposal.topic)
        else:
            print("    -> Skipped.")
            logger.info("Curated entry rejected by human: %s", proposal.topic)


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Add it to a .env file (see "
            ".env.example) or export it in your shell before running."
        )
        sys.exit(1)

    session = SQLiteSession("employee_session", SESSION_DB_PATH)

    if len(sys.argv) > 1:
        # Single-shot mode: python main.py "question here"
        await run_query(" ".join(sys.argv[1:]), session)
        return

    print("AI Enterprise Knowledge Manager — type 'exit' to quit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        await run_query(user_input, session)


if __name__ == "__main__":
    asyncio.run(main())
