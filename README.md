[README.md](https://github.com/user-attachments/files/30404865/README.md)
# AI Enterprise Knowledge Manager

A multi-agent system built on the OpenAI Agents SDK that lets employees ask
natural-language questions about company SOPs, policies, and meeting
decisions — and get sourced, structured answers instead of hunting through
a wiki.

## 1. Problem Analysis

**Business context**
Company knowledge (SOPs, HR/IT policies, meeting decisions) is scattered
across documents that employees rarely read end-to-end. The same questions
("am I eligible for remote work?", "what did we decide about the bulk
export feature?") get re-asked in Slack instead of answered from existing
documentation, and repeat answers never make it back into a durable,
searchable form.

**Stakeholders**
- Employees needing quick, accurate answers to policy/process questions
- People Ops / IT Security / Finance, who own the underlying documents
- Knowledge/documentation owners, who want recurring questions distilled
  into permanent reference material without manually re-editing docs

**Problem statement**
Build a system that can search and reason over the company's existing
documents, answer questions with accurate sourcing, remember meeting
decisions, and — with human sign-off — grow the knowledge base by curating
recurring questions into new reference entries.

**Objectives**
1. Answer document-specific questions with precise, sourced content.
2. Interpret how policies apply to a specific employee situation.
3. Recall and log meeting decisions and action items.
4. Synthesize cross-cutting questions into structured recommendations.
5. Grow the knowledge base over time, gated by human approval.

## 2. Multi-Agent Design

```
                    ┌─────────────────────┐
     employee  ───▶ │  Knowledge Search    │  (triage / entry point)
                    │  Agent               │
                    └──────────┬───────────┘
                               │ handoff
        ┌──────────────┬───────┼────────────────┬─────────────────┐
        ▼              ▼                        ▼                 ▼
 ┌─────────────┐ ┌─────────────┐        ┌───────────────┐ ┌───────────────────┐
 │  Document   │ │   Policy    │        │   Meeting     │ │  Recommendation    │
 │  Reader     │ │   Expert    │        │   Memory      │ │  Agent             │
 └─────────────┘ └─────────────┘        └───────────────┘ └─────────┬──────────┘
                                                                     │ if flagged
                                                                     │ recurring
                                                                     ▼
                                                          ┌────────────────────┐
                                                          │ Knowledge Curator  │
                                                          │ (proposes entries) │
                                                          └─────────┬──────────┘
                                                                    │
                                                           human approval (CLI)
                                                                    │
                                                                    ▼
                                                        curated_knowledge.json
```

**Roles**
| Agent | Role |
|---|---|
| Knowledge Search | Entry point/triage. Does a quick search, routes to the right specialist, or answers directly if trivial. |
| Document Reader | Reads a specific document in full and answers detail questions with precise quotes. |
| Policy Expert | Looks up policies/SOPs by Document ID, interprets how they apply to a situation, flags cross-policy overlaps. |
| Meeting Memory | Recalls past meeting decisions/action items; logs new meeting notes. |
| Recommendation Agent | Synthesizes across multiple documents into a structured, sourced recommendation; flags recurring topics for curation. |
| Knowledge Curator | Reviews flagged recommendations and proposes durable knowledge-base entries — never saves without human approval. |

**Interaction & handoff flow**
The Knowledge Search agent is the single entry point and uses the Agents
SDK's native `handoffs=[...]` mechanism to transfer control to whichever
specialist fits the question. The Recommendation Agent's structured output
(`RecommendationOutput`) carries a `needs_curator_review` flag; when true,
`main.py` (not the LLM) explicitly invokes the Curator agent and then gates
any resulting proposal behind a CLI human-approval prompt before persisting
it — this keeps the human-in-the-loop step outside LLM control by design.

**Tool integration overview**
All 6 tools are plain Python functions decorated with `@function_tool`,
operating on a local `mock_docs/` folder (no external API dependency
required beyond the OpenAI API itself):
`search_knowledge_base`, `read_document`, `list_documents`,
`get_policy_by_id`, `log_meeting_note`, `curate_knowledge_entry`.

## 3. Implementation

- **5+ specialized agents**: 6 total (see table above).
- **5+ tools**: 6 total (see above), shared across agents where relevant.
- **Agent handoffs**: native SDK handoffs from Knowledge Search to 4
  specialists.
- **Memory/context management**: `SQLiteSession` persists conversation
  history across turns and across process restarts (`conversations.db`).
- **Structured outputs**: `RecommendationOutput` and
  `CuratorProposalOutput` are Pydantic models set as each agent's
  `output_type`.
- **Human approval**: Curator proposals require an explicit `y/n` CLI
  approval in `main.py` before `curate_knowledge_entry` is called.
- **Error handling and logging**: all agent runs are wrapped in try/except
  with logging to both console and `knowledge_manager.log`.
- **Session persistence**: same `SQLiteSession`, keyed by session ID, so a
  user's history survives restarting the program.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your real OPENAI_API_KEY
python main.py
```

Or ask a single question directly:
```bash
python main.py "What's our policy on remote work eligibility?"
```

## Project Structure
```
ai-knowledge-manager/
├── main.py              # orchestration, session, error handling, approval gate
├── agents_setup.py       # 6 agent definitions + structured output schemas
├── tools.py             # 6 function tools shared across agents
├── mock_docs/            # sample SOPs, policies, meeting notes (the "knowledge base")
├── requirements.txt
├── .env.example
└── README.md
```

## Known Limitations / Next Steps
- The knowledge base is a small set of mock documents; a production version
  would connect to a real doc store (Confluence, Notion, SharePoint) and
  likely use vector search instead of substring search for scale.
- Human approval is CLI-based (`input()`); a real deployment would use a
  proper approval queue/UI (e.g., Slack approval buttons).
- No RAG/embeddings layer yet — `search_knowledge_base` is a simple
  substring search, which is the deliberate "easiest version" scope for
  this capstone. Swapping in an embeddings-based retriever is a natural
  next step if time allows.
