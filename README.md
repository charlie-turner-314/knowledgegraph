# Knowledge Graph SME Workbench

MVP Streamlit application for building a Gemma-assisted knowledge graph from locally ingested documents. SMEs upload source files, vet LLM-generated triples, and explore/query the approved graph with provenance retained for every edge.

## Features
- Upload PDFs, DOCX, PPTX, or plaintext; files are chunked and stored locally with SHA256-based deduping.
- Re-ingesting the same document regenerates triples while flagging potential duplicates for SME review.
- Gemma 3 extraction client (stubbed by default) produces candidate subject–predicate–object triples per chunk, preserving the raw LLM response for audit.
- Review queue lets the SME accept/reject/edit triples. Approved triples immediately create graph nodes/edges and provenance links back to their source chunks and SME actions.
- Graph visualization tab (NetworkX + PyVis) with document-based filters and hover tooltips showing provenance context.
- Natural-language query tab with chat-style interface. Until the Gemma query planner is integrated, it performs keyword-based graph lookups and returns supporting triples with citations.
- SQLite persistence using SQLModel keeps everything portable; no external services required.

## Getting Started
1. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **(Optional) Configure Gemma endpoint**
   - Copy `.env.example` to `.env` and set `GEMMA_ENDPOINT` + `GEMMA_API_KEY` for a local Gemma 3 server (OpenAI-compatible). Without configuration, the extractor runs in stub mode and simply records an empty triple list.
3. **Initialize the database & run Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Workflow**
   - *Ingest*: Upload a document and trigger extraction. Candidate triples appear in the review queue.
   - *Review*: Inspect triples, adjust node wording, approve or reject. Approval writes nodes/edges and attaches provenance (document/page + SME notes).
   - *Visualize*: Explore the approved graph, apply document filters, and inspect provenance via tooltips.
   - *Query*: Ask questions about the graph. The MVP uses deterministic keyword matching while the Gemma-powered NL query planner is wired for future integration.

## Project Layout
```
app/
  core/            # Configuration & logging bootstrap
  data/            # SQLModel schema & repositories
  ingestion/       # File parsing and checksum utilities
  llm/             # Gemma client & response schemas
  services/        # Orchestrators for ingestion, review, querying
  ui/              # Streamlit tab renderers
  utils/           # Helpers (upload temp storage, etc.)
resources/prompts/ # Placeholder for prompt templates
streamlit_app.py   # Streamlit entrypoint
```

## Next Steps
- Wire the Gemma 3 extraction endpoint, add Guardrails validation, and surface model confidence more prominently in the UI.
- Expand the review workflow with canonical term selection and alias management per industry taxonomy.
- Replace the keyword query fallback with the full NL→query-plan→execution loop, returning ranked answers with citations.
- Add automated tests (pytest) covering ingestion parsers, repositories, and service orchestration; integrate Ruff/Black for linting.
- Harden provenance audit by exposing SME action logs and LLM prompt/response inspection from the UI.
