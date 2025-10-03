# Knowledge Graph SME Workbench

MVP Streamlit application for building an LLM-assisted knowledge graph from locally ingested documents. SMEs upload source files, vet LLM-generated triples, and explore/query the approved graph with provenance retained for every edge.

## Features
- Upload PDFs, DOCX, PPTX, plaintext, or paste raw text; content is chunked and stored locally with SHA256-based deduping.
- Re-ingesting the same document regenerates triples while flagging potential duplicates for SME review.
- Configurable LLM extraction endpoint (stubbed by default) produces candidate subject–predicate–object triples per chunk, preserving the raw LLM response for audit.
- All LLM prompts live under `resources/prompts/` so SMEs can audit or adjust instructions (e.g., canonical terminology enforcement).
- Admin tab lets SMEs curate canonical terms/aliases and delete ingested documents along with associated graph knowledge.
- Review queue lets the SME accept/reject/edit triples. Approved triples immediately create graph nodes/edges and provenance links back to their source chunks and SME actions.
- Graph visualization tab (NetworkX + PyVis) with document-based filters and hover tooltips showing provenance context.
- Natural-language query tab with chat-style interface. An LLM orchestrates retrieval→plan→execution→answer, falling back to keyword search when the model lacks evidence and always returning the supporting triples with citations, node attributes, and tags.
- Adaptive ontology suggestions cluster similar nodes, call an LLM for rationale, and log guardrails so SMEs can approve inferred parent concepts with provenance retained.
- Graph maintenance tools can scan for unconnected but semantically similar clusters and request LLM-backed recommendations for bridging nodes or edges.
- SQLite persistence using SQLModel keeps everything portable; no external services required.

## Documentation

All logic functions now include docstrings and can be browsed via auto-generated API docs.

```bash
python generate_docs.py
```

The command writes HTML output to `docs/`, providing a navigable view of repositories, services, and UI modules so you can trace where each responsibility lives and why.

## Getting Started
1. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **(Optional) Configure LLM & embedding endpoints**
   - Copy `.env.example` to `.env` and set `LLM_ENDPOINT`/`LLM_API_KEY` for your chat-completions model. Provide `EMBEDDING_ENDPOINT` (plus optional key/deployment) if you want semantic similarity and clustering. When unset, the app falls back to stubbed extraction and fuzzy similarity.
   - Adjust `LLM_TEMPERATURE_*` values to match the behaviour expected by your model (e.g., some GPT-5 deployments require `temperature=1`).
3. **Initialize the database & run Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Workflow**
   - *Ingest*: Upload a document and trigger extraction. Candidate triples appear in the review queue.
   - *Review*: Inspect triples, adjust node wording, approve or reject. Approval writes nodes/edges and attaches provenance (document/page + SME notes).
   - *Visualize*: Explore the approved graph, apply document filters, and inspect provenance via tooltips.
   - *Query*: Ask questions about the graph. The MVP uses deterministic keyword matching while the LLM-powered NL query planner is wired for future integration.

## Project Layout
```
app/
  core/            # Configuration & logging bootstrap
  data/            # SQLModel schema & repositories
  ingestion/       # File parsing and checksum utilities
  llm/             # LLM client & response schemas
  services/        # Orchestrators for ingestion, review, querying
  ui/              # Streamlit tab renderers
  utils/           # Helpers (upload temp storage, etc.)
resources/prompts/ # Placeholder for prompt templates
streamlit_app.py   # Streamlit entrypoint
```

## Next Steps
- Wire the production LLM extraction endpoint, add guardrails validation, and surface model confidence more prominently in the UI.
- Expand the review workflow with canonical term selection and alias management per industry taxonomy.
- Expand query evaluation to cover more embeddings/backends and enrich answer ranking.
- Add automated tests (pytest) covering ingestion parsers, repositories, and service orchestration; integrate Ruff/Black for linting.
- Harden provenance audit by exposing SME action logs and LLM prompt/response inspection from the UI.

## User Stories
User Story Description: 

As a Validation Lead on the TAD platform, I require the LLP to systematically extract and structure validation-critical information from innovation proposals and SME inputs to efficiently link technical claims with traceable facts. This enables assessment of each innovation’s readiness, operational fit, and strategic alignment for deep underground mining applications.  

 

Acceptance Criteria: 

Agent ingests innovation context including scope, maturity, constraints, and technical documentation. 

Agent scans all client documents and SME inputs to extract information across 12 readiness domains: 

Ventilation & Radon Control 

Cooling Systems 

Rock Breaking, Sorting & Hauling 

TRL & Prior Testing 

Autonomy Readiness 

Production & Development Cycle Fit 

Safety & Emergency Systems 

Infrastructure Integration 

Maintainability & Survivability 

Sustainability & Energy Efficiency 

Human Factors & Operational Fit 

Known/Potential Failure Modes 

 

Workflow overview: 

 SME / Mining engineer collaboration  

SMEs and mining engineers are the primary source of technical truth. They provide trial data, innovation proposals, and operational knowledge that the LLP must interpret and validate 

SMEs upload innovation proposals, technical specifications, trials reports and supporting materials directly to the LLP system  

Parse and extract validation-critical information across domains like ventilation, cooling, autonomy, and safety. 

Identify relationships between technical concepts, constraints, and innovation goals (e.g. how cooling systems relate to depth and power infrastructure). 

Detect missing, vague, or contradictory information and generate targeted, domain-specific questions. 

Engage SMEs with intelligent, non-redundant queries to clarify assumptions and close gaps. 

Capture SME responses and update the innovation’s readiness profile in real time. 

Construct a “knowledge web” that organizes facts, dependencies, and validation logic in a traceable format. 

Success criteria: 

All validation domains are populated with structured, accurate data  

SME queries are relevant and resolved efficiently 

The knowledge web reflects a coherent traceable understanding  

 

Project Manager / initiative owner  

PMs and initiative owners need structured outputs to plan trials, allocate resources and communicate scope to stakeholders  

Automatically generate documentation based on extracted and clarified information 

Success criteria: 

Documents are complete, accurate and formatted  

Outputs align with TAD’s validation framework 

 

Management / Executive Stakeholders  

Executives require high-level insights and real-time support during strategic reviews and investment decisions  

 Provide chatbot-style interaction during meetings to answer questions, clarify technical details, and surface readiness blockers. 

Translate complex technical data into concise, decision-ready summaries. 

Capture live SME feedback and update readiness profiles dynamically. 

Support scenario-based queries (e.g. “What happens if we deploy this at 1.8km depth?”). 

Success metrics 

Executives receive clear, contextual answers  

Real-time updates reflect the most current SME input and validation status. 

Executives invest in the product  