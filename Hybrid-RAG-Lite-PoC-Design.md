
# Hybrid‑RAG‑Lite PoC Design (JSON Graph + Local Vector Store + Azure OpenAI + Streamlit)

**Date:** 2025‑10‑03  
**Author:** Drafted by M365 Copilot for Charlie Turner  
**Purpose:** Provide a minimal, developer‑ready design for a Hybrid‑RAG‑Lite PoC that routes queries to **Vector**, **Graph**, or **Hybrid** retrieval, using:

- A **local JSON file** as the knowledge graph (no graph DB)
- A **local vector store** (FAISS/Chroma) persisted to disk
- **Azure OpenAI** endpoints for chat and embeddings
- A simple **Streamlit UI** for ingestion, visualization, and Q&A

This design follows the *local vs global* reasoning ideas of **GraphRAG** and the **hybrid retrieval** principle of Azure AI Search, while keeping infrastructure to an absolute minimum.

> References used in this design:
> - **GraphRAG** (local vs global search; community summaries) [1][2]
> - **Azure AI Search** (vector/hybrid search, filtered vector, baseline concepts) [3]
> - **Semantic Kernel** multi‑agent orchestration patterns (optional future) [4]
> - **W3C PROV‑O** provenance model for traceable claims/evidence [5]

---

## 1) Goals & Non‑Goals

**Goals**
- Answer mining innovation questions using the smallest possible stack.
- Route each query to **Vector**, **Graph**, or **Hybrid** retrieval.
- Keep every asserted fact **cited** to a source (file + page) and preserve minimal **provenance**.
- Be production‑minded: pave a path to upgrade to Neo4j/Azure AI Search later **without changing UX**.

**Non‑Goals**
- No full ontology/graph database or cluster.
- No heavy multi‑agent orchestration framework (can be added later).
- No advanced community detection—use **subgraphs** and short summaries instead.

---

## 2) Architecture (Minimal)

**Runtime Components**
1. **Streamlit UI** – one page with panes for Ingest, Graph, Search, Evidence.
2. **Azure OpenAI** –
   - Chat model (e.g., GPT‑4o‑mini) for routing & synthesis.
   - Embeddings (e.g., `text-embedding-3-small`) for chunk vectors.
3. **Local Vector Store** – FAISS or Chroma; persisted to disk.
4. **Local Graph Store** – `graph.json` file with nodes/edges/evidence and PROV‑like fields.

**Why this works**
- Mirrors **GraphRAG** idea of local (entity) vs global (holistic) reasoning using a tiny subgraph instead of a graph DB. [1]
- Takes the **hybrid retrieval** concept (blend signals, apply filters) from Azure AI Search, implemented locally. [3]

---

## 3) Data Model (Files on Disk)

### 3.1 `graph.json`
A single JSON file with three arrays: `nodes`, `edges`, `evidence`.

```json
{
  "nodes": [
    {
      "id": "Tech_BEV_Loader_14t",
      "type": "Technology",
      "name": "BEV Loader 14t",
      "aliases": ["Battery Loader 14t", "LHD 14t BEV"],
      "props": {
        "trl": 7,
        "autonomy": "SemiAutonomous",
        "vendor": "OEM-X",
        "cycleStage": "Production"
      },
      "prov": { "sources": ["Proposal_2025_Q1.pdf#p4"], "lastUpdated": "2025-10-01T10:11:00Z" }
    }
  ],
  "edges": [
    {
      "source": "Tech_BEV_Loader_14t",
      "type": "PROVIDED_BY",
      "target": "Vendor_OEMX",
      "prov": { "sources": ["SpecSheet_OEMX_2025.pdf#p1"] }
    }
  ],
  "evidence": [
    {
      "id": "Trial_TR01",
      "type": "Trial",
      "title": "Mine A BEV trial report",
      "claims": [
        { "id": "Claim_Prod_10pct", "text": "+10% productivity vs diesel", "supports": ["Tech_BEV_Loader_14t"] }
      ],
      "prov": { "file": "Trial_TR01.pdf", "pages": [3,4,5], "date": "2025-06-12" }
    }
  ]
}
```

**Notes**
- **Aliases** improve entity matching.
- `prov` mirrors **W3C PROV‑O**: key source refs and timestamps for traceability. [5]

### 3.2 Vector index `chunks.idx` + `docstore.jsonl`
- Each chunk record: `{ id, text, metadata }`, where `metadata` includes:
  - `source_file`, `page_span`, `entity_links` (node IDs), `domain_tags` (Cooling, Safety...), `cycleStage`, `vendor`.
- Persist FAISS/Chroma index + plain JSONL docstore for metadata.

---

## 4) Ingestion Flow (Upload → Index → Graph)

1. **Upload** PDF/DOC/TXT → text extraction.
2. **Chunk** into ~700–1,000 tokens; attach metadata (file, page span).
3. **Embed** with Azure OpenAI embeddings → upsert into FAISS/Chroma.
4. **Graph extraction (lite)** via single LLM call:
   - Extract minimal fields: *Technology name, TRL, Autonomy, Vendor, Cycle stage, 1–3 Claims*.
   - **Upsert** nodes/edges/evidence into `graph.json`.
   - Add **provenance** (file, page ranges, timestamp) per node/edge/evidence.  

This aligns with **GraphRAG** indexer ideas (entities/relations/claims → graph) but keeps it file‑based. [1]

---

## 5) Router Agent (Vector vs Graph vs Hybrid)

### 5.1 Heuristic pre‑check (fast, no LLM)
- If query includes **compare / trade‑off / dependencies / blockers / impact vs ease / longest time / timeline / constraints** → lean **GRAPH** or **HYBRID**.
- If query names a known entity (match against `name`/`aliases`) → **GRAPH** (subgraph) or **HYBRID‑Local**.
- If query is **factoid** (e.g., TRL, vendor) → **VECTOR**.
- If query has **filterable metadata** (site/OEM/cycle) → **VECTOR** (with metadata filter) or **HYBRID**.

### 5.2 LLM classification (one cheap call)
- **Goal:** classify into `{VECTOR | GRAPH | HYBRID}` using the same chat model.
- Provide: **question text** + a short list of **top node names/aliases**.
- **Policy:**
  - Use **GRAPH** for **multi‑entity, dependency, constraints, timeline** questions.
  - Use **VECTOR** for **single fact or definition**.
  - Use **HYBRID** if both relational reasoning and text details are needed.

This reflects GraphRAG’s **local (entity) vs global (holistic)** split; your PoC simply decides the context to build. [1][2]

**Fail‑safe:** On low confidence (Section 7), escalate to **HYBRID** automatically.

---

## 6) Retrieval Flows

### 6.1 Vector‑only
1. **Embed** query (Azure OpenAI embeddings).
2. Retrieve **top‑k** (e.g., k=6) from FAISS/Chroma.
3. **Re‑rank**: boost chunks whose `entity_links` match detected entities; **dedupe** near‑duplicates.
4. Build **Answer Context** with citations: `[snippet, source_file, page]`.

### 6.2 Graph‑only (GraphRAG‑lite)
1. **Entity link** query tokens to `nodes.name/aliases` (case‑insensitive, fuzzy).
2. **Subgraph**: take seed nodes + **1–2 hop** neighbors; include node props and linked **evidence**.
3. **Summarize** subgraph into structured bullets with **citations** (single LLM call).

> This is a light‑weight take on GraphRAG’s local search over entity neighborhoods. [1]

### 6.3 Hybrid (merge graph + vectors)
1. Run **Graph‑only** subgraph build.
2. Run **Vector‑only** top‑k with **metadata filters** (entities/OEM/site/cycle if present).
3. **Merge & Rank** (simple scoring):
   - `score = w1*graph_relevance + w2*vector_similarity + w3*metadata_match + w4*freshness`
   - Graph relevance: nodes touched, hop distance, important edge types (e.g., `CONSTRAINED_BY`, `SUPPORTED_BY`).
   - Freshness from `prov.lastUpdated`.

The hybrid principle mirrors **Azure AI Search hybrid** ideas (blend multiple signals, filterable metadata), adapted to a local index. [3]

---

## 7) Answer Synthesis, Confidence & Citations

**Synthesizer role (same model, different system prompt)**
- Role: *“You are a mining innovation analyst. Use **only** the provided context. **Cite** every asserted fact like `[source p.x]`. If unknown, say so.”*
- Context sections:
  1) **Graph facts** (node properties + key edges as bullets)
  2) **Evidence claims** (IDs & pages)
  3) **Vector snippets** (top 3–6, trimmed)

**Confidence heuristic** (no extra calls):
- ≥ 2 unique **sources** cited?  
- Mean **vector score** ≥ threshold?  
- Subgraph contains ≥ 1 **seed entity**?  
- If low → show banner *“Low confidence—expanding retrieval (HYBRID)”* and auto‑requery as HYBRID.

Keeping strict citations and using PROV‑like fields meets the traceability requirement. [5]

---

## 8) Streamlit UI (One Page)

- **Ingest**: Upload file → show chunk count and sample → *Add to index & graph* → preview extracted fields (Technology, TRL, Autonomy, Vendor, 1–3 Claims) → allow quick edits.
- **Graph**: Visualize current **subgraph** (e.g., `pyvis`); select node to view properties & evidence links.
- **Search**: Query box + **Auto‑route / Force Vector / Force Graph / Force Hybrid** toggle; show chosen mode, top evidence, and **answer with citations**.
- **Evidence**: Click a citation to open the file and page span in a viewer.

---

## 9) Prompt Templates (Copy‑Ready)

### 9.1 Router Classification Prompt
```
SYSTEM:
Classify the user’s question as VECTOR, GRAPH, or HYBRID.
- Use GRAPH for questions involving dependencies, comparisons, constraints, timelines, multiple entities, or cause–effect reasoning.
- Use VECTOR for a single fact or definition likely found in a short passage.
- Use HYBRID if both relational reasoning and textual details are needed.
Answer with ONE token: VECTOR | GRAPH | HYBRID.

USER:
Question: "{{user_question}}"
Known entities (names/aliases): {{entity_list_csv}}
```

### 9.2 Synthesizer Prompt (Answer Writer)
```
SYSTEM:
You are a mining innovation analyst. Use ONLY the provided context. If a detail is not in the context, say you don’t know. Cite every asserted fact like [source p.x].

USER:
Question: "{{user_question}}"

CONTEXT — GRAPH FACTS:
{{graph_facts_bullets}}

CONTEXT — EVIDENCE CLAIMS:
{{evidence_bullets_with_ids_and_pages}}

CONTEXT — VECTOR SNIPPETS:
{{vector_snippets_with_sources}}

INSTRUCTIONS:
- Provide a concise, structured answer.
- Include a short “Assumptions & Limits” line if necessary.
- End with a bullet list of citations.
```

---

## 10) Upsert Rules (Graph JSON)

- **Nodes**: Upsert by `id` (fallback: slug of normalized name). Merge `aliases` (dedupe, lower‑case compare). Merge `props` shallowly; do not delete existing props unless explicitly null.
- **Edges**: Upsert by `(source, type, target)`. If duplicate, merge `prov.sources` (dedupe).
- **Evidence**: Upsert by `id`. Append `claims` if new. Always retain `prov` entries with new page spans.
- **Provenance**: For each insert/update, append current source file and `lastUpdated` timestamp.

---

## 11) Evaluation & Telemetry

- **Gold queries** (10–20) covering: factoid (TRL, vendor), relationship (dependencies), comparison (impact vs ease), scenario (“at 1.8 km depth”).
- **Metrics** (manual tally at PoC):
  - **Faithfulness**: are all claims cited to correct pages?
  - **Usefulness**: did the user get a directly actionable answer?
  - **Routing accuracy**: did Router pick the right mode?
- **Telemetry** (CSV): `timestamp, query, chosen_mode, k, num_sources, mean_vec_score, num_nodes_in_subgraph, latency_ms, token_usage`.
- **Fallbacks**:
  - If **no entity** matched but question uses *dependencies/constraints*, escalate to **HYBRID**.
  - If **vector top‑k** empty, broaden k and/or expand to **HYBRID**.

---

## 12) Risks & Quick Mitigations

- **Entity linking misses aliases** → maintain `aliases.json`; add/learn during ingestion.
- **Over‑eager Router** → include **manual override** toggle in UI; log routing decisions for review.
- **Hallucinations** → hard‑require citations; “use only provided context” instruction; reject uncited statements.
- **Latency** → limit to **2–3 LLM calls** per query (Router + Synthesizer ± Subgraph summary); use a **small** chat model.

---

## 13) MVP Roadmap (2–4 days)

**Day 1**
- Streamlit skeleton; Azure OpenAI env wiring (chat + embeddings wrappers).
- Initialize FAISS/Chroma; create empty `graph.json` and `docstore.jsonl`.
- Ingest pipeline: extract → chunk → embed → upsert; quick LLM extraction for nodes/edges/evidence.

**Day 2**
- Implement Router (heuristics + 1 LLM classification call).
- Graph retriever: entity linker + 1–2 hop fan‑out + evidence pull.
- Hybrid retrieval: vector + graph merge and scoring; synthesizer with citations; confidence & fallback.

**(Optional Days 3–4)**
- Add SHACL‑like checks for mandatory fields; simple SME question generator.
- Add export buttons (download `graph.json`, `docstore.jsonl`).

---

## 14) Upgrade Path

- **Graph DB**: swap `graph.json` for **Neo4j** or **Cosmos DB (Gremlin)**; keep the same subgraph API.
- **Vector service**: swap FAISS/Chroma for **Azure AI Search** (gives hybrid/vector filters, compression, and scale). [3]
- **Agent orchestration**: move routing/summarization into **Semantic Kernel** multi‑agent patterns if the workflow grows. [4]

---

## 15) References

1. **GraphRAG documentation** – Microsoft Research: local vs global search, community summaries, indexing & query patterns.  
   https://microsoft.github.io/graphrag/  
   https://github.com/microsoft/graphrag

2. **Project GraphRAG** – Microsoft Research overview and publications.  
   https://www.microsoft.com/en-us/research/project/graphrag/

3. **Azure AI Search – Vector & Hybrid search** (filtered vector, hybrid requests, compression).  
   https://learn.microsoft.com/en-us/azure/search/vector-search-overview

4. **Semantic Kernel – Agent Orchestration** (sequential, handoff, group chat patterns).  
   https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/

5. **W3C PROV‑O** – The PROV Ontology for provenance metadata (entities, activities, agents).  
   https://www.w3.org/TR/prov-o/

---

## 16) Appendix: Minimal Schemas

### 16.1 Node
```json
{
  "id": "Tech_BEV_Loader_14t",
  "type": "Technology",
  "name": "BEV Loader 14t",
  "aliases": ["Battery Loader 14t", "LHD 14t BEV"],
  "props": { "trl": 7, "autonomy": "SemiAutonomous", "vendor": "OEM-X", "cycleStage": "Production" },
  "prov": { "sources": ["Proposal_2025_Q1.pdf#p4"], "lastUpdated": "2025-10-01T10:11:00Z" }
}
```

### 16.2 Edge
```json
{
  "source": "Tech_BEV_Loader_14t",
  "type": "CONSTRAINED_BY",
  "target": "Constraint_Depth_1800m",
  "prov": { "sources": ["MineDesign_RevB.pdf#p11"] }
}
```

### 16.3 Evidence (with Claims)
```json
{
  "id": "Trial_TR01",
  "type": "Trial",
  "title": "Mine A BEV trial",
  "claims": [
    {"id": "Claim_Prod_10pct", "text": "+10% productivity vs diesel", "supports": ["Tech_BEV_Loader_14t"]}
  ],
  "prov": {"file": "Trial_TR01.pdf", "pages": [3,4,5], "date": "2025-06-12"}
}
```

### 16.4 Chunk Metadata
```json
{
  "id": "Trial_TR01_p3_c2",
  "text": "During the 2-week trial, BEV loader achieved...",
  "metadata": {
    "source_file": "Trial_TR01.pdf",
    "page_span": [3],
    "entity_links": ["Tech_BEV_Loader_14t"],
    "domain_tags": ["Production", "Autonomy", "Safety"],
    "cycleStage": "Production",
    "vendor": "OEM-X"
  }
}
```

---

**End of document.**
