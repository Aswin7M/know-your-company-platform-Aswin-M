# Know Your Company — System Architecture

## 1. Purpose

Know Your Company is a locally hosted Streamlit application for researching companies and converting public web evidence into structured company intelligence.

The architecture separates the system into distinct responsibilities:

1. User interface
2. Research and evidence collection
3. Evidence processing
4. Retrieval-augmented generation
5. Local language-model analysis
6. Deterministic grounding and validation
7. Persistence and reporting

This separation is intentional. The language model is an analysis component inside a larger evidence-processing pipeline; it is not treated as the system's source of truth.

## 2. High-Level Architecture

```text
                         +----------------------+
                         |     Streamlit UI     |
                         |       app.py         |
                         +----------+-----------+
                                    |
                         Company / Research Input
                                    |
                                    v
                         +----------------------+
                         |  Research Pipeline   |
                         | research/pipeline.py |
                         +----------+-----------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
          +------------------+              +------------------+
          | Official Website|              | DuckDuckGo Search|
          |     Crawler     |              |      / ddgs     |
          +--------+---------+              +--------+---------+
                   |                                 |
                   +----------------+----------------+
                                    |
                                    v
                         +----------------------+
                         | Evidence Processing  |
                         | Clean / Normalize /  |
                         | Deduplicate / Classify|
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Chunking + Embeddings|
                         |     rag/chunker.py   |
                         |   rag/embeddings.py  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Embedded Qdrant      |
                         | rag/vector_store.py  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Task-specific        |
                         | Retrieval            |
                         | rag/retriever.py     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Ollama / Qwen3       |
                         | ai/ollama.py         |
                         | ai/analyzer.py       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Grounding /          |
                         | Validation           |
                         | Deterministic Python |
                         +----------+-----------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
          +------------------+              +------------------+
          | JSON Persistence |              | Markdown Report  |
          | storage/         |              | storage/report.py|
          +------------------+              +------------------+
                   |                                 |
                   +----------------+----------------+
                                    |
                                    v
                         Streamlit Intelligence
                         Workspace + Grounded Chat
```

## 3. Application Layer

The Streamlit application in `app.py` is the primary user interface.

It is responsible for:

- Research input
- Research progress
- Company workspace navigation
- Intelligence tabs
- Quick actions
- Grounded chat
- Report access
- Company management
- Settings

The UI does not independently become the source of company facts. It presents the structured information produced by the research, retrieval, analysis, and validation layers.

## 4. Research Layer

The research layer is responsible for acquiring public evidence.

Relevant modules include:

- `research/pipeline.py`
- `research/company.py`
- `research/products.py`
- `research/technology.py`
- `research/competitors.py`
- `research/funding.py`
- `research/signals.py`
- `research/people.py`
- `research/gtm.py`
- `research/web_search.py`
- `research/website_parser.py`

The pipeline combines an official website crawler with DuckDuckGo-based search. Website content is bounded, cleaned, and normalized before entering the evidence pipeline.

The website crawler also honors `robots.txt` and applies configured limits to avoid unrestricted crawling.

## 5. Evidence Processing

Collected content is converted into normalized evidence records.

The processing stage includes:

- Source normalization
- Source classification
- Text cleaning
- Duplicate paragraph removal
- Contact-information redaction
- Stable source IDs
- Evidence metadata

Stable source identifiers such as `SRC-001` allow later intelligence objects to refer back to the actual collected evidence.

The application does not treat an LLM-generated source list as authoritative. Source references are associated with real retrieved evidence.

## 6. RAG Layer

The RAG layer provides the bridge between collected evidence and task-specific analysis.

### Chunking

`rag/chunker.py` performs paragraph-aware chunking with bounded chunk sizes and overlap. Very long tokens are also handled so that pathological source text does not break the chunking process.

### Embeddings

`rag/embeddings.py` provides the embedding abstraction.

The primary documented embedding model is:

```text
BAAI/bge-small-en-v1.5
```

The project also provides a deterministic fallback backend when the primary embedding model cannot be loaded.

### Vector storage

`rag/vector_store.py` uses embedded Qdrant.

This keeps the vector store local and avoids requiring a separate database service for the single-user local architecture.

### Retrieval

`rag/retriever.py` combines:

- Settings
- Embedding backend
- Vector store
- Company filtering
- Top-k retrieval

The result is a bounded evidence context for each analysis task.

## 7. AI Layer

The AI layer is implemented primarily through:

- `ai/ollama.py`
- `ai/analyzer.py`
- `ai/prompts.py`

Ollama provides the local model interface.

The validated model for the current MVP is:

```text
qwen3:1.7b
```

The model receives a bounded evidence block and a structured extraction prompt rather than being asked to research a company from its own general knowledge.

The model produces structured output that is subsequently checked by Python.

## 8. Grounding and Validation Boundary

A key architectural boundary exists between model generation and persistence.

```text
Retrieved Evidence
       |
       v
Local LLM
       |
       v
Proposed Structured Output
       |
       v
Deterministic Validation
       |
       +---- supported ----> Persist
       |
       +---- unsupported --> Not verified / rejected
```

The model is therefore not the final authority over whether a claim is accepted.

Examples of validation include:

- Product names must be present in supporting evidence.
- Funding amounts require supporting cited text.
- Funding dates, round types, and investors require evidence.
- Person names and roles must be supported together.
- Competitor relationships require explicit supporting evidence.
- Technology confidence distinguishes directly observed or company-stated technology from inferred signals.
- Invalid model-generated source IDs are not accepted as real sources.

The exact validation rules vary by intelligence module.

## 9. Structured Domain Models

The `models/` package defines explicit Pydantic structures for the intelligence domains.

Important models include:

- Company
- Product
- Technology
- Competitor
- Funding
- Growth Signal
- Person
- GTM
- Source
- Evidence
- Claim

This gives the application a stable internal representation instead of relying on arbitrary blocks of generated text.

## 10. Persistence and Reporting

The storage layer uses local JSON and Markdown.

`storage/json_store.py` persists structured company intelligence.

`storage/report.py` generates the Markdown report from the stored structured data.

This is an important design choice: the report is formatted from validated persisted information rather than asking the LLM to generate another independent report.

## 11. Grounded Chat

Grounded chat uses the same local evidence and retrieval architecture.

The flow is:

```text
User Question
     |
     v
Company-filtered Retrieval
     |
     v
Relevant Evidence
     |
     v
Ollama / Qwen3
     |
     v
Grounded Answer
     |
     v
Source / Evidence Display
```

This allows the chat interface to remain tied to the researched company corpus.

## 12. Configuration

Configuration is centralized in `config.py`.

The documented defaults include:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=300
OLLAMA_NUM_CTX=4096

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_BACKEND=auto

TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=60
MIN_SCORE=0.10
MAX_CONTEXT_CHARS=6000

MAX_WEBSITE_PAGES=8
MAX_SEARCH_QUERIES=10
MAX_RESULTS_PER_QUERY=5
MAX_SOURCE_PAGES=12
REQUEST_TIMEOUT=12
```

Configuration precedence is:

```text
Built-in defaults
        ↓
.env
        ↓
data/settings.json
```

## 13. Why the Architecture Is Local-First

The current architecture is designed for a local single-user workflow.

The main reasons are:

- No paid LLM API is required.
- Ollama provides local model inference.
- Qdrant runs in embedded mode.
- Company data remains on the local machine.
- The application can run from a Windows workstation without a separate backend service.

The architecture is therefore intentionally simpler than a multi-user production SaaS system.

## 14. Current Architectural Scope

The current version is a local MVP.

It does not currently provide:

- Multi-user authentication
- Distributed vector infrastructure
- Production API services
- Enterprise access control
- Guaranteed semantic fact verification
- Full web coverage for JavaScript-only or login-only sources

These are boundaries of the current implementation rather than missing claims about the system.

