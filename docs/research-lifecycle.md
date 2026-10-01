# Know Your Company — Research Lifecycle

## 1. Overview

The research lifecycle describes how Know Your Company converts a company name into structured, evidence-backed intelligence.

The process is deliberately staged:

```text
Company Input
    ↓
Research
    ↓
Source Collection
    ↓
Evidence Cleaning
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Storage
    ↓
Task-specific Retrieval
    ↓
Structured LLM Extraction
    ↓
Grounding Validation
    ↓
Persistence
    ↓
Workspace / Chat / Report
```

The important design principle is that evidence is collected and indexed before the language model performs the main intelligence extraction.

## 2. Stage 1 — Company Input

The user provides:

- Company name
- Optional official website
- Optional research instructions

The application uses this input to establish the research target and company-specific storage/retrieval context.

## 3. Stage 2 — Company Research

The research pipeline combines two main public information paths.

### Official website

The website crawler collects relevant public pages from the company's site.

The crawler is bounded and respects `robots.txt`.

### Web search

DuckDuckGo search through `ddgs` is used to discover additional public sources.

The application applies limits to:

- Number of queries
- Results per query
- Number of source pages
- Website pages
- Request timeout

This prevents the research stage from becoming an unrestricted crawler.

## 4. Stage 3 — Source Normalization

Each collected page becomes a normalized source/evidence record.

The processing stage handles:

- URL normalization
- Source type classification
- Title extraction
- Text extraction
- Text cleaning
- Duplicate paragraph removal
- Contact-information redaction
- Stable source IDs

A source can then be referenced consistently throughout later stages.

For example:

```text
SRC-001
SRC-002
SRC-003
```

The application preserves the relationship between a claim and the evidence used to support it.

## 5. Stage 4 — Evidence Preparation

The collected text is prepared for retrieval.

The system does not simply place every complete webpage into an LLM prompt.

Instead, source text is divided into manageable chunks.

The project uses paragraph-aware chunking with:

- Configurable chunk size
- Configurable overlap
- Long-token handling
- Token estimates

This makes retrieval more precise and keeps the model context bounded.

## 6. Stage 5 — Local Embeddings

Each evidence chunk is converted into a vector representation.

The documented primary embedding model is:

```text
BAAI/bge-small-en-v1.5
```

Embeddings are generated locally.

The application also provides a deterministic fallback embedding mechanism so that the retrieval architecture can continue operating when the primary embedding backend cannot be loaded.

## 7. Stage 6 — Vector Storage

The generated vectors are stored in embedded Qdrant.

The stored information is associated with the company being researched.

This enables company-filtered retrieval and avoids mixing evidence between separate researched companies.

The vector store is local and is excluded from Git through the project's `.gitignore`.

## 8. Stage 7 — Task-specific Retrieval

Different intelligence tasks require different evidence.

For example:

- Products need product/service evidence.
- Technology needs technology-related evidence.
- Funding needs funding events and financial evidence.
- People need public professional information.
- GTM needs evidence about customers, use cases, needs, functions, and market context.

The retriever therefore selects the most relevant chunks for the current task.

The default documented retrieval setting is:

```text
TOP_K=5
```

The retrieved evidence is bounded before it is sent to the model.

## 9. Stage 8 — Structured LLM Analysis

Ollama runs the local Qwen3 1.7B model.

The model receives:

1. The relevant task instructions
2. A bounded evidence block
3. The expected structured output format

The model is asked to extract or reason from the supplied evidence.

It is not treated as the primary web researcher.

The output is intended as a structured proposal for the next validation stage.

## 10. Stage 9 — Grounding Validation

This is one of the most important stages in the lifecycle.

The application checks the proposed intelligence against the retrieved evidence.

Examples:

### Products

The product name must be supported by the cited evidence.

### Funding

Funding amounts, dates, round types, and investors require supporting text.

Funding totals are not created by blindly adding individual rounds.

### People

The person's name and role must be supported together.

A nearby title in a document cannot automatically be used to support a different person.

### Technology

Technology that is directly observed or explicitly stated by the company's own publications can receive a stronger verification status.

Other technology signals are treated as inferred.

### Competitors

Competitor relationships require explicit evidence describing the company as a competitor, alternative, or comparable entity near the relevant name.

### Growth and GTM

Verified claims require meaningful support from the cited evidence.

## 11. Stage 10 — Verification States

The application distinguishes three important evidence states.

### Verified

The available source evidence supports the claim under the module's validation rules.

### AI Inference

The claim is reasoning derived from available evidence rather than a directly stated fact.

### Not verified

The available evidence did not support the claim sufficiently.

This distinction prevents the UI from presenting every model-generated statement as an established fact.

## 12. Stage 11 — Persistence

After validation, structured intelligence is saved locally.

The persistence layer uses JSON.

The application stores the structured company representation rather than only storing a free-form LLM answer.

This allows the same saved data to power:

- Workspace tabs
- Quick actions
- Grounded chat
- Reports

## 13. Stage 12 — Streamlit Workspace

The Streamlit interface presents the saved intelligence through separate modules.

The validated run demonstrated:

- Company overview
- Products
- Technology
- Competitors
- Funding
- Growth signals
- People
- GTM
- Sources
- Grounded AI chat

The Sources view provides access to the evidence behind the source IDs.

## 14. Stage 13 — Deterministic Reporting

The report generator uses the persisted structured data.

The report is not produced by asking the LLM to write a new independent research report.

This means the reporting stage is primarily formatting and presentation.

The report can therefore preserve:

- Structured fields
- Verification states
- Source IDs
- Evidence-backed claims

## 15. End-to-End Lifecycle

The complete process can be summarized as:

```text
1. User enters company
          ↓
2. Company research starts
          ↓
3. Official site is crawled
          ↓
4. Web search discovers additional sources
          ↓
5. Sources are normalized and cleaned
          ↓
6. Evidence is chunked
          ↓
7. Chunks are embedded
          ↓
8. Vectors are stored in Qdrant
          ↓
9. Each intelligence module retrieves evidence
          ↓
10. Qwen3 generates structured proposals
          ↓
11. Python validates claims against evidence
          ↓
12. Results become Verified / Inferred / Not verified
          ↓
13. Structured intelligence is persisted
          ↓
14. Streamlit displays the intelligence
          ↓
15. Grounded chat uses the same evidence
          ↓
16. Markdown report formats the persisted results
```

## 16. Failure Handling

The system records failure states instead of silently treating incomplete research as successful.

For example, the real Stedi validation recorded a failed leadership search while still preserving the successful results from the other research modules.

This behavior is important because incomplete research should remain visible rather than being silently converted into an apparently complete company profile.

## 17. Why the Sequence Matters

The order of operations is deliberate.

If the LLM were asked to generate the company profile first and sources were added afterward, the source layer would not necessarily prove that the generated claims came from the evidence.

The current architecture reverses that relationship:

```text
Evidence first
     ↓
Retrieval
     ↓
LLM analysis
     ↓
Validation
     ↓
Persistence
```

The evidence therefore acts as the input boundary for the reasoning layer.

