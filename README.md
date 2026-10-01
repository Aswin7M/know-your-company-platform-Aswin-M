# Know Your Company

**AI-powered company intelligence and GTM research platform — local, free, and source-grounded.**

Know Your Company is a locally hosted Streamlit application that researches companies from public web sources, converts collected evidence into structured intelligence, and provides grounded AI chat and reporting.

Designed for **company research, GTM intelligence, competitive research, and business analysis** — not clinical diagnosis or medical decision support.

> **Core principle:** **Search → Evidence → Index → Retrieve → LLM → Validate → Intelligence**

The system separates:
- **Verified** — supported by collected source evidence
- **AI Inference** — reasoning derived from available evidence
- **Not verified** — information the available evidence did not support

## Why It Was Built

The project is designed to make AI-assisted company research easier to inspect and audit.

The workflow is:

1. Collect public evidence.
2. Preserve source IDs and metadata.
3. Clean, de-duplicate, chunk, and index evidence locally.
4. Retrieve relevant evidence for each task.
5. Use a local LLM for structured analysis.
6. Apply deterministic Python grounding checks.
7. Generate reports from validated stored data.

This makes the application a **research pipeline with an AI reasoning layer**, rather than simply a chatbot.

## What It Does

Enter a company name and optionally provide its website or research instructions.

The application provides:

- Company Overview
- Products & Services
- Technology
- Competitors
- Funding
- Growth Signals
- People / Decision Makers
- GTM Intelligence
- Sources / Evidence
- Grounded AI Chat
- Markdown Reports

## Architecture

The architecture separates research, retrieval, local model reasoning, validation, and output generation.

![Know Your Company Architecture](docs/architecture-diagram.png)

Detailed architecture documentation:
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/research-lifecycle.md`](docs/research-lifecycle.md)
- [`docs/grounding.md`](docs/grounding.md)

## Screenshots

Screenshots below were captured during a real Stedi research run.

### Research Input
![Research Input](docs/screenshots/01-research-input.png)

### Company Overview
![Company Overview](docs/screenshots/02-company-overview.png)

### Technology Evidence
![Technology Evidence](docs/screenshots/03-technology-evidence.png)

### Funding
![Funding](docs/screenshots/04-funding.png)

### Source Evidence
![Source Evidence](docs/screenshots/05-source-evidence.png)

### Grounded AI Chat
![Grounded AI Chat](docs/screenshots/06-grounded-ai-chat.png)

## Key Features

### Public Web Research
- Official website crawling
- DuckDuckGo-based search through `ddgs`
- Bounded research depth and query limits
- `robots.txt` awareness
- Source normalization and classification
- Duplicate paragraph removal
- Email and phone redaction

### Local LLM
Default model:

```text
qwen3:1.7b
```

Inference runs through local Ollama, avoiding paid LLM APIs and API-key dependency.

### Local RAG

```text
Cleaned → Chunked → Embedded → Qdrant → Retrieved by company/relevance
```

Uses:
- Sentence Transformers
- `BAAI/bge-small-en-v1.5`
- Embedded Qdrant
- Company-filtered retrieval
- Paragraph-aware chunking

### Evidence Grounding

LLM output is treated as a proposal rather than automatically trusted truth. Deterministic checks validate source IDs and evidence relationships before persistence.

See [`docs/grounding.md`](docs/grounding.md) for the detailed rules.

### Grounded AI Chat

The chat layer retrieves relevant evidence before asking the local model to answer. Responses can expose the answer, evidence, inference, source list, retrieved evidence, and retrieval scores.

If relevant evidence cannot be retrieved, the system can report that the information was not verified.

### Deterministic Reports

Markdown reports are generated from saved structured intelligence rather than asking the LLM to produce a second free-form research report.

## Technology Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| Local LLM | Ollama |
| Model | Qwen3 1.7B |
| Web Search | DuckDuckGo / `ddgs` |
| Website Parsing | Requests + BeautifulSoup |
| RAG | Custom retrieval pipeline |
| Embeddings | Sentence Transformers |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Vector Store | Embedded Qdrant |
| Data Models | Pydantic |
| Persistence | JSON + Markdown |
| Testing | Pytest + Streamlit testing |
| Runtime | Windows / local single-user workflow |

## Prerequisites

Before running the project, the system needs:

- Windows 11 or compatible Windows environment
- Python 3.10+
- Ollama installed locally
- `qwen3:1.7b` downloaded through Ollama
- Internet access for dependency installation and public web research
- Sufficient RAM for local CPU inference

`setup_windows.bat` creates the Python virtual environment and installs Python dependencies.

Ollama and its model files are separate prerequisites and are not bundled in the repository.

See [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) for detailed Ollama setup and troubleshooting.

## Installation

### 1. Clone

```cmd
git clone https://github.com/Aswin7M/know-your-company-platform-Aswin-M.git
cd know-your-company-platform-Aswin-M
```

### 2. Run Windows setup

```cmd
setup_windows.bat
```

The script checks Python, creates `.venv`, installs runtime dependencies, creates local data directories, and creates `.env` from `.env.example`.

It does **not** download the Ollama model.

### 3. Install and prepare Ollama

```cmd
ollama --version
ollama pull qwen3:1.7b
ollama run qwen3:1.7b
```

The application expects:

```text
http://localhost:11434
```

Default configuration:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

### 4. Development dependencies

For testing:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Running the Application

```cmd
run.bat
```

Or:

```cmd
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### Research workflow

1. Enter a company name.
2. Optionally provide its official website and research instructions.
3. Start research.
4. Public evidence is collected and cleaned.
5. Evidence is indexed and stored locally.
6. Research modules retrieve relevant context.
7. Ollama/Qwen3 produces structured analysis.
8. Python applies grounding checks.
9. Validated intelligence is stored and displayed.

Main workspace areas:

```text
Overview
Products
Technology
Competitors
Funding
Signals
People
GTM
Sources
AI Chat
```

## Validation

### Stedi Real-World Test

The application was tested against **Stedi** using the actual Windows environment, local Ollama instance, and public web research.

**Date:** 2026-10-01  
**Model:** `qwen3:1.7b`

| Metric | Result |
|---|---:|
| Sources | 45 |
| Products | 6 |
| Technology signals | 11 |
| Competitors | 9 |
| Funding events | 3 |
| Growth signals | 8 |
| Named people | 2 |

The run produced a saved Markdown report and exposed collected sources through the Sources tab. It also demonstrated separation of verified and inferred technology signals, preservation of a failed leadership-search state, and deterministic report generation from persisted structured data.

Detailed report: [`docs/examples/stedi-report.md`](docs/examples/stedi-report.md)

### Automated Tests

```cmd
pytest
```

Documented validation snapshot:

```text
182 tests passed
0 tests failed
```

Coverage includes models, source handling, chunking, embeddings, vector storage, retrieval, Ollama client behavior, website parsing, search normalization, grounding, storage, the sample-company flow, and Streamlit UI behavior.

## Offline Sample Company

The repository includes a fictional sample company:

**Northwind Health Clearinghouse**

Its evidence uses reserved `.example` URLs, allowing the pipeline to be exercised without live company research.

## Privacy & Local-First Design

- No paid LLM API is required for the default workflow.
- No OpenAI, Anthropic, Groq, or Gemini API key is required.
- LLM inference runs through local Ollama.
- Company research is stored locally.
- Runtime company data, local vector data, and `.env` are excluded from Git.
- Emails and phone numbers are redacted from collected text.
- LinkedIn is deliberately not fetched.
- The current architecture is intended for a single-user local workflow and does not provide authentication or multi-user access control.

## Limitations

- **Small local model:** Qwen3 1.7B is practical for constrained CPU/RAM environments, but larger models can provide stronger reasoning.
- **Grounding:** deterministic checks reduce unsupported claims but cannot guarantee semantic correctness.
- **Retrieval:** information outside retrieved context may be missed.
- **Web coverage:** JavaScript-heavy sites, paywalls, login-only pages, some PDFs, and rate-limited sources may have limited coverage.
- **Performance:** CPU-only research can take minutes because multiple structured model calls may run sequentially.
- **Vector store:** embedded Qdrant is designed for the local single-user architecture, not multi-user production deployment.

## Project Structure

```text
know-your-company/
├── app.py
├── config.py
├── README.md
├── OLLAMA_SETUP.md
├── requirements.txt
├── requirements-dev.txt
├── setup_windows.bat
├── run.bat
├── ai/
├── research/
├── rag/
├── models/
├── storage/
├── tests/
└── docs/
    ├── architecture.md
    ├── architecture-diagram.png
    ├── research-lifecycle.md
    ├── grounding.md
    ├── testing.md
    ├── validation.md
    ├── examples/
    │   └── stedi-report.md
    └── screenshots/
        ├── 01-research-input.png
        ├── 02-company-overview.png
        ├── 03-technology-evidence.png
        ├── 04-funding.png
        ├── 05-source-evidence.png
        └── 06-grounded-ai-chat.png
```

Detailed implementation and validation material is maintained in `docs/`.

## Roadmap

Possible future extensions:

- LangGraph orchestration
- Multi-agent research workflows
- Incremental company re-research
- Change and signal tracking
- CRM integrations
- PostgreSQL + pgvector
- Authentication and multi-user support
- React-based frontend
- Stronger optional search providers
- Calibrated retrieval thresholds
- Larger local models where hardware permits

These are future directions rather than requirements for the current local MVP.

## Current Status

**Know Your Company v1 — Local MVP**

| Area | Status |
|---|---|
| Core application | Complete |
| Streamlit UI | Complete |
| Local Ollama integration | Validated |
| Qwen3 1.7B | Validated |
| Web research | Validated |
| RAG pipeline | Complete |
| Grounding checks | Complete |
| Structured intelligence | Complete |
| Grounded chat | Complete |
| Deterministic reporting | Complete |
| Automated tests | 182 documented passing |
| Stedi real-world validation | Complete |
| GitHub publication | Complete |

## Author

**Aswin M**

GitHub: [@Aswin7M](https://github.com/Aswin7M)

Repository: [know-your-company-platform-Aswin-M](https://github.com/Aswin7M/know-your-company-platform-Aswin-M)

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Further Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/research-lifecycle.md`](docs/research-lifecycle.md)
- [`docs/grounding.md`](docs/grounding.md)
- [`docs/testing.md`](docs/testing.md)
- [`docs/validation.md`](docs/validation.md)
- [`docs/examples/stedi-report.md`](docs/examples/stedi-report.md)
- [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md)
