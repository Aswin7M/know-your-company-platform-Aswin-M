# 🏥 Know Your Company

**AI-powered healthcare company intelligence - local, free, and source-grounded.**

Enter a healthcare company, let the app research public sources, and then ask questions about it. Every factual answer is tied to numbered sources (`SRC-001`, `SRC-002`, ...), and the app keeps three things visibly separate: **verified/source-supported information**, **AI inference**, and **what could not be verified**.

It is inspired by the simplicity of the *Know-your-PDF* project, but turns the flow from `PDF → RAG → chat` into `Company → web research → evidence/RAG → intelligence → chat`.

> **Design principle:** *Search → Evidence → Index → Retrieve → LLM → Intelligence.* Python does the research (search, fetch, clean). The LLM never invents evidence; it only analyses evidence that has been stored and indexed, and Python then **checks its output against that evidence**.

## Features

* **₹0 / $0.** No paid API, no API keys, no credit card. Runs fully locally.
* **Local LLM** via Ollama (default `qwen3:1.7b`, configurable).
* **Free web research** via DuckDuckGo (`ddgs`) plus the company's official website (polite, `robots.txt`-aware).
* **Evidence pipeline:** clean → de-duplicate → chunk → embed (sentence-transformers) → store in embedded Qdrant.
* **Structured intelligence:** company profile, products, technology, competitors, funding, growth signals, people, GTM analysis.
* **Anti-hallucination checks** (see [How grounding works](#how-grounding-works)).
* **Quick actions:** Company Summary, Products & Services, Technology, Competitors, Funding, Growth Signals, Decision Makers, GTM Analysis, Why This Company?
* **Grounded chat** with clickable sources and visible retrieval scores.
* **Markdown report** with 12 sections, clearly marking AI inference.
* **Graceful failure:** no Ollama, blocked websites, DuckDuckGo errors, malformed model JSON or embedding problems never crash the app.
* **No database:** JSON files under `data/` plus an embedded Qdrant folder. No Docker.

## Architecture

```text
                    Streamlit UI (app.py)
                 ┌──────────┴───────────┐
          Research Company          Company Chat / Quick actions
                 │                        │
      ┌──────────┴──────────┐             │
  Official website      DuckDuckGo        │
  (research/website_    (research/        │
   parser.py)            web_search.py)   │
      └──────────┬──────────┘             │
          clean + de-duplicate            │
          + redact emails/phones          │
                 │                        │
        chunk → embed → Qdrant (local) ◄──┘   rag/chunker · embeddings · vector_store · retriever
                 │                        │
           retrieve top-k evidence ───────┤
                 │                        ▼
        Ollama (ai/ollama.py)  ← anti-hallucination prompt (ai/prompts.py)
                 │
   structured JSON → GROUNDING CHECKS (research/common.py + modules)
                 │
   data/companies/<slug>/{company,evidence,sources}.json + report.md
```

## Requirements

* Windows 11 (the scripts target Windows; the Python code is cross-platform)
* 8 GB RAM, CPU only is fine - **use a small model**
* **Python 3.10 - 3.12** recommended
* [Ollama](https://ollama.com/download) (free)
* Internet access for research, and for the one-time download of the embedding model (~130 MB) and PyTorch

## Installation

```bat
setup_windows.bat
```

This checks Python, creates `.venv`, installs `requirements.txt`, creates the `data/` folders and copies `.env.example` to `.env`. It does **not** download any large AI model.

Manual install (any OS):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

## Ollama setup

See **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)**. In short:

```bash
ollama --version
ollama pull qwen3:1.7b
ollama run qwen3:1.7b
```

Configure in `.env` (or the Settings page):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

## Running the application

```bat
run.bat
```

or `streamlit run app.py`. The sidebar shows **🟢 Ollama Connected** / **🔴 Ollama Not Available**.

## Researching a company

1. Open **Home** (or **Research Company** for advanced options).
2. Enter the **Company Name**. Website and instructions are optional; if you leave the website empty the app searches for it and tells you which one it chose - please verify it.
3. Click **🔍 Research Company**. Progress is real: a step is ticked only when it actually finished; warnings (`⚠`) and failures (`✗`) are shown and the pipeline continues where possible.
4. The workspace opens with tabs: **Overview · Products · Technology · Competitors · Funding · Signals · People · GTM · Sources · AI Chat**.

The steps run in their true order: identify → website → web search → **index evidence** → overview → products → technology → competitors → funding → signals → people → GTM → save. (Indexing precedes analysis because analysis retrieves from the index.)

**Try without internet or Ollama-dependent research:** *Home → Load sample company* indexes a bundled, **fictional** company ("Northwind Health Clearinghouse (Sample)"; URLs use the reserved `.example` domain). You can also untick **Run AI analysis** to only collect and index evidence.

Research with AI analysis makes roughly eight sequential model calls. On a CPU-only laptop expect minutes, not seconds (not benchmarked).

## Company chat

Use the quick-action buttons or the chat box (workspace **AI Chat** tab, or the **💬 AI Chat** page). Each answer is structured as **Answer / Evidence / AI inference**, followed by an automatically generated **Sources** list with clickable links. You can expand **Retrieved evidence** to see exactly which chunks the model saw, with similarity scores. If nothing relevant is retrieved the app says *"Information not verified from the available sources."* without calling the model.

## Reports

**📄 Reports → Generate Markdown Report** builds a 12-section report (Executive Summary, Company Overview, Products & Services, Technology, Target Customers, Competitors, Funding, Growth Signals, People, GTM Analysis, Why This Company?, Sources) and saves it to `data/companies/<slug>/report.md`. The report is generated deterministically from the saved data (no extra model call, so it cannot add new claims). Inference sections are labelled.

## How grounding works

The model's JSON is treated as a *proposal*. Before anything is saved:

| Item | Rule |
|---|---|
| Any item's `source_ids` | Must exist in the source registry; invalid IDs are replaced with sources that really contain the item, otherwise the item is dropped. |
| Products, competitors, technologies, people | The name must literally appear (whole words) in a cited source. |
| Product pricing / funding amounts | Every figure must literally occur in the cited text. |
| Funding total | Kept only if a source states a total. **Never summed by us.** Otherwise *Not verified*. |
| Investors, dates, rounds | Each must be supported by the cited text, else removed. |
| People | Name **and role** must appear together on the same line (abbreviation-aware); neighbours' titles can't vouch for someone else. |
| Growth signals & GTM "verified" claims | Must share most of their meaningful words with the cited text (company name excluded). |
| Technology | *Verified* only if directly observed in page HTML/headers or stated by the company's own publications; everything else is *Inferred*. |
| Competitors | *Verified* only if a source explicitly calls it a competitor/alternative near the name; otherwise *Inferred*. |
| GTM confidence | Derived from how much verified evidence exists - not taken from the model. |
| Chat answers | Model-written "Sources" sections are discarded; we append real ones. Unknown `SRC-` IDs are flagged. |

Privacy: emails and phone numbers are redacted from collected text; LinkedIn and other social sites are never downloaded (only their public search snippet may be used); only public professional information about people is kept.

## Folder structure

```text
know-your-company/
├── app.py                  Streamlit UI
├── config.py               settings: defaults < .env < data/settings.json
├── ai/                     ollama.py (HTTP client) · prompts.py · analyzer.py (chat, extraction)
├── research/               web_search · website_parser · company · products · technology ·
│                           competitors · funding · signals · people · gtm · common (grounding) · pipeline
├── rag/                    chunker · embeddings · vector_store (embedded Qdrant) · retriever · sources
├── models/                 Pydantic models (company, product, technology, competitor, funding, signal,
│                           person, gtm, source) + base (lenient parsing of LLM output)
├── storage/                json_store.py · report.py
├── data/                   companies/<slug>/{company,evidence,sources}.json + report.md · vector_store/
├── tests/                  pytest suite (no Ollama / network needed)
├── examples/sample_company/   fictional evidence for offline testing
├── requirements.txt · requirements-dev.txt · pytest.ini · .env.example · .gitignore
├── OLLAMA_SETUP.md · setup_windows.bat · run.bat
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m compileall .
pytest
```

The suite mocks Ollama's HTTP API and uses a deterministic offline embedder, so **no Ollama, model download or internet is required**. It covers search normalisation, chunking, embeddings, the vector store (insert/retrieve/company filtering), retrieval, the Ollama client (errors, timeouts, malformed/empty replies), Pydantic models, source validation, storage, website parsing/`robots.txt`/404/timeouts, every grounding rule (using a fake model that deliberately fabricates products, funding, people, competitors and citations), an end-to-end sample-company flow, simulated web research, and the Streamlit UI via `streamlit.testing`.

What the tests do **not** prove: behaviour with a real Ollama model, the real embedding model, real DuckDuckGo results, or real browsers/Windows scripts. Validate those on your machine (see Troubleshooting).

## Troubleshooting

| Problem | What to do |
|---|---|
| 🔴 Ollama Not Available | Start Ollama; check `OLLAMA_BASE_URL`; see [OLLAMA_SETUP.md](OLLAMA_SETUP.md). |
| 🟡 Model missing | `ollama pull <model>` (name shown in the app). |
| Research finds little / "No usable evidence" | Give the official website explicitly; check your internet; DuckDuckGo may be rate-limiting - wait a minute and retry. |
| Many modules show ⚠ "No ... could be verified" | Expected when sources don't state the facts. Sources → check what was collected. Try a larger model or richer website. |
| Warning about the fallback embedder | The embedding model could not load (first run needs internet). Fix the network / `EMBEDDING_MODEL`, then **Companies → Re-index**. |
| "Could not open the local vector store" | Another copy of the app is running; close it (embedded Qdrant allows one process). |
| Changed the embedding model | Companies → **Re-index** (vectors from different models are kept separate, never mixed). |
| Website is JavaScript-heavy / empty | Only server-rendered text can be read; the app reports this and relies on search results instead. |
| Timeouts | Smaller model, lower `TOP_K`, higher `OLLAMA_TIMEOUT`. |

## Limitations

* **Small local models are limited.** Expect missed details and occasional awkward phrasing. The checks remove unsupported claims but cannot make a weak model insightful.
* **Grounding is lexical.** It verifies that names, figures and key words appear in the cited text. It cannot detect a wrong *interpretation* that reuses source words (e.g. negation), and it may drop a true claim that was paraphrased too loosely. "Verified" means "found in / stated by the source", not independently fact-checked - and a source itself can be wrong.
* **Retrieval bounds recall.** Only the top-k retrieved chunks are shown to the model per module, so facts in un-retrieved chunks are missed.
* **`MIN_SCORE` is uncalibrated.** The 0.10 default was not tuned against the real embedding model. Use the displayed retrieval scores to tune it. The keyword fallback embedder uses its own lower threshold and is noticeably weaker than the real model.
* **DuckDuckGo** access is unofficial and can be rate-limited or change without notice. Results vary by region.
* **Coverage:** JavaScript-rendered sites, paywalled news, PDFs and login-only pages are not read; LinkedIn is deliberately not fetched.
* **Web results can be outdated or wrong.** Always check the sources before acting.
* **Performance** on CPU-only hardware is slow; this has not been benchmarked.
* **Single user, local only** - no authentication or multi-user support by design. Embedded Qdrant supports one process at a time.
* The Windows `.bat` scripts were written carefully but could not be executed in the environment where this project was built.

## Future improvements

LangGraph / multi-agent orchestration, optional stronger search providers (disabled by default), CRM integrations, Postgres + pgvector, authentication, a React front-end, incremental re-research and change tracking, and calibrated relevance thresholds per embedding model. The code is split into small modules (`research/*`, `rag/*`, `ai/*`) so each can become an agent or service later.
