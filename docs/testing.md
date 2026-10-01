# Know Your Company — Testing and Quality Validation

## 1. Testing Strategy

The project uses several levels of validation rather than relying on a single test type.

The main categories are:

1. Automated Python tests
2. Compile-time validation
3. Offline sample-company testing
4. Local runtime validation
5. Real-world company validation

Each category answers a different question.

## 2. Automated Tests

The project includes a Pytest suite covering the major application layers.

The documented validation snapshot is:

```text
182 tests passed
0 tests failed
```

The tests cover areas including:

- AI analyzer behavior
- Streamlit application behavior
- Chunking
- End-to-end sample flow
- Embeddings
- Grounding
- Domain models
- Research modules
- Ollama client behavior
- Retrieval
- Search handling
- Source handling
- Storage
- Vector storage
- Website parsing

## 3. Test Organization

The `tests/` directory includes:

```text
tests/
├── conftest.py
├── fakes.py
├── test_analyzer.py
├── test_app.py
├── test_chunking.py
├── test_e2e.py
├── test_embeddings.py
├── test_grounding.py
├── test_models.py
├── test_modules.py
├── test_ollama.py
├── test_retrieval.py
├── test_search.py
├── test_sources.py
├── test_storage.py
├── test_vector_store.py
└── test_website_parser.py
```

The tests are deliberately distributed across the main system boundaries rather than concentrating only on the UI.

## 4. Running the Tests

After installing the development requirements:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

run:

```cmd
pytest
```

The project also supports a Python compile check:

```cmd
python -m compileall .
```

## 5. Sample Company Testing

The repository contains a fictional offline sample company.

The sample evidence is designed to exercise the research and intelligence pipeline without requiring live web research.

This is useful for testing:

- Parsing
- Evidence handling
- Structured models
- Retrieval
- Grounding
- Storage
- Reporting
- End-to-end application behavior

The sample company is intentionally fictional and should not be interpreted as real company intelligence.

## 6. Why Controlled Tests Are Important

Some external dependencies are difficult to make deterministic.

Examples include:

- Web search
- Public websites
- Local model inference
- Embedding model downloads

The automated suite therefore uses controlled/fake components where appropriate.

This allows the application logic to be tested without depending on changing internet results or model output.

## 7. Local Runtime Validation

The project was also executed on the target Windows environment.

Documented environment:

```text
Windows 11 Home Single Language
Python 3.13.2
Ollama 0.35.0
Qwen3 1.7B
CPU inference
Streamlit localhost:8501
```

The local setup was completed using the project's Windows setup script and a Python virtual environment.

## 8. Ollama Validation

Ollama was tested locally with:

```cmd
ollama --version
```

and the model:

```cmd
ollama pull qwen3:1.7b
```

The model was then run locally through:

```cmd
ollama run qwen3:1.7b
```

This confirmed that the local model boundary was operational before the real company research validation.

## 9. Real-World Validation

The strongest runtime validation was the Stedi research run performed on 1 October 2026.

The run used:

```text
Model: qwen3:1.7b
Interface: Streamlit localhost:8501
Research: public web + official website
```

Observed output:

```text
45 sources
6 products
11 technology signals
9 competitors
3 funding events
8 growth signals
2 named people
```

The application completed the research workflow and produced a saved Markdown report.

## 10. What the Stedi Run Tested

The real-world run exercised multiple layers at once:

### Research

- Public web search
- Official website collection
- Source processing

### Intelligence extraction

- Company overview
- Products
- Technology
- Competitors
- Funding
- Growth signals
- People
- GTM

### RAG

- Evidence indexing
- Retrieval
- Company-specific evidence context

### Grounding

- Source-linked claims
- Verified versus inferred technology
- Evidence-backed competitor information
- Source display

### UI

- Research progress
- Workspace tabs
- Sources
- Grounded chat
- Report access

## 11. Failure Visibility

The Stedi run also demonstrated that incomplete research can be recorded explicitly.

The research-status panel recorded one failed leadership search rather than silently presenting the People module as if every leadership lookup had succeeded.

This is an important quality property because an incomplete external search should remain distinguishable from a successful search that simply found no information.

## 12. Compile Validation

The project also uses:

```cmd
python -m compileall .
```

This provides a basic syntax/compilation check across the Python source tree.

It complements the behavioral tests rather than replacing them.

## 13. What Has Not Been Proven

The current test and validation record does not establish:

- Production-scale performance
- Multi-user concurrency
- Enterprise security
- Perfect semantic hallucination detection
- Complete web coverage
- Accuracy of every external source
- Performance on large company batches
- Robustness across every website technology
- Reliability of every possible Ollama model

These should remain outside the project's current claims.

## 14. Testing Philosophy

The testing strategy follows a simple principle:

```text
Test deterministic logic with automated tests.
Test the complete local application with runtime validation.
Test external research behavior with real-world runs.
Record limitations instead of hiding them.
```

This provides a more useful picture of the project than reporting only a unit-test count.

## 15. Current Quality Snapshot

At the documented validation point:

```text
Automated tests:       182 passed
Compile validation:    Clean
Local Streamlit:       Passed
Ollama + Qwen3:        Passed
Sample company:        Passed
Real Stedi run:        Completed
```

These results describe the observed project state during the documented validation session and should not be interpreted as a permanent guarantee for future versions.

