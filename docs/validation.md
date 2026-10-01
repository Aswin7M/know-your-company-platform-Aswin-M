# Know Your Company — Validation Record

## 1. Purpose

This document records the real-world validation performed after the local installation of Know Your Company.

The purpose is to distinguish demonstrated behavior from planned functionality and from assumptions.

The main real-world validation target was:

```text
Stedi
```

Validation date:

```text
1 October 2026
```

## 2. Validation Environment

The run was performed on the actual local Windows environment rather than only in an isolated development sandbox.

Documented environment:

```text
Operating System: Windows 11 Home Single Language
Python:            3.13.2
Ollama:            0.35.0
Model:             qwen3:1.7b
Interface:         Streamlit
Local URL:         http://localhost:8501
Inference:         CPU
```

The application was connected to the local Ollama instance and had access to public web research for the validation run.

## 3. Stedi Research Result

The completed run collected:

| Metric | Observed result |
|---|---:|
| Sources | 45 |
| Products | 6 |
| Technology signals | 11 |
| Competitors | 9 |
| Funding events | 3 |
| Growth signals | 8 |
| Named people | 2 |

The generated report was:

```text
stedi-report.md
```

The report contains source IDs and explicitly distinguishes AI inference from verified facts.

## 4. What Was Demonstrated

### Company Overview

The Overview workspace rendered structured company information with source-linked information.

### Products

The Products tab displayed structured product/service records including fields such as:

- Healthcare use case
- Features
- Target users
- Integrations
- Pricing status

### Technology

The Technology tab demonstrated the distinction between:

- Directly observed or stated technology
- Inferred technology signals

This is an important part of the project's confidence model.

### Competitors

The Competitors tab displayed categorized competitor records with source IDs and overlap explanations.

### Funding

The Funding tab displayed individual funding events and reported funding information with source references.

### Growth Signals

The Signals tab displayed growth, partnership, and funding-related signals with source references.

### People

The People tab displayed public professional information captured by the research process.

The run contained two named people.

The research-status panel also recorded a failed leadership search instead of silently hiding the failure.

### GTM

The GTM tab separated verified evidence from AI-inferred fields such as:

- ICP
- Needs
- Pain points
- Functions
- Opportunities
- Outreach angles

### Sources

The Sources tab exposed the collected source records and allowed the user to inspect the underlying pages associated with source IDs.

### Grounded Chat

The application also provided the grounded chat interface using the researched evidence and local retrieval architecture.

## 5. Report Generation

The Stedi run produced a Markdown report from the persisted structured intelligence.

The report layer is deterministic.

In other words, the application does not perform another unrestricted LLM generation step to rewrite the entire research result.

Instead:

```text
Research
   ↓
Structured intelligence
   ↓
Validation
   ↓
Persistence
   ↓
Markdown formatting
```

This keeps report generation separate from model reasoning.

## 6. Validation of Evidence Handling

The run demonstrated that the application can preserve source references through the research and intelligence process.

The report uses source IDs such as:

```text
SRC-001
SRC-002
...
```

The application uses real collected source records rather than accepting arbitrary model-generated citations.

## 7. Validation of Confidence Handling

The real run demonstrated that the application can distinguish between different evidence states.

For example, technology information can be shown as directly supported or inferred rather than treating every detected technology as equally confirmed.

Similarly, GTM information is separated into verified evidence and AI-inferred reasoning.

## 8. Validation of Failure Handling

An important result from the Stedi run was that not every research module was required to succeed for the application to complete.

A failed leadership search was explicitly recorded.

This is preferable to silently converting a failed external lookup into an empty or apparently successful result.

The behavior demonstrates that external research failures can remain visible within the application's research state.

## 9. Relationship to Automated Tests

The Stedi run should not be confused with the automated test suite.

The project has:

```text
182 automated tests
```

Those tests primarily validate deterministic application behavior.

The Stedi run validates the interaction of multiple real components:

```text
Web research
    +
Official website
    +
Local embeddings
    +
Qdrant
    +
Ollama / Qwen3
    +
Grounding
    +
Streamlit UI
    +
Report generation
```

The two forms of validation complement each other.

## 10. Earlier Validation Limitations

Before the successful local Stedi run, some components had not been tested against real external services in the development environment.

Those earlier limitations were superseded for the Stedi scenario by the successful local validation performed on 1 October 2026.

However, this does not mean that every possible external website, search query, model configuration, or hardware configuration has been validated.

## 11. Interpretation of the Results

The Stedi run demonstrates that the current local MVP can perform a complete company-research workflow against a real company using public web evidence and a local language model.

It does not establish:

- Universal factual accuracy
- Complete coverage of Stedi
- Accuracy of every source
- Production-scale throughput
- Generalized performance across all companies
- Perfect hallucination prevention

The observed counts should therefore be understood as the output of one completed research run.

## 12. Reproducibility

To reproduce the application locally:

```cmd
git clone https://github.com/Aswin7M/know-your-company-platform-Aswin-M.git
cd know-your-company-platform-Aswin-M
setup_windows.bat
ollama pull qwen3:1.7b
run.bat
```

The exact research output can vary because public websites and search results change over time.

The project should therefore be evaluated based on its architecture and evidence-handling process as well as individual research counts.

## 13. Validation Status

At the documented point:

```text
Core implementation:       Complete
Automated tests:            182 passed
Local Streamlit runtime:   Passed
Ollama + Qwen3:             Passed
Sample company:             Passed
Real Stedi validation:      Completed
GitHub publication:         Completed
```

This document records the observed state of the project at the time of validation.
