# Know Your Company — Grounding and Evidence Validation

## 1. Purpose

Grounding is the mechanism used by Know Your Company to reduce unsupported company claims.

The application does not assume that a language model is correct simply because it produced a plausible answer.

Instead, the system separates:

- Evidence collection
- Evidence retrieval
- Model reasoning
- Deterministic validation
- Persistence

The central rule is:

> A model-generated claim is a proposal until the application checks it against the available evidence.

## 2. Evidence as the Source of Truth

The system begins with collected public evidence.

Each source receives a stable identifier such as:

```text
SRC-001
SRC-002
SRC-003
```

Evidence records preserve source metadata and extracted text.

The LLM receives selected evidence from this corpus during structured analysis.

This prevents the model from being the only place where a fact exists.

## 3. Why Source IDs Are Important

A language model can generate a source identifier that looks valid even when that identifier does not exist.

For this reason, the application does not blindly accept a model-generated source list.

Instead, real source IDs from the retrieved evidence are used by the application.

The result is:

```text
Real source
     ↓
Stable source ID
     ↓
Retrieved evidence
     ↓
LLM proposal
     ↓
Validation
     ↓
Stored claim
```

rather than:

```text
LLM
 ↓
Invented citation
 ↓
Accepted as fact
```

## 4. Verification States

The project uses three main states.

### Verified

A claim is considered Verified when the available evidence supports it according to the relevant validation rules.

This is an evidence-status label.

It does not mean that the source has been independently fact-checked outside the application.

### AI Inference

An AI Inference is a reasoning result derived from the available evidence.

It should not be presented as if the source directly stated the conclusion.

### Not verified

Not verified means that the available evidence did not sufficiently support the claim.

The application prefers this state over filling an evidence gap with unsupported generated content.

## 5. Module-specific Grounding Rules

Different types of company information require different checks.

### 5.1 Products

Product names must be present in cited evidence.

The purpose is to avoid accepting a product name that exists only in the model output.

Product attributes such as features, users, integrations, and pricing are also subject to the available evidence.

### 5.2 Funding

Funding claims require direct supporting evidence for relevant fields.

The validation model considers:

- Funding amount
- Funding date
- Round type
- Investors
- Funding totals

A total is not automatically created by adding individual funding rounds.

If a source does not explicitly support a reported total, the system does not treat a calculated total as equivalent to a sourced total.

### 5.3 People

A person's name and role need to be supported together.

A title appearing near a different person's name cannot automatically be used to validate the first person's role.

This is intended to reduce incorrect person-role associations.

### 5.4 Technology

Technology has an important distinction between observed and inferred information.

Technology can receive stronger verification when it is:

- Directly observed in page markup or headers, or
- Explicitly stated by the company's own publications

Other technology signals are treated as inferred rather than automatically verified.

This distinction is useful because technology fingerprinting can produce plausible signals that are not necessarily confirmed by the company.

### 5.5 Competitors

A company is not treated as a verified competitor simply because the model considers it similar.

The source should explicitly support a competitor, alternative, or comparable relationship near the competitor name.

### 5.6 Growth Signals

Growth-related claims require meaningful overlap with supporting evidence.

The system does not treat a generic company description as sufficient evidence for an unrelated growth claim.

### 5.7 GTM

GTM confidence is based on the quantity and quality of verified supporting evidence rather than blindly accepting the language model's own confidence value.

This keeps model confidence separate from evidence confidence.

## 6. Evidence Retrieval Boundary

The model does not receive the entire research corpus for every task.

The retriever selects relevant chunks and constructs a bounded evidence block.

The documented default configuration includes:

```text
TOP_K=5
MAX_CONTEXT_CHARS=6000
```

The purpose is to provide task-specific evidence while keeping the local model's context manageable.

## 7. What Grounding Does Not Guarantee

Grounding reduces unsupported claims, but it does not provide independent truth verification.

The project's documented grounding approach is primarily lexical/evidence based.

For example, a claim may contain words that occur in the supporting source while still representing a questionable interpretation.

Therefore:

```text
Verified
```

should be understood as:

```text
Supported by the application's available source evidence
```

not:

```text
Independently proven to be true
```

## 8. Grounding and Hallucination

The system is designed to reduce several common failure modes.

### Unsupported facts

A model can produce a plausible fact without evidence.

The validation stage can reject or downgrade it.

### Fabricated citations

A model can invent source IDs.

The application uses real source IDs from retrieved evidence instead of trusting arbitrary model-generated references.

### Incorrect person-role association

The validation stage requires the person and role to be supported together.

### Unsupported funding numbers

Amounts and other funding details require supporting source text.

### Inferred technology presented as confirmed

Technology evidence is separated into directly observed/stated information and inferred signals.

## 9. Grounding Flow

```text
                 Research Evidence
                        |
                        v
                Stable Source IDs
                        |
                        v
                  Chunking / RAG
                        |
                        v
                Relevant Retrieval
                        |
                        v
                 Bounded Context
                        |
                        v
                 Qwen3 / Ollama
                        |
                        v
              Structured Proposal
                        |
                        v
          +-----------------------------+
          | Deterministic Validation    |
          |                             |
          | Source IDs                  |
          | Names                       |
          | Roles                       |
          | Amounts                     |
          | Dates                       |
          | Relationships               |
          | Evidence overlap            |
          +-------------+---------------+
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Verified   Inference   Not Verified
             |          |          |
             +----------+----------+
                        |
                        v
                    Persistence
```

## 10. Why Deterministic Validation Is Separate From the LLM

The validation layer is intentionally implemented in Python rather than delegated entirely to another model.

This makes several rules explicit and repeatable.

For example:

```text
if a funding amount is not supported:
    do not treat the amount as verified
```

The application can therefore apply the same rule consistently across research runs.

## 11. Relationship Between RAG and Grounding

RAG and grounding solve different problems.

### RAG

RAG answers:

> Which parts of the research corpus should the model see?

### Grounding

Grounding answers:

> Does the model's proposed output have sufficient support in the evidence it was given?

The combined architecture is:

```text
Research Corpus
      ↓
RAG Retrieval
      ↓
Relevant Evidence
      ↓
LLM Analysis
      ↓
Grounding Validation
      ↓
Structured Intelligence
```

RAG without grounding can still produce unsupported interpretations.

Grounding without retrieval would be less practical because the model would not have a focused evidence context.

## 12. Practical Interpretation

The grounding system should be described accurately in technical and portfolio material.

Appropriate description:

> "The application uses evidence retrieval and deterministic validation rules to reduce unsupported LLM-generated company claims."

Avoid describing it as:

> "The system completely eliminates hallucinations."

The implementation provides a grounding mechanism and validation layer; it does not mathematically guarantee that every accepted interpretation is correct.

