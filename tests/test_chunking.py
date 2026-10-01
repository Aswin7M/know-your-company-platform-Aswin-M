import pytest

from models.source import Evidence
from rag.chunker import CHARS_PER_TOKEN, chunk_all, chunk_evidence, estimate_tokens, split_text


def _evidence(text, sid="SRC-001"):
    return Evidence(source_id=sid, title="Acme Products", url="https://acme.example/p",
                    source_type="official_website", content=text)


def test_chunks_are_created_and_bounded():
    text = "\n".join(f"Sentence number {i} about healthcare claims processing and eligibility checks." for i in range(200))
    chunks = split_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 3
    assert all(len(c) <= 100 * CHARS_PER_TOKEN + 1 for c in chunks)


def test_metadata_is_preserved_on_every_chunk():
    chunks = chunk_evidence(_evidence("word " * 600), "Acme", "acme", chunk_size=100, overlap=10)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert (c.company, c.company_slug, c.source_id) == ("Acme", "acme", "SRC-001")
        assert c.url == "https://acme.example/p" and c.title == "Acme Products"
        assert c.source_type == "official_website" and c.chunk_index == i


def test_overlap_carries_text_forward():
    # every word is unique so we can tell exactly which words were carried into the next chunk
    lines = [" ".join(f"w{i}x{j}" for j in range(12)) for i in range(60)]
    chunks = split_text("\n".join(lines), chunk_size=120, overlap=30)
    assert len(chunks) > 2
    carried = chunks[1].split()[0]                      # first word of chunk 2 ...
    assert carried in chunks[0]                         # ... was already in chunk 1
    assert chunks[0].rfind(carried) > len(chunks[0]) - 30 * CHARS_PER_TOKEN - 20   # ... from its tail end


def test_no_overlap_when_zero():
    chunks = split_text("\n".join("line %03d %s" % (i, "x" * 60) for i in range(50)), chunk_size=60, overlap=0)
    joined = "\n".join(chunks)
    assert joined.count("line 000") == 1


def test_empty_and_whitespace_text():
    assert split_text("") == [] and split_text("   \n  \n") == []


def test_short_text_is_single_chunk():
    assert split_text("Just one short paragraph.") == ["Just one short paragraph."]


def test_pathological_token_is_split_not_looped():
    chunks = split_text("A" * 10_000, chunk_size=100, overlap=10)
    assert len(chunks) > 1 and all(len(c) <= 100 * CHARS_PER_TOKEN + 1 for c in chunks)


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)


def test_overlap_is_clamped():
    assert split_text("word " * 500, chunk_size=50, overlap=5000)       # must terminate


def test_chunk_all_and_token_estimate():
    chunks = chunk_all([_evidence("a b c " * 50, "SRC-001"), _evidence("d e f " * 50, "SRC-002")], "Acme", "acme", 50, 5)
    assert {c.source_id for c in chunks} == {"SRC-001", "SRC-002"}
    assert estimate_tokens("x" * 350) == 101
