from pathlib import Path

from karllo.training.data import analyze_dataset, iter_examples, parse_qa_book, split_indices


def test_parse_qa_book():
    text = "Question: A?\nAnswer: B\n\nQuestion: C?\nAnswer: D\n"
    ex = parse_qa_book(text)
    assert len(ex) == 2
    assert ex[0]["prompt"] == "A?"
    assert ex[0]["response"] == "B"


def test_jsonl_and_duplicates(tmp_path: Path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"text": "hello"}\n{"text": "hello"}\n{"prompt": "q", "response": "a"}\n\nnot-json\n',
        encoding="utf-8",
    )
    kinds = [e["kind"] for e in iter_examples(p)]
    assert "text" in kinds
    assert "sft" in kinds
    assert "malformed" in kinds
    analysis = analyze_dataset(p)
    assert analysis["counts"]["duplicates"] >= 1
    assert analysis["counts"]["malformed"] >= 1
    assert analysis["example_count"] >= 2


def test_txt_streaming_seed():
    seed = Path(__file__).resolve().parents[1] / "data" / "seed" / "pilot_corpus.txt"
    analysis = analyze_dataset(seed)
    assert analysis["example_count"] >= 1
    assert analysis["chars"] > 100


def test_split_indices():
    train, val = split_indices(20, 0.1)
    assert len(train) + len(val) == 20
    assert len(val) >= 1
