from karllo.model.tokenizer import KarlloTokenizer, train_bpe


def test_roundtrip_bytes():
    tok = KarlloTokenizer(vocab_size=512)
    text = "Karllo AI trains a Transformer."
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_special_tokens_and_chat():
    tok = KarlloTokenizer(vocab_size=512)
    ids = tok.encode_chat(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert ids[0] == tok.bos_id
    assert tok.user_id in ids
    assert tok.assistant_id in ids


def test_bpe_learns_merges():
    texts = ["hello hello hello world world"] * 20
    tok = train_bpe(texts, vocab_size=300)
    assert len(tok.merges) > 0
    ids = tok.encode("hello world")
    assert all(i < tok.vocab_size for i in ids)
    assert "hello" in tok.decode(ids)
