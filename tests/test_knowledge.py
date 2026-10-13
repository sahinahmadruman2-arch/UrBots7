from karllo.retrieval.knowledge import KnowledgeBase, embedding_status


def test_bm25_retrieval(tmp_path, monkeypatch):
    from karllo.retrieval import knowledge as kn

    monkeypatch.setattr(kn, "KNOWLEDGE_DIR", tmp_path)
    kb = KnowledgeBase()
    kb.root = tmp_path
    kb._docs_index = tmp_path / "index.json"
    kb.add_document("planets", "Jupiter is the largest planet. Mars is the red planet.")
    kb.add_document("training", "Gradient descent updates model weights during training.")
    res = kb.search("largest planet", k=2)
    assert res["hits"]
    assert "Jupiter" in res["hits"][0]["text"]
    assert "does not train" in res["note"]
    st = embedding_status()
    assert st["method"] in ("bm25", "sentence-transformers")
