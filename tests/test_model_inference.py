import torch

from karllo.model.config import KarlloConfig
from karllo.model.tokenizer import KarlloTokenizer
from karllo.model.transformer import KarlloTransformer


def test_forward_and_loss_shape():
    cfg = KarlloConfig.tiny()
    model = KarlloTransformer(cfg)
    tok = KarlloTokenizer(vocab_size=cfg.vocab_size)
    ids = torch.tensor([tok.encode("abcde", add_special=True)], dtype=torch.long)
    # pad/truncate to at least 4
    if ids.size(1) < 4:
        pad = torch.zeros((1, 4 - ids.size(1)), dtype=torch.long)
        ids = torch.cat([ids, pad], dim=1)
    targets = ids.clone()
    logits, loss = model(ids, targets=targets)
    assert logits.shape == (1, ids.size(1), cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_generate_yields_tokens():
    cfg = KarlloConfig.tiny()
    model = KarlloTransformer(cfg)
    tok = KarlloTokenizer(vocab_size=cfg.vocab_size)
    idx = torch.tensor([[tok.bos_id]], dtype=torch.long)
    tokens = list(model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=10, eos_id=None))
    assert len(tokens) == 5
    assert all(isinstance(t, int) for t in tokens)


def test_random_model_is_untrained_fingerprint_stable_without_step():
    cfg = KarlloConfig.tiny()
    model = KarlloTransformer(cfg)
    a = model.weight_fingerprint()
    b = model.weight_fingerprint()
    assert a == b
