import numpy as np

from musicgen.sampling import sample_next
from musicgen.tokens import PAD, VOCAB


def test_greedy_picks_largest_and_never_pad():
    lg = np.zeros(VOCAB)
    lg[PAD] = 100.0
    lg[42] = 5.0
    assert sample_next(lg, mode="argmax") == 42


def test_top_k_one_is_deterministic():
    lg = np.random.default_rng(0).normal(size=VOCAB)
    lg[PAD] = 0
    best = int(np.argmax(np.where(np.arange(VOCAB) == PAD, -np.inf, lg)))
    for s in range(5):
        assert sample_next(lg, top_k=1, rng=np.random.default_rng(s)) == best


def test_sampling_stays_in_top_k():
    lg = np.arange(VOCAB, dtype=float)       # larger id = larger logit
    rng = np.random.default_rng(1)
    picks = {sample_next(lg, top_k=5, top_p=1.0, rng=rng) for _ in range(200)}
    assert picks <= set(range(VOCAB - 5, VOCAB))
