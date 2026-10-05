import random

from musicgen.tokens import (LOW, HIGH, VOCAB, PAD, events_to_tokens, tokens_to_events)


def make_events(n=60, seed=0):
    rng = random.Random(seed)
    events, t = [], 0.0
    for _ in range(n):
        t += rng.choice([0.0, 0.05, 0.12, 0.5, 1.7])        # includes a gap longer than 1 s
        p = rng.randint(LOW, HIGH)
        dur = rng.choice([0.2, 0.5, 2.3])
        v = rng.randint(20, 120)
        events.append((t, "on", p, v))
        events.append((t + dur, "off", p, 0))
    return events


def test_tokens_in_range():
    toks = events_to_tokens(make_events())
    assert toks and all(PAD < t < VOCAB for t in toks)


def test_round_trip_notes_and_times():
    events = make_events()
    back = tokens_to_events(events_to_tokens(events))
    ons_in = sorted((round(t, 2), p) for t, k, p, v in events if k == "on")
    ons_out = sorted((round(t, 2), p) for t, k, p, v in back if k == "on")
    assert len(ons_in) == len(ons_out)
    for (t1, p1), (t2, p2) in zip(ons_in, ons_out):
        assert abs(t1 - t2) <= 0.011       # 10 ms time grid
    # first pitch comparison on sorted lists of pitches only
    assert sorted(p for _, p in ons_in) == sorted(p for _, p in ons_out)


def test_out_of_range_pitch_is_dropped():
    assert events_to_tokens([(0.0, "on", 10, 80), (1.0, "off", 10, 0)]) == []


def test_repairs_bad_sequences():
    # note-off for a key that is not down is ignored; a key still down is released at the end
    toks = events_to_tokens([(0.0, "on", 60, 80)])
    ev = tokens_to_events(toks)
    assert [e[1] for e in ev] == ["on", "off"]
