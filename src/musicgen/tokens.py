"""Vocabulary and conversion between note events and token ids (pure python, no torch needed).

Token vocabulary (309 ids, in the spirit of Oore et al. 2018):
  0 pad | 1..88 note_on (piano keys A0..C8) | 89..176 note_off | 177..276 time_shift (10 ms steps) |
  277..308 velocity (32 bins)
"""

LOW, HIGH = 21, 108                   # MIDI note numbers of the piano keys
N_PITCH = HIGH - LOW + 1              # 88
N_SHIFT = 100                         # time-shift bins
SHIFT_MS = 10                         # one bin = 10 ms, so one shift token covers up to 1 s
N_VEL = 32                            # velocity bins
PAD = 0
ON0 = 1                               # note_on tokens   1..88
OFF0 = ON0 + N_PITCH                  # note_off tokens  89..176
SHIFT0 = OFF0 + N_PITCH               # time_shift       177..276  (token SHIFT0+k = (k+1)*10 ms)
VEL0 = SHIFT0 + N_SHIFT               # velocity         277..308
VOCAB = VEL0 + N_VEL                  # 309


def events_to_tokens(events):
    """events: list of (time_seconds, 'on'|'off', pitch, velocity), in any order."""
    events = sorted(events, key=lambda e: (e[0], 0 if e[1] == "off" else 1))   # offs before ons
    toks, t_bins = [], 0
    for t, kind, pitch, vel in events:
        if not (LOW <= pitch <= HIGH):
            continue
        target = int(round(t * 1000 / SHIFT_MS))
        gap = target - t_bins
        while gap > 0:
            step = min(gap, N_SHIFT)
            toks.append(SHIFT0 + step - 1)
            gap -= step
        t_bins = max(t_bins, target)
        if kind == "on":
            toks.append(VEL0 + min(N_VEL - 1, vel * N_VEL // 128))
            toks.append(ON0 + pitch - LOW)
        else:
            toks.append(OFF0 + pitch - LOW)
    return toks


def tokens_to_events(tokens):
    """Inverse of events_to_tokens. Also repairs impossible sequences a model may produce
    (key pressed twice, release of a key that is not down, keys still down at the end)."""
    events, sounding = [], set()
    t_bins, vel = 0, 64
    for tok in tokens:
        tok = int(tok)
        sec = t_bins * SHIFT_MS / 1000
        if tok == PAD:
            continue
        if SHIFT0 <= tok < VEL0:
            t_bins += tok - SHIFT0 + 1
        elif VEL0 <= tok < VOCAB:
            b = tok - VEL0
            vel = max(1, min(127, b * 128 // N_VEL + 128 // N_VEL // 2))
        elif ON0 <= tok < OFF0:
            p = tok - ON0 + LOW
            if p in sounding:
                events.append((sec, "off", p, 0))
            sounding.add(p)
            events.append((sec, "on", p, vel))
        elif OFF0 <= tok < SHIFT0:
            p = tok - OFF0 + LOW
            if p in sounding:
                sounding.discard(p)
                events.append((sec, "off", p, 0))
    end = t_bins * SHIFT_MS / 1000 + 0.05
    for p in sorted(sounding):
        events.append((end, "off", p, 0))
    return events
