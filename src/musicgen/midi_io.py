"""Read a MIDI file into note events, and write token ids back to a MIDI file."""
from .tokens import tokens_to_events


def read_midi_events(path):
    """MIDI file -> list of (time_seconds, 'on'|'off', pitch, velocity). The sustain pedal is ignored."""
    import mido
    t, events = 0.0, []
    for msg in mido.MidiFile(path):          # iterating gives playback order, .time = delta in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append((t, "on", msg.note, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            events.append((t, "off", msg.note, 0))
    return events


def write_midi(tokens, path):
    """Token ids -> piano MIDI file (120 BPM, 960 ticks per second)."""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    last = 0
    for t, kind, p, v in sorted(tokens_to_events(tokens), key=lambda e: (e[0], 0 if e[1] == "off" else 1)):
        tick = int(round(t * 960))
        dt, last = tick - last, tick
        if kind == "on":
            track.append(mido.Message("note_on", note=p, velocity=v, time=dt))
        else:
            track.append(mido.Message("note_off", note=p, velocity=0, time=dt))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(path)
