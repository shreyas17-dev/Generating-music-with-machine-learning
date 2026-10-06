"""Manual check of the audio tools (not collected by pytest).

    python tests/audio_roundtrip.py path/to/piano.wav --soundfont path/to/soundfont.sf2

audio -> MIDI -> audio. Prints how many notes were found and where the files were written.
"""
import argparse
import os

from musicgen.audio_io import audio_to_midi, midi_to_audio
from musicgen.midi_io import read_midi_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--out", default="outputs/audio_test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mid, wav = os.path.join(a.out, "transcribed.mid"), os.path.join(a.out, "resynth.wav")
    audio_to_midi(a.audio, mid)
    n = sum(1 for e in read_midi_events(mid) if e[1] == "on")
    print(f"transcribed {n} notes -> {mid}")
    midi_to_audio(mid, wav, soundfont=a.soundfont)
    print("rendered ->", wav)


if __name__ == "__main__":
    main()
