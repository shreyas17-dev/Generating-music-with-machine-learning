"""audio -> MIDI (Basic Pitch) and MIDI -> audio (FluidSynth command line).

Both functions are the only places that touch these tools, so either can be swapped
(e.g. for ByteDance's piano_transcription_inference) without changing the rest of the project.
"""
import os
import shutil
import subprocess
from pathlib import Path


def audio_to_midi(audio_path, midi_path):
    """Transcribe a (solo piano) recording to a MIDI file. Accepts wav / mp3 / flac / ogg."""
    try:
        from basic_pitch.inference import predict
    except ModuleNotFoundError as e:
        if e.name == "basic_pitch":
            raise ImportError("basic-pitch is not installed. Run: pip install -r requirements-audio.txt "
                              "(or pass --midi-input to skip transcription)") from e
        raise        # basic-pitch is installed but one of ITS dependencies is missing: show the real error
    if not Path(audio_path).exists():
        raise FileNotFoundError(audio_path)
    _, midi_data, _ = predict(str(audio_path))        # midi_data is a pretty_midi.PrettyMIDI
    os.makedirs(os.path.dirname(os.path.abspath(midi_path)), exist_ok=True)
    midi_data.write(str(midi_path))


def _find_fluidsynth():
    exe = os.environ.get("FLUIDSYNTH") or shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError("fluidsynth not found. Install it and add it to PATH "
                                "(or set the FLUIDSYNTH environment variable to the full path of fluidsynth.exe).")
    return exe


def midi_to_audio(midi_path, audio_path, soundfont=None, sample_rate=44100):
    """Render a MIDI file to a .wav with FluidSynth. `soundfont` = path to a .sf2 file
    (or set the SOUNDFONT environment variable)."""
    soundfont = soundfont or os.environ.get("SOUNDFONT")
    if not soundfont or not Path(soundfont).exists():
        raise FileNotFoundError("soundfont (.sf2) not found. Pass --soundfont <file.sf2> or set SOUNDFONT.")
    exe = _find_fluidsynth()
    os.makedirs(os.path.dirname(os.path.abspath(audio_path)), exist_ok=True)
    # options must come BEFORE the soundfont and MIDI file (FluidSynth 2.5+ rejects them afterwards)
    cmd = [exe, "-ni", "-F", str(audio_path), "-r", str(sample_rate), str(soundfont), str(midi_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not Path(audio_path).exists():
        raise RuntimeError("fluidsynth failed:\n" + res.stdout + res.stderr)
