# Audio setup (Windows)

Audio is only needed for `audio -> MIDI` (transcription) and `MIDI -> audio` (listening). Everything else works without it.

## audio -> MIDI : Basic Pitch
    pip install -r requirements-audio.txt
Use Python 3.10 or 3.11 (newer versions may not have wheels for basic-pitch's dependencies).
For mp3 input also install ffmpeg (https://ffmpeg.org, add its `bin` folder to PATH); wav input needs nothing extra.

## MIDI -> audio : FluidSynth + a soundfont
1. Download the Windows zip from https://github.com/FluidSynth/fluidsynth/releases , unzip to e.g. `C:\tools\fluidsynth`.
2. Add `C:\tools\fluidsynth\bin` to PATH, or set `$env:FLUIDSYNTH = "C:\tools\fluidsynth\bin\fluidsynth.exe"`.
3. Get a General-MIDI soundfont (`.sf2`), e.g. FluidR3_GM.sf2 (what the Colab notebook used) and keep it OUTSIDE git.
   Then pass `--soundfont path\to\FluidR3_GM.sf2` or set `$env:SOUNDFONT = "path\to\FluidR3_GM.sf2"`.

## Check
    python tests/audio_roundtrip.py some_piano.wav --soundfont path\to\FluidR3_GM.sf2
