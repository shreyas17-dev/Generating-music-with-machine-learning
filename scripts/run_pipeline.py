"""audio in -> audio out.

    python scripts/run_pipeline.py --input song.mp3 --model B --prompt-seconds 10 --out outputs/
    python scripts/run_pipeline.py --midi-input piece.mid --model A --prompt-seconds 10     # skip transcription

Steps: audio_to_midi -> keep events before --prompt-seconds -> tokens -> model continues by --max-new tokens
       -> write_midi -> midi_to_audio.   Run it with --model A and --model B on the same input to compare.
"""
import argparse
import os

from musicgen.audio_io import audio_to_midi, midi_to_audio
from musicgen.midi_io import read_midi_events, write_midi
from musicgen.registry import load_by_name
from musicgen.sampling import generate
from musicgen.tokens import events_to_tokens

MAX_PROMPT_TOKENS = 1024        # the models were trained on windows of 1024 tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="audio file (wav/mp3/...), solo piano works best")
    src.add_argument("--midi-input", help="use this MIDI file instead of transcribing audio")
    ap.add_argument("--model", default="B", choices=["A", "B", "C"])
    ap.add_argument("--prompt-seconds", type=float, default=10.0)
    ap.add_argument("--max-new", type=int, default=800, help="number of tokens to generate (hard cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--greedy", action="store_true", help="always take the most likely token")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--soundfont", default=None, help="path to a .sf2 file (or set SOUNDFONT)")
    ap.add_argument("--no-audio", action="store_true", help="stop after writing the MIDI")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1. audio -> MIDI
    if a.midi_input:
        midi_in = a.midi_input
    else:
        midi_in = os.path.join(a.out, "input.mid")
        print("transcribing audio ...")
        audio_to_midi(a.input, midi_in)
        print("  ->", midi_in)

    # 2. keep the first --prompt-seconds, make tokens
    events = [e for e in read_midi_events(midi_in) if e[0] < a.prompt_seconds]
    prompt = events_to_tokens(events)
    if not prompt:
        raise SystemExit(f"No piano notes found in the first {a.prompt_seconds}s of {midi_in}. "
                         "Try a larger --prompt-seconds or a cleaner recording.")
    if len(prompt) > MAX_PROMPT_TOKENS:
        print(f"prompt has {len(prompt)} tokens; keeping the last {MAX_PROMPT_TOKENS}")
        prompt = prompt[-MAX_PROMPT_TOKENS:]
    print(f"prompt: {len(events)} events -> {len(prompt)} tokens")
    write_midi(prompt, os.path.join(a.out, "prompt.mid"))

    # 3. continue with the chosen model
    model = load_by_name(a.model, a.ckpt_dir)
    sampling = dict(mode="argmax") if a.greedy else dict(
        mode="sample", temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)
    tokens = generate(model, prompt, max_new=a.max_new, seed=a.seed, **sampling)

    # 4. MIDI + audio
    out_mid = os.path.join(a.out, f"output_{a.model}.mid")
    write_midi(tokens, out_mid)
    print("MIDI  ->", out_mid)
    if a.no_audio:
        return
    out_wav = os.path.join(a.out, f"output_{a.model}.wav")
    try:
        midi_to_audio(out_mid, out_wav, soundfont=a.soundfont)
        print("audio ->", out_wav)
    except (FileNotFoundError, RuntimeError) as e:
        print("could not render audio (the MIDI is still saved):", e)


if __name__ == "__main__":
    main()
