import argparse
import os
from pathlib import Path

import torch
from api import TextToSpeech

from tortoise.utils.audio import get_voices, load_audio

"""
Dumps the conditioning latents for the specified voice to disk. These are expressive latents which can be used for
other ML models, or can be augmented manually and fed back into Tortoise to affect vocal qualities.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        type=str,
        help="Selects the voice to convert to conditioning latents",
        default="pat2",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where to store outputs.",
        default=str(Path(__file__).parents[1] / "results" / "conditioning_latents"),
    )
    args = parser.parse_args()
    output = Path(args.output_path)
    output.mkdir(exist_ok=True, parents=True)

    print("Loading text to speech model...")
    tts = TextToSpeech()
    voices = get_voices()
    selected_voices = args.voice.split(",")
    for voice in selected_voices:
        print(f"Generating `pth` file for {voice!r}.")
        cond_paths = voices[voice]
        conds = []
        for cond_path in cond_paths:
            c = load_audio(cond_path, 22050)
            conds.append(c)
        conditioning_latents = tts.get_conditioning_latents(conds)
        file_path = output / f"{voice}.pth"
        print(f"Saving file to {file_path!r}.")
        torch.save(conditioning_latents, str(output / f"{voice}.pth"))
