"""Make any piece of text sound like GlaDOS."""
import tempfile
import typing
from pathlib import Path

import click
import torch
import torchaudio

from glados.utils.auto_tune import Pitch, Scale, autotune
from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import load_voices

_DEFAULT_VOICE = "glados"
_ROOT_DIR = Path(__file__).parents[1]


def _get_valid_pitches() -> list[str]:
    """Return all valid pitches to tune to."""
    valid: list[str] = []
    for p in Pitch:
        valid.extend((f"{p.name}:min", f"{p.name}:maj"))
    return valid


def _save(
    gen: torch.Tensor | list[torch.Tensor],
    dbg_state: typing.Any | None,
) -> list[Path]:
    """Save generated tensors."""
    if not isinstance(gen, list):
        gen = [gen]
    output_files: list[Path] = []
    for i, array in enumerate(gen):
        p = Path(tempfile.gettempdir()) / f"{_DEFAULT_VOICE}_{i}.wav"
        torchaudio.save(  # pylint: disable=no-member
            str(p),
            array.squeeze(0).cpu(),
            24_000,
        )
        output_files.append(p)
    if dbg_state:
        ds_path = _ROOT_DIR / "debug_states"
        ds_path.mkdir(exist_ok=True, parents=True)
        torch.save(dbg_state, str(ds_path / f"do_tts_debug_{_DEFAULT_VOICE}.pth"))
    return output_files


@click.command()
@click.option("-t", "--text", help="Text to speak.", type=str, required=True)
@click.option(
    "-p",
    "--preset",
    help="Inference quality preset.",
    type=click.Choice(["ultra_fast", "fast", "standard", "high_quality"]),
    default="fast",
    show_default=True,
)
@click.option(
    "-op",
    "--output_path",
    help="Where to store outputs.",
    default=_ROOT_DIR / "results",
    type=Path,
    show_default=True,
)
@click.option(
    "-md",
    "--model-dir",
    help="Where to find pretrained model checkpoints. Tortoise automatically \
    downloads these to .models, so this should only be specified if you have \
    custom checkpoints.",
    default=MODELS_DIR,
)
@click.option(
    "-c",
    "--candidates",
    help="How many output candidates to produce per-voice.",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="Random seed which can be used to reproduce results.",
    default=None,
)
@click.option(
    "-d",
    "--produce-debug-state",
    is_flag=True,
    default=False,
    help="Whether to produce debug_state.pth, which can aid in reproducing problems.",
    show_default=True,
)
@click.option(
    "-cvvp",
    "--cvvp-amount",
    type=float,
    default=0.0,
    help="How much the CVVP model should influence the output. Increasing \
    this can in some cases reduce the likelihood of multiple speakers.",
    show_default=True,
)
@click.option(
    "-mk",
    "--music-key",
    type=click.Choice(_get_valid_pitches()),
    help="Pitch:Key to scale to.",
    default=None,
    show_default=True,
)
def do_tts(  # pylint: disable=too-many-locals
    text: str,
    preset: str = "fast",
    output_path: Path = _ROOT_DIR / "results",
    model_dir: str = MODELS_DIR,
    candidates: int = 3,
    seed: int | None = None,
    produce_debug_state: bool = False,
    cvvp_amount: float = 0,
    music_key: str | None = None,
):
    """GlaDOS-ify any text."""
    output_path.mkdir(parents=True, exist_ok=True)
    tts = TextToSpeech(models_dir=model_dir)
    voice_samples, conditioning_latents = load_voices((_DEFAULT_VOICE,))
    gen, dbg_state = tts.tts_with_preset(
        text=text,
        k=candidates,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset,
        use_deterministic_seed=seed,
        return_deterministic_state=True,
        cvvp_amount=cvvp_amount,
    )
    output_files = _save(
        gen,
        dbg_state if produce_debug_state else None,
    )
    for of in output_files:
        autotune(of, None if not music_key else Scale.from_str(music_key))


if __name__ == "__main__":
    do_tts()  # pylint: disable=no-value-for-parameter
