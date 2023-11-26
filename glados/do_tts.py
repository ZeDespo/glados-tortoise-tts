"""Make any piece of text sound like GlaDOS."""
import io
import typing
from pathlib import Path

import click
import soundfile
import torch
import torchaudio
from psola.core import functools

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


@functools.cache
def _load_tts(
    model_dir: str,
) -> tuple[TextToSpeech, list[torch.Tensor], list[torch.Tensor]]:
    tts = TextToSpeech(models_dir=model_dir)
    voice_samples, conditioning_latents = load_voices((_DEFAULT_VOICE,))
    return tts, voice_samples, conditioning_latents  # type: ignore


def _save(
    autotuned: soundfile.SoundFile,
    sr: float,
    output_path: Path,
    filename: str,
    file_prefix: str,
) -> Path:
    """Save the soundfile to the file."""
    filename = filename.replace(_DEFAULT_VOICE, file_prefix)
    output_path = output_path / filename
    soundfile.write(output_path, autotuned, sr)
    return output_path


def _tensors_to_bytesio(
    gen: torch.Tensor | list[torch.Tensor],
    dbg_state: typing.Any | None,
) -> list[io.BytesIO]:
    """Save generated tensors."""
    if not isinstance(gen, list):
        gen = [gen]
    output_buffers: list[io.BytesIO] = []
    for i, array in enumerate(gen):
        buffer = io.BytesIO()
        buffer.name = f"{_DEFAULT_VOICE}_{i}.wav"
        torchaudio.save(  # pylint: disable=no-member
            buffer,
            array.squeeze(0).cpu(),
            24_000,
            format="wav",
        )
        buffer.seek(0, io.SEEK_SET)
        output_buffers.append(buffer)
    if dbg_state:
        ds_path = _ROOT_DIR / "debug_states"
        ds_path.mkdir(exist_ok=True, parents=True)
        torch.save(dbg_state, str(ds_path / f"do_tts_debug_{_DEFAULT_VOICE}.pth"))
    return output_buffers


def do_tts(  # pylint: disable=too-many-locals
    text: str,
    preset: str = "fast",
    output_path: Path = _ROOT_DIR / "results",
    output_base_name: str = _DEFAULT_VOICE,
    model_dir: str = MODELS_DIR,
    candidates: int = 3,
    seed: int | None = None,
    produce_debug_state: bool = False,
    cvvp_amount: float = 0,
    music_key: str | None = None,
) -> list[Path]:
    """Do the actual glados stuff"""
    output_path.mkdir(parents=True, exist_ok=True)
    tts, voice_samples, conditioning_latents = _load_tts(model_dir)
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
    output_buffers = _tensors_to_bytesio(
        gen,
        dbg_state if produce_debug_state else None,
    )
    output_paths: list[Path] = []
    for buffers in output_buffers:
        autotuned, sr = autotune(
            buffers, None if not music_key else Scale.from_str(music_key)
        )
        op = _save(autotuned, sr, output_path, buffers.name, output_base_name)
        output_paths.append(op)
    return output_paths


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
    "--output-path",
    help="Where to store outputs.",
    default=_ROOT_DIR / "results",
    type=Path,
    show_default=True,
)
@click.option(
    "-obn",
    "--output-base-name",
    help="The base prefix to save the file as.",
    default=_DEFAULT_VOICE,
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
def gladosify(*args, **kwargs) -> None:
    """Say any text with Glados' AI voice model."""
    do_tts(*args, **kwargs)


if __name__ == "__main__":
    gladosify()  # pylint: disable=no-value-for-parameter
