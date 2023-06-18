"""autotune"""
import enum
import io
from pathlib import Path
from typing import NamedTuple

import librosa
import librosa.display
import numpy as np
import psola
import scipy.signal as sig
import soundfile as sf

_SEMITONES_IN_OCTAVE = 12


class Pitch(enum.IntEnum):
    """Key to scale to."""

    A = enum.auto()
    B = enum.auto()
    C = enum.auto()
    D = enum.auto()
    E = enum.auto()
    F = enum.auto()
    G = enum.auto()


class Key(enum.IntEnum):
    """Minor / Major tone"""

    MIN = enum.auto()
    MAJ = enum.auto()


class Scale(NamedTuple):
    """Scale to autotune to"""

    pitch: Pitch
    key: Key

    @classmethod
    def from_str(cls, s: str) -> "Scale":
        """Convert from librosa scale to enum"""
        pitch, key = s.split(":")
        return cls(getattr(Pitch, pitch.upper()), getattr(Key, key.upper()))

    @property
    def to_str(self) -> str:
        """Conver to Librosa scale format"""
        return f"{self.pitch.name}:{self.key.name.lower()}"


def _autotune(
    audio: np.ndarray, sample_rate: float, scale: Scale | None
) -> sf.SoundFile:
    """Autotune to pitch with PYIN algorithm + PSOLA algorithm."""
    # Set some basis parameters.
    frame_length = 2048
    hop_length = frame_length // 4
    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    wav_snippet, _, __ = librosa.pyin(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
        sr=sample_rate,
        fmin=fmin,  # type: ignore
        fmax=fmax,  # type: ignore
    )
    corrected_wav_snippet: np.ndarray
    if scale:
        corrected_wav_snippet = _tune_to_scale(wav_snippet, scale.to_str)
    else:
        corrected_wav_snippet = _tune_to_closest(wav_snippet)

    # Pitch-shifting using the PSOLA algorithm.
    return psola.vocode(
        audio,
        sample_rate=int(sample_rate),
        target_pitch=corrected_wav_snippet,
        fmin=fmin,  # type: ignore
        fmax=fmax,  # type: ignore
    )


def _degrees_from(scale: str):
    """Return the pitch classes (degrees) that correspond to the given scale"""
    degrees = librosa.key_to_degrees(scale)
    degrees = np.concatenate(
        (
            degrees,
            [degrees[0] + _SEMITONES_IN_OCTAVE],
        )
    )
    return degrees


def _tune_to_closest(wav_snippet: np.ndarray):
    """Round the given pitch values to the nearest MIDI note numbers"""
    midi_note = np.around(librosa.hz_to_midi(wav_snippet))
    nan_indices = np.isnan(wav_snippet)
    midi_note[nan_indices] = np.nan
    return librosa.midi_to_hz(midi_note)


def _tune_to_scale(wav_snippet: np.ndarray, scale: str) -> np.ndarray:
    """Map each pitch in the wav_snippet array to the closest
    pitch belonging to the given scale."""
    sanitized_pitch = np.zeros_like(wav_snippet)
    for i in np.arange(wav_snippet.shape[0]):
        sanitized_pitch[i] = _tune_to_scale_helper(wav_snippet[i], scale)
    smoothed_sanitized_pitch = sig.medfilt(sanitized_pitch, kernel_size=11)
    smoothed_sanitized_pitch[np.isnan(smoothed_sanitized_pitch)] = sanitized_pitch[
        np.isnan(smoothed_sanitized_pitch)
    ]
    return smoothed_sanitized_pitch


def _tune_to_scale_helper(wav_snippet: np.ndarray, scale: str):
    """Return the pitch closest to wav_snippet that belongs to the given scale"""
    if np.isnan(wav_snippet):
        return np.nan
    degrees = _degrees_from(scale)
    midi_note = librosa.hz_to_midi(wav_snippet)
    degree = midi_note % _SEMITONES_IN_OCTAVE
    degree_id = np.argmin(np.abs(degrees - degree))
    degree_difference = degree - degrees[degree_id]
    midi_note -= degree_difference
    return librosa.midi_to_hz(midi_note)


def autotune(
    filepath: Path | io.BytesIO, scale: Scale | None = None
) -> tuple[sf.SoundFile, float]:
    """
    Autotune some audio recording

    :param filepath: Path to some file containing sound to autotune.
    :param scale: If provided, autotune to this exact pitch.
    """
    source: str | io.BytesIO = (
        filepath if isinstance(filepath, io.BytesIO) else str(filepath)
    )
    y, sr = librosa.load(source, sr=None, mono=False)
    if y.ndim > 1:
        print("Converting stereo sound to mono.")
        y = y[0, :]
    return _autotune(y, sr, scale), sr

    # filepath = filepath.parent / (
    #     filepath.stem + '_pitch_corrected' + filepath.suffix
    # )
    # sf.write(str(filepath), pitch_corrected_y, sr)
