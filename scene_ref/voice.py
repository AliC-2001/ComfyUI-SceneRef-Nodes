"""A character's voice: one short recording, cleaned up for MiniMax H3.

H3 ref2va takes standalone reference audio (MiniMaxH3ReferenceToVideo's ref_audio_N, at
most 3) and the prompt points at it as <Audio N>. Its audio VAE works at 32 kHz; ComfyUI
resamples, but a clip that is mostly silence, clipped, or two minutes long makes a poor
reference. So a voice is stored and used as: mono, 32 kHz, silence trimmed from both
ends, loudness evened out (about -20 dBFS RMS, peaks below -1 dBFS), 3 to 30 seconds.
"""
from __future__ import annotations

import io
import os
import tempfile
import wave

import torch

VOICE_SR = 32000
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm", ".mp4")


def _resample(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == VOICE_SR:
        return wav
    import comfy.audio
    return comfy.audio.resample(wav, sr, VOICE_SR)


def prepare(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, list[str]]:
    """A [L] mono 32 kHz clip from [C, L] or [B, C, L] audio, plus notes for the user."""
    notes = []
    w = waveform.float()
    while w.ndim > 2:
        w = w[0]
    if w.ndim == 2:
        w = w.mean(0)
    w = _resample(w[None], int(sample_rate))[0]
    # Silence at both ends: 20 ms frames more than 40 dB below the loudest frame.
    frame = VOICE_SR // 50
    n = w.shape[0] // frame
    if n:
        rms = w[:n * frame].reshape(n, frame).pow(2).mean(1).sqrt()
        loud = (rms > rms.max() * 0.01).nonzero().flatten()
        if loud.numel():
            a, b = max(0, int(loud[0]) - 5), min(n, int(loud[-1]) + 6)  # keep 100 ms around speech
            w = w[a * frame:b * frame]
    seconds = w.shape[0] / VOICE_SR
    if seconds > MAX_SECONDS:
        w = w[:int(MAX_SECONDS * VOICE_SR)]
        notes.append(f"voice cut to the first {MAX_SECONDS:.0f} s (of {seconds:.0f} s)")
    elif seconds < MIN_SECONDS:
        notes.append(f"voice is only {seconds:.1f} s; 5 to 15 s of clear speech works best")
    peak = float(w.abs().max()) if w.numel() else 0.0
    if peak <= 1e-4:
        raise ValueError("SceneRef: the voice recording is silent.")
    if peak >= 0.999:
        notes.append("voice recording is clipped (distorted); a cleaner recording works better")
    rms = float(w.pow(2).mean().sqrt())
    gain = min(10 ** (-20 / 20) / max(rms, 1e-6), 10 ** (-1 / 20) / peak)
    return (w * gain).clamp(-1, 1), notes


def to_audio(clip: torch.Tensor) -> dict:
    """ComfyUI's AUDIO: {waveform [1, 1, L], sample_rate}."""
    return {"waveform": clip.reshape(1, 1, -1).float(), "sample_rate": VOICE_SR}


def wav_bytes(clip: torch.Tensor) -> bytes:
    pcm = (clip.clamp(-1, 1) * 32767).round().short().numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(VOICE_SR)
        f.writeframes(pcm)
    return buf.getvalue()


def load_file(path: str) -> tuple[torch.Tensor, int]:
    """Any audio (or video with sound) file ComfyUI can decode: ([C, L] float, sample rate)."""
    from comfy_extras.nodes_audio import load
    return load(path)


def load_bytes(data: bytes, ext: str = ".wav") -> tuple[torch.Tensor, int]:
    """Audio held in memory (a .char member). PyAV wants a file, so it goes through a temp one."""
    if ext == ".wav":
        try:
            with wave.open(io.BytesIO(data)) as f:
                if f.getsampwidth() == 2:
                    raw = torch.frombuffer(bytearray(f.readframes(f.getnframes())), dtype=torch.int16)
                    ch = f.getnchannels()
                    return raw.float().div(32768).reshape(-1, ch).t(), f.getframerate()
        except wave.Error:
            pass
    fd, tmp = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return load_file(tmp)
    finally:
        os.remove(tmp)


def seconds(clip: torch.Tensor) -> float:
    return clip.shape[0] / VOICE_SR if clip is not None else 0.0
