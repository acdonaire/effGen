"""
Media preset — bundles audio, image, and media processing tools.

Exported symbol: MEDIA_PRESET (a PresetConfig instance).
The preset is automatically registered in the global PRESETS dict when this
module is imported.
"""

from __future__ import annotations

from .registry import PRESETS, PresetConfig

MEDIA_PRESET = PresetConfig(
    name="media",
    description=(
        "Media processing agent with AudioTranscribeTool for speech-to-text. "
        "Handles audio files (MP3, WAV, OGG, FLAC) using local faster-whisper "
        "or HuggingFace Inference API."
    ),
    tool_names=[
        "audio_transcribe",
    ],
    system_prompt=(
        "You are a media processing assistant. "
        "Use the audio_transcribe tool to convert speech audio files to text. "
        "When given an audio file path, transcribe it and return the full transcript. "
        "For long recordings, summarize key points after transcribing."
    ),
    max_iterations=8,
    temperature=0.3,
    tags=["media", "audio", "transcription", "speech-to-text"],
)

# Register the preset so create_agent("media") works
PRESETS["media"] = MEDIA_PRESET
