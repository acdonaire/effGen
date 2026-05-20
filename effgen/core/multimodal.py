"""
Multimodal input helpers for effGen.

image_from / audio_from / video_from each accept the widest possible set of
input types (bytes, path, URL, PIL.Image, np.ndarray) and return the
corresponding ContentPart.
"""

from __future__ import annotations

import io
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image as PILImage

from effgen.core.messages import AudioPart, ImagePart, VideoPart
from effgen.errors import InvalidMultimodalContent

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    ImageSource: TypeAlias = bytes | str | Path | PILImage.Image | np.ndarray  # noqa: UP040
else:
    ImageSource: TypeAlias = object  # noqa: UP040
AudioSource: TypeAlias = bytes | str | Path  # noqa: UP040
VideoSource: TypeAlias = bytes | str | Path  # noqa: UP040


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_image_mime(data: bytes) -> str:
    """Detect image MIME from magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # safe fallback for PIL-produced data


_MIME_ALIASES: dict[str, str] = {
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/m4a",
    "audio/x-flac": "audio/flac",
    "audio/x-mpeg": "audio/mpeg",
}


def _normalise_mime(mime: str) -> str:
    """Normalise non-canonical MIME aliases to the canonical form."""
    return _MIME_ALIASES.get(mime, mime)


def _detect_audio_mime(data: bytes) -> str:
    """Detect audio MIME from magic bytes / extension hint."""
    if data[:3] == b"ID3" or data[:2] == b"\xff\xfb" or data[:2] == b"\xff\xf3":
        return "audio/mpeg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/wav"


def _fetch_url(url: str) -> bytes:
    """Fetch raw bytes from a URL."""
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — URL comes from user
        return resp.read()


def _path_to_bytes(path: str | Path) -> bytes:
    p = Path(path)
    if not p.exists():
        raise InvalidMultimodalContent("source", f"File not found: {p}")
    return p.read_bytes()


# ---------------------------------------------------------------------------
# image_from
# ---------------------------------------------------------------------------

def image_from(source: ImageSource, mime: str | None = None) -> ImagePart:
    """Create an ImagePart from bytes, path, URL, PIL.Image, or np.ndarray.

    Args:
        source: The image data in any supported form.
        mime: Override the detected MIME type (e.g. "image/webp").

    Returns:
        ImagePart ready to embed in a Message.
    """
    data: bytes

    if isinstance(source, bytes | bytearray):
        data = bytes(source)

    elif isinstance(source, str | Path):
        src = str(source)
        if src.startswith("http://") or src.startswith("https://"):
            data = _fetch_url(src)
            if mime is None:
                # Try to infer from URL extension
                ext = os.path.splitext(src.split("?")[0])[-1].lower()
                mime_guess, _ = mimetypes.guess_type(f"x{ext}")
                if mime_guess and mime_guess.startswith("image/"):
                    mime = mime_guess
        else:
            data = _path_to_bytes(source)
            if mime is None:
                mime_guess, _ = mimetypes.guess_type(str(source))
                if mime_guess and mime_guess.startswith("image/"):
                    mime = mime_guess

    else:
        # Try PIL.Image
        try:
            from PIL import Image as _PIL  # type: ignore[import]
            if isinstance(source, _PIL.Image):
                buf = io.BytesIO()
                fmt = source.format or "PNG"
                source.save(buf, format=fmt)
                data = buf.getvalue()
                if mime is None:
                    mime = f"image/{fmt.lower()}"
            else:
                raise TypeError
        except (ImportError, TypeError):
            pass
        else:
            detected = mime or _detect_image_mime(data)
            return ImagePart(image=data, mime=detected)

        # Try np.ndarray
        try:
            import numpy as np  # type: ignore[import]
            if isinstance(source, np.ndarray):
                try:
                    from PIL import Image as _PIL  # type: ignore[import]
                    img = _PIL.fromarray(source)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    data = buf.getvalue()
                    if mime is None:
                        mime = "image/png"
                except ImportError:
                    raise InvalidMultimodalContent(
                        "image",
                        "Pillow is required to convert np.ndarray to image bytes. "
                        "Install with: pip install Pillow"
                    )
            else:
                raise TypeError
        except (ImportError, TypeError):
            raise InvalidMultimodalContent(
                "image",
                f"Cannot convert source of type {type(source).__name__} to ImagePart. "
                "Accepted types: bytes, str (path or URL), Path, PIL.Image, np.ndarray."
            )
        else:
            detected = mime or _detect_image_mime(data)
            return ImagePart(image=data, mime=detected)

    detected = mime or _detect_image_mime(data)
    return ImagePart(image=data, mime=detected)


# ---------------------------------------------------------------------------
# audio_from
# ---------------------------------------------------------------------------

def audio_from(source: AudioSource, mime: str | None = None) -> AudioPart:
    """Create an AudioPart from bytes, path, or URL.

    Args:
        source: The audio data in any supported form.
        mime: Override the detected MIME type (e.g. "audio/wav").

    Returns:
        AudioPart ready to embed in a Message.
    """
    data: bytes

    if isinstance(source, bytes | bytearray):
        data = bytes(source)

    elif isinstance(source, str | Path):
        src = str(source)
        if src.startswith("http://") or src.startswith("https://"):
            data = _fetch_url(src)
            if mime is None:
                ext = os.path.splitext(src.split("?")[0])[-1].lower()
                mime_guess, _ = mimetypes.guess_type(f"x{ext}")
                if mime_guess and mime_guess.startswith("audio/"):
                    mime = _normalise_mime(mime_guess)
        else:
            if mime is None:
                mime_guess, _ = mimetypes.guess_type(str(source))
                if mime_guess and mime_guess.startswith("audio/"):
                    mime = _normalise_mime(mime_guess)
            data = _path_to_bytes(source)

    else:
        raise InvalidMultimodalContent(
            "audio",
            f"Cannot convert source of type {type(source).__name__} to AudioPart. "
            "Accepted types: bytes, str (path or URL), Path."
        )

    detected = mime or _detect_audio_mime(data)
    return AudioPart(audio=data, mime=detected)


# ---------------------------------------------------------------------------
# video_from
# ---------------------------------------------------------------------------

def video_from(
    source: VideoSource,
    fps: float = 1.0,
    mime: str = "image/jpeg",
    max_frames: int = 16,
) -> VideoPart:
    """Create a VideoPart by sampling keyframes from a video file.

    Requires ffmpeg to be installed on the system PATH.

    Args:
        source: Path, URL, or raw bytes of the video.
        fps: Frames per second to sample (default 1).
        mime: MIME type for each extracted frame (default image/jpeg).
        max_frames: Hard cap on extracted frames (default 16).

    Returns:
        VideoPart containing sampled frame bytes.
    """
    if fps <= 0:
        raise InvalidMultimodalContent("video_frames", "fps must be positive")
    if max_frames < 1:
        raise InvalidMultimodalContent("video_frames", "max_frames must be at least 1")

    from effgen.multimodal.video_pre import VideoSource as _VideoSource

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        source = _fetch_url(source)

    vs = _VideoSource(source)
    try:
        frames = vs.sample_frames(fps=fps, max_frames=max_frames)
        return VideoPart(frames=[frame.image for frame in frames], fps=fps, mime=mime)
    finally:
        vs.cleanup()
