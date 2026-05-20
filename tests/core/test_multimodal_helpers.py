"""
Tests for effgen.core.multimodal — image_from, audio_from, video_from helpers.
Tests use only bytes, in-memory objects, and local temp paths (no live URLs).
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from effgen.core import multimodal as multimodal_module
from effgen.core.messages import AudioPart, ImagePart, VideoPart
from effgen.core.multimodal import audio_from, image_from, video_from
from effgen.errors import InvalidMultimodalContent, MissingSystemDependency

# ---------------------------------------------------------------------------
# Minimal valid file bytes
# ---------------------------------------------------------------------------

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64   # minimal PNG magic
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF_BYTES = b"GIF89a" + b"\x00" * 64
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 20

WAV_BYTES = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
MP3_BYTES = b"ID3" + b"\x00" * 30
FLAC_BYTES = b"fLaC" + b"\x00" * 30
OGG_BYTES = b"OggS" + b"\x00" * 30


# ---------------------------------------------------------------------------
# image_from — bytes
# ---------------------------------------------------------------------------

class TestImageFromBytes:
    def test_png_bytes(self):
        part = image_from(PNG_BYTES)
        assert isinstance(part, ImagePart)
        assert part.mime == "image/png"
        assert part.image == PNG_BYTES

    def test_jpeg_bytes(self):
        part = image_from(JPEG_BYTES)
        assert part.mime == "image/jpeg"

    def test_gif_bytes(self):
        part = image_from(GIF_BYTES)
        assert part.mime == "image/gif"

    def test_webp_bytes(self):
        part = image_from(WEBP_BYTES)
        assert part.mime == "image/webp"

    def test_mime_override(self):
        part = image_from(PNG_BYTES, mime="image/webp")
        assert part.mime == "image/webp"

    def test_bytearray_accepted(self):
        part = image_from(bytearray(PNG_BYTES))
        assert isinstance(part, ImagePart)


# ---------------------------------------------------------------------------
# image_from — local path (str and Path)
# ---------------------------------------------------------------------------

class TestImageFromPath:
    def _write_tmp(self, suffix: str, data: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_png_path_str(self):
        p = self._write_tmp(".png", PNG_BYTES)
        try:
            part = image_from(p)
            assert isinstance(part, ImagePart)
            assert part.mime == "image/png"
        finally:
            os.unlink(p)

    def test_jpeg_path_pathlib(self):
        from pathlib import Path
        p = self._write_tmp(".jpg", JPEG_BYTES)
        try:
            part = image_from(Path(p))
            assert part.mime == "image/jpeg"
        finally:
            os.unlink(p)

    def test_nonexistent_path_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            image_from("/nonexistent/path/to/image.png")

    def test_webp_path(self):
        p = self._write_tmp(".webp", WEBP_BYTES)
        try:
            part = image_from(p, mime="image/webp")
            assert part.mime == "image/webp"
        finally:
            os.unlink(p)


# ---------------------------------------------------------------------------
# image_from — URL
# ---------------------------------------------------------------------------

class TestImageFromUrl:
    def test_png_url(self, monkeypatch):
        monkeypatch.setattr(multimodal_module, "_fetch_url", lambda url: PNG_BYTES)
        part = image_from("https://example.test/image.png")
        assert isinstance(part, ImagePart)
        assert part.mime == "image/png"
        assert part.image == PNG_BYTES

    def test_url_without_extension_uses_magic_bytes(self, monkeypatch):
        monkeypatch.setattr(multimodal_module, "_fetch_url", lambda url: JPEG_BYTES)
        part = image_from("https://example.test/image")
        assert part.mime == "image/jpeg"


# ---------------------------------------------------------------------------
# image_from — PIL.Image (optional)
# ---------------------------------------------------------------------------

class TestImageFromPIL:
    def test_pil_image(self):
        pytest.importorskip("PIL", reason="Pillow not installed")
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
        part = image_from(img)
        assert isinstance(part, ImagePart)
        assert part.mime in {"image/png", "image/jpeg"}

    def test_numpy_array(self):
        pytest.importorskip("numpy", reason="numpy not installed")
        pytest.importorskip("PIL", reason="Pillow required for np.ndarray conversion")
        import numpy as np
        arr = np.zeros((5, 5, 3), dtype=np.uint8)
        part = image_from(arr)
        assert isinstance(part, ImagePart)
        assert part.mime == "image/png"


# ---------------------------------------------------------------------------
# image_from — invalid source type
# ---------------------------------------------------------------------------

class TestImageFromInvalidSource:
    def test_int_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            image_from(12345)  # type: ignore[arg-type]

    def test_dict_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            image_from({"url": "http://example.com"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# audio_from — bytes
# ---------------------------------------------------------------------------

class TestAudioFromBytes:
    def test_wav_bytes(self):
        part = audio_from(WAV_BYTES)
        assert isinstance(part, AudioPart)
        assert part.mime == "audio/wav"
        assert part.audio == WAV_BYTES

    def test_mp3_bytes(self):
        part = audio_from(MP3_BYTES)
        assert part.mime == "audio/mpeg"

    def test_flac_bytes(self):
        part = audio_from(FLAC_BYTES)
        assert part.mime == "audio/flac"

    def test_ogg_bytes(self):
        part = audio_from(OGG_BYTES)
        assert part.mime == "audio/ogg"

    def test_mime_override(self):
        part = audio_from(WAV_BYTES, mime="audio/wav")
        assert part.mime == "audio/wav"

    def test_bytearray_accepted(self):
        part = audio_from(bytearray(WAV_BYTES))
        assert isinstance(part, AudioPart)


# ---------------------------------------------------------------------------
# audio_from — local path
# ---------------------------------------------------------------------------

class TestAudioFromPath:
    def _write_tmp(self, suffix: str, data: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    def test_wav_path_str(self):
        p = self._write_tmp(".wav", WAV_BYTES)
        try:
            part = audio_from(p)
            assert isinstance(part, AudioPart)
            assert part.mime == "audio/wav"
        finally:
            os.unlink(p)

    def test_mp3_path_pathlib(self):
        from pathlib import Path
        p = self._write_tmp(".mp3", MP3_BYTES)
        try:
            part = audio_from(Path(p), mime="audio/mp3")
            assert part.mime == "audio/mp3"
        finally:
            os.unlink(p)

    def test_nonexistent_path_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            audio_from("/nonexistent/path/to/audio.wav")


# ---------------------------------------------------------------------------
# audio_from — URL
# ---------------------------------------------------------------------------

class TestAudioFromUrl:
    def test_wav_url(self, monkeypatch):
        monkeypatch.setattr(multimodal_module, "_fetch_url", lambda url: WAV_BYTES)
        part = audio_from("https://example.test/audio.wav")
        assert isinstance(part, AudioPart)
        assert part.mime == "audio/wav"
        assert part.audio == WAV_BYTES

    def test_url_without_extension_uses_magic_bytes(self, monkeypatch):
        monkeypatch.setattr(multimodal_module, "_fetch_url", lambda url: FLAC_BYTES)
        part = audio_from("https://example.test/audio")
        assert part.mime == "audio/flac"


# ---------------------------------------------------------------------------
# audio_from — invalid source
# ---------------------------------------------------------------------------

class TestAudioFromInvalidSource:
    def test_int_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            audio_from(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# video_from — ffmpeg missing
# ---------------------------------------------------------------------------

class TestVideoFromFfmpegMissing:
    def test_raises_missing_dep_when_ffmpeg_absent(self, monkeypatch):
        import shutil as _shutil
        monkeypatch.setattr(_shutil, "which", lambda name: None)
        with pytest.raises(MissingSystemDependency) as exc:
            video_from(b"\x00" * 10)
        assert "ffmpeg" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# video_from — with ffmpeg present (integration, skip if not installed)
# ---------------------------------------------------------------------------

class TestVideoFromIntegration:
    @pytest.fixture(autouse=True)
    def require_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")

    def _make_video(self, video_path: str, color: str) -> bool:
        """Create a 1-second solid-colour video. Tries x264 first, falls back to mpeg4.
        Returns True on success.
        """
        import subprocess
        for vcodec in ("libx264", "libopenh264", "mpeg4"):
            result = subprocess.run(
                [
                    "ffmpeg", "-f", "lavfi", "-i", f"color=c={color}:s=32x32:d=1",
                    "-vcodec", vcodec, "-pix_fmt", "yuv420p",
                    video_path, "-y",
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return True
        return False

    def test_video_from_path(self, tmp_path):
        """Create a tiny synthetic video with ffmpeg and then sample it back."""
        video_path = str(tmp_path / "test.mp4")
        if not self._make_video(video_path, "blue"):
            pytest.skip("ffmpeg cannot create a test video with any known encoder")

        part = video_from(video_path, fps=1.0)
        assert isinstance(part, VideoPart)
        assert len(part.frames) >= 1
        assert all(isinstance(f, bytes) and len(f) > 0 for f in part.frames)
        assert part.fps == 1.0

    def test_video_from_bytes(self, tmp_path):
        """Same but pass raw bytes instead of path."""
        video_path = str(tmp_path / "test.mp4")
        if not self._make_video(video_path, "red"):
            pytest.skip("ffmpeg cannot create a test video with any known encoder")

        with open(video_path, "rb") as video_file:
            raw_bytes = video_file.read()
        part = video_from(raw_bytes, fps=1.0)
        assert isinstance(part, VideoPart)
        assert len(part.frames) >= 1

    def test_video_from_url(self, tmp_path, monkeypatch):
        """Fetch video bytes from a URL-like source and sample frames."""
        video_path = str(tmp_path / "test.mp4")
        if not self._make_video(video_path, "green"):
            pytest.skip("ffmpeg cannot create a test video with any known encoder")

        with open(video_path, "rb") as video_file:
            raw_bytes = video_file.read()
        monkeypatch.setattr(multimodal_module, "_fetch_url", lambda url: raw_bytes)

        part = video_from("https://example.test/video.mp4", fps=1.0)
        assert isinstance(part, VideoPart)
        assert len(part.frames) >= 1

    def test_invalid_source_type_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            video_from(12345)  # type: ignore[arg-type]

    def test_non_positive_fps_raises(self):
        with pytest.raises(InvalidMultimodalContent):
            video_from(b"\x00" * 10, fps=0)
