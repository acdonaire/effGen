"""
QR code reading tool for the effGen framework.

Backend: pyzbar + Pillow (local, zero network).

System requirement: libzbar shared library must be present.
- Ubuntu/Debian:  sudo apt-get install libzbar0
- Fedora/RHEL:    sudo dnf install zbar
- macOS:          brew install zbar
- conda:          conda install -c conda-forge zbar

Operations:
- read: Decode QR codes (and other barcodes) from an image file or raw bytes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

from ..base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

logger = logging.getLogger(__name__)


def _read_qr(
    image_path: str | None = None,
    image_base64: str | None = None,
) -> dict[str, Any]:
    """Decode QR/barcodes from an image file or base64 PNG bytes."""
    try:
        from PIL import Image
        from pyzbar import pyzbar
    except ImportError as exc:
        raise RuntimeError(
            "pyzbar and Pillow are required for QRReadTool. "
            "Install with: pip install pyzbar Pillow\n"
            "You also need the libzbar system library:\n"
            "  Ubuntu/Debian: sudo apt-get install libzbar0\n"
            "  Fedora/RHEL:   sudo dnf install zbar\n"
            "  macOS:         brew install zbar\n"
            "  conda:         conda install -c conda-forge zbar"
        ) from exc

    try:
        if image_path:
            path = Path(image_path)
            if not path.exists():
                return {
                    "success": False,
                    "data": {"codes": [], "count": 0},
                    "codes": [],
                    "count": 0,
                    "error": f"File not found: {image_path}",
                }
            img = Image.open(path).convert("RGB")
        elif image_base64:
            import io

            if image_base64.startswith("data:image/") and "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw = base64.b64decode(image_base64, validate=True)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            return {
                "success": False,
                "data": {"codes": [], "count": 0},
                "codes": [],
                "count": 0,
                "error": "Provide either 'image_path' or 'image_base64'.",
            }
    except Exception as exc:
        return {
            "success": False,
            "data": {"codes": [], "count": 0},
            "codes": [],
            "count": 0,
            "error": f"Failed to load image: {exc}",
        }

    try:
        decoded_objects = pyzbar.decode(img)
    except Exception as exc:
        return {
            "success": False,
            "data": {"codes": [], "count": 0},
            "codes": [],
            "count": 0,
            "error": f"pyzbar decode error: {exc}",
        }

    codes = []
    for obj in decoded_objects:
        rect = obj.rect
        codes.append(
            {
                "data": obj.data.decode("utf-8", errors="replace"),
                "type": obj.type,
                "rect": {
                    "left": rect.left,
                    "top": rect.top,
                    "width": rect.width,
                    "height": rect.height,
                },
            }
        )

    return {
        "success": True,
        "data": {"codes": codes, "count": len(codes)},
        "codes": codes,
        "count": len(codes),
        "error": None,
    }


class QRReadTool(BaseTool):
    """Decode QR codes and barcodes from images (local, no network required)."""

    def __init__(self) -> None:
        super().__init__(
            metadata=ToolMetadata(
                name="qr_read",
                description=(
                    "Decode QR codes (and other barcodes) from an image file or base64 PNG. "
                    "Returns a list of decoded strings and their bounding-box positions. "
                    "Fully local — no network required. "
                    "Requires the libzbar system library (see docs for install instructions). "
                    "Operations: read (decode QR/barcode from image)."
                ),
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Operation to perform. Currently: 'read'.",
                        required=True,
                        enum=["read"],
                    ),
                    ParameterSpec(
                        name="image_path",
                        type=ParameterType.STRING,
                        description="Path to the image file to decode. Mutually exclusive with image_base64.",
                        required=False,
                    ),
                    ParameterSpec(
                        name="image_base64",
                        type=ParameterType.STRING,
                        description="Base64-encoded PNG/JPEG bytes to decode. Mutually exclusive with image_path.",
                        required=False,
                    ),
                ],
                timeout_seconds=30,
                tags=["qr", "qrcode", "barcode", "image", "decode", "local", "offline"],
                examples=[
                    {"operation": "read", "image_path": "/tmp/my_qr.png"},
                    {"operation": "read", "image_base64": "<base64-encoded-png>"},
                ],
            )
        )

    async def _execute(
        self,
        operation: str,
        image_path: str | None = None,
        image_base64: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        op = operation.lower()

        if op == "read":
            if not image_path and not image_base64:
                raise ValueError("Provide 'image_path' or 'image_base64' for the read operation.")
            return await asyncio.to_thread(_read_qr, image_path, image_base64)

        raise ValueError(f"Unknown operation: {operation!r}. Use 'read'.")
