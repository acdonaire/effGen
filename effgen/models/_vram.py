"""How much GPU memory is actually free right now.

Both the model loader and the transformers engine size their placement
decisions from this, and they have to agree: if one reads the card's capacity
and the other reads what is unused, the same machine gets two different answers
about whether a model fits.
"""

from __future__ import annotations


def free_vram_gb() -> float:
    """Return free VRAM in GB across the visible CUDA devices, or 0.0 if none.

    Reports currently-free memory rather than total capacity, so a decision
    made from it accounts for whatever else is already resident on the card.

    Returns:
        Free memory in gibibytes, summed over the visible devices.
    """
    # Imported here rather than at module scope: torch is heavy, and a caller
    # that never touches a GPU should not pay for it at import time.
    import torch

    if not torch.cuda.is_available():
        return 0.0
    free_bytes = 0
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes += torch.cuda.mem_get_info(index)[0]
        except Exception:  # noqa: BLE001 - one unreadable device is not fatal
            pass
    return free_bytes / (1024**3)
