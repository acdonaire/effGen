"""
effgen.security — Security utilities for effGen.

Sub-modules:
  supply_chain  — startup integrity verification (EFFGEN_VERIFY_HASHES=1).
"""

from effgen.security.supply_chain import (
    HashDriftWarning,
    VerificationResult,
    verify_installed_hashes,
    verify_on_startup,
)

__all__ = [
    "HashDriftWarning",
    "VerificationResult",
    "verify_installed_hashes",
    "verify_on_startup",
]
