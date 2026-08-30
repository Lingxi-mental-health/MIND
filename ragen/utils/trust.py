"""Central switch for HuggingFace ``trust_remote_code``.

``trust_remote_code=True`` makes ``from_pretrained`` execute whatever Python
ships inside the model repository. For the models this project uses (Qwen3-4B /
Qwen3-8B) that is unnecessary — their architectures are supported natively by
``transformers`` — so the default here is ``False``.

If you deliberately need a model that requires custom code, opt in per process::

    export MIND_TRUST_REMOTE_CODE=1

Only do that for a model whose repository you have actually reviewed.
"""
from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}


def trust_remote_code() -> bool:
    """Return whether custom model code may be executed (default: ``False``)."""
    return os.environ.get("MIND_TRUST_REMOTE_CODE", "").strip().lower() in _TRUTHY


__all__ = ["trust_remote_code"]
