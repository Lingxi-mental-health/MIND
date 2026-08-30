"""Patient simulators (§4 ``Experimental Setup`` and Appendix C).

Two settings are supported:

* ``PsySim-Std``  — fixed-prompt patient that strictly answers in one
  sentence about its self-report state, used as the controlled benchmark.
* ``PsySim-Adapt`` — same backbone but with adaptive behavioural traits
  (vague, evasive, anxious, suspicious …) sampled per-case to test
  robustness under simulator variation.

Both simulators receive a reconstructed *patient profile* (see
``MIND/scripts/data_process/extract_medical_data.py``) and a doctor question;
they return one patient utterance.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..utils.llm_client import LLMClient


@dataclass
class PatientProfile:
    """Reconstructed patient profile fed to the simulator."""

    description: str
    diagnosis: str = ""
    recommendation: str = ""
    extra: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# System prompts.
# ---------------------------------------------------------------------------

PSYSIM_STD_SYSTEM = (
    "You are a patient interacting with a doctor. Answer each medical question "
    "from the doctor concisely in a single sentence, strictly describing your "
    "symptoms based on your self-report state. Do not mention diagnoses or "
    "recommendations. If a question is unrelated to your self-report state, "
    "respond exactly with: \"Sorry, I cannot answer your question.\"\n\n"
    "Your self-report state: {description}"
)

# Behavioural style modifiers used by PsySim-Adapt.
_PSYSIM_ADAPT_TRAITS = (
    "speak in vague metaphors and avoid direct answers",
    "answer reluctantly, often with 'I don't know' or partial details",
    "be anxious and add catastrophic interpretations to every symptom",
    "be suspicious; question why the doctor is asking",
    "be brief and slightly irritable, but factually consistent",
)


def _adaptive_system(description: str, seed: int) -> str:
    rng = random.Random(seed)
    trait = rng.choice(_PSYSIM_ADAPT_TRAITS)
    return (
        "You are a patient interacting with a doctor. Adopt the following "
        f"behavioural trait while remaining factually consistent with your "
        f"self-report state: {trait}. Always answer in one or two short "
        f"sentences and never reveal a diagnosis.\n\n"
        f"Your self-report state: {description}"
    )


# ---------------------------------------------------------------------------
# Simulator API.
# ---------------------------------------------------------------------------

@dataclass
class PatientSim:
    """Generates patient utterances via an LLM.

    The class deliberately exposes a one-method interface ``respond(question)``
    so that any duck-typed object can be plugged into :class:`MindEnv`.
    """

    llm: LLMClient
    profile: "PatientProfile"
    mode: str = "psysim_std"  # or ``psysim_adapt``
    seed: int = 0
    max_response_length: int = 512

    def respond(self, question: str) -> str:
        if self.mode == "psysim_std":
            system = PSYSIM_STD_SYSTEM.format(description=self.profile.description)
        elif self.mode == "psysim_adapt":
            system = _adaptive_system(self.profile.description, seed=self.seed)
        else:
            raise ValueError(f"Unknown PsySim mode: {self.mode}")
        return self.llm.chat(
            system=system,
            user=question,
            max_tokens=self.max_response_length,
            temperature=0.0,
        )


# Backwards-compat alias.
PatientSimulator = PatientSim


__all__ = ["PatientProfile", "PatientSim", "PatientSimulator"]
