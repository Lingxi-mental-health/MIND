"""MIND multi-turn psychiatric consultation environment.

Implements the operational semantics of §3 + §3.2 + Algorithm 1:

1. ``reset`` initialises an empty :class:`EvidenceState`.
2. ``stage_i_step`` consumes a Stage-I output, reconstructs
   :math:`(\\mathcal{O}_t,\\mathcal{M}_t,\\mathcal{D}_t)` from the current
   retrieval state, populates :math:`(\\mathcal{S}_t,\\mathcal{R}_t)` via
   :class:`PRBStateOperator`, and returns the state to be fed into Stage II.
3. ``stage_ii_step`` consumes a Stage-II output, applies state-conditioned
   rectification, simulates the patient response, updates
   :math:`\\mathcal{O}_t,\\mathcal{M}_t`, and returns ``(observation, reward,
   done, info)``.

The environment is intentionally simulator-agnostic: any object that exposes
``respond(question) -> str`` is accepted as the patient (see
:class:`PatientSim`).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .patient_sim import PatientSim
from ..evaluation.label_mapping import canonical_label
from ..evidence_state.parser import build_initial_state, infer_active_differentials
from ..evidence_state.state import (
    EvidenceState,
    MissingChecks,
    ObservedEvidence,
    SupportSet,
    ReliabilityMetadata,
)
from ..prb.state_operator import PRBStateOperator
from ..rectification.rectifier import Rectifier, RectificationDecision
from ..reward.aggregator import MindRewardManager, TurnRewardResult
from ..reward.process_reward import ProcessRewardScores
from ..utils.format import (
    StageIIOutput,
    parse_stage_i,
    parse_stage_ii,
)


@dataclass
class MindEnvConfig:
    max_turns: int = 10                # Table 13: L = 10
    max_response_tokens: int = 2048
    reliability_threshold: float = 0.2  # Table 13: τ_rel = 0.2
    top_k: int = 4                      # default retrieval depth in §F


@dataclass
class TurnLog:
    turn: int
    stage_i_raw: str = ""
    rag_query: str = ""
    stage_ii_raw: str = ""
    answer: str = ""
    patient_response: str = ""
    reward: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    rectification: Optional[Dict[str, Any]] = None


@dataclass
class MindEnv:
    """Stateful environment for one consultation episode."""

    config: MindEnvConfig
    prb_operator: PRBStateOperator
    reward_manager: MindRewardManager
    patient: PatientSim
    rectifier: Optional[Rectifier] = None

    # ---- runtime state (reset every episode) ----
    profile: Dict[str, Any] = field(default_factory=dict)
    state: EvidenceState = field(default_factory=EvidenceState)
    history: List[Dict[str, str]] = field(default_factory=list)
    turn: int = 0
    finished: bool = False
    diagnosis_made: bool = False
    log: List[TurnLog] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Episode lifecycle.
    # ------------------------------------------------------------------
    def reset(self, profile: Dict[str, Any], seed: int = 0) -> EvidenceState:
        random.seed(seed)
        self.profile = profile
        self.state = build_initial_state(profile.get("initial_retrieval_state", ""))
        self.history = []
        self.turn = 0
        self.finished = False
        self.diagnosis_made = False
        self.log = []
        if self.rectifier is not None:
            self.rectifier.new_episode()
        return self.state

    # ------------------------------------------------------------------
    # Stage I — retrieval-state construction.
    # ------------------------------------------------------------------
    def stage_i_step(self, stage_i_raw: str) -> EvidenceState:
        """Reconstruct all evidence-state fields from the current Stage I.

        The policy verbalises the dialogue history as ``q_t`` every turn.  We
        parse that current retrieval state into ``O_t/M_t/D_t`` before PRB
        lookup, rather than retaining the fields created only at ``reset``.
        """

        parsed = parse_stage_i(stage_i_raw)
        if parsed.is_valid:
            rag_query = parsed.rag_query
            current = build_initial_state(rag_query)
            self.state.O = current.O
            self.state.M = current.M
            self.state.D = current.D
        else:
            # A malformed Stage-I output must not erase the last valid typed
            # state.  Reuse its rendering only as a best-effort PRB query.
            rag_query = self._fallback_rag_query()
        self.prb_operator.populate(self.state, rag_query)
        self.log.append(TurnLog(turn=self.turn + 1, stage_i_raw=stage_i_raw, rag_query=rag_query))
        return self.state

    def _fallback_rag_query(self) -> str:
        """If Stage I is malformed, fall back to a render of E_t fields."""

        return (
            f"observed: {self.state.O.render()}; "
            f"missing: {self.state.M.render()}; "
            f"differentials: {self.state.D.render()}"
        )

    # ------------------------------------------------------------------
    # Stage II — state-to-action reasoning.
    # ------------------------------------------------------------------
    def stage_ii_step(
        self,
        stage_ii_raw: str,
        process_scores: Optional[ProcessRewardScores] = None,
    ) -> Tuple[EvidenceState, float, bool, Dict[str, Any]]:
        """Consume the policy's Stage-II output and step the environment."""

        if self.finished:
            return self.state, 0.0, True, {"reason": "already_finished"}

        prev_state = _shallow_copy(self.state)
        self.turn += 1
        last_turn = (self.turn >= self.config.max_turns)

        # 1. Optional rectification.
        decision = None
        if self.rectifier is not None:
            decision = self.rectifier(
                raw_response=stage_ii_raw,
                state=self.state,
                history_questions=[t.answer for t in self.log if t.answer],
                last_turn=last_turn,
                made_diagnosis=False,
            )
            stage_ii_raw = decision.final_response

        parsed: StageIIOutput = parse_stage_ii(stage_ii_raw)

        # 2. Diagnosis terminal branch.
        if parsed.is_diagnosis():
            diag_pair = parsed.parse_diagnosis() or ("", "")
            predicted, _ = diag_pair
            gold = self.profile.get("gold_label", "")
            term_r = self.reward_manager.terminal(
                predicted_diagnosis=predicted, gold_diagnosis=gold,
            )
            self.diagnosis_made = True
            self.finished = True
            log = self.log[-1] if self.log else TurnLog(turn=self.turn)
            log.stage_ii_raw = stage_ii_raw
            log.answer = parsed.answer
            log.reward = term_r
            log.breakdown = {"terminal": term_r}
            log.rectification = _decision_to_dict(decision)
            if self.log and self.log[-1].turn == self.turn:
                self.log[-1] = log
            else:
                self.log.append(log)
            return self.state, term_r, True, {
                "predicted": canonical_label(predicted),
                "gold": canonical_label(gold),
                "terminal_reward": term_r,
            }

        # 3. Simulate patient response → update O / M.
        question = parsed.answer or self._fallback_question()
        patient_response = self.patient.respond(question)
        self.history.append({"role": "doctor", "content": question})
        self.history.append({"role": "patient", "content": patient_response})
        self._update_observed_from_patient(patient_response)

        # 4. Format / repetition / budget signals.
        format_signals = self._format_signals(stage_ii_raw, question, last_turn=last_turn)

        # 5. Aggregate per-turn reward.
        turn_result: TurnRewardResult = self.reward_manager.turn_reward(
            prev_state=prev_state,
            curr_state=self.state,
            process_scores=process_scores,
            format_signals=format_signals,
            retrieval_score=_retrieval_score(self.state),
        )

        log = self.log[-1] if self.log and self.log[-1].turn == self.turn else TurnLog(turn=self.turn)
        log.stage_ii_raw = stage_ii_raw
        log.answer = question
        log.patient_response = patient_response
        log.reward = turn_result.total
        log.breakdown = turn_result.breakdown
        log.rectification = _decision_to_dict(decision)
        if self.log and self.log[-1].turn == self.turn:
            self.log[-1] = log
        else:
            self.log.append(log)

        if last_turn:
            self.finished = True
        return self.state, turn_result.total, self.finished, {
            "patient_response": patient_response,
            "format_signals": format_signals,
        }

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _fallback_question(self) -> str:
        if self.state.M.fields:
            target = self.state.M.fields[0]
            return f"Could you tell me more about your {target.replace('_', ' ')}?"
        return "Could you tell me more about your current symptoms?"

    def _format_signals(
        self, raw: str, question: str, *, last_turn: bool
    ) -> Dict[str, float]:
        fp = self.reward_manager.format_penalty
        signals: Dict[str, float] = {}
        signals["stage_ii"] = fp.stage_ii_compliance(raw)
        signals["repetition"] = fp.repetition_penalty(
            question, history=[t.answer for t in self.log if t.answer]
        )
        signals["budget"] = fp.budget_penalty(
            last_turn=last_turn, made_diagnosis=False
        )
        return signals

    def _update_observed_from_patient(self, response: str) -> None:
        """Best-effort interim O_t / M_t / D_t update from the patient response.

        Stage I reconstructs the complete state from history on the next turn;
        this routine keeps the state coherent between the patient response and
        that reconstruction, and makes the information-gain reward non-trivial.
        """

        low = response.lower()
        observed = self.state.O.confirmed
        missing = self.state.M.fields[:]
        for field in list(missing):
            keyword = field.replace("_", " ").split()[0]
            if keyword and keyword.lower() in low:
                self.state.M.remove(field)
                observed[field] = response.strip()

        self.state.D = infer_active_differentials(self.state.O)


def _shallow_copy(state: EvidenceState) -> EvidenceState:
    return EvidenceState(
        O=ObservedEvidence(dict(state.O.confirmed), dict(state.O.negated)),
        M=MissingChecks(list(state.M.fields)),
        D=state.D,
        S=state.S,
        R=state.R,
    )


def _retrieval_score(state: EvidenceState) -> float:
    if not state.R.scores:
        return 0.0
    admitted = [s for s, ok in zip(state.R.scores, state.R.gated_in) if ok]
    if not admitted:
        return 0.0
    return float(sum(admitted) / len(admitted))


def _decision_to_dict(decision: Optional[RectificationDecision]) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    return {
        "self_retry": decision.used_self_retry,
        "fallback": decision.used_fallback,
        "triggers": [t.kind.value for t in decision.triggers],
    }


__all__ = ["MindEnv", "MindEnvConfig", "TurnLog"]
