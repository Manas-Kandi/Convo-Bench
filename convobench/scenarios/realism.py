"""Realism-focused scenarios for Agent Factors research.

Implements:
- PartialObservabilityReconciliation: agents get different documents and must reconcile
- InterruptDrivenPlanning: mid-run updates force plan revisions
- ProtocolHandoffExperiment: evaluates handoff protocol formats
- FailureModeSuite: injects typical multi-agent failure mode stressors

These scenarios are designed to resemble real agentic work and expose AF failure modes.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Optional

from convobench.core.environment import ScenarioEnvironment, Tool
from convobench.core.types import Message, MessageRole
from convobench.scenarios.base import Scenario, ScenarioConfig, ScenarioRegistry


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@ScenarioRegistry.register("partial_observability_reconciliation")
class PartialObservabilityReconciliation(Scenario):
    """Agents receive different documents and must reconcile into a single consistent brief."""

    def __init__(
        self,
        chain_length: int = 3,
        domain: str = "incident",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.domain = domain
        super().__init__(config)
        self.config.chain_length = chain_length

        self._docs: list[dict[str, Any]] = []
        self._canonical: dict[str, Any] = {}

    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="partial_observability_reconciliation",
            description="Agents see different docs; must reconcile into consistent summary",
            category="coordination",
            difficulty="hard",
            chain_length=3,
            constraints=[
                "Do not invent facts not present in the documents",
                "Flag conflicts explicitly",
                "Preserve all dates, IDs, and numeric values exactly",
            ],
        )

    def create_instance(self):
        rng = random.Random(self._seed)

        # Generate docs with overlapping and conflicting fields
        incident_id = f"INC-{rng.randint(1000,9999)}"
        service = rng.choice(["payments", "auth", "search", "notifications"]) 
        start_time = rng.choice(["2026-01-03 09:14 UTC", "2026-01-03 09:16 UTC"])
        region = rng.choice(["us-east", "us-west", "eu-central"])

        doc_a = {
            "source": "pager",
            "incident_id": incident_id,
            "service": service,
            "start_time": start_time,
            "symptom": "Elevated 500s",
            "region": region,
        }
        doc_b = {
            "source": "dashboard",
            "incident_id": incident_id,
            "service": service,
            "start_time": start_time,
            "error_rate": rng.choice(["12%", "15%"]),
            "primary_metric": "http_5xx",
        }
        # Introduce a possible conflict
        doc_c = {
            "source": "slack",
            "incident_id": incident_id,
            "suspected_cause": rng.choice(["deploy", "db", "cache"]),
            "start_time": rng.choice([start_time, "2026-01-03 09:10 UTC"]),
            "note": "Team debating root cause; confirm before action",
        }

        self._docs = [doc_a, doc_b, doc_c][: self.config.chain_length]

        # Canonical truth is the union but conflicts are represented explicitly
        self._canonical = {
            "incident_id": incident_id,
            "service": service,
            "start_time": start_time,
            "region": region,
            "required": {
                "error_rate": doc_b["error_rate"],
                "primary_metric": doc_b["primary_metric"],
            },
            "conflicts_expected": [
                "start_time" if doc_c["start_time"] != start_time else None,
            ],
        }
        self._canonical["conflicts_expected"] = [c for c in self._canonical["conflicts_expected"] if c]

        return super().create_instance()

    def generate_initial_message(self) -> Message:
        content = (
            "You are helping on an ops incident. You will receive one document. "
            "Your job is to extract facts exactly, flag any uncertainties/conflicts, and produce a handoff brief."
        )
        return Message(role=MessageRole.USER, content=content, metadata={"seed": self._seed})

    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(
            name="partial_observability_env",
            initial_state={
                "docs": self._docs,
                "doc_count": len(self._docs),
                "handoff_protocol": "structured",
            },
            constraints=self.config.constraints,
        )

        def get_doc(args, state):
            idx = int(args.get("index", 0))
            docs = state.get("docs", [])
            if idx < 0 or idx >= len(docs):
                raise ValueError("doc index out of range")
            return docs[idx]

        env.register_tool(
            Tool(
                name="get_document",
                description="Fetch a specific document by index",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer", "minimum": 0}},
                    "required": ["index"],
                },
                handler=get_doc,
            )
        )
        return env

    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": self._canonical,
            "constraints": self.config.constraints,
            "preserved_information": {
                "incident_id": self._canonical.get("incident_id"),
                "service": self._canonical.get("service"),
                "start_time": self._canonical.get("start_time"),
            },
            "required_actions": ["get_document"],
            "constraint_checks": ["no_hallucinations", "conflicts_flagged"],
        }


@ScenarioRegistry.register("interrupt_driven_planning")
class InterruptDrivenPlanning(Scenario):
    """Agents plan, then receive an interrupt requiring revision without losing constraints."""

    def __init__(
        self,
        num_agents: int = 3,
        domain: str = "project",
        config: Optional[ScenarioConfig] = None,
    ):
        self.num_agents = num_agents
        self.domain = domain
        super().__init__(config)
        self.config.chain_length = num_agents
        self._initial_task: dict[str, Any] = {}
        self._interrupt: dict[str, Any] = {}

    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="interrupt_driven_planning",
            description="Mid-run updates require replanning while preserving constraints",
            category="planning",
            difficulty="hard",
            chain_length=3,
            constraints=[
                "Maintain the original goal",
                "Incorporate new constraint without dropping existing ones",
                "Explicitly note plan changes after interrupt",
            ],
        )

    def create_instance(self):
        rng = random.Random(self._seed)
        self._initial_task = {
            "goal": "Ship v1 onboarding flow",
            "deadline": rng.choice(["2026-02-01", "2026-02-05"]),
            "team": ["design", "frontend", "backend"],
            "constraints": ["No backend schema changes", "Must support mobile"],
        }
        self._interrupt = {
            "type": "new_requirement",
            "message": rng.choice([
                "Legal requires explicit consent checkbox",
                "Analytics must be added to every step",
                "Design requests accessibility AA compliance",
            ]),
        }
        return super().create_instance()

    def generate_initial_message(self) -> Message:
        content = f"""You're coordinating a small project. Create a step-by-step plan.

TASK:
{json.dumps(self._initial_task, indent=2)}

After you draft a plan, you may receive an update that forces revisions."""
        return Message(role=MessageRole.USER, content=content, metadata={"task": self._initial_task})

    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(
            name="interrupt_env",
            initial_state={
                "interrupt": self._interrupt,
                "interrupt_delivered": False,
            },
            constraints=self.config.constraints,
        )

        def get_interrupt(args, state):
            if state.get("interrupt_delivered") is True:
                return {"status": "none"}
            state.set("interrupt_delivered", True)
            return state.get("interrupt")

        env.register_tool(
            Tool(
                name="get_interrupt",
                description="Fetch an interrupt/update that may require replanning",
                parameters={"type": "object", "properties": {}},
                handler=get_interrupt,
            )
        )
        return env

    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": {
                "goal": self._initial_task["goal"],
                "deadline": self._initial_task["deadline"],
                "must_include": self._initial_task["constraints"] + [self._interrupt["message"]],
            },
            "constraints": self.config.constraints,
            "required_actions": ["get_interrupt"],
            "constraint_checks": ["replans_after_interrupt"],
        }


@ScenarioRegistry.register("protocol_handoff_experiment")
class ProtocolHandoffExperiment(Scenario):
    """Same task but handoff protocol varies: freeform vs structured vs checksum readback."""

    def __init__(
        self,
        chain_length: int = 3,
        protocol: str = "structured",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.protocol = protocol
        super().__init__(config)
        self.config.chain_length = chain_length
        self._payload: dict[str, Any] = {}

    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="protocol_handoff_experiment",
            description="Compare handoff protocols under identical task content",
            category="relay",
            difficulty="medium",
            chain_length=3,
            constraints=[
                "Preserve all numbers and IDs",
                "Do not add new fields",
            ],
        )

    def create_instance(self):
        rng = random.Random(self._seed)
        self._payload = {
            "ticket": f"SUP-{rng.randint(10000,99999)}",
            "customer": rng.choice(["Acme Co", "Globex", "Initech"]),
            "issue": "Login loop after password reset",
            "priority": rng.choice(["P1", "P2"]),
            "workaround": "Clear cookies and retry",
        }
        return super().create_instance()

    def generate_initial_message(self) -> Message:
        content = f"""You are handling a support ticket. Prepare a handoff.

DATA:
{json.dumps(self._payload, indent=2)}

Use the required handoff protocol: {self.protocol}.
"""
        return Message(role=MessageRole.USER, content=content)

    def setup_environment(self) -> ScenarioEnvironment:
        env = ScenarioEnvironment(
            name="protocol_env",
            initial_state={"protocol": self.protocol},
            constraints=self.config.constraints,
        )

        def get_protocol(args, state):
            return state.get("protocol")

        env.register_tool(
            Tool(
                name="get_handoff_protocol",
                description="Get required handoff protocol for this run",
                parameters={"type": "object", "properties": {}},
                handler=get_protocol,
            )
        )
        return env

    def get_ground_truth(self) -> dict[str, Any]:
        checksum = _sha1(json.dumps(self._payload, sort_keys=True))
        return {
            "expected_final_output": {"payload": self._payload, "checksum": checksum, "protocol": self.protocol},
            "constraints": self.config.constraints,
            "required_actions": ["get_handoff_protocol"],
            "constraint_checks": ["protocol_followed"],
        }


@ScenarioRegistry.register("failure_mode_suite")
class FailureModeSuite(Scenario):
    """Stress suite for typical multi-agent failure modes."""

    def __init__(
        self,
        chain_length: int = 4,
        mode: str = "spec_drift",
        config: Optional[ScenarioConfig] = None,
    ):
        self.chain_length = chain_length
        self.mode = mode
        super().__init__(config)
        self.config.chain_length = chain_length

        self._spec: dict[str, Any] = {}

    def _default_config(self) -> ScenarioConfig:
        return ScenarioConfig(
            name="failure_mode_suite",
            description="Synthetic suite for silent failure propagation, overconfidence, misalignment, spec drift",
            category="adversarial",
            difficulty="hard",
            chain_length=4,
            constraints=["Preserve the spec exactly", "Flag uncertainty instead of guessing"],
        )

    def create_instance(self):
        rng = random.Random(self._seed)
        self._spec = {
            "mode": self.mode,
            "requirements": [
                "Return output as JSON",
                "Include all keys",
                "Do not assume missing fields",
            ],
            "numbers": {
                "budget": rng.choice(["$5,000", "$10,000"]),
                "deadline": rng.choice(["2026-01-15", "2026-01-20"]),
            },
            "trap": rng.choice([
                "A previous agent will confidently claim an incorrect deadline",
                "A previous agent will omit the budget silently",
                "A previous agent will introduce an extra requirement",
            ]),
        }
        return super().create_instance()

    def generate_initial_message(self) -> Message:
        content = f"""You're part of a team writing a requirements brief.

SPEC:
{json.dumps(self._spec, indent=2)}

Be careful: prior messages may contain errors or drift depending on the mode.
"""
        return Message(role=MessageRole.USER, content=content)

    def setup_environment(self) -> ScenarioEnvironment:
        return ScenarioEnvironment(
            name="failure_mode_env",
            initial_state={"mode": self.mode},
            constraints=self.config.constraints,
        )

    def get_ground_truth(self) -> dict[str, Any]:
        return {
            "expected_final_output": self._spec,
            "constraints": self.config.constraints,
            "required_actions": [],
            "constraint_checks": ["no_spec_drift"],
        }
