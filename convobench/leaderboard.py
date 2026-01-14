"""Leaderboard submission format + validator.

Goal: a stable, public submission format pinned to scenario pack version.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class LeaderboardSubmission(BaseModel):
    submission_version: str = "0.1"

    team: str
    contact: Optional[str] = None

    scenario_pack_id: str
    scenario_pack_version: str

    model_ids: List[str]
    af_variables: Dict[str, Any] = Field(default_factory=dict)

    # Summarized results
    metrics: Dict[str, Any]

    # Optional artifacts
    artifact_links: List[str] = Field(default_factory=list)

    @field_validator("model_ids")
    @classmethod
    def _non_empty_models(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("model_ids must be non-empty")
        return v


def validate_submission(data: dict[str, Any]) -> LeaderboardSubmission:
    return LeaderboardSubmission.model_validate(data)
