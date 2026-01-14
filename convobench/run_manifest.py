"""Run manifest helpers.

Provides utilities to build a RunManifest from runtime configuration.
"""

from __future__ import annotations

from typing import Optional

from convobench.spec import AFVariables, RunManifest, ScenarioPack, ToolConfig


def build_manifest(
    *,
    run_id: str,
    code_version: Optional[str],
    scenario_pack: Optional[ScenarioPack],
    af_variables: Optional[AFVariables],
    tool_config: Optional[ToolConfig],
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        code_version=code_version,
        scenario_pack_version=scenario_pack.version if scenario_pack else None,
        af_variables=af_variables or (scenario_pack.af_variables if scenario_pack else AFVariables()),
        tool_config=tool_config or ToolConfig(),
        scenarios=scenario_pack.scenarios if scenario_pack else [],
    )
