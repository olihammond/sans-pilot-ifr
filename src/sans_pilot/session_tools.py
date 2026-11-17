"""Small, flat MCP tools for guided analysis (the ``guided`` tool profile).

Every tool takes a handful of scalar arguments and returns compact text-only
JSON with a ``next_steps`` list, so models with small context windows or weak
structured-output support can drive an analysis one decision at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from sans_pilot import procedure, sessions
from sans_pilot.artifacts import artifact_result, create_run_directory
from sans_pilot.files import get_user_id_from_request
from sans_pilot.workers import run_cancellable_worker

SessionId = Annotated[
  str, Field(description="The session_id returned by start-sans-analysis.")
]

GUIDED_INSTRUCTIONS = """
Guided SANS (small-angle neutron scattering) model fitting.

Follow this procedure in order:
1. `list-uploaded-sans-files` to find the data file (skip if the name is known).
2. `run-sans-guided-analysis` with the file name. It loads the data, ranks
   models against a curve library, fits the top suggestions and selects the best.
3. Read `candidates`, `selected_fit.diagnostics` and `features`. If the user
   described the sample, prefer the candidate consistent with it and switch
   with `use-sans-candidate`.
4. If diagnostics list issues, act on ONE suggestion with `set-sans-parameter`,
   `set-sans-polydispersity` or `set-sans-q-range`, then `run-sans-fit`.
   Stop after the verdict is good or acceptable, or after three refinements.
5. `export-sans-analysis`, then report the model, fitted values with
   uncertainties, reduced chi-squared, verdict and remaining issues.

Rules:
- Never invent parameter names, values or statistics; quote tool results.
- A low chi-squared means a model CAN describe the data, not that it is right.
- Prefer the simpler model when candidates are marked close_to_best.
- Pass the session_id from step 2 to every later call.
"""


def start_sans_analysis(
  file: Annotated[
    str | None, Field(description="Stored name of an uploaded data file.")
  ] = None,
  example: Annotated[
    str | None, Field(description="Name of a bundled example dataset.")
  ] = None,
) -> dict[str, Any]:
  """Load one dataset into a new analysis session."""
  return sessions.create_session(
    file=file, example=example, user_id=get_user_id_from_request()
  )


async def suggest_sans_models(
  session_id: SessionId,
  top: Annotated[int, Field(ge=1, le=10)] = 5,
) -> dict[str, Any]:
  """Rank candidate models for the session data using the curve library."""
  return await run_cancellable_worker(
    procedure.suggest,
    session_id,
    get_user_id_from_request(),
    top,
    operation_name="suggest-sans-models",
  )


async def fit_sans_suggestions(
  session_id: SessionId,
  top: Annotated[int, Field(ge=1, le=procedure.MAX_CANDIDATES)] = 3,
  extra_models: Annotated[
    str | None,
    Field(description="Extra models to fit too, e.g. 'sphere,cylinder@hardsphere'."),
  ] = None,
  include_baselines: Annotated[
    bool,
    Field(description="Also fit one or two simple shapes chosen from the curve."),
  ] = True,
) -> dict[str, Any]:
  """Fit the top suggested models, rank the fits, and select the best."""
  return await run_cancellable_worker(
    procedure.fit_suggestions,
    session_id,
    get_user_id_from_request(),
    top,
    extra_models,
    include_baselines,
    operation_name="fit-sans-suggestions",
  )


async def run_sans_guided_analysis(
  file: Annotated[
    str | None, Field(description="Stored name of an uploaded data file.")
  ] = None,
  example: Annotated[
    str | None, Field(description="Name of a bundled example dataset.")
  ] = None,
  top: Annotated[int, Field(ge=1, le=procedure.MAX_CANDIDATES)] = 3,
  extra_models: Annotated[
    str | None,
    Field(description="Extra models to fit too, e.g. 'sphere,cylinder@hardsphere'."),
  ] = None,
  q_min: float | None = None,
  q_max: float | None = None,
  include_baselines: Annotated[
    bool,
    Field(description="Also fit one or two simple shapes chosen from the curve."),
  ] = True,
) -> dict[str, Any]:
  """Run the fixed procedure: load, suggest models, fit the top ones, rank."""
  return await run_cancellable_worker(
    procedure.guided_analysis,
    file,
    example,
    get_user_id_from_request(),
    top,
    extra_models,
    q_min,
    q_max,
    include_baselines,
    operation_name="run-sans-guided-analysis",
  )


def use_sans_candidate(
  session_id: SessionId,
  model: Annotated[
    str, Field(description="A fitted candidate's model name, e.g. 'cylinder'.")
  ],
) -> dict[str, Any]:
  """Switch the session to another candidate that was already fitted."""
  return procedure.use_candidate(session_id, get_user_id_from_request(), model)


def set_sans_model(
  session_id: SessionId,
  model: Annotated[str, Field(description="Exact sasmodels name, e.g. 'sphere'.")],
  structure_factor: Annotated[
    str | None, Field(description="Optional: hardsphere, hayter_msa, ...")
  ] = None,
) -> dict[str, Any]:
  """Set the session's model; parameters restart from the model defaults."""
  return sessions.set_model(
    session_id,
    model=model,
    structure_factor=structure_factor,
    user_id=get_user_id_from_request(),
  )


def set_sans_parameter(
  session_id: SessionId,
  name: str,
  value: float | None = None,
  min: float | None = None,
  max: float | None = None,
  vary: bool | None = None,
) -> dict[str, Any]:
  """Change one parameter's start value, bounds, or whether it is fitted."""
  return sessions.set_parameter(
    session_id,
    name=name,
    value=value,
    minimum=min,
    maximum=max,
    vary=vary,
    user_id=get_user_id_from_request(),
  )


def set_sans_polydispersity(
  session_id: SessionId,
  parameter: str,
  width: Annotated[
    float, Field(ge=0, le=2, description="Relative width; 0.1 means 10%.")
  ],
  distribution: str = "gaussian",
  vary: bool = False,
) -> dict[str, Any]:
  """Give one size parameter a size distribution."""
  return sessions.set_polydispersity(
    session_id,
    parameter=parameter,
    width=width,
    distribution=distribution,
    vary=vary,
    user_id=get_user_id_from_request(),
  )


def set_sans_q_range(
  session_id: SessionId,
  q_min: float | None = None,
  q_max: float | None = None,
) -> dict[str, Any]:
  """Restrict the fitted Q range in 1/Å; omit both limits to reset it."""
  return sessions.set_q_range(
    session_id, q_min=q_min, q_max=q_max, user_id=get_user_id_from_request()
  )


def _fit_worker(session_id: str, method: str, user_id: str | None) -> dict[str, Any]:
  return sessions.run_fit(session_id, method=method, user_id=user_id)


async def run_sans_fit(
  session_id: SessionId,
  method: Annotated[
    str, Field(description="BUMPS method: lm (default), amoeba, or de.")
  ] = "lm",
) -> dict[str, Any]:
  """Fit the session's model and return statistics with a text verdict."""
  return await run_cancellable_worker(
    _fit_worker,
    session_id,
    method,
    get_user_id_from_request(),
    operation_name="run-sans-fit",
  )


def get_sans_analysis(session_id: SessionId) -> dict[str, Any]:
  """Return the session's data, model, parameters, last fit and next steps."""
  return sessions.session_status(session_id, user_id=get_user_id_from_request())


def _export_worker(
  session_id: str, user_id: str | None, output_dir: Path
) -> dict[str, Any]:
  return sessions.export_session(session_id, user_id, output_dir)


async def export_sans_analysis(session_id: SessionId) -> Any:
  """Write the Markdown report, IQuana analysis file, and fit plot."""
  output_dir = create_run_directory("export-sans-analysis")
  user_id = get_user_id_from_request()
  result = await run_cancellable_worker(
    _export_worker,
    session_id,
    user_id,
    output_dir,
    operation_name="export-sans-analysis",
    output_dir=output_dir,
  )
  return artifact_result(result["summary"], result["artifacts"], user_id=user_id)


def register_session_tools(mcp: FastMCP) -> None:
  """Register the flat guided-analysis tools."""
  mcp.tool(
    name="run-sans-guided-analysis",
    description=(
      "Run the fixed analysis procedure on one dataset: load it, rank models "
      "against the Sassafras curve library, fit the top suggestions, rank the "
      "fits and select the best. Returns a session_id for refinement."
    ),
  )(run_sans_guided_analysis)
  mcp.tool(
    name="start-sans-analysis",
    description=(
      "Load one uploaded file or bundled example into a new analysis session "
      "and describe the curve shape in text. Returns a session_id."
    ),
  )(start_sans_analysis)
  mcp.tool(
    name="suggest-sans-models",
    description=(
      "Rank candidate models for the session data by matching it against the "
      "Sassafras curve library. Suggestions describe shape only."
    ),
  )(suggest_sans_models)
  mcp.tool(
    name="fit-sans-suggestions",
    description=(
      "Fit the top suggested models (and any extra_models), rank the fits, "
      "and make the best one the session's model."
    ),
  )(fit_sans_suggestions)
  mcp.tool(
    name="use-sans-candidate",
    description="Switch the session to another already-fitted candidate model.",
  )(use_sans_candidate)
  mcp.tool(
    name="set-sans-model",
    description=(
      "Set the session's model by exact sasmodels name, with an optional "
      "structure factor. Returns its parameters."
    ),
  )(set_sans_model)
  mcp.tool(
    name="set-sans-parameter",
    description=(
      "Change one parameter of the session's model: value, min, max, and "
      "vary (true to fit it, false to hold it fixed)."
    ),
  )(set_sans_parameter)
  mcp.tool(
    name="set-sans-polydispersity",
    description="Give one size parameter a distribution of relative width.",
  )(set_sans_polydispersity)
  mcp.tool(
    name="set-sans-q-range",
    description="Restrict the fitted Q range of the session, or reset it.",
  )(set_sans_q_range)
  mcp.tool(
    name="run-sans-fit",
    description=(
      "Fit the session's model. Returns fitted values, reduced chi-squared, "
      "a verdict, issues and suggested next actions as text."
    ),
  )(run_sans_fit)
  mcp.tool(
    name="get-sans-analysis",
    description="Show the session's current model, parameters and last fit.",
  )(get_sans_analysis)
  mcp.tool(
    name="export-sans-analysis",
    description=(
      "Write the session's Markdown report, reloadable IQuana analysis file, "
      "SasView parameter file and fit plot."
    ),
  )(export_sans_analysis)

  @mcp.prompt(
    name="sans-guided-analysis",
    description="The fixed step-by-step procedure for analysing one SANS dataset.",
  )
  def guided_prompt(file: str, sample: str = "") -> str:
    known = f"\nWhat is known about the sample: {sample}" if sample else ""
    return (
      f"Analyse the SANS data file '{file}'.{known}\n" + GUIDED_INSTRUCTIONS.strip()
    )
