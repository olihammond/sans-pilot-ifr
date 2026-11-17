"""Persistent analysis sessions behind the small, flat guided tools.

A session is a directory holding the loaded data, a small metadata file, and
an IQuana analysis file (``IQuana.save_analysis``).  Every guided tool call
reloads that state, changes one thing, and saves it again, so calls can run in
separate worker processes and a session survives a server restart.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import signal
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from iquana import IQuana, get_all_models, get_structure_factors
from sasdata.dataloader.data_info import Data1D

from sans_pilot.analyses._fitting_helpers import (
  filter_actionable_warnings,
  format_fit_result,
  normalize_value,
)
from sans_pilot.artifacts import artifact_root, safe_path_component
from sans_pilot.datasets import inspect_data, load_data_source
from sans_pilot.diagnostics import (
  THRESHOLDS,
  active_arrays,
  assess_fit,
  describe_features,
  is_undetermined,
  residual_diagnostics,
  scale_limit,
)
from sans_pilot.runtime import render_runtime, scientific_runtime
from sans_pilot.schemas import ExampleDataSource, UploadDataSource

SESSION_VERSION = 1
_SESSION_ID = re.compile(r"s-[0-9a-f]{8}")
_DATA_ARRAYS = ("x", "y", "dy", "dx", "dxl", "dxw")
_MAX_HISTORY = 200
FIT_SECONDS = 300.0
# Held fixed when starting values come from a guess: contrast is degenerate
# with scale, and solvent conditions are known rather than fitted.
_FIXED_BY_DEFAULT_PREFIXES = ("sld",)
_FIXED_BY_DEFAULT_NAMES = frozenset(
  {
    "temperature",
    "dielectconst",
    "concentration_salt",
    "structure_factor_mode",
    "radius_effective_mode",
  }
)


class SessionError(ValueError):
  """A guided-tool request that cannot run in the session's current state."""


class FitTimeout(RuntimeError):
  """A fit exceeded its wall-clock limit and was interrupted."""


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
  """Interrupt the enclosed fit after ``seconds`` of wall-clock time.

  BUMPS' Levenberg-Marquardt can spin in its inner solver for minutes on
  ill-conditioned problems, which an iteration cap does not bound.  The limit
  uses SIGALRM, so it applies on POSIX in a process's main thread (how the
  scientific workers run); elsewhere the block runs unlimited.
  """
  usable = (
    hasattr(signal, "SIGALRM")
    and threading.current_thread() is threading.main_thread()
    and seconds > 0
  )
  if not usable:
    yield
    return

  def interrupt(_signum: int, _frame: Any) -> None:
    raise FitTimeout(f"exceeded {seconds:g} s")

  previous = signal.signal(signal.SIGALRM, interrupt)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


@dataclass(slots=True)
class Session:
  """One analysis session and the files that hold its state."""

  id: str
  directory: Path
  meta: dict[str, Any] = field(default_factory=dict)

  @property
  def data_path(self) -> Path:
    return self.directory / "data.json"

  @property
  def analysis_path(self) -> Path:
    return self.directory / "analysis.json"

  @property
  def fitted_path(self) -> Path:
    """The analysis as it stood at the last completed fit."""
    return self.directory / "fitted.json"

  @property
  def meta_path(self) -> Path:
    return self.directory / "session.json"

  @property
  def candidates_dir(self) -> Path:
    return self.directory / "candidates"

  @property
  def has_model(self) -> bool:
    return self.meta.get("model") is not None and self.analysis_path.is_file()

  def mark_fitted(self) -> None:
    """Record that the saved analysis and ``last_fit`` describe the same fit."""
    shutil.copyfile(self.analysis_path, self.fitted_path)
    self.meta["pending_changes"] = []

  def conditional(self) -> list[str]:
    """Parameters currently fixed at their own fitted values."""
    return list(self.meta.setdefault("conditional_on", []))

  def note_change(self, description: str) -> list[str]:
    """Record a setting changed after the last fit and not yet fitted."""
    pending = self.meta.setdefault("pending_changes", [])
    if self.meta.get("last_fit") is not None:
      pending.append(description)
    return list(pending)

  def log(self, action: str, **detail: Any) -> None:
    history = self.meta.setdefault("history", [])
    history.append({"action": action, **normalize_value(detail)})
    del history[:-_MAX_HISTORY]

  def save(self) -> None:
    self.meta["updated_at"] = time.time()
    temporary = self.meta_path.with_name(f".{self.meta_path.name}.{uuid.uuid4().hex}")
    temporary.write_text(
      json.dumps(self.meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, self.meta_path)


def sessions_root(user_id: str | None) -> Path:
  """Return the per-user directory that holds guided analysis sessions."""
  owner = safe_path_component(user_id) if user_id else "_local"
  return artifact_root() / "sessions" / owner


def load_session(session_id: str, *, user_id: str | None) -> Session:
  """Open an existing session owned by the current user."""
  value = session_id.strip()
  if not _SESSION_ID.fullmatch(value):
    raise SessionError(
      "session_id must look like 's-1a2b3c4d'. Use the session_id returned by "
      "start-sans-analysis."
    )
  directory = sessions_root(user_id) / value
  meta_path = directory / "session.json"
  if not meta_path.is_file():
    raise SessionError(
      f"Session '{value}' was not found. Call start-sans-analysis to create one."
    )
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  return Session(id=value, directory=directory, meta=meta)


@contextlib.contextmanager
def quiet_science(collected: list[str] | None = None) -> Iterator[None]:
  """Serialize IQuana calls, silence console output, and collect warnings."""
  with scientific_runtime(), warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    with contextlib.redirect_stdout(io.StringIO()):
      yield
  if collected is not None:
    collected.extend(str(item.message) for item in captured)


def drop_points_without_uncertainty(data: Any) -> tuple[Any, list[float]]:
  """Remove points whose dI is zero, negative or missing.

  BUMPS weights each point by 1/dI and refuses a dataset in which any point
  cannot be weighted, so a single bad dI makes every fit fail.  Such points
  carry no usable weight and are removed; their Q values are returned so the
  removal can be reported.  Data with no usable dI at all is left alone (it
  is fitted with |I| weighting instead).
  """
  dy = getattr(data, "dy", None)
  if dy is None:
    return data, []
  dy = np.asarray(dy, dtype=float)
  usable = np.isfinite(dy) & (dy > 0)
  if usable.all() or not usable.any():
    return data, []
  x = np.asarray(data.x, dtype=float)
  arrays: dict[str, Any] = {}
  for name in _DATA_ARRAYS:
    values = getattr(data, name, None)
    if values is None:
      continue
    values = np.asarray(values, dtype=float)
    arrays[name] = values[usable] if values.shape == usable.shape else None
  cleaned = Data1D(
    x=arrays["x"], y=arrays["y"], dy=arrays.get("dy"), dx=arrays.get("dx")
  )
  for name in ("dxl", "dxw"):
    if arrays.get(name) is not None:
      setattr(cleaned, name, arrays[name])
  for name in ("filename", "title", "_xunit", "_yunit"):
    if hasattr(data, name):
      setattr(cleaned, name, getattr(data, name))
  return cleaned, [float(value) for value in x[~usable]]


def dropped_points_note(dropped: list[float]) -> str:
  shown = ", ".join(f"{value:.4g}" for value in dropped[:5])
  more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
  return (
    f"{len(dropped)} point(s) with zero or missing intensity uncertainty (dI) "
    f"were excluded, at Q = {shown}{more} Å⁻¹. Such points cannot be weighted "
    "in a fit; check them in the reduction."
  )


def _dump_data(data: Any, path: Path) -> None:
  payload: dict[str, Any] = {
    "name": str(getattr(data, "filename", "") or getattr(data, "title", "") or ""),
    "y_unit": str(getattr(data, "_yunit", "") or ""),
  }
  for name in _DATA_ARRAYS:
    values = getattr(data, name, None)
    if values is None:
      continue
    array = np.asarray(values, dtype=float)
    if array.ndim == 1 and array.size:
      payload[name] = [None if not np.isfinite(v) else float(v) for v in array]
  path.write_text(json.dumps(payload), encoding="utf-8")


def read_data(session: Session) -> Data1D:
  """Rebuild the session's dataset exactly as it was loaded."""
  payload = json.loads(session.data_path.read_text(encoding="utf-8"))

  def array(name: str) -> np.ndarray | None:
    values = payload.get(name)
    if values is None:
      return None
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)

  data = Data1D(x=array("x"), y=array("y"), dx=array("dx"), dy=array("dy"))
  for name in ("dxl", "dxw"):
    values = array(name)
    if values is not None:
      setattr(data, name, values)
  data.filename = payload.get("name") or "session-data"
  if payload.get("y_unit"):
    data._yunit = payload["y_unit"]
  return data


def open_fitter(session: Session, warnings_out: list[str] | None = None) -> IQuana:
  """Load the session's IQuana analysis (model, parameters, last fit)."""
  if not session.has_model:
    raise SessionError(
      "No model is set for this session. Call set-sans-model, or "
      "fit-sans-suggestions to try the suggested models."
    )
  with quiet_science(warnings_out):
    return IQuana.load_analysis(str(session.analysis_path))


def save_fitter(session: Session, fitter: IQuana) -> None:
  with quiet_science():
    fitter.save_analysis(str(session.analysis_path))


def _number(value: Any) -> Any:
  normalized = normalize_value(value)
  if isinstance(normalized, float):
    return float(f"{normalized:.6g}")
  return normalized


def parameter_table(fitter: IQuana, *, descriptions: bool = False) -> dict[str, Any]:
  """Return every parameter as a small row a model can read and edit."""
  table: dict[str, Any] = {}
  for name, config in fitter.params.items():
    row = {
      "value": _number(config.get("value")),
      "min": _number(config.get("min")),
      "max": _number(config.get("max")),
      "vary": bool(config.get("vary", False)),
    }
    if descriptions and config.get("description"):
      row["description"] = str(config["description"])
    table[str(name)] = row
  return table


def polydispersity_table(fitter: IQuana) -> dict[str, Any]:
  table: dict[str, Any] = {}
  if not fitter.is_polydispersity_enabled():
    return table
  for name in fitter.get_polydisperse_parameters():
    config = fitter.get_pd_param(name)
    if not config.get("pd"):
      continue
    table[str(name)] = {
      "width": _number(config.get("pd")),
      "distribution": config.get("pd_type"),
      "vary": bool(config.get("vary", False)),
    }
  return table


def model_expression(session: Session) -> str | None:
  model = session.meta.get("model")
  if not model:
    return None
  if model.get("structure_factor"):
    return f"{model['model']}@{model['structure_factor']}"
  return str(model["model"])


def _next_steps(session: Session) -> list[str]:
  if not session.has_model:
    return [
      "suggest-sans-models to rank models against the curve library",
      "fit-sans-suggestions to fit the top suggestions automatically",
      "set-sans-model if the sample's shape is already known",
    ]
  if session.meta.get("last_fit") is None:
    return [
      "set-sans-parameter to set start values, bounds and which parameters vary",
      "run-sans-fit to fit the model",
    ]
  return [
    "act on diagnostics.suggestions, then run-sans-fit again",
    "export-sans-analysis when the fit is acceptable",
  ]


def _track_conditional(
  session: Session, name: str, *, value: float | None, vary: bool | None
) -> str | None:
  """Notice a fitted parameter being fixed at (or near) the value the fit gave.

  Fixing a parameter at its own fitted value, exactly or rounded, leaves the
  fit unchanged and only shrinks the other uncertainties.  A supplied value
  counts as independent only when it differs from the fitted value by more
  than the fit's own uncertainty, or when the user declared it through
  ``fixed_parameters``.
  """
  conditional = session.meta.setdefault("conditional_on", [])
  record = session.meta.setdefault("fixed_after_fit", {})
  fitted = (session.meta.get("last_fit") or {}).get("fitted") or {}
  declared = name in (session.meta.get("fixed_parameters") or {})

  if vary is True:
    if name in conditional:
      conditional.remove(name)
    record.pop(name, None)
    return None
  previous = fitted.get(name)
  if vary is not False or previous is None:
    # Not a newly fixed fitted parameter; an explicit new value on an already
    # fixed one replaces whatever was recorded.
    if value is not None and name in record:
      record[name]["value"] = value
    return None

  fit_value, fit_error = previous.get("value"), previous.get("stderr")
  undetermined = is_undetermined(fit_value, fit_error)
  record[name] = {
    "value": fit_value if value is None else value,
    "fitted_value": fit_value,
    "fitted_stderr": fit_error,
    "undetermined": undetermined,
  }
  if undetermined:
    # The fit was insensitive to this parameter, so holding it anywhere
    # reasonable neither changes the fit nor tightens the other uncertainties.
    if name in conditional:
      conditional.remove(name)
    return None
  tolerance = 0.0
  if isinstance(fit_value, (int, float)):
    tolerance = 0.02 * abs(fit_value)
    if isinstance(fit_error, (int, float)) and math.isfinite(fit_error):
      tolerance = max(tolerance, abs(fit_error))
  same = value is None or (
    isinstance(fit_value, (int, float)) and abs(value - fit_value) <= tolerance
  )
  if same and not declared:
    if name not in conditional:
      conditional.append(name)
    return (
      f"{name} is now fixed at the value the last fit gave it "
      f"({fit_value} ± {fit_error}). This cannot improve the fit: chi-squared "
      "stays the same while the other uncertainties shrink and correlations "
      "with it disappear. The next fit will be marked as having conditional "
      "uncertainties. Fix a parameter only at a value known independently of "
      f"this fit; otherwise set {name} vary=true again."
    )
  if name in conditional:
    conditional.remove(name)
  return None


def _unfitted(pending: list[str]) -> dict[str, Any]:
  """Tell the caller that changed settings only take effect after a fit."""
  result: dict[str, Any] = {"next_steps": ["run-sans-fit to fit with the new settings"]}
  if pending:
    result["unfitted_changes"] = pending
    result["note"] = (
      "The reported fit predates these changes. Call run-sans-fit before "
      "finishing, or the report will show the previous fit."
    )
  return result


def data_summary(session: Session) -> dict[str, Any]:
  info = session.meta.get("data", {})
  return {
    key: info[key]
    for key in (
      "points_active",
      "actual_q_range",
      "has_intensity_errors",
      "resolution_type",
    )
    if key in info
  }


def create_session(
  *,
  file: str | None,
  example: str | None,
  user_id: str | None,
) -> dict[str, Any]:
  """Load one dataset into a new session and describe its shape in text."""
  if (file is None) == (example is None):
    raise SessionError("Provide exactly one of 'file' or 'example'.")
  source = (
    UploadDataSource(kind="upload", file=file)
    if file is not None
    else ExampleDataSource(kind="example", name=example or "")
  )
  notes: list[str] = []
  with quiet_science(notes):
    loaded = load_data_source(source, user_id=user_id)
  data, dropped = drop_points_without_uncertainty(loaded.data)
  if dropped:
    notes.append(dropped_points_note(dropped))

  session_id = f"s-{uuid.uuid4().hex[:8]}"
  directory = sessions_root(user_id) / session_id
  directory.mkdir(parents=True, exist_ok=False)
  session = Session(id=session_id, directory=directory)
  _dump_data(data, session.data_path)

  info = inspect_data(data)
  if dropped:
    info["excluded_points_without_dI"] = dropped
  features = describe_features(data)
  session.meta = {
    "version": SESSION_VERSION,
    "id": session_id,
    "created_at": time.time(),
    "source": loaded.provenance,
    "data": info,
    "features": features,
    "q_range": {"q_min": None, "q_max": None},
    "model": None,
    "last_fit": None,
    "candidates": [],
    "history": [],
  }
  session.log("start", source=loaded.provenance.get("file_name"))
  session.save()
  return {
    "session_id": session_id,
    "source": {
      key: loaded.provenance[key]
      for key in ("kind", "file_name", "name", "format")
      if key in loaded.provenance
    },
    "data": data_summary(session),
    "features": features,
    "warnings": filter_actionable_warnings(notes),
    "next_steps": _next_steps(session),
  }


def apply_scale_cap(fitter: IQuana, expression: str) -> None:
  """Keep ``scale`` at or below its limit when it is a volume fraction."""
  cap = scale_limit(expression)
  if cap is None or "scale" not in fitter.params:
    return
  row = fitter.params["scale"]
  value = float(row.get("value") or 0.0)
  high = row.get("max")
  high = (
    cap if high is None or not math.isfinite(float(high)) else min(float(high), cap)
  )
  value = min(value, high)
  low = row.get("min")
  low = None if low is None else min(float(low), value)
  fitter.set_param("scale", value=value, min=low, max=high)


def new_fitter(session: Session, model: str, structure_factor: str | None) -> IQuana:
  """Build a fitter on the session data with the session's Q range applied."""
  fitter = IQuana()
  fitter.set_data(read_data(session))
  fitter.set_model(model)
  if structure_factor:
    fitter.set_structure_factor(structure_factor)
  q_range = session.meta.get("q_range") or {}
  if q_range.get("q_min") is not None or q_range.get("q_max") is not None:
    fitter.set_q_range(qmin=q_range.get("q_min"), qmax=q_range.get("q_max"))
  if session.meta.get("data", {}).get("resolution_type") == "none":
    # State the choice IQuana would otherwise warn about on every fit.
    fitter.set_resolution("none")
  if not session.meta.get("data", {}).get("has_intensity_errors", True):
    # BUMPS cannot weight points without dI; |I| weighting (constant relative
    # error) is the stated substitute and matches the curve-library scoring.
    fitter.set_weighting("abs")
  apply_scale_cap(fitter, f"{model}@{structure_factor}" if structure_factor else model)
  return fitter


def validate_model_names(model: str, structure_factor: str | None) -> None:
  if model not in get_all_models():
    close = [name for name in sorted(get_all_models()) if model.lower() in name]
    hint = f" Similar names: {', '.join(close[:8])}." if close else ""
    raise SessionError(
      f"Unknown model '{model}'. Use an exact sasmodels name such as sphere, "
      f"cylinder, core_shell_sphere or lamellar.{hint}"
    )
  if structure_factor and structure_factor not in get_structure_factors():
    raise SessionError(
      f"Unknown structure factor '{structure_factor}'. Choose one of: "
      + ", ".join(get_structure_factors())
      + "."
    )


def set_model(
  session_id: str,
  *,
  model: str,
  structure_factor: str | None,
  user_id: str | None,
) -> dict[str, Any]:
  """Replace the session's model; parameters restart from sasmodels defaults."""
  session = load_session(session_id, user_id=user_id)
  model = model.strip()
  structure_factor = (structure_factor or "").strip() or None
  if "@" in model and structure_factor is None:
    model, _, structure_factor = model.partition("@")
  validate_model_names(model, structure_factor)
  with quiet_science():
    fitter = new_fitter(session, model, structure_factor)
  session.meta["model"] = {"model": model, "structure_factor": structure_factor}
  session.meta["last_fit"] = None
  session.meta["pending_changes"] = []
  session.meta["conditional_on"] = []
  session.meta["fixed_after_fit"] = {}
  session.fitted_path.unlink(missing_ok=True)
  save_fitter(session, fitter)
  session.log("set_model", model=model_expression(session))
  session.save()
  return {
    "session_id": session.id,
    "model": model_expression(session),
    "parameters": parameter_table(fitter, descriptions=True),
    "polydisperse_parameters": [
      str(name) for name in fitter.get_polydisperse_parameters()
    ],
    "next_steps": _next_steps(session),
  }


def set_parameter(
  session_id: str,
  *,
  name: str,
  value: float | None,
  minimum: float | None,
  maximum: float | None,
  vary: bool | None,
  user_id: str | None,
) -> dict[str, Any]:
  """Change one parameter's value, bounds, or vary flag."""
  session = load_session(session_id, user_id=user_id)
  fitter = open_fitter(session)
  name = name.strip()
  if name not in fitter.params:
    raise SessionError(
      f"'{name}' is not a parameter of {model_expression(session)}. "
      "Parameters: " + ", ".join(fitter.params) + "."
    )
  changes: dict[str, Any] = {
    key: item
    for key, item in (
      ("value", value),
      ("min", minimum),
      ("max", maximum),
      ("vary", vary),
    )
    if item is not None
  }
  if not changes:
    raise SessionError("Give at least one of value, min, max or vary.")
  # A value outside its bounds cannot be saved or fitted: widen the bounds to
  # hold it rather than leaving the session in an unusable state.
  current = fitter.params[name]
  low = changes.get("min", current.get("min"))
  high = changes.get("max", current.get("max"))
  target = changes.get("value", current.get("value"))
  if low is not None and high is not None and float(low) > float(high):
    raise SessionError(f"min ({low}) is larger than max ({high}) for {name}.")
  cap = scale_limit(model_expression(session)) if name == "scale" else None
  if cap is not None:
    if target is not None and float(target) > cap:
      raise SessionError(
        f"scale cannot exceed {cap:g} in {model_expression(session)}: there it "
        "is a volume fraction for absolute intensities. A larger scale means the data is not in cm^-1, the SLDs "
        "are wrong, or the model is wrong. Set SANS_PILOT_SCALE_MAX for data "
        "in arbitrary units."
      )
    if high is not None and float(high) > cap:
      changes["max"] = cap
  widened = False
  if target is not None and low is not None and float(target) < float(low):
    changes["min"], widened = float(target), True
  if target is not None and high is not None and float(target) > float(high):
    changes["max"], widened = float(target), True
  with quiet_science():
    fitter.set_param(name, **changes)
  save_fitter(session, fitter)
  warning = _track_conditional(session, name, value=value, vary=vary)
  pending = session.note_change(
    f"{name}: " + ", ".join(f"{key}={item}" for key, item in changes.items())
  )
  session.log("set_parameter", name=name, **changes)
  session.save()
  return {
    "session_id": session.id,
    "parameter": {name: parameter_table(fitter)[name]},
    "varying": [key for key, row in parameter_table(fitter).items() if row["vary"]],
    **({"warning": warning} if warning else {}),
    **(
      {"bounds_note": f"The bounds of {name} were widened to include its value."}
      if widened
      else {}
    ),
    **_unfitted(pending),
  }


def set_polydispersity(
  session_id: str,
  *,
  parameter: str,
  width: float,
  distribution: str,
  vary: bool,
  user_id: str | None,
) -> dict[str, Any]:
  """Give one size parameter a distribution of the stated relative width."""
  session = load_session(session_id, user_id=user_id)
  fitter = open_fitter(session)
  supported = [str(name) for name in fitter.get_polydisperse_parameters()]
  if parameter not in supported:
    raise SessionError(
      f"'{parameter}' cannot be polydisperse in {model_expression(session)}. "
      "Supported: " + (", ".join(supported) or "none") + "."
    )
  with quiet_science():
    fitter.enable_polydispersity(True)
    fitter.set_pd_param(parameter, pd_width=width, pd_type=distribution, vary=vary)
  save_fitter(session, fitter)
  session.log(
    "set_polydispersity", parameter=parameter, width=width, distribution=distribution
  )
  pending = session.note_change(
    f"polydispersity of {parameter}: width={width}, distribution={distribution}"
  )
  session.save()
  return {
    "session_id": session.id,
    "polydispersity": polydispersity_table(fitter),
    **_unfitted(pending),
  }


def set_q_range(
  session_id: str,
  *,
  q_min: float | None,
  q_max: float | None,
  user_id: str | None,
) -> dict[str, Any]:
  """Restrict (or, with both limits omitted, reset) the fitted Q range."""
  session = load_session(session_id, user_id=user_id)
  session.meta["q_range"] = {"q_min": q_min, "q_max": q_max}
  data = read_data(session)
  with quiet_science():
    probe = IQuana()
    probe.set_data(data)
    if q_min is not None or q_max is not None:
      probe.set_q_range(qmin=q_min, qmax=q_max)
    session.meta["data"] = inspect_data(probe.data)
    session.meta["features"] = describe_features(probe.data)
  if session.has_model:
    fitter = open_fitter(session)
    with quiet_science():
      if q_min is None and q_max is None:
        fitter.reset_q_range()
      else:
        fitter.set_q_range(qmin=q_min, qmax=q_max)
    save_fitter(session, fitter)
  pending = session.note_change(f"Q range: q_min={q_min}, q_max={q_max}")
  session.log("set_q_range", q_min=q_min, q_max=q_max)
  session.save()
  return {
    "session_id": session.id,
    "data": data_summary(session),
    **(_unfitted(pending) if pending else {"next_steps": _next_steps(session)}),
  }


def fit_residuals(fitter: IQuana) -> tuple[dict[str, Any], bool]:
  """Compute residual diagnostics for the fitter's current parameters."""
  arrays = active_arrays(fitter.data)
  theory = np.asarray(fitter.calculate(), dtype=float)
  index = arrays["index"]
  if theory.shape == index.shape:
    theory = theory[index]
  elif theory.shape != arrays["q"].shape:
    return {}, "di" in arrays
  return (
    residual_diagnostics(arrays["q"], arrays["i"], theory, arrays.get("di")),
    "di" in arrays,
  )


def summarize_fit(
  fitter: IQuana,
  fit_result: dict[str, Any],
  conditional_on: list[str] | None = None,
  expression: str | None = None,
) -> dict[str, Any]:
  """Reduce an IQuana fit result to statistics, parameters, and a verdict."""
  formatted = format_fit_result(fit_result)
  residuals, has_uncertainties = fit_residuals(fitter)
  fitted: dict[str, Any] = {}
  fixed: dict[str, Any] = {}
  for name, entry in (formatted.get("parameters") or {}).items():
    if not isinstance(entry, dict):
      continue
    if entry.get("fixed"):
      fixed[name] = _number(entry.get("value"))
    else:
      fitted[name] = {
        "value": _number(entry.get("value")),
        "stderr": _number(entry.get("stderr")),
      }
      if is_undetermined(entry.get("value"), entry.get("stderr")):
        fitted[name]["undetermined"] = True
  statistics = {
    key: _number(formatted.get(key))
    for key in (
      "method",
      "reduced_chisq",
      "chisq",
      "dof",
      "n_points",
      "n_free",
      "converged",
    )
    if key in formatted
  }
  chisq, n_points, n_free = (
    formatted.get("chisq"),
    formatted.get("n_points"),
    formatted.get("n_free"),
  )
  if (
    isinstance(chisq, (int, float))
    and isinstance(n_points, int)
    and isinstance(n_free, int)
  ):
    statistics["bic"] = _number(chisq + n_free * math.log(max(n_points, 1)))
  return {
    "statistics": statistics,
    "fitted": fitted,
    "fixed": fixed,
    "polydispersity": polydispersity_table(fitter),
    "diagnostics": assess_fit(
      formatted,
      residuals,
      has_uncertainties=has_uncertainties,
      conditional_on=conditional_on,
      scale_cap=scale_limit(expression),
    ),
    "conditional_on": list(conditional_on or []),
    "residuals": residuals,
  }


def run_fit(
  session_id: str,
  *,
  method: str,
  user_id: str | None,
) -> dict[str, Any]:
  """Fit the session's model and return statistics with a text verdict."""
  session = load_session(session_id, user_id=user_id)
  notes: list[str] = []
  fitter = open_fitter(session, notes)
  varying = [name for name, row in parameter_table(fitter).items() if row["vary"]]
  if not varying and not fitter.get_varying_pd_params():
    raise SessionError(
      "No parameter is set to vary. Call set-sans-parameter with vary=true "
      "for the parameters to fit (usually scale, background and the sizes)."
    )
  try:
    with quiet_science(notes), time_limit(FIT_SECONDS):
      fit_result = fitter.fit(engine="bumps", method=method.strip() or "lm")
  except FitTimeout as exc:
    raise SessionError(
      f"The fit {exc} and was stopped; the session is unchanged. Try "
      "method='amoeba', fix a poorly determined parameter, or narrow the bounds."
    ) from exc
  with quiet_science(notes):
    summary = summarize_fit(
      fitter, fit_result, session.conditional(), model_expression(session)
    )
  save_fitter(session, fitter)
  session.meta["last_fit"] = summary
  session.mark_fitted()
  session.log(
    "fit",
    model=model_expression(session),
    method=method,
    reduced_chisq=summary["statistics"].get("reduced_chisq"),
    verdict=summary["diagnostics"]["verdict"],
  )
  session.save()
  return {
    "session_id": session.id,
    "model": model_expression(session),
    **summary,
    "warnings": filter_actionable_warnings(notes),
    "next_steps": _next_steps(session),
  }


def session_status(session_id: str, *, user_id: str | None) -> dict[str, Any]:
  """Report where the analysis stands so a caller can resume it."""
  session = load_session(session_id, user_id=user_id)
  status: dict[str, Any] = {
    "session_id": session.id,
    "source": session.meta.get("source", {}).get("file_name"),
    "data": data_summary(session),
    "features": session.meta.get("features"),
    "model": model_expression(session),
  }
  if session.has_model:
    fitter = open_fitter(session)
    status["parameters"] = parameter_table(fitter)
    status["polydispersity"] = polydispersity_table(fitter)
  status["last_fit"] = session.meta.get("last_fit")
  if session.meta.get("pending_changes"):
    status["unfitted_changes"] = session.meta["pending_changes"]
  status["candidates"] = session.meta.get("candidates") or []
  status["diagnostic_thresholds"] = THRESHOLDS
  status["next_steps"] = _next_steps(session)
  return status


def report_markdown(session: Session, fitter: IQuana | None) -> str:
  """Write the session as a self-contained Markdown report."""
  meta = session.meta
  source = meta.get("source", {})
  data = meta.get("data", {})
  lines = [
    f"# SANS analysis report: {source.get('file_name') or source.get('name')}",
    "",
    f"- Session: `{session.id}`",
    f"- Points fitted: {data.get('points_active')} of {data.get('points_total')}",
  ]
  if data.get("excluded_points_without_dI"):
    lines.append(f"- {dropped_points_note(data['excluded_points_without_dI'])}")
  restricted = meta.get("q_range") or {}
  if restricted.get("q_min") is not None or restricted.get("q_max") is not None:
    lines.append(
      f"- **Q range restricted** (q_min={restricted.get('q_min')}, "
      f"q_max={restricted.get('q_max')}): points outside it were not fitted."
    )
  q_range = data.get("actual_q_range") or {}
  if q_range:
    lines.append(f"- Q range: {q_range.get('min'):.4g} to {q_range.get('max'):.4g} Å⁻¹")
  lines.append(f"- Intensity uncertainties: {data.get('has_intensity_errors')}")
  lines.append(f"- Resolution: {data.get('resolution_type')}")

  features = meta.get("features") or {}
  if features.get("hints"):
    lines += ["", "## Curve shape", ""]
    lines += [f"- {hint}" for hint in features["hints"]]
    guinier = features.get("guinier")
    if guinier:
      lines.append(
        f"- Guinier estimate: Rg ≈ {guinier['rg_angstrom']} Å, "
        f"I(0) ≈ {guinier['i0']} ({guinier['points_used']} points)."
      )

  candidates = meta.get("candidates") or []
  if candidates:
    lines += [
      "",
      "## Models tried",
      "",
      "| Rank | Model | χ²/dof | BIC | Free | Verdict | Library χ²/N |",
      "|---|---|---|---|---|---|---|",
    ]
    for candidate in candidates:
      lines.append(
        "| {rank} | {model} | {reduced} | {bic} | {free} | {verdict} | {lib} |".format(
          rank=candidate.get("rank", "-"),
          model=candidate.get("model"),
          reduced=candidate.get("reduced_chisq", "-"),
          bic=candidate.get("bic", "-"),
          free=candidate.get("n_free", "-"),
          verdict=candidate.get("verdict") or candidate.get("error") or "-",
          lib=candidate.get("library_chisq_n", "-"),
        )
      )
    lines += [
      "",
      "A low χ² means a model *can* describe the curve, not that it is the "
      "right description of the sample. Models with similar χ²/dof are not "
      "distinguished by this data alone.",
    ]

  lines += ["", f"## Selected model: `{model_expression(session) or 'none'}`", ""]
  last_fit = meta.get("last_fit")
  if last_fit is not None:
    if fitter is not None and fitter.fit_result is not None:
      with quiet_science():
        lines.append(fitter.get_fit_report().to_markdown())
    else:
      # No IQuana result to render (settings changed after the fit and no
      # fitted snapshot exists): fall back to the stored summary.
      lines += ["| Statistic | Value |", "|---|---|"]
      lines += [f"| {key} | {value} |" for key, value in last_fit["statistics"].items()]
      lines += ["", "| Parameter | Value | Std. error |", "|---|---|---|"]
      lines += [
        f"| {name} | {entry['value']} | {entry['stderr']} |"
        for name, entry in last_fit.get("fitted", {}).items()
      ]
    fixed_later = meta.get("fixed_after_fit") or {}
    if fixed_later:
      lines += [
        "",
        "Parameters fixed after having been fitted:",
        "",
        "| Parameter | Fixed at | Previously fitted |",
        "|---|---|---|",
        *[
          f"| {name} | {entry['value']} | "
          + (
            f"{entry['fitted_value']} (undetermined by the data)"
            if entry.get("undetermined")
            else f"{entry['fitted_value']} ± {entry['fitted_stderr']}"
          )
          + " |"
          for name, entry in fixed_later.items()
        ],
      ]
    notes = (meta.get("procedure_notes") or {}).get(model_expression(session) or "")
    if notes:
      lines += ["", "Notes from the automatic procedure:", ""]
      lines += [f"- {note}" for note in notes]
    pending = meta.get("pending_changes") or []
    if pending:
      lines += [
        "",
        "**Settings changed after this fit and not fitted** (the results "
        "above do not include them):",
        "",
        *[f"- {item}" for item in pending],
      ]
    diagnostics = last_fit.get("diagnostics", {})
    lines += ["", "## Diagnostics", "", f"**Verdict: {diagnostics.get('verdict')}**"]
    if diagnostics.get("issues"):
      lines += ["", "Issues:"] + [f"- {item}" for item in diagnostics["issues"]]
    if diagnostics.get("suggestions"):
      lines += ["", "Suggestions:"] + [
        f"- {item}" for item in diagnostics["suggestions"]
      ]
    lines += [
      "",
      "Verdict thresholds (heuristic): "
      + "; ".join(f"{key}: {value}" for key, value in THRESHOLDS.items())
      + ".",
    ]
  else:
    lines.append("No fit has been run for this model yet.")

  history = meta.get("history") or []
  if history:
    lines += ["", "## Steps taken", ""]
    for number, entry in enumerate(history, start=1):
      detail = ", ".join(
        f"{key}={value}" for key, value in entry.items() if key != "action"
      )
      lines.append(f"{number}. {entry['action']}" + (f" ({detail})" if detail else ""))
  return "\n".join(lines) + "\n"


def _write_notebook(
  session: Session, analysis_path: Path, output_dir: Path
) -> dict[str, Path]:
  """Write an IQuana notebook (and its data snapshot) for the session's fit."""
  from sans_pilot.notebook import build_notebook, safe_stem

  analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
  source = session.meta.get("source", {})
  name = str(source.get("file_name") or source.get("name") or "data")
  model = model_expression(session) or analysis["configuration"]["model_name"]
  stem = safe_stem(f"{safe_stem(name)}_{model.replace('@', '_')}")
  last_fit = session.meta.get("last_fit") or {}
  notes: list[str] = []
  excluded = session.meta.get("data", {}).get("excluded_points_without_dI")
  if excluded:
    notes.append(dropped_points_note(excluded))
  known = session.meta.get("fixed_parameters") or {}
  if known:
    notes.append(
      "Held at values supplied as known: "
      + ", ".join(f"{key} = {value:g}" for key, value in known.items())
      + "."
    )
  candidate = next(
    (
      item
      for item in session.meta.get("candidates") or []
      if item.get("model") == model
    ),
    {},
  )
  notes += list(candidate.get("procedure_notes") or [])
  for item in session.meta.get("pending_changes") or []:
    notes.append(f"Changed after this fit and not included: {item}.")
  built = build_notebook(
    analysis,
    title=f"{name} · {model}",
    stem=stem,
    reduced_chisq=(last_fit.get("statistics") or {}).get("reduced_chisq"),
    verdict=(last_fit.get("diagnostics") or {}).get("verdict"),
    notes=notes,
  )
  written: dict[str, Path] = {}
  for file_name, text in built["files"].items():
    path = output_dir / file_name
    path.write_text(text, encoding="utf-8")
    written[file_name] = path
  notebook_path = output_dir / f"{stem}.ipynb"
  notebook_path.write_text(
    json.dumps(built["notebook"], indent=1, ensure_ascii=False), encoding="utf-8"
  )
  written[notebook_path.name] = notebook_path
  return written


def export_session(
  session_id: str,
  user_id: str | None,
  output_dir: Path,
  include_plot: bool = True,
) -> dict[str, Any]:
  """Write the report, analysis file, parameter file, and fit plot."""
  session = load_session(session_id, user_id=user_id)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  fitter = open_fitter(session) if session.has_model else None
  pending = session.meta.get("pending_changes") or []
  if fitter is not None and pending and session.fitted_path.is_file():
    # Export the last completed fit, not half-applied settings.
    with quiet_science():
      fitter = IQuana.load_analysis(str(session.fitted_path))
  has_fit = (
    fitter is not None
    and session.meta.get("last_fit") is not None
    and fitter.fit_result is not None
  )
  artifacts: dict[str, Path] = {}

  report_path = output_dir / "report.md"
  report_path.write_text(report_markdown(session, fitter), encoding="utf-8")
  artifacts[report_path.name] = report_path

  if fitter is not None:
    analysis_path = output_dir / "analysis.json"
    with quiet_science():
      fitter.save_analysis(str(analysis_path))
    artifacts[analysis_path.name] = analysis_path

    try:
      artifacts.update(_write_notebook(session, analysis_path, output_dir))
    except Exception as exc:  # the report and analysis file stand
      session.log("notebook_failed", error=str(exc)[:200])

    if has_fit:
      if include_plot:
        plot_path = output_dir / "fit_plot.png"
        try:
          with quiet_science(), render_runtime():
            fitter.plot_results(
              show_residuals=True, log_scale=True, show=False
            ).write_image(plot_path)
          artifacts[plot_path.name] = plot_path
        except Exception as exc:  # rendering needs a browser; the report stands
          session.log("plot_failed", error=str(exc)[:200])

  html_path = output_dir / "report.html"
  try:
    from sans_pilot.report_html import build_html_report

    with quiet_science():
      page = build_html_report(
        session, fitter, model=model_expression(session), has_fit=has_fit
      )
    html_path.write_text(page, encoding="utf-8")
    artifacts[html_path.name] = html_path
  except Exception as exc:  # the Markdown report and analysis file stand
    session.log("html_report_failed", error=str(exc)[:200])

  session.log("export", files=sorted(artifacts))
  session.save()
  mime_types = {
    ".md": "text/markdown",
    ".json": "application/json",
    ".txt": "text/plain",
    ".ipynb": "application/x-ipynb+json",
    ".csv": "text/csv",
    ".png": "image/png",
    ".html": "text/html",
  }
  return {
    "summary": {
      "session_id": session.id,
      "model": model_expression(session),
      "last_fit": session.meta.get("last_fit"),
      "unfitted_changes": pending,
      "artifacts": [
        {"name": name, "mime_type": mime_types.get(path.suffix, "text/plain")}
        for name, path in artifacts.items()
      ],
    },
    "artifacts": artifacts,
  }


def orientation_parameters(expression: str) -> set[str]:
  """Return parameters that have no effect on orientationally averaged 1D data."""
  try:
    from sasmodels.core import load_model_info

    parameters = load_model_info(expression).parameters.kernel_parameters
    return {p.name for p in parameters if p.type in ("orientation", "magnetic")}
  except Exception:
    return {"theta", "phi", "psi"}


def default_free_parameters(fitter: IQuana, expression: str) -> list[str]:
  """Choose the parameters a first automatic fit should vary."""
  free: list[str] = []
  skipped = orientation_parameters(expression)
  for name, config in fitter.params.items():
    if name in skipped:
      continue
    if name in _FIXED_BY_DEFAULT_NAMES or name.startswith(_FIXED_BY_DEFAULT_PREFIXES):
      continue
    if name.endswith("_mode") or "_sld" in name or name.startswith("n_"):
      continue
    value = config.get("value")
    if not isinstance(value, (int, float, np.integer, np.floating)):
      continue
    free.append(str(name))
  return free
