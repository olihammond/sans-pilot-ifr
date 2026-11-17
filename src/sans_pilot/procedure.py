"""The fixed analysis procedure: suggest models, fit the top ones, rank them.

``guided_analysis`` runs the whole procedure for one dataset without any
model-side decisions, so its outcome does not depend on how well a language
model follows instructions:

1. load the data and describe its shape;
2. ask the Sassafras curve library for the best-matching models ("Guess");
3. fit the top suggestions with IQuana from the library's starting values;
4. rank the fits and keep the best one in the session for refinement.
"""

from __future__ import annotations

import math
import shutil
import time
from typing import Any

import numpy as np
from iquana import IQuana

from sans_pilot.analyses._fitting_helpers import filter_actionable_warnings
from sans_pilot.artifacts import safe_path_component
from sans_pilot.diagnostics import active_arrays, baseline_models
from sans_pilot.guess import CAUTION, GuessUnavailable, guess_status, suggest_models
from sans_pilot.sessions import (
  Session,
  SessionError,
  _next_steps,
  create_session,
  data_summary,
  default_free_parameters,
  load_session,
  model_expression,
  new_fitter,
  parameter_table,
  quiet_science,
  read_data,
  set_q_range,
  summarize_fit,
  time_limit,
  validate_model_names,
)

MAX_CANDIDATES = 6
MAX_BASELINES = 2
CANDIDATE_FIT_SECONDS = 20.0
CANDIDATE_METHOD_SECONDS = 30.0
DEFAULT_FIT_STEPS = 200
LIMITED_FIT_STEPS = 30
RANKING_NOTE = (
  "Ranked by BIC = chi-squared + n_free * ln(n_points), which penalises extra "
  "free parameters; lower is better. Candidates within 2 of the best BIC are "
  "marked close: this data does not separate them."
)
RANKING_NOTE_NO_ERRORS = (
  "The data has no intensity uncertainties, so candidates are ranked by "
  "reduced chi-squared and the values are only comparable with each other."
)


def _split_models(text: str | None) -> list[tuple[str, str | None]]:
  """Parse 'sphere, cylinder@hardsphere' into (model, structure_factor) pairs."""
  result: list[tuple[str, str | None]] = []
  for item in (text or "").replace(";", ",").split(","):
    name = item.strip()
    if not name:
      continue
    model, _, structure_factor = name.partition("@")
    pair = (model.strip(), structure_factor.strip() or None)
    validate_model_names(*pair)
    if pair not in result:
      result.append(pair)
  return result


def _expression(model: str, structure_factor: str | None) -> str:
  return f"{model}@{structure_factor}" if structure_factor else model


def _hard_limits(expression: str) -> dict[str, tuple[float, float]]:
  """Return sasmodels' physical parameter limits for a model expression."""
  try:
    from sasmodels.core import load_model_info

    parameters = load_model_info(expression).parameters.kernel_parameters
    return {p.name: (float(p.limits[0]), float(p.limits[1])) for p in parameters}
  except Exception:
    return {}


def suggest(session_id: str, user_id: str | None, top: int) -> dict[str, Any]:
  """Rank models for the session data and remember their starting values."""
  session = load_session(session_id, user_id=user_id)
  top = max(1, min(int(top), 10))
  try:
    with quiet_science():
      probe = IQuana()
      probe.set_data(read_data(session))
      _apply_q_range(probe, session)
      guess = suggest_models(probe.data, top=top, starting_values=min(top, 3))
  except GuessUnavailable as exc:
    return {
      "session_id": session.id,
      "available": False,
      "reason": str(exc),
      "features": session.meta.get("features"),
      "next_steps": [
        "Choose a model from features.hints and call set-sans-model",
        "or call fit-sans-suggestions with extra_models='sphere,cylinder'",
      ],
    }
  session.meta["suggestions"] = guess["suggestions"]
  session.log("suggest", top=[item["expression"] for item in guess["suggestions"]])
  session.save()
  return {
    "session_id": session.id,
    "available": True,
    "library": guess["library"],
    "suggestions": [
      {
        "rank": rank,
        "model": item["expression"],
        "library_chisq_n": item["library_chisq_n"],
        **(
          {"chisq_n_after_polish": item["start"]["chisq_n"]} if "start" in item else {}
        ),
        **({"note": item["note"]} if item.get("note") else {}),
      }
      for rank, item in enumerate(guess["suggestions"], start=1)
    ],
    "caution": CAUTION,
    "next_steps": [
      "fit-sans-suggestions to fit the top suggestions and rank the fits",
      "set-sans-model to choose one model yourself",
    ],
  }


def _apply_q_range(fitter: IQuana, session: Session) -> None:
  q_range = session.meta.get("q_range") or {}
  if q_range.get("q_min") is not None or q_range.get("q_max") is not None:
    fitter.set_q_range(qmin=q_range.get("q_min"), qmax=q_range.get("q_max"))


def _solve_scale_background(fitter: IQuana) -> None:
  """Set scale and background by weighted linear least squares."""
  if "scale" not in fitter.params or "background" not in fitter.params:
    return
  arrays = active_arrays(fitter.data)
  fitter.set_param("scale", value=1.0)
  fitter.set_param("background", value=0.0)
  theory = np.asarray(fitter.calculate(), dtype=float)
  if theory.shape == arrays["index"].shape:
    theory = theory[arrays["index"]]
  if theory.shape != arrays["q"].shape:
    return
  weight = (
    1.0 / arrays["di"]
    if "di" in arrays
    else 1.0 / np.maximum(np.abs(arrays["i"]), 1e-30)
  )
  finite = np.isfinite(theory) & np.isfinite(weight) & (weight > 0)
  if int(np.sum(finite)) < 3 or not np.any(theory[finite] != 0):
    return
  design = np.c_[theory[finite], np.ones(int(np.sum(finite)))] * weight[finite, None]
  (scale, background), *_ = np.linalg.lstsq(
    design, arrays["i"][finite] * weight[finite], rcond=None
  )
  if math.isfinite(scale) and scale > 0 and math.isfinite(background):
    fitter.set_param("scale", value=float(scale))
    fitter.set_param("background", value=float(background))


def _configure_candidate(
  fitter: IQuana,
  expression: str,
  start: dict[str, Any] | None,
) -> list[str]:
  """Apply starting values and choose bounded free parameters for one fit."""
  notes: list[str] = []
  values = dict((start or {}).get("parameters") or {})
  # Control parameters (for example n_shells) change which others exist.
  ordered = sorted(values, key=lambda name: not name.startswith("n_"))
  for name in ordered:
    if name.endswith("_mode") or name not in fitter.params:
      continue
    value = values[name]
    if not isinstance(value, (int, float)) or not math.isfinite(value):
      continue
    current = fitter.params[name]
    low, high = current.get("min"), current.get("max")
    try:
      fitter.set_param(
        name,
        value=float(value),
        min=min(float(low), float(value)) if low is not None else None,
        max=max(float(high), float(value)) if high is not None else None,
      )
    except Exception as exc:
      notes.append(f"Start value for {name} was not applied: {exc}")

  for name, setting in ((start or {}).get("polydispersity") or {}).items():
    try:
      fitter.enable_polydispersity(True)
      fitter.set_pd_param(
        name, pd_width=float(setting["width"]), pd_type=setting["distribution"]
      )
    except Exception as exc:
      notes.append(f"Polydispersity for {name} was not applied: {exc}")

  if start is None:
    _solve_scale_background(fitter)

  limits = _hard_limits(expression)
  spread = 4.0 if start is not None else 10.0
  arrays = active_arrays(fitter.data)
  typical = float(np.median(np.abs(arrays["i"]))) if arrays["i"].size else 1.0
  for name in default_free_parameters(fitter, expression):
    value = float(fitter.params[name]["value"])
    hard_low, hard_high = limits.get(name, (-math.inf, math.inf))
    if name == "background":
      span = max(abs(value), typical * 1e-3, 1e-6)
      low, high = value - 3 * span, value + 3 * span
    elif name == "scale":
      if value <= 0:
        continue
      low, high = value / 10.0, value * 10.0
    elif value > 0:
      low, high = value / spread, value * spread
    else:
      continue
    low, high = max(low, hard_low), min(high, hard_high)
    if not low < value < high:
      continue
    fitter.set_param(name, min=low, max=high, vary=True)
  return notes


def _fit_candidate(
  session: Session,
  model: str,
  structure_factor: str | None,
  start: dict[str, Any] | None,
  library_chisq_n: float | None,
  origin: str,
) -> dict[str, Any]:
  expression = _expression(model, structure_factor)
  candidate: dict[str, Any] = {"model": expression, "origin": origin}
  if library_chisq_n is not None:
    candidate["library_chisq_n"] = library_chisq_n
  warnings_out: list[str] = []
  started = time.perf_counter()
  try:
    with quiet_science(warnings_out):
      fitter = new_fitter(session, model, structure_factor)
      notes = _configure_candidate(fitter, expression, start)
      # Library starts are already polished, so LM converges directly (simplex
      # is the fallback).  From model defaults, a simplex search runs first.
      methods = ("lm", "amoeba") if start is not None else ("amoeba", "lm")
      # Give every candidate the same time budget: iterations are capped from
      # the measured cost of one curve, so a slow model (numerical integration,
      # polydispersity, smearing) cannot stall the procedure.
      fitter.calculate()
      timer = time.perf_counter()
      fitter.calculate()
      per_curve = max(time.perf_counter() - timer, 1e-4)
      free = sum(1 for row in fitter.params.values() if row.get("vary"))
      steps = int(CANDIDATE_FIT_SECONDS / (per_curve * (free + 2)))
      options: dict[str, Any] = {"steps": max(2, min(steps, DEFAULT_FIT_STEPS))}
      if steps < LIMITED_FIT_STEPS:
        notes.append(
          f"{expression} is slow to evaluate ({per_curve:.2f} s per curve): its "
          f"automatic fit was limited to {options['steps']} iterations. Refine "
          "it with run-sans-fit if needed."
        )
        candidate["limited_iterations"] = True
      fit_result: dict[str, Any] | None = None
      for method in methods:
        try:
          with time_limit(CANDIDATE_METHOD_SECONDS):
            fit_result = fitter.fit(engine="bumps", method=method, **options)
        except Exception as exc:
          notes.append(f"{expression}: {method} failed: {str(exc)[:160]}")
          continue
        if start is not None:
          break
      if fit_result is None:
        raise RuntimeError("; ".join(notes) or "no fit result")
      summary = summarize_fit(fitter, fit_result)
  except Exception as exc:
    candidate["error"] = f"fit failed: {str(exc)[:200]}"
    return candidate

  session.candidates_dir.mkdir(exist_ok=True)
  path = session.candidates_dir / f"{safe_path_component(expression)}.json"
  with quiet_science():
    fitter.save_analysis(str(path))
  statistics = summary["statistics"]
  candidate.update(
    {
      "reduced_chisq": statistics.get("reduced_chisq"),
      "bic": statistics.get("bic"),
      "n_free": statistics.get("n_free"),
      "verdict": summary["diagnostics"]["verdict"],
      "fitted": {name: entry["value"] for name, entry in summary["fitted"].items()},
      "issues": summary["diagnostics"]["issues"][:3],
      "started_from": "curve library" if start is not None else "model defaults",
      "seconds": round(time.perf_counter() - started, 1),
      "_summary": summary,
      "_notes": filter_actionable_warnings([*notes, *warnings_out]),
    }
  )
  return candidate


def _rank(candidates: list[dict[str, Any]], *, calibrated: bool) -> None:
  """Order candidates in place, best first, and mark statistical ties."""
  key = "bic" if calibrated else "reduced_chisq"

  def score(candidate: dict[str, Any]) -> float:
    value = candidate.get(key)
    if "error" in candidate or not isinstance(value, (int, float)):
      return math.inf
    return float(value) if math.isfinite(value) else math.inf

  candidates.sort(key=score)
  best = score(candidates[0]) if candidates else math.inf
  for rank, candidate in enumerate(candidates, start=1):
    candidate["rank"] = rank
    if calibrated and rank > 1 and math.isfinite(score(candidate)):
      candidate["close_to_best"] = bool(score(candidate) - best < 2.0)


def _select(session: Session, expression: str) -> dict[str, Any]:
  """Make one fitted candidate the session's current model."""
  path = session.candidates_dir / f"{safe_path_component(expression)}.json"
  if not path.is_file():
    raise SessionError(
      f"No fitted candidate named '{expression}'. Fitted candidates: "
      + (
        ", ".join(
          item["model"]
          for item in session.meta.get("candidates") or []
          if "error" not in item
        )
        or "none"
      )
      + "."
    )
  shutil.copyfile(path, session.analysis_path)
  model, _, structure_factor = expression.partition("@")
  session.meta["model"] = {
    "model": model,
    "structure_factor": structure_factor or None,
  }
  summaries = session.meta.get("candidate_summaries") or {}
  session.meta["last_fit"] = summaries.get(expression)
  return session.meta["last_fit"] or {}


def fit_suggestions(
  session_id: str,
  user_id: str | None,
  top: int,
  extra_models: str | None,
  include_baselines: bool = True,
) -> dict[str, Any]:
  """Fit the top library suggestions, simple baselines and named models."""
  session = load_session(session_id, user_id=user_id)
  top = max(1, min(int(top), MAX_CANDIDATES))
  extras = _split_models(extra_models)

  # (model, structure factor, library start, library score, origin)
  plan: list[tuple[str, str | None, dict[str, Any] | None, float | None, str]] = []
  guess_note: str | None = None
  status = guess_status()
  if status["available"]:
    baselines = (
      _split_models(",".join(baseline_models(session.meta.get("features") or {})))
      if include_baselines
      else []
    )
    also = list(dict.fromkeys([*extras, *baselines[:MAX_BASELINES]]))
    with quiet_science():
      probe = IQuana()
      probe.set_data(read_data(session))
      _apply_q_range(probe, session)
      guess = suggest_models(
        probe.data, top=max(top, 5), starting_values=top, also=also
      )
    session.meta["suggestions"] = guess["suggestions"]
    for item in guess["suggestions"]:
      if len(plan) >= top:
        break
      if item["loadable"] and "start" in item:
        plan.append(
          (
            item["model"],
            item["structure_factor"],
            item["start"],
            item["library_chisq_n"],
            "curve library",
          )
        )
    for item in guess.get("requested") or []:
      pair = (item["model"], item["structure_factor"])
      if any(pair == (p[0], p[1]) for p in plan):
        continue
      origin = "requested" if pair in extras else "simple baseline"
      plan.append((*pair, item.get("start"), None, origin))
  else:
    guess_note = status["reason"]
    plan = [(*pair, None, None, "requested") for pair in extras]

  if not plan:
    return {
      "session_id": session.id,
      "status": "needs_model",
      "reason": guess_note or "No loadable suggestion was found.",
      "features": session.meta.get("features"),
      "next_steps": [
        "Call fit-sans-suggestions again with extra_models, for example "
        "'sphere,cylinder,core_shell_sphere', chosen from features.hints",
      ],
    }

  limit = MAX_CANDIDATES + MAX_BASELINES
  candidates = [_fit_candidate(session, *item) for item in plan[:limit]]
  calibrated = bool(session.meta.get("data", {}).get("has_intensity_errors"))
  _rank(candidates, calibrated=calibrated)

  session.meta["candidate_summaries"] = {
    item["model"]: item.pop("_summary") for item in candidates if "_summary" in item
  }
  notes = [note for item in candidates for note in item.pop("_notes", [])]
  session.meta["candidates"] = candidates
  succeeded = [item for item in candidates if "error" not in item]
  selected: dict[str, Any] = {}
  if succeeded:
    selected = _select(session, succeeded[0]["model"])
  session.log(
    "fit_suggestions",
    tried=[item["model"] for item in candidates],
    selected=succeeded[0]["model"] if succeeded else None,
  )
  session.save()

  result: dict[str, Any] = {
    "session_id": session.id,
    "status": "fitted" if succeeded else "all_fits_failed",
    "candidates": candidates,
    "ranking": RANKING_NOTE if calibrated else RANKING_NOTE_NO_ERRORS,
    "selected_model": model_expression(session) if succeeded else None,
    "selected_fit": selected or None,
    "caution": CAUTION,
    "warnings": list(dict.fromkeys(notes))[:10],
    "next_steps": [
      "use-sans-candidate to switch to another fitted candidate",
      *_next_steps(session),
    ]
    if succeeded
    else ["set-sans-model to choose a model yourself"],
  }
  if guess_note:
    result["guess_unavailable"] = guess_note
  return result


def use_candidate(session_id: str, user_id: str | None, model: str) -> dict[str, Any]:
  """Switch the session to another candidate fitted by fit-sans-suggestions."""
  session = load_session(session_id, user_id=user_id)
  last_fit = _select(session, model.strip())
  session.log("use_candidate", model=model.strip())
  session.save()
  with quiet_science():
    fitter = IQuana.load_analysis(str(session.analysis_path))
  return {
    "session_id": session.id,
    "model": model_expression(session),
    "parameters": parameter_table(fitter),
    "last_fit": last_fit or None,
    "next_steps": _next_steps(session),
  }


def guided_analysis(
  file: str | None,
  example: str | None,
  user_id: str | None,
  top: int,
  extra_models: str | None,
  q_min: float | None,
  q_max: float | None,
  include_baselines: bool = True,
) -> dict[str, Any]:
  """Run the whole fixed procedure for one dataset."""
  started = create_session(file=file, example=example, user_id=user_id)
  session_id = started["session_id"]
  if q_min is not None or q_max is not None:
    set_q_range(session_id, q_min=q_min, q_max=q_max, user_id=user_id)
  fitted = fit_suggestions(session_id, user_id, top, extra_models, include_baselines)
  session = load_session(session_id, user_id=user_id)
  return {
    "session_id": session_id,
    "source": started["source"],
    "data": data_summary(session),
    "features": session.meta.get("features"),
    **{key: value for key, value in fitted.items() if key != "session_id"},
    "warnings": list(
      dict.fromkeys([*started["warnings"], *fitted.get("warnings", [])])
    ),
  }
