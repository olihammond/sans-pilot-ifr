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
from sans_pilot.diagnostics import (
  RUNS_Z_SYSTEMATIC,
  active_arrays,
  baseline_models,
)
from sans_pilot.guess import CAUTION, GuessUnavailable, guess_status, suggest_models
from sans_pilot.sessions import (
  Session,
  SessionError,
  _next_steps,
  create_session,
  data_summary,
  default_free_parameters,
  fit_residuals,
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
EXTRA_STARTS = 6
RESTART_SECONDS = 25.0
RESTART_MAX_CURVE_SECONDS = 0.5
DEFAULT_FIT_STEPS = 200
DEGENERATE_CORRELATION = 0.999
# A multi-shell model whose shell count is about one is fitted alongside its
# single-shell counterpart: (parameter holding the count, simpler model).
SINGLE_SHELL_COUNTERPART = {
  "multilayer_vesicle": ("n_shells", "vesicle"),
  "onion": ("n_shells", "core_shell_sphere"),
  "core_multi_shell": ("n", "core_shell_sphere"),
}
# Parameter names of a multi-shell model translated to its counterpart's.
COUNTERPART_PARAMETERS = {
  "vesicle": {"thick_shell": "thickness"},
  "core_shell_sphere": {
    "radius_core": "radius",
    "thickness1": "thickness",
    "sld_out1": "sld_shell",
    "sld1": "sld_shell",
  },
}
# Parameters that multiply the intensity in most models: these are treated as
# degenerate with the scale at a lower correlation than other parameters.
PREFACTOR_NAMES = ("volfraction", "vol_frac")
PREFACTOR_CORRELATION = 0.95
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


def alternative_starts(
  parameters: dict[str, Any], expression: str, count: int
) -> list[dict[str, float]]:
  """Propose other starting points for the free structural parameters.

  The first proposal mirrors every size about their geometric mean (which
  turns a prolate start into an oblate one); the rest are log-uniform draws
  inside the current bounds.  Draws are seeded by the model name so repeated
  runs give the same result.
  """
  free = {
    name: row
    for name, row in parameters.items()
    if row.get("vary") and name not in ("scale", "background")
  }
  usable = {
    name: (float(row["value"]), float(row["min"]), float(row["max"]))
    for name, row in free.items()
    if row.get("min") is not None
    and row.get("max") is not None
    and float(row["min"]) > 0
    and math.isfinite(float(row["max"]))
    and float(row["value"]) > 0
  }
  if not usable:
    return []
  starts: list[dict[str, float]] = []
  if len(usable) >= 2:
    centre = math.exp(
      sum(math.log(value) for value, _, _ in usable.values()) / len(usable)
    )
    starts.append(
      {
        name: min(max(centre * centre / value, low * 1.001), high * 0.999)
        for name, (value, low, high) in usable.items()
      }
    )
  seed = sum(ord(character) for character in expression)
  generator = np.random.default_rng(seed)
  while len(starts) < count:
    starts.append(
      {
        name: float(
          math.exp(generator.uniform(math.log(low * 1.001), math.log(high * 0.999)))
        )
        for name, (_, low, high) in usable.items()
      }
    )
  return starts[:count]


def _solve_scale_background(fitter: IQuana, keep_bounds: bool = False) -> None:
  """Set scale and background by weighted linear least squares.

  With ``keep_bounds`` the solved values are clipped into the existing bounds
  (used for restarts, where the bounds are already chosen).
  """
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
    values = {"scale": float(scale), "background": float(background)}
    for name, value in values.items():
      row = fitter.params[name]
      low, high = row.get("min"), row.get("max")
      if keep_bounds and low is not None and high is not None:
        span = float(high) - float(low)
        value = min(max(value, float(low) + 1e-6 * span), float(high) - 1e-6 * span)
      fitter.set_param(name, value=value)


def parse_fixed_parameters(text: str | None) -> dict[str, float]:
  """Parse 'sld=0.4, sld_solvent=6.36' into independently known values."""
  fixed: dict[str, float] = {}
  for item in (text or "").replace(";", ",").split(","):
    if not item.strip():
      continue
    name, separator, value = item.partition("=")
    try:
      if not separator or not name.strip():
        raise ValueError
      fixed[name.strip()] = float(value)
    except ValueError:
      raise SessionError(
        f"fixed_parameters entry '{item.strip()}' must look like name=number, "
        "for example 'sld=0.4,sld_solvent=6.36'."
      ) from None
  return fixed


def _configure_candidate(
  fitter: IQuana,
  expression: str,
  start: dict[str, Any] | None,
  fixed: dict[str, float] | None = None,
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

  # Independently known values (for example SLDs) replace the library's and
  # stay fixed; the scale is re-solved because the contrast has changed.
  known = {
    name: value for name, value in (fixed or {}).items() if name in fitter.params
  }
  for name, value in known.items():
    current = fitter.params[name]
    low, high = current.get("min"), current.get("max")
    fitter.set_param(
      name,
      value=value,
      min=min(float(low), value) if low is not None else None,
      max=max(float(high), value) if high is not None else None,
      vary=False,
    )
  if start is None or known:
    _solve_scale_background(fitter)

  limits = _hard_limits(expression)
  spread = 4.0 if start is not None else 10.0
  arrays = active_arrays(fitter.data)
  typical = float(np.median(np.abs(arrays["i"]))) if arrays["i"].size else 1.0
  for name in default_free_parameters(fitter, expression):
    if name in known:
      continue
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


def _degenerate_with_scale(fit_result: dict[str, Any]) -> list[str]:
  """Name free parameters whose effect is indistinguishable from the scale."""
  labels = [str(label) for label in fit_result.get("cov_labels") or []]
  cov = fit_result.get("cov")
  if cov is None or "scale" not in labels:
    return []
  matrix = np.asarray(cov, dtype=float)
  if matrix.shape != (len(labels), len(labels)):
    return []
  index = labels.index("scale")
  names = []
  for other, label in enumerate(labels):
    if other == index or label == "background":
      continue
    denominator = math.sqrt(abs(matrix[index, index] * matrix[other, other]))
    if not denominator:
      continue
    limit = (
      PREFACTOR_CORRELATION
      if label.startswith(PREFACTOR_NAMES)
      else DEGENERATE_CORRELATION
    )
    if abs(matrix[index, other] / denominator) >= limit:
      names.append(label)
  return names


def _resolve_scale_degeneracy(
  fitter: IQuana,
  fit_result: dict[str, Any],
  initial: dict[str, float],
  options: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
  """Hold parameters that only rescale the intensity, then refit.

  A volume fraction that merely multiplies the curve cannot be separated from
  ``scale``: freeing both gives each a huge, meaningless uncertainty.  Such
  parameters go back to their starting value and stay fixed; only their
  product with the scale is determined, and the report says so.
  """
  names = _degenerate_with_scale(fit_result)
  if not names:
    return fit_result, []
  for name in names:
    fitter.set_param(
      name, value=initial.get(name, fitter.params[name]["value"]), vary=False
    )
  with time_limit(CANDIDATE_METHOD_SECONDS):
    refit = fitter.fit(engine="bumps", method="lm", **options)
  notes = [
    f"{name} only rescales the intensity in this model, so it cannot be "
    f"separated from scale. It is held at {fitter.params[name]['value']:.4g} "
    "(its starting value); only the product with scale is determined."
    for name in names
  ]
  return refit, notes


def _bic(fit_result: dict[str, Any]) -> float:
  chisq, free, points = (
    fit_result.get("chisq"),
    fit_result.get("n_free"),
    fit_result.get("n_points"),
  )
  if (
    not isinstance(chisq, (int, float))
    or not isinstance(free, (int, float))
    or not isinstance(points, (int, float))
  ):
    return math.inf
  return float(chisq) + float(free) * math.log(max(float(points), 1.0))


def _refine_polydispersity(
  session: Session,
  fitter: IQuana,
  fit_result: dict[str, Any],
  expression: str,
) -> tuple[IQuana, dict[str, Any], str | None]:
  """Let a size-distribution width vary when the residuals call for it.

  The curve library only scans a few fixed widths.  If the best fit still has
  systematic residuals, one width (an existing one, or a new 10% distribution
  on the first free size) is fitted on a copy; the copy replaces the fit only
  when BIC improves, so the extra parameter has to earn its place.
  """
  residuals, _ = fit_residuals(fitter)
  if not residuals or (residuals.get("runs_z") or 0.0) >= RUNS_Z_SYSTEMATIC:
    return fitter, fit_result, None
  supported = [str(name) for name in fitter.get_polydisperse_parameters()]
  free = {name for name, row in fitter.params.items() if row.get("vary")}
  active = [
    name
    for name in supported
    if fitter.is_polydispersity_enabled() and fitter.get_pd_param(name).get("pd")
  ]
  target = next(iter(active), None) or next(
    (name for name in supported if name in free), None
  )
  if target is None:
    return fitter, fit_result, None

  path = session.directory / "_pd_trial.json"
  try:
    fitter.save_analysis(str(path))
    trial = IQuana.load_analysis(str(path))
    trial.enable_polydispersity(True)
    current = trial.get_pd_param(target)
    trial.set_pd_param(
      target,
      pd_width=float(current.get("pd") or 0.1),
      pd_type=str(current.get("pd_type") or "gaussian"),
      vary=True,
    )
    trial_options, _, _ = _fit_budget(trial)
    with time_limit(CANDIDATE_METHOD_SECONDS):
      trial_result = trial.fit(engine="bumps", method="lm", **trial_options)
  except Exception:
    return fitter, fit_result, None
  finally:
    path.unlink(missing_ok=True)
  if _bic(trial_result) < _bic(fit_result) - 2.0:
    width = float(trial.get_pd_param(target).get("pd") or 0.0)
    return (
      trial,
      trial_result,
      f"The polydispersity of {target} was fitted ({width:.3g} relative width) "
      "because the residuals were systematic; BIC improved.",
    )
  return fitter, fit_result, None


def _fit_budget(fitter: IQuana) -> tuple[dict[str, Any], float, int]:
  """Cap iterations from the measured cost of one curve.

  Every candidate gets the same time budget, so a slow model (numerical
  integration, polydispersity, smearing) cannot stall the procedure.
  Returns (fit options, seconds per curve, number of free parameters).
  """
  fitter.calculate()
  timer = time.perf_counter()
  fitter.calculate()
  per_curve = max(time.perf_counter() - timer, 1e-4)
  free = sum(1 for row in fitter.params.values() if row.get("vary"))
  steps = int(CANDIDATE_FIT_SECONDS / (per_curve * (free + 2)))
  return {"steps": max(2, min(steps, DEFAULT_FIT_STEPS))}, per_curve, free


def _fit_candidate(
  session: Session,
  model: str,
  structure_factor: str | None,
  start: dict[str, Any] | None,
  library_chisq_n: float | None,
  origin: str,
  fixed: dict[str, float] | None = None,
  start_note: str | None = None,
) -> dict[str, Any]:
  expression = _expression(model, structure_factor)
  candidate: dict[str, Any] = {"model": expression, "origin": origin}
  if start_note:
    candidate["note"] = start_note
  if library_chisq_n is not None:
    candidate["library_chisq_n"] = library_chisq_n
  warnings_out: list[str] = []
  started = time.perf_counter()
  try:
    with quiet_science(warnings_out):
      fitter = new_fitter(session, model, structure_factor)
      notes = _configure_candidate(fitter, expression, start, fixed)
      initial = {
        str(name): float(row["value"])
        for name, row in fitter.params.items()
        if isinstance(row.get("value"), (int, float))
      }
      # Library starts are already polished, so LM converges directly (simplex
      # is the fallback).  From model defaults, a simplex search runs first.
      methods = ("lm", "amoeba") if start is not None else ("amoeba", "lm")
      options, per_curve, free = _fit_budget(fitter)
      if options["steps"] < LIMITED_FIT_STEPS:
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

      # A single start can settle in a local minimum (for example the prolate
      # solution for an oblate ellipsoid).  Cheap models are refitted from a
      # few other starting points inside the same bounds; the best fit wins.
      restarts = 0
      has_spread = bool((start or {}).get("polydispersity"))
      if free > 2 and (per_curve <= RESTART_MAX_CURVE_SECONDS or has_spread):
        best = float(fit_result.get("chisq", math.inf))
        deadline = time.perf_counter() + RESTART_SECONDS
        # The library's polydispersity comes from a coarse scan and is held
        # fixed, which can blur the shape; restarts are monodisperse, and the
        # first one repeats the original start without the size distribution.
        trials: list[dict[str, float]] = [{}] if has_spread else []
        trials += alternative_starts(fitter.params, expression, EXTRA_STARTS)
        plain = {**start, "polydispersity": {}} if start is not None else None
        for trial in trials:
          if time.perf_counter() > deadline:
            break
          restarts += 1
          try:
            other = new_fitter(session, model, structure_factor)
            _configure_candidate(other, expression, plain, fixed)
            for name, value in trial.items():
              other.set_param(name, value=value)
            _solve_scale_background(other, keep_bounds=True)
            other_options, other_cost, _ = _fit_budget(other)
            if other_cost > RESTART_MAX_CURVE_SECONDS:
              break
            with time_limit(CANDIDATE_METHOD_SECONDS):
              other_result = other.fit(engine="bumps", method="lm", **other_options)
          except Exception:
            continue
          chisq = float(other_result.get("chisq", math.inf))
          if math.isfinite(chisq) and chisq < best * (1.0 - 1e-4):
            best, fitter, fit_result = chisq, other, other_result
            candidate["improved_by_restart"] = True
            candidate.pop("limited_iterations", None)
      candidate["starts_tried"] = restarts + 1

      procedure_notes: list[str] = []
      final_options, _, _ = _fit_budget(fitter)
      fit_result, degenerate = _resolve_scale_degeneracy(
        fitter, fit_result, initial, final_options
      )
      procedure_notes += degenerate
      fitter, fit_result, spread_note = _refine_polydispersity(
        session, fitter, fit_result, expression
      )
      if spread_note:
        procedure_notes.append(spread_note)
      summary = summarize_fit(fitter, fit_result)
      if procedure_notes:
        candidate["procedure_notes"] = procedure_notes
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
  session.meta["conditional_on"] = []
  session.meta["fixed_after_fit"] = {}
  session.mark_fitted()
  return session.meta["last_fit"] or {}


def _probe_data(session: Session) -> Any:
  with quiet_science():
    probe = IQuana()
    probe.set_data(read_data(session))
    _apply_q_range(probe, session)
  return probe.data


def fit_suggestions(
  session_id: str,
  user_id: str | None,
  top: int,
  extra_models: str | None,
  include_baselines: bool = True,
  fixed_parameters: str | None = None,
) -> dict[str, Any]:
  """Fit the top library suggestions, simple baselines and named models.

  Candidates already fitted in this session under the same Q range and fixed
  parameters are kept; a later call with ``extra_models`` fits only the new
  ones and re-ranks everything.
  """
  session = load_session(session_id, user_id=user_id)
  top = max(1, min(int(top), MAX_CANDIDATES))
  extras = _split_models(extra_models)
  if fixed_parameters is not None:
    session.meta["fixed_parameters"] = parse_fixed_parameters(fixed_parameters)
  fixed: dict[str, float] = dict(session.meta.get("fixed_parameters") or {})

  context = {"q_range": session.meta.get("q_range"), "fixed": fixed}
  previous = [
    dict(item)
    for item in session.meta.get("candidates") or []
    if session.meta.get("candidates_context") == context
  ]
  from_library = sum(1 for item in previous if item.get("origin") == "curve library")
  reuse = bool(previous) and top <= max(from_library, 1)
  known_models = {item["model"] for item in previous} if reuse else set()

  # Each entry: model, structure factor, library start, library score, origin, note
  plan: list[dict[str, Any]] = []
  guess_note: str | None = None
  status = guess_status()
  new_extras = [pair for pair in extras if _expression(*pair) not in known_models]
  if status["available"] and (new_extras or not reuse):
    baselines = (
      _split_models(",".join(baseline_models(session.meta.get("features") or {})))
      if include_baselines and not reuse
      else []
    )
    also = list(dict.fromkeys([*new_extras, *baselines[:MAX_BASELINES]]))

    def usable(expression: str) -> bool:
      """Check the fitting engine can set the model up and evaluate it."""
      model, _, structure_factor = expression.partition("@")
      try:
        new_fitter(session, model, structure_factor or None).calculate()
      except Exception:
        return False
      return True

    data = _probe_data(session)
    with quiet_science():
      guess = suggest_models(
        data,
        top=1 if reuse else max(top, 5) + 3,
        starting_values=0 if reuse else top,
        also=also,
        usable=usable,
      )
      if not reuse:
        # A multi-shell suggestion with about one shell is really its
        # single-shell counterpart: fit that simpler model as well, starting
        # from the multi-shell values.  If the engine cannot set the
        # multi-shell model up at all, ask the library for the counterpart.
        mapped: list[dict[str, Any]] = []
        unknown: list[tuple[str, str | None]] = []
        for item in guess["suggestions"]:
          counter, alternative = SINGLE_SHELL_COUNTERPART.get(
            item["model"], (None, None)
          )
          if not alternative:
            continue
          pair = (alternative, item["structure_factor"])
          if pair in also or pair in unknown:
            continue
          if any(pair == (m["model"], m["structure_factor"]) for m in mapped):
            continue
          start = item.get("start")
          if start is None:
            if not item["loadable"] and not unknown:
              unknown.append(pair)
            continue
          shells = start["parameters"].get(counter)
          if not isinstance(shells, (int, float)) or shells >= 1.5:
            continue
          names = COUNTERPART_PARAMETERS.get(alternative, {})
          mapped.append(
            {
              "model": alternative,
              "structure_factor": item["structure_factor"],
              "expression": _expression(alternative, item["structure_factor"]),
              "origin": "single-shell counterpart",
              "start": {
                "parameters": {
                  names.get(name, name): value
                  for name, value in start["parameters"].items()
                },
                "polydispersity": {
                  names.get(name, name): value
                  for name, value in start["polydispersity"].items()
                },
              },
            }
          )
        if unknown:
          more = suggest_models(data, top=1, starting_values=0, also=unknown)
          for entry in more.get("requested") or []:
            entry["origin"] = "single-shell counterpart"
          mapped += more.get("requested") or []
        guess["requested"] = [*(guess.get("requested") or []), *mapped]
    if not reuse:
      session.meta["suggestions"] = guess["suggestions"]
      for item in guess["suggestions"]:
        if len(plan) >= top:
          break
        if item["loadable"] and "start" in item:
          plan.append(
            {
              "model": item["model"],
              "structure_factor": item["structure_factor"],
              "start": item["start"],
              "library_chisq_n": item["library_chisq_n"],
              "origin": "curve library",
            }
          )
    for item in guess.get("requested") or []:
      pair = (item["model"], item["structure_factor"])
      if any(pair == (p["model"], p["structure_factor"]) for p in plan):
        continue
      plan.append(
        {
          "model": pair[0],
          "structure_factor": pair[1],
          "start": item.get("start"),
          "library_chisq_n": None,
          "origin": item.get("origin")
          or ("requested" if pair in extras else "simple baseline"),
          "start_note": item.get("note"),
        }
      )
  elif not status["available"]:
    guess_note = status["reason"]
    plan = [
      {
        "model": pair[0],
        "structure_factor": pair[1],
        "start": None,
        "library_chisq_n": None,
        "origin": "requested",
      }
      for pair in new_extras
    ]

  if not plan and not previous:
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
  fitted_now = [
    _fit_candidate(
      session,
      item["model"],
      item["structure_factor"],
      item["start"],
      item["library_chisq_n"],
      item["origin"],
      fixed,
      item.get("start_note"),
    )
    for item in plan[:limit]
  ]
  candidates = [*(previous if reuse else []), *fitted_now]
  calibrated = bool(session.meta.get("data", {}).get("has_intensity_errors"))
  _rank(candidates, calibrated=calibrated)

  summaries = dict(session.meta.get("candidate_summaries") or {}) if reuse else {}
  summaries.update(
    {item["model"]: item.pop("_summary") for item in fitted_now if "_summary" in item}
  )
  session.meta["candidate_summaries"] = summaries
  notes = [note for item in fitted_now for note in item.pop("_notes", [])]
  procedure_notes = dict(session.meta.get("procedure_notes") or {}) if reuse else {}
  procedure_notes.update(
    {
      item["model"]: item["procedure_notes"]
      for item in fitted_now
      if item.get("procedure_notes")
    }
  )
  session.meta["procedure_notes"] = procedure_notes
  session.meta["candidates"] = candidates
  session.meta["candidates_context"] = context
  succeeded = [item for item in candidates if "error" not in item]
  new_models = {item["model"] for item in fitted_now if "error" not in item}
  # On a first run the best candidate becomes the session's model.  Later
  # calls leave the current (possibly refined) model alone unless a newly
  # fitted candidate ranks first.
  if succeeded and (not reuse or succeeded[0]["model"] in new_models):
    _select(session, succeeded[0]["model"])
  session.log(
    "fit_suggestions",
    fitted=[item["model"] for item in fitted_now],
    kept=[item["model"] for item in previous] if reuse else [],
    selected=model_expression(session),
  )
  session.save()

  result: dict[str, Any] = {
    "session_id": session.id,
    "status": "fitted" if succeeded else "all_fits_failed",
    "newly_fitted": [item["model"] for item in fitted_now],
    "candidates": candidates,
    "ranking": RANKING_NOTE if calibrated else RANKING_NOTE_NO_ERRORS,
    "selected_model": model_expression(session),
    "selected_fit": session.meta.get("last_fit"),
    "caution": CAUTION,
    "warnings": list(dict.fromkeys(notes))[:10],
    "next_steps": [
      "use-sans-candidate to switch to another fitted candidate",
      *_next_steps(session),
    ]
    if succeeded
    else ["set-sans-model to choose a model yourself"],
  }
  if fixed:
    result["fixed_parameters"] = fixed
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
  fixed_parameters: str | None = None,
) -> dict[str, Any]:
  """Run the whole fixed procedure for one dataset."""
  started = create_session(file=file, example=example, user_id=user_id)
  session_id = started["session_id"]
  if q_min is not None or q_max is not None:
    set_q_range(session_id, q_min=q_min, q_max=q_max, user_id=user_id)
  if fixed_parameters:
    parse_fixed_parameters(fixed_parameters)
  fitted = fit_suggestions(
    session_id, user_id, top, extra_models, include_baselines, fixed_parameters
  )
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
