"""Text-only data features and fit diagnostics for models that cannot view plots.

Everything here is a deterministic heuristic computed from numbers IQuana
already returned.  The thresholds are deliberately simple and are reported
alongside every verdict so a reader can judge them; they do not replace
looking at the fit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

REDUCED_CHISQ_GOOD = 2.0
REDUCED_CHISQ_ACCEPTABLE = 5.0
REDUCED_CHISQ_OVERFIT = 0.3
RUNS_Z_SYSTEMATIC = -3.0
REGION_BIAS_SIGMA = 1.0
POORLY_DETERMINED_RELATIVE_ERROR = 0.5

THRESHOLDS: dict[str, Any] = {
  "good": (
    f"reduced chi-squared <= {REDUCED_CHISQ_GOOD:g}, residuals not systematic, "
    "uncertainties not conditional"
  ),
  "acceptable": f"reduced chi-squared <= {REDUCED_CHISQ_ACCEPTABLE:g}",
  "poor": f"reduced chi-squared > {REDUCED_CHISQ_ACCEPTABLE:g}",
  "systematic_residuals": f"runs-test z < {RUNS_Z_SYSTEMATIC:g}",
  "poorly_determined": (
    f"standard error > {POORLY_DETERMINED_RELATIVE_ERROR:.0%} of the value"
  ),
}


UNDETERMINED_STDERR = 1e7


def is_undetermined(value: Any, stderr: Any) -> bool:
  """Return whether a fitted parameter is not constrained by the data at all.

  IQuana reports a very large standard error when the fit is insensitive to a
  parameter; an error ten times the value says the same thing.
  """
  if not isinstance(stderr, (int, float)) or not math.isfinite(stderr):
    return False
  if abs(stderr) >= UNDETERMINED_STDERR:
    return True
  return (
    isinstance(value, (int, float)) and value != 0 and abs(stderr) > 10 * abs(value)
  )


def _round(value: float, digits: int = 4) -> float:
  """Round to significant digits so responses stay short."""
  if value == 0 or not math.isfinite(value):
    return value
  return float(f"{value:.{digits}g}")


def active_arrays(data: Any) -> dict[str, np.ndarray]:
  """Return finite, unmasked points inside the dataset's active Q range."""
  q = np.asarray(data.x, dtype=float)
  intensity = np.asarray(data.y, dtype=float)
  keep = np.isfinite(q) & np.isfinite(intensity) & (q > 0)
  mask = getattr(data, "mask", None)
  if mask is not None and np.asarray(mask).shape == keep.shape:
    keep &= ~np.asarray(mask, dtype=bool)
  q_min = getattr(data, "qmin", None)
  q_max = getattr(data, "qmax", None)
  if q_min is not None:
    keep &= q >= float(q_min)
  if q_max is not None:
    keep &= q <= float(q_max)

  result = {"index": keep, "q": q[keep], "i": intensity[keep]}
  for name, attribute in (("di", "dy"), ("dq", "dx")):
    values = getattr(data, attribute, None)
    if values is None:
      continue
    array = np.asarray(values, dtype=float)
    if array.shape == keep.shape and np.any(np.isfinite(array) & (array > 0)):
      result[name] = array[keep]
  return result


def _log_slope(q: np.ndarray, intensity: np.ndarray) -> float | None:
  positive = (intensity > 0) & (q > 0)
  if int(np.sum(positive)) < 4:
    return None
  slope, _ = np.polyfit(np.log(q[positive]), np.log(intensity[positive]), 1)
  return float(slope)


def _guinier(q: np.ndarray, intensity: np.ndarray) -> dict[str, Any] | None:
  """Shrink a low-Q window until the Guinier limit q_max * Rg <= 1.3 holds."""
  positive = intensity > 0
  q, intensity = q[positive], intensity[positive]
  if q.size < 6:
    return None
  for count in range(min(q.size, 40), 4, -1):
    slope, intercept = np.polyfit(q[:count] ** 2, np.log(intensity[:count]), 1)
    if slope >= 0:
      continue
    rg = math.sqrt(-3.0 * slope)
    if q[count - 1] * rg <= 1.3:
      return {
        "rg_angstrom": _round(rg),
        "i0": _round(math.exp(intercept)),
        "points_used": int(count),
        "q_max_times_rg": _round(float(q[count - 1] * rg), 3),
      }
  return None


def _peak(q: np.ndarray, intensity: np.ndarray) -> dict[str, Any] | None:
  """Find an interference peak: a maximum that rises above the low-Q level.

  Form-factor oscillations also produce local maxima, but those sit below the
  low-Q intensity; a structure-factor or correlation peak rises above it.
  """
  if q.size < 15:
    return None
  smooth = np.convolve(intensity, np.ones(5) / 5.0, mode="valid")
  centres = q[2:-2]
  index = int(np.argmax(smooth))
  if index < 3 or index > smooth.size - 4:
    return None
  low_q_level = float(np.median(smooth[:3]))
  if low_q_level <= 0 or smooth[index] < 1.15 * low_q_level:
    return None
  q_peak = float(centres[index])
  return {
    "q_peak": _round(q_peak),
    "spacing_angstrom": _round(2.0 * math.pi / q_peak),
    "height_over_low_q": _round(float(smooth[index] / low_q_level), 3),
  }


def describe_features(data: Any) -> dict[str, Any]:
  """Summarize the shape of an I(Q) curve in numbers and plain-language hints."""
  arrays = active_arrays(data)
  q, intensity = arrays["q"], arrays["i"]
  order = np.argsort(q)
  q, intensity = q[order], intensity[order]
  if q.size < 5:
    return {"hints": ["Too few active points to describe the curve shape."]}

  tail = max(3, q.size // 20)
  background = float(np.median(intensity[-tail:]))
  edge = max(5, q.size // 8)
  low_slope = _log_slope(q[:edge], intensity[:edge])

  # Decay region: above the low-Q plateau and above the flat background.
  excess = intensity - background
  top = float(np.max(excess)) if np.any(excess > 0) else 0.0
  decay = (excess < 0.5 * top) & (excess > 0.02 * top) if top > 0 else excess > 0
  decay_slope = _log_slope(q[decay], excess[decay]) if np.sum(decay) >= 5 else None

  features: dict[str, Any] = {
    "points": int(q.size),
    "q_min": _round(float(q[0])),
    "q_max": _round(float(q[-1])),
    "dynamic_range": (
      _round(float(np.max(intensity) / background), 3) if background > 0 else None
    ),
    "high_q_background_estimate": _round(background),
    "low_q_log_slope": None if low_slope is None else _round(low_slope, 3),
    "decay_log_slope": None if decay_slope is None else _round(decay_slope, 3),
    "guinier": _guinier(q, intensity),
    "peak": _peak(q, intensity),
  }

  hints: list[str] = []
  if low_slope is not None:
    guinier = features["guinier"] or {}
    rolling_off = guinier.get("points_used", 0) >= 8 and -1.5 < low_slope <= -0.3
    if rolling_off:
      hints.append(
        "The curve is still rolling off towards a low-Q plateau that the Q "
        f"range does not fully reach (Guinier Rg about "
        f"{guinier['rg_angstrom']} Å): finite-sized particles. The "
        "low-Q slope is not a reliable shape indicator here."
      )
    elif low_slope > 0.3:
      hints.append(
        "Intensity falls towards low Q (suppressed forward scattering): "
        "particles repel each other, so a structure factor is needed "
        "(hardsphere for uncharged, hayter_msa for charged particles)."
      )
    elif low_slope > -0.3:
      hints.append(
        "Low-Q plateau: compact, non-interacting particles are plausible "
        "(sphere, ellipsoid, core_shell_sphere)."
      )
    elif low_slope > -1.5:
      hints.append(
        "Low-Q slope near -1: elongated objects are plausible "
        "(cylinder, flexible_cylinder)."
      )
    elif low_slope > -2.7:
      hints.append(
        "Low-Q slope near -2: flat objects or chains are plausible "
        "(lamellar, core_shell_bicelle, polymer_excl_volume, gaussian coil)."
      )
    else:
      hints.append(
        "Steep low-Q upturn: large structures, aggregation, or Porod "
        "scattering from sharp interfaces (power_law, porod, fractal)."
      )
  if features["peak"] is not None and not (low_slope and low_slope > 0.3):
    hints.append(
      "An interference peak is present: consider a structure factor "
      "(hardsphere, hayter_msa) or an ordered/lamellar model."
    )
  if decay_slope is not None and decay_slope < -3.5:
    hints.append("Decay slope near -4: sharp particle interfaces (Porod law).")
  if "di" not in arrays:
    hints.append(
      "No intensity uncertainties: fits weight each point by |I| (constant "
      "relative error) and chi-squared is not statistically calibrated."
    )
  if "dq" not in arrays:
    hints.append("No Q-resolution column: models are evaluated unsmeared.")
  features["hints"] = hints
  return features


def baseline_models(features: dict[str, Any]) -> list[str]:
  """Name one or two simple shapes consistent with the curve's low-Q behaviour.

  The curve library often ranks flexible many-parameter models first because
  they can imitate simpler shapes.  Fitting these plain shapes alongside the
  library's suggestions lets the ranking prefer them when they fit as well.
  """
  slope = features.get("low_q_log_slope")
  if not isinstance(slope, (int, float)):
    return []
  if slope > 0.3:
    return ["sphere@hardsphere", "sphere@hayter_msa"]
  guinier = features.get("guinier")
  if guinier and guinier.get("points_used", 0) >= 8 and -1.5 < slope <= -0.3:
    return ["sphere", "ellipsoid"]
  if slope > -0.3:
    models = ["sphere", "ellipsoid"]
  elif slope > -1.5:
    models = ["cylinder", "ellipsoid"]
  elif slope > -2.7:
    models = ["lamellar", "cylinder"]
  else:
    models = ["power_law"]
  if features.get("peak") is not None and slope > -1.5:
    models = ["sphere@hardsphere", models[0]]
  return models


def residual_diagnostics(
  q: np.ndarray,
  intensity: np.ndarray,
  theory: np.ndarray,
  sigma: np.ndarray | None,
) -> dict[str, Any]:
  """Describe where and how systematically the model misses the data."""
  finite = np.isfinite(theory) & np.isfinite(intensity)
  if sigma is not None:
    finite &= np.isfinite(sigma) & (sigma > 0)
  q, intensity, theory = q[finite], intensity[finite], theory[finite]
  if q.size < 6:
    return {"points": int(q.size)}

  if sigma is not None:
    residual = (intensity - theory) / sigma[finite]
    kind = "normalized (data - model) / dI"
  else:
    scale = np.maximum(np.abs(theory), 1e-30)
    residual = (intensity - theory) / scale
    kind = "relative (data - model) / model"

  order = np.argsort(q)
  q, residual = q[order], residual[order]
  signs = residual > 0
  n_pos, n_neg = int(np.sum(signs)), int(np.sum(~signs))
  runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
  runs_z: float | None = None
  if n_pos and n_neg:
    total = n_pos + n_neg
    expected = 2.0 * n_pos * n_neg / total + 1.0
    variance = (
      2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - total) / (total**2 * (total - 1.0))
    )
    if variance > 0:
      runs_z = (runs - expected) / math.sqrt(variance)

  regions: dict[str, Any] = {}
  for name, part_q, part in zip(
    ("low_q", "mid_q", "high_q"),
    np.array_split(q, 3),
    np.array_split(residual, 3),
    strict=True,
  ):
    regions[name] = {
      "q_range": [_round(float(part_q[0])), _round(float(part_q[-1]))],
      "mean_residual": _round(float(np.mean(part)), 3),
      "rms_residual": _round(float(np.sqrt(np.mean(part**2))), 3),
    }
  worst = int(np.argmax(np.abs(residual)))
  return {
    "residual_kind": kind,
    "points": int(q.size),
    "runs": runs,
    "runs_z": None if runs_z is None else _round(runs_z, 3),
    "regions": regions,
    "largest_residual": {
      "q": _round(float(q[worst])),
      "value": _round(float(residual[worst]), 3),
    },
  }


def assess_fit(
  fit_result: dict[str, Any],
  residuals: dict[str, Any] | None,
  *,
  has_uncertainties: bool,
  conditional_on: list[str] | None = None,
) -> dict[str, Any]:
  """Turn fit statistics into a verdict, a list of issues, and next actions.

  ``conditional_on`` names parameters that were fixed at their own fitted
  values, which leaves the fit unchanged but shrinks the other uncertainties.
  """
  issues: list[str] = []
  suggestions: list[str] = []
  if conditional_on:
    names = ", ".join(conditional_on)
    issues.append(
      f"Uncertainties are conditional: {names} held at fitted values, not "
      "independently known ones. The reported standard errors understate the "
      "real uncertainty and hide correlations with those parameters."
    )
    suggestions.append(
      f"Let {names} vary again (set-sans-parameter vary=true) for honest "
      "uncertainties, unless their values are known independently."
    )
  reduced = fit_result.get("reduced_chisq")
  reduced = float(reduced) if isinstance(reduced, (int, float)) else None

  if reduced is None or not math.isfinite(reduced):
    verdict = "unknown"
    issues.append("The fit did not report a finite reduced chi-squared.")
  elif not has_uncertainties:
    verdict = "uncalibrated"
    issues.append(
      "The data has no intensity uncertainties, so reduced chi-squared "
      f"({_round(reduced)}) cannot be compared with 1. Judge the fit from the "
      "residual pattern and the plot."
    )
  elif reduced <= REDUCED_CHISQ_GOOD:
    verdict = "good"
  elif reduced <= REDUCED_CHISQ_ACCEPTABLE:
    verdict = "acceptable"
  else:
    verdict = "poor"

  if has_uncertainties and reduced is not None and reduced < REDUCED_CHISQ_OVERFIT:
    issues.append(
      f"Reduced chi-squared {_round(reduced)} is well below 1: uncertainties "
      "may be overestimated or the model has more freedom than the data needs."
    )

  on_bounds: list[str] = []
  for entry in fit_result.get("on_bounds") or []:
    # IQuana reports (name, "min" | "max") pairs.
    if isinstance(entry, (list, tuple)) and entry:
      parts = [str(part) for part in entry]
      side = f" (at its {parts[1]})" if len(parts) > 1 else ""
      on_bounds.append(f"{parts[0]}{side}")
    else:
      on_bounds.append(str(entry))
  if on_bounds:
    names = ", ".join(item.split(" ")[0] for item in on_bounds)
    issues.append("Parameters resting on a bound: " + ", ".join(on_bounds) + ".")
    suggestions.append(
      f"Widen the bounds of {names} with set-sans-parameter, or fix them if "
      "the bound is physical."
    )

  correlated = fit_result.get("strong_correlations") or []
  if correlated:
    shown = "; ".join(
      f"{' & '.join(pair['parameters'])} ({pair['correlation']:+.2f})"
      for pair in correlated[:4]
    )
    more = f"; and {len(correlated) - 4} more" if len(correlated) > 4 else ""
    issues.append(f"Strongly correlated parameters: {shown}{more}.")
    suggestions.append(
      "The data cannot separate the correlated parameters. If one of a pair "
      "is known independently (composition, another measurement), set it to "
      "that value with set-sans-parameter value=... vary=false. Never fix a "
      "parameter at its fitted value: the fit stays the same and the other "
      "uncertainties become misleadingly small. Otherwise report the "
      "correlation as a limitation."
    )

  parameters = fit_result.get("parameters") or {}
  poorly: list[str] = []
  undetermined: list[str] = []
  for name, entry in parameters.items():
    if not isinstance(entry, dict) or entry.get("fixed"):
      continue
    value, error = entry.get("value"), entry.get("stderr")
    if not isinstance(value, (int, float)) or not isinstance(error, (int, float)):
      continue
    if name == "background" and abs(error) < UNDETERMINED_STDERR:
      # A background near zero has a large relative error by construction.
      continue
    if is_undetermined(value, error):
      undetermined.append(str(name))
    elif value != 0 and abs(error) > POORLY_DETERMINED_RELATIVE_ERROR * abs(value):
      poorly.append(str(name))
  if undetermined:
    issues.append(
      "Undetermined parameters (the data does not constrain them): "
      + ", ".join(undetermined)
      + "."
    )
    suggestions.append(
      "Fix "
      + ", ".join(undetermined)
      + " at a physically sensible value with set-sans-parameter vary=false; "
      "the fit is insensitive to them, so this does not bias the others."
    )
  if poorly:
    issues.append("Poorly determined parameters: " + ", ".join(poorly) + ".")
    suggestions.append(
      "The uncertainty of " + ", ".join(poorly) + " exceeds half the value. "
      "Set them to independently known values if you have them, or choose a "
      "simpler model; do not fix them at their fitted values."
    )

  if residuals and residuals.get("runs_z") is not None:
    if residuals["runs_z"] < RUNS_Z_SYSTEMATIC:
      issues.append(
        "Residuals are systematic, not random (runs-test z = "
        f"{residuals['runs_z']}): the model shape does not match the data."
      )
      for name, label in (
        ("low_q", "low Q"),
        ("mid_q", "intermediate Q"),
        ("high_q", "high Q"),
      ):
        region = residuals["regions"][name]
        if abs(region["mean_residual"]) < REGION_BIAS_SIGMA:
          continue
        direction = "below" if region["mean_residual"] > 0 else "above"
        issues.append(
          f"The model lies {direction} the data at {label} "
          f"(Q {region['q_range'][0]} to {region['q_range'][1]})."
        )
        if name == "low_q":
          suggestions.append(
            "Low-Q misfit: add a structure factor, try a larger or more "
            "elongated shape, or raise q_min with set-sans-q-range."
          )
        elif name == "high_q":
          suggestions.append(
            "High-Q misfit: let background vary, or lower q_max with set-sans-q-range."
          )
        else:
          suggestions.append(
            "Intermediate-Q misfit: add polydispersity with "
            "set-sans-polydispersity or try a different shape."
          )
      if verdict == "good":
        verdict = "acceptable"

  if verdict == "poor" and not suggestions:
    suggestions.append(
      "Try another model from suggest-sans-models, or let more parameters vary."
    )
  if verdict == "good" and conditional_on:
    verdict = "acceptable"
  if verdict == "good" and not issues:
    suggestions.append(
      "The fit is statistically consistent with the data. Check the "
      "parameters are physically sensible for the sample before reporting."
    )

  return {
    "verdict": verdict,
    "reduced_chisq": None if reduced is None else _round(reduced),
    "issues": list(dict.fromkeys(issues)),
    "suggestions": list(dict.fromkeys(suggestions)),
  }
