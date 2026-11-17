"""Model suggestions from the Sassafras curve library ("Guess").

Sassafras ranks sasmodels models for a dataset by matching it, on shape only,
against about 1.3 million simulated curves (SAS-55M-20k plus an S(Q)
sub-library), then derives starting values from the closest curves.  This
module calls that implementation directly; it adds no ranking logic of its own.

Sassafras and its curve library are optional.  Install them with::

    pip install "sans-pilot[guess]"
    python -m sassafras.library fetch      # ~290 MB, once
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from sans_pilot.diagnostics import active_arrays

CAUTION = (
  "A low chi-squared means a model CAN describe the curve shape, not that it "
  "is the right description of the sample. Check each suggestion against "
  "what is known about the sample."
)
ASSUMED_RELATIVE_UNCERTAINTY = 0.05
INSTALL_HINT = (
  'Install with: pip install "sans-pilot[guess]" and then '
  "python -m sassafras.library fetch"
)


class GuessUnavailable(RuntimeError):
  """Sassafras or its curve library is not installed."""


def guess_status() -> dict[str, Any]:
  """Report whether the Guess backend and its curve library are usable."""
  if os.environ.get("SANS_PILOT_GUESS", "").strip().lower() in {"off", "0", "false"}:
    return {"available": False, "reason": "Guess is disabled (SANS_PILOT_GUESS=off)."}
  try:
    from sassafras.library import Library  # pyright: ignore[reportMissingImports]
  except ImportError:
    return {
      "available": False,
      "reason": "The sassafras package is not installed. " + INSTALL_HINT,
    }
  folders = [
    str(folder)
    for folder in (
      Library.find(None),
      Library.find(None, "library_sq", "SASSAFRAS_LIBRARY_SQ"),
    )
    if folder
  ]
  if not folders:
    return {
      "available": False,
      "reason": "The Sassafras curve library is not downloaded. " + INSTALL_HINT,
    }
  return {"available": True, "folders": folders}


def _number(value: Any) -> Any:
  if isinstance(value, (float, np.floating)):
    return float(f"{float(value):.6g}") if math.isfinite(value) else None
  if isinstance(value, np.integer):
    return int(value)
  return value


def _sassafras_dataset(data: Any) -> Any:
  """Copy the active points of a sasdata Data1D into a Sassafras dataset."""
  from sassafras.backend import Dataset  # pyright: ignore[reportMissingImports]

  arrays = active_arrays(data)
  q, intensity = arrays["q"], arrays["i"]
  di, dq = arrays.get("di"), arrays.get("dq")
  keep = np.ones(q.size, dtype=bool)
  if di is not None:
    keep &= np.isfinite(di) & (di > 0)
  order = np.argsort(q[keep])

  def pick(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.ascontiguousarray(values[keep][order])

  if int(np.sum(keep)) < 8:
    raise ValueError(
      "Guess needs at least 8 valid points in the active Q range; this dataset "
      f"has {int(np.sum(keep))}."
    )
  sigma = pick(di)
  if sigma is None:
    # Sassafras scores dI-less data with 1/|I| weights; stating that as an
    # explicit relative uncertainty gives the same ranking and lets its
    # evaluation and polish steps run.
    values = np.abs(np.ascontiguousarray(intensity[keep][order]))
    floor = max(float(np.median(values)) * 1e-3, 1e-30)
    sigma = ASSUMED_RELATIVE_UNCERTAINTY * np.maximum(values, floor)
  return Dataset(
    id="d1",
    label="sans-pilot",
    source="upload:sans-pilot",
    q=pick(q),
    i=pick(intensity),
    di=sigma,
    dq=pick(dq),
  )


def _start(values: dict[str, Any]) -> dict[str, Any]:
  return {
    "parameters": {
      str(name): _number(value) for name, value in values["params"].items()
    },
    "polydispersity": {
      str(name): {
        "width": _number(setting["width"]),
        "distribution": setting["type"],
      }
      for name, setting in values["pd"].items()
    },
    "chisq_n": _number(values.get("chisq_n")),
    "polished": bool(values.get("polished")),
  }


def suggest_models(
  data: Any,
  *,
  top: int = 5,
  starting_values: int = 3,
  also: list[tuple[str, str | None]] | None = None,
) -> dict[str, Any]:
  """Rank models for a dataset and attach starting values to the best few.

  Returns ``suggestions`` best first.  Each carries the library score
  (``library_chisq_n``, the chi-squared per point of its closest library curve
  after a scale/background solve); the first ``starting_values`` loadable ones
  also carry ``start``: parameters, polydispersity, and the score after
  Sassafras' short polish.  ``also`` names further (model, structure factor)
  pairs; their library starting values come back under ``requested``.
  """
  status = guess_status()
  if not status["available"]:
    raise GuessUnavailable(status["reason"])

  from sassafras.backend import (  # pyright: ignore[reportMissingImports]
    Session,
    UserError,
  )
  from sassafras.guess import Guesser  # pyright: ignore[reportMissingImports]

  session = Session()
  dataset = _sassafras_dataset(data)
  session.datasets[dataset.id] = dataset
  guesser = Guesser(session)
  try:
    ranked = guesser.suggest(
      {"dataset": dataset.id, "top": int(top), "with_values": False}
    )
  except UserError as exc:
    raise ValueError(str(exc)) from exc

  suggestions: list[dict[str, Any]] = []
  with_values = 0
  for entry in ranked["suggestions"]:
    suggestion: dict[str, Any] = {
      "model": entry["form"],
      "structure_factor": entry["structure_factor"],
      "expression": entry["model"],
      "library_chisq_n": _number(entry["chisq_n"]),
      "relative_score": _number(entry["weight"]),
      "loadable": bool(entry["loadable"]),
    }
    if entry.get("reason"):
      suggestion["note"] = entry["reason"]
    if entry["loadable"] and with_values < starting_values:
      with_values += 1
      try:
        values = guesser.starting_values(
          {
            "dataset": dataset.id,
            "model": entry["form"],
            "structure_factor": entry["structure_factor"],
          }
        )
      except UserError as exc:
        suggestion["note"] = str(exc)
      else:
        suggestion["start"] = _start(values)
        if values.get("note"):
          suggestion["note"] = values["note"]
    suggestions.append(suggestion)

  requested: list[dict[str, Any]] = []
  for model, structure_factor in also or []:
    entry = {
      "model": model,
      "structure_factor": structure_factor,
      "expression": f"{model}@{structure_factor}" if structure_factor else model,
    }
    try:
      values = guesser.starting_values(
        {"dataset": dataset.id, "model": model, "structure_factor": structure_factor}
      )
    except Exception as exc:
      entry["note"] = str(exc)[:200]
    else:
      if values.get("params"):
        entry["start"] = _start(values)
      if values.get("note"):
        entry["note"] = values["note"]
    requested.append(entry)

  result_note = (
    {}
    if "di" in active_arrays(data)
    else {
      "note": (
        "The data has no intensity uncertainties; library scores assume "
        f"{ASSUMED_RELATIVE_UNCERTAINTY:.0%} relative uncertainty."
      )
    }
  )
  return {
    **result_note,
    "source": "Sassafras curve library",
    "library": ranked.get("library"),
    "points_used": int(ranked.get("used_points", dataset.q.size)),
    "suggestions": suggestions,
    "requested": requested,
    "caution": CAUTION,
  }
