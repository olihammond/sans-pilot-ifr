"""Export a fitted session as a ready-to-run IQuana Jupyter notebook.

The notebook mirrors the notebook export of the Sassafras GUI: it loads a
snapshot of exactly the data that was fitted, sets resolution, weighting, Q
range, model, structure factor, every parameter (value, bounds, vary) and
polydispersity with explicit IQuana calls, checks that it reproduces the
reported chi-squared, then refits, reports and saves.  It is built from the
session's IQuana analysis file, so it carries the state SANS-pilot fitted
rather than a translation of it.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import math
import re
from typing import Any

import numpy as np

METHOD_CALLS = {
  "lm": "fitter.fit(engine='bumps', method='lm')",
  "amoeba": "fitter.fit(engine='bumps', method='amoeba')",
  "de": "fitter.fit(engine='bumps', method='de')",
}
# Parameters IQuana sets itself once the structure factor is chosen.
MANAGED = {"radius_effective_mode"}


def _version(name: str) -> str | None:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return None


def safe_stem(text: str) -> str:
  """A file-name stem from a data file name and a model expression."""
  text = re.sub(
    r"\.(dat|xml|h5|hdf5|nxs|txt|csv|abs|cor|pdh|ses)$", "", text, flags=re.I
  )
  return re.sub(r"[^\w.-]+", "_", text).strip("_") or "data"


def _literal(value: Any) -> str:
  """A Python literal that round-trips the value exactly."""
  if isinstance(value, str):
    if value.lower() in ("infinity", "inf"):
      return "float('inf')"
    if value.lower() in ("-infinity", "-inf"):
      return "-float('inf')"
    return repr(value)
  number = float(value)
  if math.isinf(number):
    return "float('inf')" if number > 0 else "-float('inf')"
  if number == int(number) and abs(number) < 1e15:
    return f"{number:.1f}"
  return repr(number)


def _bound(value: Any) -> str:
  """A readable literal for a fit bound (no need for last-bit precision)."""
  if isinstance(value, str) or math.isinf(float(value)):
    return _literal(value)
  text = f"{float(value):.10g}"
  return text if any(mark in text for mark in ".e") else text + ".0"


def data_csv(analysis: dict[str, Any]) -> str:
  """The fitted data as CSV text: Q (1/Å), I, dI and, when present, dQ."""
  arrays = analysis["data"]["arrays"]
  columns = [arrays["x"], arrays["y"]]
  header = ["Q_invA", "I"]
  dy = arrays.get("dy")
  if dy is not None:
    columns.append(dy)
    header.append("dI")
    dx = arrays.get("dx")
    if dx is not None and np.any(np.asarray(dx, dtype=float) > 0):
      columns.append(dx)
      header.append("dQ")
  rows = [
    ",".join(repr(float(column[index])) for column in columns)
    for index in range(len(columns[0]))
  ]
  return ",".join(header) + "\n" + "\n".join(rows) + "\n"


def _resolution_call(resolution: dict[str, Any]) -> str:
  mode = str(resolution.get("mode") or "none")
  extra = ", ".join(
    f"{key}={_literal(resolution[key])}"
    for key in ("dq_over_q", "slit_length", "slit_width")
    if resolution.get(key) is not None
  )
  comment = {
    "none": "   # the data carry no dQ",
    "data": "   # dQ from the data",
  }.get(mode, "")
  return f"fitter.set_resolution({mode!r}{', ' + extra if extra else ''}){comment}"


def _setup_lines(analysis: dict[str, Any], csv_name: str) -> list[str]:
  lines = [
    "from iquana import IQuana",
    "",
    "fitter = IQuana()",
    f"fitter.load_data({csv_name!r})   # Q (1/Å), I, dI: exactly the points sans-pilot fitted",
    _resolution_call(analysis.get("resolution") or {}),
  ]
  weighting = (analysis.get("weighting") or {}).get("mode") or "dI"
  comment = (
    "   # the data had no usable dI: points weighted by |I|"
    if weighting == "abs"
    else ""
  )
  lines.append(f"fitter.set_weighting({weighting!r}){comment}")
  window = analysis.get("fit_range") or {}
  limits = [
    f"{key}={_literal(window[key])}"
    for key in ("qmin", "qmax")
    if window.get(key) is not None
  ]
  if limits:
    lines.append(f"fitter.set_q_range({', '.join(limits)})")
  return lines


def _model_lines(analysis: dict[str, Any]) -> tuple[list[str], int]:
  """IQuana calls that rebuild the model; returns (lines, free parameter count)."""
  configuration = analysis["configuration"]
  lines = [f"fitter.set_model({configuration['model_name']!r})"]
  structure = configuration.get("structure_factor") or {}
  skip = set(MANAGED)
  if structure.get("name"):
    mode = structure.get("radius_effective_mode") or "unconstrained"
    lines.append(
      f"fitter.set_structure_factor({structure['name']!r}, "
      f"radius_effective_mode={mode!r})"
    )
    if mode == "link_radius":
      skip.add("radius_effective")
  else:
    skip.add("structure_factor_mode")
  lines.append("")
  params = configuration.get("params") or {}
  width = max((len(name) for name in params), default=4) + 2
  free = 0
  for name, row in params.items():
    if name in skip:
      continue
    pad = " " * (width - len(name))
    if row.get("vary"):
      free += 1
      lines.append(
        f"fitter.set_param({name!r},{pad}value={_literal(row['value'])}, "
        f"min={_bound(row['min'])}, max={_bound(row['max'])}, vary=True)"
      )
    else:
      lines.append(
        f"fitter.set_param({name!r},{pad}value={_literal(row['value'])}, vary=False)"
      )
  spread = (configuration.get("polydispersity") or {}).get("params") or {}
  active = {
    name: row
    for name, row in spread.items()
    if float(row.get("pd") or 0.0) > 0 or row.get("vary")
  }
  if active:
    lines += ["", "fitter.enable_polydispersity(True)"]
    for name, row in active.items():
      free += bool(row.get("vary"))
      lines.append(
        f"fitter.set_pd_param({name!r}, pd_width={_literal(row.get('pd') or 0.0)}, "
        f"pd_n={int(row.get('pd_n') or 35)}, "
        f"pd_nsigma={_literal(row.get('pd_nsigma') or 3.0)}, "
        f"pd_type={str(row.get('pd_type') or 'gaussian')!r}, "
        f"vary={bool(row.get('vary'))})"
      )
  links = configuration.get("links") or {}
  if links:
    lines.append("")
    lines += [
      f"fitter.link_params({follower!r}, to={target!r})"
      for follower, target in links.items()
    ]
  lines += ["", "fitter.get_params()"]
  return lines, free


def _code(lines: list[str]) -> dict[str, Any]:
  return {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": "\n".join(lines),
  }


def _markdown(lines: list[str]) -> dict[str, Any]:
  return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def build_notebook(
  analysis: dict[str, Any],
  *,
  title: str,
  stem: str,
  reduced_chisq: float | None = None,
  verdict: str | None = None,
  notes: list[str] | None = None,
) -> dict[str, Any]:
  """Return ``{'notebook': nbformat dict, 'files': {name: text}}``."""
  csv_name = f"{stem}_data.csv"
  model_lines, free = _model_lines(analysis)
  weighting = (analysis.get("weighting") or {}).get("mode") or "dI"
  weights = "np.abs(d.y[sel])" if weighting == "abs" else "d.dy[sel]"
  method = str((analysis.get("result") or {}).get("method") or "lm")
  now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

  heading = [
    f"# {title}",
    "",
    f"<sub>Exported from sans-pilot {_version('sans-pilot') or ''} on {now} "
    f"(IQuana {_version('iquana') or 'not installed'})</sub>",
  ]
  if reduced_chisq is not None:
    heading += [
      "",
      f"sans-pilot result: χ²/dof = {reduced_chisq:.6g}"
      + (f", verdict **{verdict}**" if verdict else "")
      + ". Diagnostics and the comparison with other models are in the "
      "sans-pilot report.",
    ]
  if notes:
    heading += [""] + [f"* {note}" for note in notes]

  check = [
    "import numpy as np",
    "",
    "# the model at the exported values, with the same resolution, weighting and Q window",
    "model_I = fitter.calculate()",
    "d = fitter.data",
    "sel = np.isfinite(model_I)",
    f"chi2 = np.sum(((d.y[sel] - model_I[sel]) / {weights}) ** 2)",
    f"n_free = {free}",
    "print(f'χ²/dof = {chi2 / (sel.sum() - n_free):.6g} over {sel.sum()} points'"
    + (
      f" + '   (sans-pilot: {reduced_chisq:.6g})'" if reduced_chisq is not None else ""
    )
    + ")",
    "fitter.plot_model(show_residuals=True, log_scale=True)",
  ]
  if free:
    fit_intro = [
      "## Refit",
      "",
      "Free parameters and bounds are those sans-pilot used. Change any "
      "`set_param` line above (value, `min`, `max`, `vary`) and rerun from there.",
    ]
    fit_cells = [
      _code(
        [
          f"result = {METHOD_CALLS.get(method, METHOD_CALLS['lm'])}",
          "fitter.plot_results(show_residuals=True, log_scale=True)",
        ]
      ),
      _code(["fitter.get_fit_report()"]),
    ]
  else:
    fit_intro = [
      "## Refit",
      "",
      "Nothing is set to vary: set `vary=True` (with `min`/`max`) above first.",
    ]
    fit_cells = [
      _code(
        [
          "# result = fitter.fit(engine='bumps', method='lm')",
          "# fitter.plot_results(show_residuals=True, log_scale=True)",
        ]
      )
    ]
  save = [
    f"# fitter.save_analysis({stem + '_analysis.json'!r})   # reopen with IQuana.load_analysis",
    f"# fitter.report({stem + '_report.html'!r})",
  ]
  cells = [
    _markdown(heading),
    _code(_setup_lines(analysis, csv_name)),
    _markdown(["## Model and parameters (as fitted by sans-pilot)"]),
    _code(model_lines),
    _markdown(["## Check: reproduce the sans-pilot fit"]),
    _code(check),
    _markdown(fit_intro),
    *fit_cells,
    _markdown(["## Save (optional)"]),
    _code(save),
  ]
  for index, cell in enumerate(cells):
    cell["id"] = f"cell-{index}"
  notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
      "kernelspec": {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
      },
      "language_info": {"name": "python"},
      "sans_pilot": {
        "version": _version("sans-pilot"),
        "exported": now,
        "model": analysis["configuration"]["model_name"],
        "data": csv_name,
      },
    },
    "cells": cells,
  }
  return {"notebook": notebook, "files": {csv_name: data_csv(analysis)}}
