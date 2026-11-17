"""A single-file, human-readable HTML report with interactive figures."""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np
from iquana import IQuana

from sans_pilot.artifacts import safe_path_component
from sans_pilot.diagnostics import THRESHOLDS, active_arrays

EXTRA_MARKER = "<!--SANS-PILOT-EXTRA-->"
_STYLE = """
:root { color-scheme: light dark; --fg:#1c1c1c; --bg:#fdfdfc; --muted:#6b6b6b;
  --line:#d9d9d6; --card:#f4f4f1; --good:#1a7f37; --ok:#9a6700; --poor:#b42318; }
@media (prefers-color-scheme: dark) { :root { --fg:#e8e8e6; --bg:#161616;
  --muted:#a0a0a0; --line:#3a3a3a; --card:#202020; --good:#4ac26b; --ok:#d4a72c;
  --poor:#ff7b72; } }
body { font: 15px/1.55 system-ui, sans-serif; color: var(--fg);
  background: var(--bg); max-width: 980px; margin: 2rem auto; padding: 0 1.2rem; }
h1 { font-size: 1.5rem; margin-bottom: .2rem; } h2 { font-size: 1.15rem;
  margin-top: 2.2rem; border-bottom: 1px solid var(--line); padding-bottom: .3rem; }
.sub { color: var(--muted); margin-top: 0; }
.cards { display: flex; flex-wrap: wrap; gap: .8rem; margin: 1.2rem 0; }
.card { background: var(--card); border-radius: 8px; padding: .7rem 1rem;
  min-width: 150px; } .card b { display: block; font-size: 1.15rem; }
.card span { color: var(--muted); font-size: .85rem; }
.good { color: var(--good); } .acceptable, .uncalibrated { color: var(--ok); }
.poor, .unknown { color: var(--poor); }
table { border-collapse: collapse; width: 100%; margin: .6rem 0; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .85rem; }
td.num { font-variant-numeric: tabular-nums; }
tr.selected td { font-weight: 600; }
.note { background: var(--card); border-left: 3px solid var(--ok);
  padding: .6rem .9rem; border-radius: 4px; margin: .8rem 0; }
.warn { border-left-color: var(--poor); }
details { margin: .6rem 0; } summary { cursor: pointer; color: var(--muted); }
.small { color: var(--muted); font-size: .85rem; }
"""


def _e(value: Any) -> str:
  return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int = 4) -> str:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return _e(value if value is not None else "–")
  return _e(f"{value:.{digits}g}")


def _table(
  headers: list[str], rows: list[list[str]], selected: int | None = None
) -> str:
  head = "".join(f"<th>{_e(item)}</th>" for item in headers)
  body = "".join(
    f"<tr{' class=selected' if index == selected else ''}>"
    + "".join(f"<td>{cell}</td>" for cell in row)
    + "</tr>"
    for index, row in enumerate(rows)
  )
  return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _list(items: list[str]) -> str:
  return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def markdown_to_html(text: str) -> str:
  """Render the small Markdown subset language models write in summaries."""

  def inline(value: str) -> str:
    value = _e(value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", value)

  out: list[str] = []
  lines = text.strip().splitlines()
  index = 0
  while index < len(lines):
    line = lines[index].rstrip()
    if not line.strip():
      index += 1
    elif line.lstrip().startswith("|"):
      rows = []
      while index < len(lines) and lines[index].lstrip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
          rows.append(cells)
        index += 1
      if rows:
        head = "".join(f"<th>{inline(cell)}</th>" for cell in rows[0])
        body = "".join(
          "<tr>" + "".join(f"<td>{inline(cell)}</td>" for cell in row) + "</tr>"
          for row in rows[1:]
        )
        out.append(
          f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )
    elif re.match(r"\s*([-*•]|\d+\.)\s+", line):
      items = []
      while index < len(lines) and re.match(r"\s*([-*•]|\d+\.)\s+", lines[index]):
        items.append(re.sub(r"^\s*([-*•]|\d+\.)\s+", "", lines[index]))
        index += 1
      out.append(
        "<ul>" + "".join(f"<li>{inline(item)}</li>" for item in items) + "</ul>"
      )
    elif line.startswith("#"):
      out.append(f"<h3>{inline(line.lstrip('#').strip())}</h3>")
      index += 1
    else:
      paragraph = []
      while (
        index < len(lines)
        and lines[index].strip()
        and not lines[index].lstrip().startswith(("|", "#"))
        and not re.match(r"\s*([-*•]|\d+\.)\s+", lines[index])
      ):
        paragraph.append(lines[index].strip())
        index += 1
      out.append("<p>" + "<br>".join(inline(item) for item in paragraph) + "</p>")
  return "\n".join(out)


def _figure_html(figure: Any, *, first: bool) -> str:
  """Embed a Plotly figure; the first one carries plotly.js so the file works offline."""
  figure.update_layout(margin={"l": 60, "r": 20, "t": 30, "b": 50})
  return figure.to_html(
    full_html=False,
    include_plotlyjs=True if first else False,
    config={"displaylogo": False, "responsive": True},
  )


def _candidates_figure(session: Any, data: Any) -> Any | None:
  """Overlay every fitted candidate on the data."""
  import plotly.graph_objects as go

  candidates = [
    item for item in session.meta.get("candidates") or [] if "error" not in item
  ]
  if len(candidates) < 2:
    return None
  arrays = active_arrays(data)
  order = np.argsort(arrays["q"])
  figure = go.Figure()
  figure.add_trace(
    go.Scatter(
      x=arrays["q"][order],
      y=arrays["i"][order],
      error_y=(
        {"type": "data", "array": arrays["di"][order], "width": 0, "thickness": 1}
        if "di" in arrays
        else None
      ),
      mode="markers",
      name="data",
      marker={"size": 5, "color": "#7a7a7a"},
    )
  )
  drawn = 0
  for candidate in candidates[:6]:
    path = session.candidates_dir / f"{safe_path_component(candidate['model'])}.json"
    if not path.is_file():
      continue
    try:
      fitter = IQuana.load_analysis(str(path))
      theory = np.asarray(fitter.calculate(), dtype=float)
    except Exception:
      continue
    if theory.shape == arrays["index"].shape:
      theory = theory[arrays["index"]]
    if theory.shape != arrays["q"].shape:
      continue
    figure.add_trace(
      go.Scatter(
        x=arrays["q"][order],
        y=theory[order],
        mode="lines",
        name=f"{candidate.get('rank', '')}. {candidate['model']} "
        f"(χ²/dof {_num(candidate.get('reduced_chisq'), 3)})",
        line={"width": 2 if candidate.get("rank") == 1 else 1.2},
      )
    )
    drawn += 1
  if not drawn:
    return None
  figure.update_xaxes(type="log", title="Q (Å⁻¹)")
  figure.update_yaxes(type="log", title="I(Q)")
  figure.update_layout(
    height=460, template="plotly_white", legend={"orientation": "h", "y": -0.2}
  )
  return figure


def build_html_report(
  session: Any,
  fitter: IQuana | None,
  *,
  model: str | None,
  has_fit: bool,
) -> str:
  """Return the session as one self-contained HTML page."""
  meta = session.meta
  source = meta.get("source", {})
  data = meta.get("data", {})
  last_fit = meta.get("last_fit") or {}
  diagnostics = last_fit.get("diagnostics") or {}
  statistics = last_fit.get("statistics") or {}
  verdict = diagnostics.get("verdict") or "no fit"
  name = source.get("file_name") or source.get("name") or "dataset"
  q_range = data.get("actual_q_range") or {}

  parts: list[str] = [
    f"<h1>SANS analysis: {_e(name)}</h1>",
    f"<p class=sub>Session {_e(session.id)} · {_e(data.get('points_active'))} points"
    f" · Q {_num(q_range.get('min'))} to {_num(q_range.get('max'))} Å⁻¹</p>",
    "<div class=cards>",
    f"<div class=card><span>Selected model</span><b>{_e(model or 'none')}</b></div>",
    f"<div class=card><span>Verdict</span><b class={_e(verdict.split()[0])}>"
    f"{_e(verdict)}</b></div>",
    f"<div class=card><span>χ²/dof</span><b>{_num(statistics.get('reduced_chisq'))}</b></div>",
    f"<div class=card><span>Free parameters</span><b>{_e(statistics.get('n_free', '–'))}</b></div>",
    "</div>",
  ]

  pending = meta.get("pending_changes") or []
  if pending:
    parts.append(
      "<div class='note warn'><b>Settings changed after this fit and not "
      "fitted</b> (results below do not include them):" + _list(pending) + "</div>"
    )
  conditional = last_fit.get("conditional_on") or []
  if conditional:
    parts.append(
      "<div class='note warn'><b>Uncertainties are conditional.</b> "
      + _e(", ".join(conditional))
      + " were held at their own fitted values, so the standard errors below "
      "understate the real uncertainty.</div>"
    )

  first_figure = True
  if has_fit and fitter is not None:
    parts.append("<h2>Fit</h2>")
    try:
      figure = fitter.plot_results(show_residuals=True, log_scale=True, show=False)
      figure.update_layout(height=560)
      parts.append(_figure_html(figure, first=first_figure))
      first_figure = False
    except Exception as exc:
      parts.append(f"<p class=small>The fit figure could not be drawn: {_e(exc)}</p>")

    fitted = last_fit.get("fitted") or {}
    parts.append("<h2>Fitted parameters</h2>")
    parts.append(
      _table(
        ["Parameter", "Value", "Std. error", "Relative"],
        [
          [
            _e(key),
            _num(entry.get("value"), 5),
            _num(entry.get("stderr"), 3),
            (
              _num(100 * abs(entry["stderr"] / entry["value"]), 2) + " %"
              if isinstance(entry.get("stderr"), (int, float))
              and isinstance(entry.get("value"), (int, float))
              and entry["value"]
              else "–"
            ),
          ]
          for key, entry in fitted.items()
        ],
      )
    )
    fixed = last_fit.get("fixed") or {}
    if fixed:
      parts.append(
        "<details><summary>Fixed parameters</summary>"
        + _table(
          ["Parameter", "Value"],
          [[_e(key), _num(value, 6)] for key, value in fixed.items()],
        )
        + "</details>"
      )
    fixed_later = meta.get("fixed_after_fit") or {}
    if fixed_later:
      parts.append(
        "<p><b>Fixed after having been fitted</b></p>"
        + _table(
          ["Parameter", "Fixed at", "Previously fitted"],
          [
            [
              _e(key),
              _num(entry.get("value"), 6),
              f"{_num(entry.get('fitted_value'), 6)} ± "
              f"{_num(entry.get('fitted_stderr'), 3)}",
            ]
            for key, entry in fixed_later.items()
          ],
        )
      )
    known = meta.get("fixed_parameters") or {}
    if known:
      parts.append(
        "<p class=small>Supplied as independently known: "
        + _e(", ".join(f"{key} = {value:g}" for key, value in known.items()))
        + "</p>"
      )

    parts.append("<h2>Diagnostics</h2>")
    if diagnostics.get("issues"):
      parts.append("<b>Issues</b>" + _list(diagnostics["issues"]))
    else:
      parts.append("<p>No issues were detected.</p>")
    if diagnostics.get("suggestions"):
      parts.append("<b>Suggestions</b>" + _list(diagnostics["suggestions"]))
    parts.append(
      "<p class=small>Verdict thresholds are heuristic: "
      + _e("; ".join(f"{key}: {value}" for key, value in THRESHOLDS.items()))
      + ".</p>"
    )

  candidates = meta.get("candidates") or []
  if candidates:
    parts.append("<h2>Models compared</h2>")
    try:
      figure = _candidates_figure(session, fitter.data) if fitter is not None else None
    except Exception:
      figure = None
    if figure is not None:
      parts.append(_figure_html(figure, first=first_figure))
      first_figure = False
    selected_row = next(
      (index for index, item in enumerate(candidates) if item.get("model") == model),
      None,
    )
    parts.append(
      _table(
        ["Rank", "Model", "χ²/dof", "BIC", "Free", "Verdict", "Origin"],
        [
          [
            _e(item.get("rank", "–")),
            _e(item.get("model")),
            _num(item.get("reduced_chisq")),
            _num(item.get("bic"), 5),
            _e(item.get("n_free", "–")),
            f"<span class={_e(str(item.get('verdict') or 'poor').split()[0])}>"
            f"{_e(item.get('verdict') or item.get('error'))}</span>",
            _e(item.get("origin", "")),
          ]
          for item in candidates
        ],
        selected=selected_row,
      )
    )
    parts.append(
      "<p class=small>Ranked by BIC, which penalises extra free parameters. A "
      "low χ² means a model <i>can</i> describe the curve, not that it is the "
      "right description of the sample.</p>"
    )

  features = meta.get("features") or {}
  if features.get("hints"):
    parts.append("<h2>Curve shape</h2>" + _list(features["hints"]))

  parts.append(EXTRA_MARKER)

  history = meta.get("history") or []
  if history:
    steps = [
      entry["action"]
      + (
        " ("
        + ", ".join(f"{key}={value}" for key, value in entry.items() if key != "action")
        + ")"
        if len(entry) > 1
        else ""
      )
      for entry in history
    ]
    parts.append(
      "<details><summary>Steps taken</summary><ol>"
      + "".join(f"<li>{_e(step)}</li>" for step in steps)
      + "</ol></details>"
    )

  return (
    "<!doctype html><html lang=en><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width, initial-scale=1'>"
    f"<title>SANS analysis: {_e(name)}</title><style>{_STYLE}</style></head><body>"
    + "\n".join(parts)
    + "</body></html>"
  )


def series_markdown(summary: dict[str, Any], fit_report: str) -> str:
  """Write a joint series analysis as Markdown."""
  fit = summary["selected_fit"]
  statistics = fit["statistics"]
  lines = [
    f"# SANS series analysis: {len(summary['files'])} curves",
    "",
    f"- Selected model: `{summary['selected_model']}`",
    f"- Verdict: **{fit['verdict']}** (joint χ²/dof {statistics['reduced_chisq']})",
    f"- Reference curve for model suggestions: {summary['reference_file']}",
  ]
  if summary.get("fixed_parameters"):
    lines.append(f"- Known values (all curves): {summary['fixed_parameters']}")
  if summary.get("note"):
    lines.append(f"- {summary['note']}")
  lines += [
    "",
    "## Shared parameters",
    "",
    "| Parameter | Value | Std. error |",
    "|---|---|---|",
    *[
      f"| {name} | {row['value']} | {row['stderr']} |"
      for name, row in fit["shared"].items()
    ],
    "",
    "## Curves",
    "",
    "| File | Known values | Background | χ²/N |",
    "|---|---|---|---|",
    *[
      f"| {item['file']} | "
      + (", ".join(f"{k}={v:g}" for k, v in item["known"].items()) or "-")
      + f" | {item.get('background', '-')} | {item.get('reduced_chisq')} |"
      for item in fit["datasets"]
    ],
    "",
    "## Models compared (joint fits)",
    "",
    "| Rank | Model | χ²/dof | BIC | Free | Worst curve χ²/N | Origin |",
    "|---|---|---|---|---|---|---|",
    *[
      "| {rank} | {model} | {chi} | {bic} | {free} | {worst} | {origin} |".format(
        rank=item.get("rank", "-"),
        model=item["model"],
        chi=item.get("reduced_chisq", item.get("error", "-")),
        bic=item.get("bic", "-"),
        free=item.get("n_free", "-"),
        worst=item.get("worst_curve_reduced_chisq", "-"),
        origin=item.get("origin", ""),
      )
      for item in summary["candidates"]
    ],
    "",
    summary["ranking"],
    "",
    summary["caution"],
  ]
  if fit["issues"]:
    lines += ["", "## Issues", ""] + [f"- {item}" for item in fit["issues"]]
  lines += ["", "## Full fit report", "", fit_report]
  return "\n".join(lines) + "\n"


def build_series_html(summary: dict[str, Any], figure: Any | None) -> str:
  """Return a joint series analysis as one self-contained HTML page."""
  fit = summary["selected_fit"]
  statistics = fit["statistics"]
  verdict = fit["verdict"]
  parts: list[str] = [
    f"<h1>SANS series analysis: {len(summary['files'])} curves</h1>",
    f"<p class=sub>{_e(', '.join(summary['files']))}</p>",
    "<div class=cards>",
    f"<div class=card><span>Selected model</span><b>{_e(summary['selected_model'])}</b></div>",
    f"<div class=card><span>Verdict</span><b class={_e(verdict)}>{_e(verdict)}</b></div>",
    f"<div class=card><span>Joint χ²/dof</span><b>{_num(statistics['reduced_chisq'])}</b></div>",
    f"<div class=card><span>Free parameters</span><b>{_e(statistics['n_free'])}</b></div>",
    f"<div class=card><span>Starts tried</span><b>{_e(fit['starts_tried'])}</b></div>",
    "</div>",
  ]
  if summary.get("note"):
    parts.append(f"<div class=note>{_e(summary['note'])}</div>")

  if figure is not None:
    parts.append("<h2>Joint fit</h2>")
    parts.append(_figure_html(figure, first=True))

  parts.append("<h2>Shared parameters</h2>")
  parts.append(
    _table(
      ["Parameter", "Value", "Std. error", "Relative"],
      [
        [
          _e(name),
          _num(row.get("value"), 6),
          _num(row.get("stderr"), 3),
          (
            _num(100 * abs(row["stderr"] / row["value"]), 2) + " %"
            if isinstance(row.get("stderr"), (int, float))
            and isinstance(row.get("value"), (int, float))
            and row["value"]
            else "–"
          ),
        ]
        for name, row in fit["shared"].items()
      ],
    )
  )
  if fit.get("free_slds"):
    parts.append(
      "<p class=small>Fitted from the contrast variation: "
      + _e(", ".join(fit["free_slds"]))
      + ".</p>"
    )
  if summary.get("fixed_parameters"):
    parts.append(
      "<p class=small>Supplied as known for all curves: "
      + _e(", ".join(f"{k} = {v:g}" for k, v in summary["fixed_parameters"].items()))
      + "</p>"
    )

  parts.append("<h2>Curves</h2>")
  parts.append(
    _table(
      ["File", "Known values", "Background", "Points", "χ²/N"],
      [
        [
          _e(item["file"]),
          _e(", ".join(f"{k} = {v:g}" for k, v in item["known"].items()) or "–"),
          _num(item.get("background")),
          _e(item.get("points")),
          _num(item.get("reduced_chisq")),
        ]
        for item in fit["datasets"]
      ],
    )
  )

  parts.append("<h2>Diagnostics</h2>")
  parts.append(
    _list(fit["issues"]) if fit["issues"] else "<p>No issues were detected.</p>"
  )

  parts.append("<h2>Models compared (joint fits)</h2>")
  parts.append(
    _table(
      ["Rank", "Model", "χ²/dof", "BIC", "Free", "Worst curve χ²/N", "Origin"],
      [
        [
          _e(item.get("rank", "–")),
          _e(item["model"]),
          _num(item.get("reduced_chisq")) if "error" not in item else _e(item["error"]),
          _num(item.get("bic"), 6),
          _e(item.get("n_free", "–")),
          _num(item.get("worst_curve_reduced_chisq")),
          _e(item.get("origin", "")),
        ]
        for item in summary["candidates"]
      ],
      selected=next(
        (
          index
          for index, item in enumerate(summary["candidates"])
          if item["model"] == summary["selected_model"]
        ),
        None,
      ),
    )
  )
  parts.append(f"<p class=small>{_e(summary['ranking'])} {_e(summary['caution'])}</p>")
  parts.append(EXTRA_MARKER)
  return (
    "<!doctype html><html lang=en><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width, initial-scale=1'>"
    f"<title>SANS series analysis</title><style>{_STYLE}</style></head><body>"
    + "\n".join(parts)
    + "</body></html>"
  )
