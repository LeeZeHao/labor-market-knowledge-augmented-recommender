"""Render ``ood_case_study.py`` JSONL output as a standalone HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OOD_COMPANIES = Path(
    "/home/leeze/career_experiments_2/data/ood_companies.parquet"
)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {path} is not a JSON object"
                )
            records.append(record)
    return records


def read_ood_companies(path: Path, column: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"OOD-company file not found: {path}")
    frame = pd.read_parquet(path, columns=[column])
    return {str(value) for value in frame[column].dropna().tolist()}


def company_chip(value: Any, ood_companies: set[str]) -> str:
    company = str(value)
    is_ood = company in ood_companies
    classes = "chip company ood-company" if is_ood else "chip company"
    badge = '<span class="ood-badge">OOD</span>' if is_ood else ""
    return f'<span class="{classes}">{escape(company)}{badge}</span>'


def render_sequence(
    values: Iterable[Any],
    *,
    company_values: bool,
    ood_companies: set[str],
) -> str:
    values = list(values)
    if not values:
        return '<span class="muted">Empty</span>'
    if company_values:
        return '<div class="chips">' + "".join(
            company_chip(value, ood_companies) for value in values
        ) + "</div>"
    if len(values) > 100 and all(
        isinstance(value, (int, float)) for value in values
    ):
        preview = ", ".join(escape(value) for value in values[:8])
        full = escape(json.dumps(values, ensure_ascii=False))
        return (
            f'<details><summary>{len(values):,} numeric values '
            f'(preview: {preview}, …)</summary><pre>{full}</pre></details>'
        )
    return "<ul>" + "".join(
        f"<li>{render_value(value, company_values=company_values, ood_companies=ood_companies)}</li>"
        for value in values
    ) + "</ul>"


def render_value(
    value: Any,
    *,
    company_values: bool,
    ood_companies: set[str],
) -> str:
    if value is None:
        return '<span class="muted">null</span>'
    if isinstance(value, list):
        return render_sequence(
            value,
            company_values=company_values,
            ood_companies=ood_companies,
        )
    if isinstance(value, dict):
        return render_mapping(value, ood_companies)
    if company_values:
        return company_chip(value, ood_companies)
    if isinstance(value, str) and len(value) > 500:
        return (
            f'<details><summary>Show text ({len(value):,} characters)</summary>'
            f'<div class="long-text">{escape(value)}</div></details>'
        )
    return escape(value)


def is_company_field(field: str) -> bool:
    field = field.lower()
    return field in {
        "company",
        "companies",
        "target_company",
        "pos_company_id",
        "history_company_ids",
    }


def render_mapping(
    mapping: dict[str, Any], ood_companies: set[str]
) -> str:
    rows = []
    for key, value in mapping.items():
        rows.append(
            "<tr>"
            f'<th scope="row">{escape(key)}</th>'
            f"<td>{render_value(value, company_values=is_company_field(key), ood_companies=ood_companies)}</td>"
            "</tr>"
        )
    return '<div class="table-wrap"><table>' + "".join(rows) + "</table></div>"


def render_result(
    label: str,
    result: dict[str, Any],
    *,
    company_task: bool,
    ood_companies: set[str],
) -> str:
    correct = bool(result.get("correct_top1"))
    status = "Correct" if correct else "Incorrect"
    css_class = "result correct" if correct else "result incorrect"
    predicted = render_value(
        result.get("predicted_id"),
        company_values=company_task,
        ood_companies=ood_companies,
    )
    truth = render_value(
        result.get("ground_truth_id"),
        company_values=company_task,
        ood_companies=ood_companies,
    )
    predicted_score = float(result.get("predicted_score", float("nan")))
    truth_score = float(result.get("ground_truth_score", float("nan")))
    return f"""
      <section class="{css_class}">
        <div class="result-heading"><h3>{escape(label)}</h3><span>{status}</span></div>
        <dl>
          <dt>Top-1 prediction</dt><dd>{predicted}</dd>
          <dt>Prediction score</dt><dd>{predicted_score:.6f}</dd>
          <dt>Ground truth</dt><dd>{truth}</dd>
          <dt>Ground-truth rank</dt><dd>{escape(result.get('ground_truth_rank'))}</dd>
          <dt>Ground-truth score</dt><dd>{truth_score:.6f}</dd>
        </dl>
      </section>
    """


def render_career_history(
    row: dict[str, Any], ood_companies: set[str]
) -> str:
    """Render aligned past roles, companies, and tenures in one table."""

    roles = row.get("roles") or []
    companies = row.get("companies") or []
    tenures = row.get("tenures") or []
    if not isinstance(roles, list):
        roles = [roles]
    if not isinstance(companies, list):
        companies = [companies]
    if not isinstance(tenures, list):
        tenures = [tenures]
    history_length = max(len(roles), len(companies), len(tenures))
    if history_length == 0:
        return '<p class="muted">No career history is available.</p>'

    rows = []
    for index in range(history_length):
        role = roles[index] if index < len(roles) else None
        company = companies[index] if index < len(companies) else None
        tenure = tenures[index] if index < len(tenures) else None
        company_is_ood = company is not None and str(company) in ood_companies
        status = (
            '<span class="history-status ood-status">OOD company</span>'
            if company_is_ood
            else '<span class="history-status known-status">Not OOD</span>'
        )
        rows.append(
            "<tr>"
            f'<td class="position-number">{index + 1}</td>'
            f"<td>{render_value(role, company_values=False, ood_companies=ood_companies)}</td>"
            f"<td>{render_value(company, company_values=True, ood_companies=ood_companies)}</td>"
            f"<td>{render_value(tenure, company_values=False, ood_companies=ood_companies)}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap career-history"><table>'
        "<thead><tr><th>#</th><th>Past role</th><th>Past company</th>"
        "<th>Tenure</th><th>Company status</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_case(
    record: dict[str, Any],
    case_number: int,
    ood_companies: set[str],
) -> str:
    task = str(record.get("prediction_task", "unknown"))
    company_task = task == "company"
    row = record.get("row", {})
    candidates = record.get("candidate_ids", [])
    target_company = row.get("target_company") if isinstance(row, dict) else None
    target_marker = (
        company_chip(target_company, ood_companies)
        if target_company is not None
        else '<span class="muted">Unavailable</span>'
    )
    candidate_html = render_sequence(
        candidates,
        company_values=company_task,
        ood_companies=ood_companies,
    )
    return f"""
    <article class="case" data-search="{escape(json.dumps(record, ensure_ascii=False).lower())}">
      <header class="case-header">
        <div>
          <span class="eyebrow">Case {case_number}</span>
          <h2>OOD row {escape(record.get('ood_row_index'))}</h2>
        </div>
        <div class="case-meta"><span>{escape(task.title())} prediction</span> Target company: {target_marker}</div>
      </header>
      <div class="results-grid">
        {render_result('DIN · none', record.get('none_result', {}), company_task=company_task, ood_companies=ood_companies)}
        {render_result('DIN · kar_base', record.get('kar_base_result', {}), company_task=company_task, ood_companies=ood_companies)}
        {render_result('DIN · kar_ours', record.get('kar_ours_result', {}), company_task=company_task, ood_companies=ood_companies)}
      </div>
      <section class="history-section">
        <h3>Career history</h3>
        <p class="section-note">Past roles and companies are aligned by position. Companies from the original OOD list are explicitly marked.</p>
        {render_career_history(row, ood_companies) if isinstance(row, dict) else '<p class="muted">History unavailable</p>'}
      </section>
      <details>
        <summary>Source OOD row</summary>
        {render_mapping(row, ood_companies) if isinstance(row, dict) else escape(row)}
      </details>
      <details>
        <summary>All {len(candidates)} candidates</summary>
        {candidate_html}
      </details>
    </article>
    """


def build_html(
    records: list[dict[str, Any]],
    source_path: Path,
    ood_companies: set[str],
) -> str:
    cases = "".join(
        render_case(record, index, ood_companies)
        for index, record in enumerate(records, start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DIN OOD case study</title>
  <style>
    :root {{ --ink:#17212b; --muted:#66717e; --paper:#f5f7f9; --card:#fff;
      --line:#dce2e8; --green:#18794e; --green-bg:#e9f7ef; --red:#b42318;
      --red-bg:#fff0ee; --ood:#8a3ffc; --ood-bg:#f2eaff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 system-ui,-apple-system,sans-serif; }}
    .page-header {{ padding:34px max(24px,calc((100vw - 1200px)/2)); background:#152536; color:white; }}
    .page-header h1 {{ margin:0 0 6px; font-size:30px; }}
    .page-header p {{ margin:4px 0; color:#cbd7e3; }}
    .toolbar {{ position:sticky; top:0; z-index:3; display:flex; gap:14px; align-items:center;
      padding:13px max(24px,calc((100vw - 1200px)/2)); background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); }}
    input {{ width:min(520px,65vw); padding:10px 13px; border:1px solid #aeb9c4; border-radius:8px; font:inherit; }}
    main {{ max-width:1200px; margin:24px auto; padding:0 24px 60px; }}
    .case {{ margin:0 0 26px; padding:22px; background:var(--card); border:1px solid var(--line); border-radius:14px; box-shadow:0 4px 14px #1825330b; }}
    .case-header,.result-heading {{ display:flex; justify-content:space-between; gap:16px; align-items:center; }}
    .case-header h2,.result h3 {{ margin:0; }} .eyebrow {{ color:var(--muted); text-transform:uppercase; font-size:12px; letter-spacing:.08em; }}
    .case-meta {{ text-align:right; }} .case-meta>span {{ padding:5px 8px; border-radius:6px; background:#e8eef5; font-weight:650; }}
    .results-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:20px 0; }}
    .result {{ padding:15px; border-radius:10px; border:1px solid; }}
    .correct {{ background:var(--green-bg); border-color:#9bd5b7; }} .incorrect {{ background:var(--red-bg); border-color:#f1b3ac; }}
    .correct .result-heading span {{ color:var(--green); }} .incorrect .result-heading span {{ color:var(--red); }}
    dl {{ display:grid; grid-template-columns:max-content 1fr; gap:5px 12px; margin:12px 0 0; }} dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }}
    details {{ margin-top:12px; border-top:1px solid var(--line); padding-top:12px; }} summary {{ cursor:pointer; font-weight:700; }}
    .table-wrap {{ overflow:auto; margin-top:10px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px 10px; border-bottom:1px solid #edf0f3; vertical-align:top; text-align:left; }} th {{ width:190px; color:#465361; }}
    .history-section {{ margin:20px 0 14px; padding:16px; background:#f8fafc; border:1px solid var(--line); border-radius:10px; }}
    .history-section h3 {{ margin:0; }} .section-note {{ margin:4px 0 10px; color:var(--muted); }}
    .career-history th {{ width:auto; white-space:nowrap; }} .career-history .position-number {{ width:42px; color:var(--muted); text-align:center; }}
    .history-status {{ display:inline-block; padding:3px 7px; border-radius:999px; font-size:12px; font-weight:700; white-space:nowrap; }}
    .ood-status {{ color:#5721a8; background:var(--ood-bg); border:1px solid #b994ff; }}
    .known-status {{ color:#52606d; background:#edf1f4; border:1px solid #ccd5dc; }}
    ul {{ margin:4px 0; padding-left:22px; }} pre,.long-text {{ white-space:pre-wrap; overflow-wrap:anywhere; max-height:420px; overflow:auto; background:#f7f8fa; padding:12px; border-radius:7px; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:6px; }} .chip {{ display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border:1px solid #cdd5dd; border-radius:999px; background:#f8fafb; }}
    .ood-company {{ background:var(--ood-bg); border-color:#b994ff; color:#5721a8; font-weight:650; box-shadow:inset 3px 0 var(--ood); }}
    .ood-badge {{ padding:1px 5px; border-radius:999px; background:var(--ood); color:white; font-size:10px; letter-spacing:.05em; }}
    .legend {{ display:inline-flex; align-items:center; gap:7px; }} .legend-dot {{ width:12px; height:12px; border-radius:3px; background:var(--ood); }} .muted {{ color:var(--muted); }}
    .empty {{ padding:50px; text-align:center; color:var(--muted); }}
    @media (max-width:850px) {{ .results-grid {{ grid-template-columns:1fr; }} .case-header {{ align-items:flex-start; flex-direction:column; }} .case-meta {{ text-align:left; }} }}
  </style>
</head>
<body>
  <header class="page-header">
    <h1>DIN OOD case study</h1>
    <p>{len(records):,} cases from {escape(source_path.name)}</p>
    <p class="legend"><span class="legend-dot"></span>Purple labels are companies in the original OOD-company list ({len(ood_companies):,} companies).</p>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter by user, role, company, prediction…">
    <span id="visible-count">{len(records):,} cases</span>
  </div>
  <main>{cases if cases else '<div class="empty">No records found in the JSONL file.</div>'}</main>
  <script>
    const search = document.getElementById('search');
    const cases = Array.from(document.querySelectorAll('.case'));
    const count = document.getElementById('visible-count');
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      for (const item of cases) {{
        const show = !query || item.dataset.search.includes(query);
        item.hidden = !show;
        if (show) visible += 1;
      }}
      count.textContent = `${{visible.toLocaleString()}} cases`;
    }});
  </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DIN OOD case-study JSONL into standalone HTML."
    )
    parser.add_argument("input", type=Path, help="Case-study JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path; defaults to the input path with .html suffix.",
    )
    parser.add_argument(
        "--ood_companies",
        type=Path,
        default=DEFAULT_OOD_COMPANIES,
        help="Parquet containing the original OOD company list.",
    )
    parser.add_argument("--company_column", default="company")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (args.output or input_path.with_suffix(".html")).expanduser().resolve()
    records = read_records(input_path)
    ood_companies = read_ood_companies(
        args.ood_companies.expanduser().resolve(), args.company_column
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(records, input_path, ood_companies), encoding="utf-8"
    )
    print(f"Cases rendered: {len(records)}")
    print(f"OOD companies loaded: {len(ood_companies)}")
    print(f"HTML written to: {output_path}")


if __name__ == "__main__":
    main()
