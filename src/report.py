"""
Report layer: given a successful run, have an NLP expert describe the analysis methods
and a dream/psychology expert interpret the outputs and write conclusions. Saves report.md in the run dir.
"""

import json
import os

from src.evaluator import get_latest_run_dir
from src.executor import _project_root

_RUNS_DIR = "outputs"
_REPORT_FILENAME = "report.md"
_MAX_CSV_CHARS = 6000


def _list_output_files(run_dir: str) -> list[str]:
    """Return non-script, non-metadata files in run_dir (e.g. *.csv, *.png)."""
    out = []
    for f in os.listdir(run_dir):
        if f.startswith("step_") and f.endswith(".py"):
            continue
        if f in ("run_result.json", "fix_log.txt", "plan_snapshot.json", _REPORT_FILENAME):
            continue
        path = os.path.join(run_dir, f)
        if os.path.isfile(path):
            out.append(f)
    return sorted(out)


def _read_text_outputs(run_dir: str, file_list: list[str]) -> str:
    """Read CSV (and other text) files; return concatenated content, truncated."""
    parts = []
    for f in file_list:
        if not f.endswith(".csv") and not f.endswith(".txt"):
            continue
        path = os.path.join(run_dir, f)
        try:
            with open(path, encoding="utf-8", errors="replace") as fp:
                content = fp.read()
            if len(content) > _MAX_CSV_CHARS:
                content = content[:_MAX_CSV_CHARS] + "\n... (truncated)"
            parts.append(f"--- {f} ---\n{content}")
        except OSError:
            pass
    return "\n\n".join(parts) if parts else "(No CSV/text outputs read)"


NLP_EXPERT_SYSTEM = """You are an NLP expert. You will be given the research question and the list of pipeline steps (action names and descriptions) that were executed in a dream-analysis run.

Your task: Write a single section titled "## Analysis methods" in clear, non-technical human language. Explain what was done in the analysis: data loading, preprocessing (e.g. tokenization, lemmatization), topic modeling (e.g. LDA), how topics were labeled, sentiment/affect scoring, grouping by life epoch, and aggregation/visualization. Use plain language so a dream researcher or psychologist can understand the pipeline without needing to read code. Keep it concise (about one short paragraph per major phase). Output only the markdown section, no preamble."""

DREAM_EXPERT_SYSTEM = """You are a dream psychology expert. You will be given the research question, a list of output files produced by the analysis (e.g. trajectory plots, statistical comparison CSV), and optionally the contents of CSV/text outputs.

Your task: Write a single section titled "## Interpretation and conclusions" in clear human language. Interpret what the outputs suggest about the diarist and comparison subjects: how themes and emotional tone evolve across life epochs, any notable patterns or contrasts, and limitations (e.g. sample size, exploratory flags). Base your interpretation on the file names and any provided CSV content; you do not see the actual plot images, so describe what such outputs typically show and what the statistics suggest. Keep it to a few paragraphs. Output only the markdown section, no preamble."""


def generate_report(
    llm: "object",
    run_dir: str | None = None,
    base_path: str | None = None,
) -> str:
    """
    Load run context (plan, run_result, output files). Have NLP expert write methods section,
    dream expert write interpretation section. Combine and save as run_dir/report.md.
    Returns a short message with the report path or an error.
    """
    root = base_path or _project_root()
    run_dir = run_dir or get_latest_run_dir(root)
    if not run_dir or not os.path.isdir(run_dir):
        return "No run directory found. Run the executor first and use a successful run."
    if not os.path.isabs(run_dir):
        run_dir = os.path.normpath(os.path.join(root, run_dir))

    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.isfile(run_result_path):
        return "run_result.json not found in run dir."
    with open(run_result_path, encoding="utf-8") as f:
        run_result = json.load(f)
    if not run_result.get("success"):
        return "Run did not complete successfully. Generate a report only for a successful run (or run 'report' after re_run completes)."

    plan_path = os.path.join(run_dir, "plan_snapshot.json")
    if not os.path.isfile(plan_path):
        return "plan_snapshot.json not found in run dir; cannot describe methods."
    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    agreed_question = run_result.get("agreed_question") or plan.get("agreed_question", "")
    steps = plan.get("steps") or []
    step_descriptions = "\n".join(
        f"  Step {s.get('id')} [{s.get('action')}]: {s.get('description')}"
        for s in steps
    )

    output_files = _list_output_files(run_dir)
    text_content = _read_text_outputs(run_dir, output_files)

    # NLP expert: methods section
    user_nlp = (
        f"Research question: {agreed_question}\n\n"
        f"Pipeline steps executed:\n{step_descriptions}\n\n"
        "Write the '## Analysis methods' section."
    )
    methods_section = llm.generate(system_prompt=NLP_EXPERT_SYSTEM, user_message=user_nlp)
    methods_section = methods_section.strip()
    if not methods_section.startswith("##"):
        methods_section = "## Analysis methods\n\n" + methods_section

    # Dream/psychology expert: interpretation section
    user_dream = (
        f"Research question: {agreed_question}\n\n"
        f"Output files from the run: {', '.join(output_files)}\n\n"
        f"Contents of CSV/text outputs (for context):\n{text_content}\n\n"
        "Write the '## Interpretation and conclusions' section."
    )
    interp_section = llm.generate(system_prompt=DREAM_EXPERT_SYSTEM, user_message=user_dream)
    interp_section = interp_section.strip()
    if not interp_section.startswith("##"):
        interp_section = "## Interpretation and conclusions\n\n" + interp_section

    # Combine report
    report_lines = [
        "# Dream analysis report",
        "",
        f"**Research question:** {agreed_question}",
        "",
        f"*Run: {os.path.basename(run_dir)}*",
        "",
        "---",
        "",
        methods_section,
        "",
        "---",
        "",
        interp_section,
        "",
        "---",
        "",
        "## Output files",
        "",
    ]
    for f in output_files:
        report_lines.append(f"- {f}")
    report_lines.append("")

    report_path = os.path.join(run_dir, _REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return f"Report written to {report_path}\nSections: Analysis methods (NLP expert), Interpretation and conclusions (dream psychology expert), Output files."
