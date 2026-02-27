"""
Planning layer: turn the chosen ideation (question + approach) into a
structured, executable plan. The planner sees the ideation and a sample
of the corpus so it can propose concrete steps.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

# Default paths (relative to project root).
_DEFAULT_IDEATION_PATH = "ideation/latest.json"
_DEFAULT_DATA_PATH = "data/clean_dreams.csv"
_PLANNING_OUTPUT_DIR = "planning"
_DEFAULT_DATA_SAMPLE_ROWS = 25

# Truncate long narrative in sample so context stays manageable.
_MAX_NARRATIVE_CHARS = 120


def _project_root() -> str:
    return os.path.abspath(os.environ.get("AGENT_PROJECT_ROOT") or os.getcwd())


def load_ideation(path: str | None = None) -> dict:
    """Load ideation JSON from path (default: ideation/latest.json)."""
    root = _project_root()
    resolved = os.path.normpath(os.path.join(root, path or _DEFAULT_IDEATION_PATH))
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


def load_data_sample(
    csv_path: str | None = None,
    n_rows: int = _DEFAULT_DATA_SAMPLE_ROWS,
    base_path: str | None = None,
) -> str:
    """
    Return the first n_rows of the corpus CSV as a string for the planner.
    Long narratives are truncated to _MAX_NARRATIVE_CHARS chars.
    """
    root = base_path or _project_root()
    resolved = os.path.normpath(os.path.join(root, csv_path or _DEFAULT_DATA_PATH))
    try:
        import csv as csv_module
    except ImportError:
        return "(Could not load csv module)"

    rows: list[list[str]] = []
    with open(resolved, encoding="utf-8", newline="", errors="replace") as f:
        reader = csv_module.reader(f)
        header = next(reader, None)
        if not header:
            return "(Empty file)"
        rows.append(header)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            # Truncate narrative (typically column index 3) if present
            if len(row) > 3 and len(row[3]) > _MAX_NARRATIVE_CHARS:
                row = list(row)
                row[3] = row[3][:_MAX_NARRATIVE_CHARS] + "..."
            rows.append(row)

    # Format as a simple table: header then rows, comma-separated or tab for readability
    lines = [",".join(header)]
    for r in rows[1:]:
        # Escape newlines in cells for single-line display
        cells = [c.replace("\n", " ").replace("\r", "") for c in r]
        lines.append(",".join(cells))
    return "\n".join(lines) + f"\n\n(... first {len(rows)-1} rows of corpus; total rows ~16.5k ...)"


PLANNER_SYSTEM = """You are a planning agent for a dream-analysis NLP workflow. You receive:
1) An ideation result: the agreed research question, rationale, and high-level approach.
2) A sample of the dream corpus (first N rows of the CSV) so you see the actual columns and content.

Your task: produce a structured, executable plan. Each step should be concrete enough that a developer (or an executor agent) can implement it. Use the exact column names and file paths you see in the data sample (e.g. dream_id, subject, date, narrative, word_count; data/clean_dreams.csv).

Output a single JSON object with this exact structure (no other text before or after the JSON):
{
  "steps": [
    {
      "id": "1",
      "action": "short_action_name",
      "description": "One clear sentence of what this step does.",
      "inputs": ["list", "of", "input", "paths or names"],
      "outputs": ["list", "of", "output", "paths or names"],
      "output_spec": "Exact format the next step(s) will consume: e.g. 'DataFrame with columns dream_id, narrative, processed_narrative' or 'Pickle: dict with keys lda_model, vectorizer for topic labeling'.",
      "notes": "Edge cases: e.g. 'If preprocessing removes all tokens, keep original text or a placeholder so downstream vectorization never sees empty documents.'"
    }
  ]
}

Rules:
- action: short snake_case label (e.g. load_corpus, run_topic_model, aggregate_by_decade, export_results).
- inputs: file paths or outputs from previous steps (e.g. ["data/clean_dreams.csv"] or ["topic_model", "corpus_df"]).
- outputs: what this step produces (e.g. ["corpus_df"] or ["outputs/topics.csv"]).
- output_spec (required): For each step, state exactly what you write so the NEXT step can use it. E.g. step 3: "Pickle at model path: dict with keys 'lda_model', 'vectorizer'; step 4 will load this and call vectorizer.get_feature_names_out()". Step 2: "DataFrame with columns from input plus 'processed_narrative' (string); no row may have empty processed_narrative for every document—if all tokens removed, keep original narrative or a placeholder so step 3's vectorizer does not get empty vocabulary."
- notes (required for steps that can fail on edge cases): E.g. "Empty vocabulary: ensure preprocessing never yields a corpus where every document has zero terms; use min_df=1 or retain fallback text."
- Order steps so dependencies come first.
- Be specific: real paths (data/clean_dreams.csv), column names (narrative, processed_narrative), and exact pickle/key contracts between steps.
- Topic labels: Include a step that produces interpretable topic labels (e.g. after LDA: extract top N terms per topic from the model and vectorizer). Output something like topic_labels_df with columns topic_id and top_terms (or theme_label). The final visualization step must take this as an input so trajectories and plots can use human-readable theme names.
- Final step (comparison/visualization): (1) Inputs must include the aggregated trajectory data AND the topic labels (e.g. topic_labels_df). (2) Outputs must include: theme_trajectories.csv (columns: life_stage, topic_id, topic_label or top_terms, value, group e.g. diarist/others), valence_trajectories.csv or affect_trajectories.csv (life_stage, metric, value, group), statistical_comparison_results.csv, and plot(s). (3) Plots must use interpretable theme labels on the axes (from topic_labels), not generic 'theme1', 'theme2'. These CSVs are required so the report tool and readers can interpret the trajectory data.
"""


@dataclass
class PlanStep:
    id: str
    action: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    output_spec: str = ""
    notes: str = ""


@dataclass
class PlanResult:
    steps: list[PlanStep]
    agreed_question: str
    rationale: str
    approach: str
    source_ideation: str
    timestamp: str
    data_sample_preview: str = ""  # optional: first/last line of sample for reference


def _parse_plan_json(text: str) -> list[PlanStep] | None:
    """Extract JSON from the planner response (allow markdown code block)."""
    # Try raw JSON first
    text = text.strip()
    # Strip markdown code block if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
        steps = data.get("steps") or []
        return [
            PlanStep(
                id=str(s.get("id", i)),
                action=str(s.get("action", "")),
                description=str(s.get("description", "")),
                inputs=list(s.get("inputs") or []),
                outputs=list(s.get("outputs") or []),
                output_spec=str(s.get("output_spec") or ""),
                notes=str(s.get("notes") or ""),
            )
            for i, s in enumerate(steps, 1)
        ]
    except (json.JSONDecodeError, TypeError):
        return None


def run_planning(
    llm: "object",
    ideation_path: str | None = None,
    data_path: str | None = None,
    data_sample_rows: int = _DEFAULT_DATA_SAMPLE_ROWS,
    base_path: str | None = None,
) -> PlanResult:
    """
    Run the planner: load ideation + data sample, call LLM, return structured plan.
    llm must have .generate(system_prompt: str, user_message: str) -> str.
    """
    base = base_path or _project_root()
    ideation_path = ideation_path or _DEFAULT_IDEATION_PATH
    resolved_ideation = os.path.normpath(os.path.join(base, ideation_path))

    ideation = load_ideation(ideation_path)
    agreed_question = ideation.get("agreed_question", "")
    rationale = ideation.get("rationale", "")
    approach = ideation.get("approach", "")

    data_sample = load_data_sample(
        csv_path=data_path or _DEFAULT_DATA_PATH,
        n_rows=data_sample_rows,
        base_path=base,
    )

    user_message = (
        "IDEATION:\n"
        f"Agreed question: {agreed_question}\n\n"
        f"Rationale: {rationale}\n\n"
        f"Approach:\n{approach}\n\n"
        "DATA SAMPLE (first rows of the corpus CSV):\n"
        "---\n"
        f"{data_sample}\n"
        "---\n\n"
        "Produce the JSON plan with a 'steps' array as specified."
    )

    response = llm.generate(system_prompt=PLANNER_SYSTEM, user_message=user_message)
    steps = _parse_plan_json(response)
    if not steps:
        steps = [
            PlanStep(
                id="1",
                action="parse_failed",
                description="Planner did not return valid JSON. Raw response: " + response[:500],
                inputs=[],
                outputs=[],
            )
        ]

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return PlanResult(
        steps=steps,
        agreed_question=agreed_question,
        rationale=rationale,
        approach=approach,
        source_ideation=resolved_ideation,
        timestamp=timestamp,
        data_sample_preview=data_sample.split("\n")[0] + " ..." if data_sample else "",
    )


def plan_to_dict(result: PlanResult) -> dict:
    """Serialize PlanResult for JSON save."""
    return {
        "agreed_question": result.agreed_question,
        "rationale": result.rationale,
        "approach": result.approach,
        "source_ideation": result.source_ideation,
        "timestamp": result.timestamp,
        "steps": [
            {
                "id": s.id,
                "action": s.action,
                "description": s.description,
                "inputs": s.inputs,
                "outputs": s.outputs,
                "output_spec": s.output_spec,
                "notes": s.notes,
            }
            for s in result.steps
        ],
    }


def format_plan(result: PlanResult) -> str:
    """Human-readable summary of the plan."""
    lines = [
        "--- Plan ---",
        "",
        f"Source ideation: {result.source_ideation}",
        f"Timestamp: {result.timestamp}",
        "",
        "Agreed question:",
        result.agreed_question,
        "",
        "Steps:",
    ]
    for s in result.steps:
        lines.append(f"  {s.id}. [{s.action}] {s.description}")
        if s.inputs:
            lines.append(f"      inputs:  {s.inputs}")
        if s.outputs:
            lines.append(f"      outputs: {s.outputs}")
        if s.output_spec:
            lines.append(f"      output_spec: {s.output_spec}")
        if s.notes:
            lines.append(f"      notes: {s.notes}")
        lines.append("")
    return "\n".join(lines)


def save_plan(
    result: PlanResult,
    output_dir: str | None = None,
    base_path: str | None = None,
) -> list[str]:
    """
    Write the plan to the repo. Timestamped + latest, same pattern as ideation.
    Returns list of absolute paths written.
    """
    base = base_path or _project_root()
    out = os.path.join(base, output_dir or _PLANNING_OUTPUT_DIR)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    timestamp = result.timestamp
    safe_ts = timestamp.replace(":", "-")

    payload = plan_to_dict(result)

    # Timestamped
    ts_json = os.path.join(out, f"{safe_ts}.json")
    with open(ts_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    written.append(os.path.abspath(ts_json))

    ts_md = os.path.join(out, f"{safe_ts}.md")
    with open(ts_md, "w", encoding="utf-8") as f:
        f.write(format_plan(result))
        f.write(f"\n\n---\nSaved at {timestamp} UTC.\n")
    written.append(os.path.abspath(ts_md))

    # Latest
    latest_json = os.path.join(out, "latest.json")
    latest_md = os.path.join(out, "latest.md")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    written.append(os.path.abspath(latest_json))
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(format_plan(result))
        f.write(f"\n\n---\nSaved at {timestamp} UTC.\n")
    written.append(os.path.abspath(latest_md))

    return written


if __name__ == "__main__":
    from src.main import GPTAgent

    llm = GPTAgent()
    result = run_planning(llm, data_sample_rows=_DEFAULT_DATA_SAMPLE_ROWS)
    paths = save_plan(result)
    print(format_plan(result))
    print("\nSaved to:", ", ".join(paths))
    print("Timestamped copy in planning/ for history; use latest.json for executor.")
