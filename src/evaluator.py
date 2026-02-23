"""
Evaluator: on a failed executor run, inspect all generated steps and run_result,
propose fixes to any step (not just the failed one), apply edits, and record a fix log.
"""

import json
import os
import re
from datetime import datetime

from src.executor import _project_root, run_script_with_auto_install

_RUNS_DIR = "outputs"
_FIX_LOG = "fix_log.txt"
_EDIT_START = re.compile(r"^---EDIT\s+(\S+)\s*$", re.MULTILINE)
_EDIT_END = re.compile(r"^---END---\s*$", re.MULTILINE)


def _normalize_edit_filename(fname: str) -> str:
    """Strip trailing --- so 'step_3.py---' becomes 'step_3.py'."""
    fname = fname.strip()
    if fname.endswith("---"):
        fname = fname[:-3].rstrip()
    return fname


def get_latest_run_dir(base_path: str | None = None) -> str | None:
    """Return the most recent run directory under outputs/, or None if none exist."""
    root = base_path or _project_root()
    runs = os.path.join(root, _RUNS_DIR)
    if not os.path.isdir(runs):
        return None
    dirs = [d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d)) and d.startswith("run_")]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return os.path.join(runs, dirs[0])


def load_run_context(run_dir: str) -> dict:
    """Load run_result.json and all step_*.py file contents for the evaluator."""
    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.isfile(run_result_path):
        return {"run_result": None, "steps": [], "error": "run_result.json not found"}
    with open(run_result_path, encoding="utf-8") as f:
        run_result = json.load(f)

    steps = []
    for s in run_result.get("steps", []):
        step_id = s.get("step_id", "")
        script_path = s.get("script_path", "")
        if not script_path:
            script_path = os.path.join(run_dir, f"step_{step_id}.py")
        if os.path.isfile(script_path):
            with open(script_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        else:
            content = ""
        steps.append({
            "step_id": step_id,
            "action": s.get("action", ""),
            "returncode": s.get("returncode", -1),
            "stdout": (s.get("stdout") or "")[:1500],
            "stderr": (s.get("stderr") or "")[:1500],
            "script_path": script_path,
            "content": content,
        })
    return {"run_result": run_result, "steps": steps}


EVALUATOR_SYSTEM = """You are an evaluator for a failed pipeline run. You will be given:
1) The full run_result.json (which step failed, returncodes, stdout/stderr for each step).
2) The complete source code of every step script (step_1.py through step_N.py).

Your job: identify the root cause of the failure. The failing step might be failing because of a bug in an *earlier* step (e.g. step 4 fails because step 3 did not write a required file). You may fix *any* of the step scripts, not just the one that failed.

Output format (strict):
1) First line must be: SUMMARY: <one-line plain English summary of what you changed and why>.
2) Then for each file you want to replace, use this exact format:
   ---EDIT step_N.py---
   <full new contents of the file, line by line>
   ---END---

Rules:
- Only output edits for step_*.py files that exist in the run (e.g. step_1.py, step_2.py, ...).
- Use ---EDIT filename--- and ---END--- exactly as shown. You can edit one or multiple files.
- Preserve the intent of the pipeline; only fix bugs or add missing outputs (e.g. if step 3 must save a vectorizer for step 4, add that to step 3).
- Write error messages to stderr and exit with sys.exit(1) on failure.
- If you cannot fix the run, output only: SUMMARY: <reason you cannot fix> and no ---EDIT--- blocks.
- CRITICAL: You will be given a FIX_LOG of previous attempts for this run. Do NOT repeat the same approach. If the log shows "min_df", "__empty__", "processed_narrative" fallback, or similar fixes were already tried and the run still fails, infer the real root cause and try something different (e.g. different column, different vectorizer settings, or fix an earlier step that produces bad input)."""


def _parse_edits(response: str) -> tuple[list[tuple[str, str]], str]:
    """Parse SUMMARY and ---EDIT filename--- ... ---END--- blocks. Returns ([(filename, content), ...], summary)."""
    summary = ""
    lines = response.strip().split("\n")
    if lines and lines[0].strip().upper().startswith("SUMMARY:"):
        summary = lines[0].strip()
        lines = lines[1:]

    edits: list[tuple[str, str]] = []
    text = "\n".join(lines)
    pos = 0
    while True:
        m = _EDIT_START.search(text, pos)
        if not m:
            break
        fname = _normalize_edit_filename(m.group(1))
        if not fname.endswith(".py") or ".." in fname or "/" in fname or "\\" in fname:
            pos = m.end()
            continue
        start = m.end()
        end_m = _EDIT_END.search(text, start)
        if not end_m:
            break
        content = text[start : end_m.start()].rstrip()
        edits.append((fname, content))
        pos = end_m.end()
    return edits, summary


def _apply_edits(run_dir: str, edits: list[tuple[str, str]]) -> list[str]:
    """Write edits to run_dir. Only allow step_*.py. Returns list of paths written."""
    run_dir = os.path.abspath(run_dir)
    written = []
    for fname, content in edits:
        base = os.path.basename(fname)
        if not base.startswith("step_") or not base.endswith(".py"):
            continue
        if ".." in base:
            continue
        path = os.path.join(run_dir, base)
        if not os.path.abspath(path).startswith(run_dir):
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        written.append(path)
    return written


def _append_fix_log(run_dir: str, summary: str, files_changed: list[str]) -> str:
    """Append an entry to run_dir/fix_log.txt. Returns path to fix_log."""
    path = os.path.join(run_dir, _FIX_LOG)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {ts} ===\n")
        f.write(f"SUMMARY: {summary}\n")
        f.write(f"FILES CHANGED: {', '.join(os.path.basename(p) for p in files_changed)}\n")
    return path


def get_from_step_from_fix_log(run_dir: str) -> int | None:
    """
    Parse fix_log.txt in run_dir; return the minimum step number from the last
    FILES CHANGED line (e.g. step_3.py, step_4.py -> 3). Return None if no log or no steps.
    """
    path = os.path.join(run_dir, _FIX_LOG)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Last FILES CHANGED line
    matches = list(re.finditer(r"FILES CHANGED:\s*(.+)", content))
    if not matches:
        return None
    line = matches[-1].group(1).strip()
    if not line:
        return None
    step_nums = []
    for part in re.split(r"[\s,]+", line):
        part = part.strip()
        m = re.match(r"step_(\d+)\.py", part)
        if m:
            step_nums.append(int(m.group(1)))
    return min(step_nums) if step_nums else None


def get_fix_log_recent(run_dir: str, max_entries: int = 15) -> str:
    """Return the last max_entries block(s) from fix_log.txt for context. Returns '' if no log."""
    path = os.path.join(run_dir, _FIX_LOG)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    blocks = re.split(r"\n=== ", content)
    # First segment may be empty or preamble
    entries = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block or block.startswith("==="):
            continue
        entries.append(block)
    recent = entries[-max_entries:] if len(entries) > max_entries else entries
    if not recent:
        return ""
    return "FIX_LOG (previous attempts – do not repeat these approaches):\n\n" + "\n\n".join(recent)


def evaluate_and_fix(
    llm: "object",
    run_dir: str | None = None,
    base_path: str | None = None,
) -> str:
    """
    Load run context, ask the LLM to propose fixes to any steps, apply edits, and record in fix_log.
    Returns a human-readable summary of what was done.
    """
    root = base_path or _project_root()
    run_dir = run_dir or get_latest_run_dir(root)
    if not run_dir or not os.path.isdir(run_dir):
        return "No run directory found. Run the executor first, then run evaluate after a failure."
    if not os.path.isabs(run_dir):
        run_dir = os.path.normpath(os.path.join(root, run_dir))
    if not os.path.isdir(run_dir):
        return f"Run directory not found: {run_dir}"

    ctx = load_run_context(run_dir)
    if ctx.get("error"):
        return f"Could not load run context: {ctx['error']}"

    run_result = ctx["run_result"]
    if run_result and run_result.get("success"):
        return "Run succeeded; nothing to fix. Use evaluate after a failed run."

    fix_log_text = get_fix_log_recent(run_dir)
    # Build user message: fix_log first (so model avoids repeating), then run_result, then step code
    parts = []
    if fix_log_text:
        parts.append(fix_log_text)
        parts.append("\n\n--- END FIX_LOG ---\n\n")
    parts.append("RUN_RESULT (failed run):\n")
    parts.append(json.dumps(run_result, indent=2)[:8000])
    if len(json.dumps(run_result)) > 8000:
        parts.append("\n... (truncated)")
    parts.append("\n\n--- STEP SCRIPTS (full source) ---\n")
    for s in ctx["steps"]:
        parts.append(f"\n--- step_{s['step_id']}.py (action: {s['action']}, returncode: {s['returncode']}) ---\n")
        parts.append(s["content"])
        parts.append("\n")

    user_message = "".join(parts)
    response = llm.generate(system_prompt=EVALUATOR_SYSTEM, user_message=user_message)
    edits, summary = _parse_edits(response)

    if not edits:
        _append_fix_log(run_dir, summary or "No edits proposed.", [])
        return f"Evaluator response (no file edits applied):\n{summary or response[:500]}"

    written = _apply_edits(run_dir, edits)
    _append_fix_log(run_dir, summary, written)
    return f"Applied {len(written)} fix(s) to {run_dir}\nSummary: {summary}\nFiles changed: {', '.join(os.path.basename(p) for p in written)}\nFix log appended to {os.path.join(run_dir, _FIX_LOG)}"


def re_run_existing(
    run_dir: str | None = None,
    from_step: int | None = None,
    base_path: str | None = None,
) -> str:
    """
    Re-execute step_*.py scripts in the run dir. If from_step is set (or inferred from
    fix_log), only run that step and later; otherwise run all. Preserves previous
    step results for steps before from_step. Overwrites run_result.json.
    """
    root = base_path or _project_root()
    run_dir = run_dir or get_latest_run_dir(root)
    if not run_dir or not os.path.isdir(run_dir):
        return "No run directory found."
    if not os.path.isabs(run_dir):
        run_dir = os.path.normpath(os.path.join(root, run_dir))
    if not os.path.isdir(run_dir):
        return f"Run directory not found: {run_dir}"

    run_result_path = os.path.join(run_dir, "run_result.json")
    if not os.path.isfile(run_result_path):
        return "run_result.json not found; cannot determine step order."

    with open(run_result_path, encoding="utf-8") as f:
        existing = json.load(f)

    all_step_files = sorted(
        f for f in os.listdir(run_dir)
        if f.startswith("step_") and f.endswith(".py")
        and re.match(r"step_\d+\.py", f)
    )
    all_step_files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    if from_step is None:
        from_step = get_from_step_from_fix_log(run_dir)
    step_files_to_run = [
        f for f in all_step_files
        if int(re.search(r"\d+", f).group()) >= (from_step or 1)
    ]
    if not step_files_to_run:
        return "No steps to run."

    # Keep previous results for steps before from_step
    first_run_step = int(re.search(r"\d+", step_files_to_run[0]).group())
    previous_steps = [s for s in existing.get("steps", []) if int(str(s.get("step_id", 0))) < first_run_step]

    results = list(previous_steps)
    all_ok = True
    for step_file in step_files_to_run:
        script_path = os.path.join(run_dir, step_file)
        returncode, stdout, stderr = run_script_with_auto_install(script_path, root)
        step_id = re.search(r"\d+", step_file).group()
        results.append({
            "step_id": step_id,
            "action": next((s.get("action", "") for s in existing.get("steps", []) if str(s.get("step_id")) == step_id), ""),
            "returncode": returncode,
            "stdout": stdout[:2000] + ("..." if len(stdout) > 2000 else ""),
            "stderr": stderr[:2000] + ("..." if len(stderr) > 2000 else ""),
            "script_path": script_path,
            "fixed_and_retried": False,
        })
        if returncode != 0:
            all_ok = False
            break

    new_payload = {
        "run_dir": run_dir,
        "plan_path": existing.get("plan_path", ""),
        "agreed_question": existing.get("agreed_question", ""),
        "success": all_ok,
        "timestamp": existing.get("timestamp", ""),
        "re_run": True,
        "re_run_from_step": first_run_step,
        "steps": results,
    }
    with open(run_result_path, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, indent=2, ensure_ascii=False)

    lines = [
        f"Re-ran {run_dir}",
        f"From step: {first_run_step} (ran {len(step_files_to_run)} step(s); {len(previous_steps)} unchanged)",
        f"Success: {all_ok}",
        "",
    ]
    for r in results:
        status = "ok" if r["returncode"] == 0 else "FAILED"
        lines.append(f"  {r['step_id']}. [{r.get('action', '')}] {status}")
        if r.get("returncode", 0) != 0:
            err = (r.get("stderr") or r.get("stdout") or "").strip()[:300]
            lines.append(f"      error: {err}")
    return "\n".join(lines)
