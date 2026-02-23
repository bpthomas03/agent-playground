"""
Ideation layer: two experts (dream psychology + NLP) converse to agree on
a research question that is both psychologically interesting and tractable
given the dream corpus and text-analysis methods.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime

# Shared context about the corpus (no need to send full CSV in the loop).
CORPUS_BRIEF = """
DREAM CORPUS
- About 16,500 dream reports in a CSV: dream_id, subject, date, narrative, word_count.
- Narratives are free text: first-person descriptions of dreams (e.g. from dream diaries).
- Multiple subjects; at least one long-running series (e.g. one person over decades).
- No existing labels or codes—only the text and metadata. Analysis would be unsupervised
  or use derived features (e.g. topic models, keyword counts, sentiment lexicons).
"""

PSYCHOLOGY_SYSTEM = """You are an expert in dream psychology and dream research. You know:
- Classic and contemporary theories (Freud, Jung, cognitive, neurocognitive, continuity hypothesis).
- What kinds of questions are meaningful and interpretable in dream science (themes, symbolism, affect, development over time, individual differences).
- How to phrase research questions that can be argued from dream *reports* (written narratives), not from sleep-lab measures.

Your role in this dialogue: propose or refine research questions that would be interesting and psychologically meaningful to ask about this dream corpus. Be concrete. When the NLP expert responds, take their feasibility constraints seriously and suggest alternatives or refinements so you converge on one agreed question."""

NLP_SYSTEM = """You are an expert in natural language processing and corpus-based text analysis. You know:
- What is tractable with ~16k short documents: topic modeling (LDA, NMF), keyword/lexicon counts, simple clustering, temporal trends, basic sentiment or affect lexicons, readability/length stats.
- What is not tractable or is underpowered without labels: supervised classification without annotations, fine-grained semantic parsing, rare-event detection in raw text.
- Data format: CSV with narrative text and metadata (subject, date, word_count). No existing annotations.

Your role in this dialogue: respond to the psychology expert's ideas with what is feasible (and how), and what is not. Propose concrete methods (e.g. "topic model then compare topic prevalence by decade"). Push toward one agreed research question and a high-level approach that you could implement."""

CONVERGENCE_INSTRUCTION = """
Converge on a single research question and a short plan. In your reply, include a final block exactly in this format (so it can be parsed):

--- AGREED ---
Question: [one clear sentence]
Rationale: [1-2 sentences: why interesting psychologically, why tractable with NLP]
Approach: [2-4 short bullets: what we will do with the data and methods]
--- END ---
"""


@dataclass
class IdeationResult:
    """Structured output from the ideation dialogue."""
    agreed_question: str
    rationale: str
    approach: str
    full_conversation: list[tuple[str, str]]  # (role, message)


def _parse_agreement(text: str) -> tuple[str, str, str] | None:
    """Extract agreed question, rationale, approach from --- AGREED --- block."""
    start = text.find("--- AGREED ---")
    end = text.find("--- END ---")
    if start == -1 or end == -1 or end <= start:
        return None
    block = text[start:end].replace("--- AGREED ---", "").strip()

    def get_field(label: str) -> str:
        low = block.lower()
        key = label.lower()
        pos = low.find(key)
        if pos == -1:
            return ""
        pos += len(key)
        # Take rest of line and any following lines until next field or end
        rest = block[pos:].strip()
        for next_key in ["question:", "rationale:", "approach:"]:
            if next_key != key:
                idx = rest.lower().find(next_key)
                if idx != -1:
                    rest = rest[:idx].strip()
        return rest.strip()

    question = get_field("Question:")
    rationale = get_field("Rationale:")
    approach = get_field("Approach:")
    if question or rationale or approach:
        return (question, rationale, approach)
    return None


def run_ideation(llm: "object", num_rounds: int = 4) -> IdeationResult:
    """
    Run the ideation dialogue between psychology and NLP experts.
    llm must have .generate(system_prompt: str, user_message: str) -> str.
    With 4 rounds, NLP expert speaks last and outputs the AGREED block.
    """
    conversation: list[tuple[str, str]] = []
    user_content = (
        "Below is a brief description of the dream corpus we have. "
        "Propose 1–3 research questions that would be psychologically interesting to ask about this corpus. "
        "Be specific enough that we can later refine with the NLP expert.\n\n"
        + CORPUS_BRIEF.strip()
    )

    for round_no in range(num_rounds):
        # Psychology expert speaks first in round 0; then we alternate.
        is_psychology_turn = (round_no % 2 == 0)
        system = PSYCHOLOGY_SYSTEM if is_psychology_turn else NLP_SYSTEM
        role = "Psychology expert" if is_psychology_turn else "NLP expert"

        # Build user message: initial brief + full conversation so far.
        if round_no == 0:
            full_user = user_content
        else:
            transcript = "\n\n".join(
                f"{r}: {m}" for r, m in conversation
            )
            full_user = (
                "Initial context:\n" + user_content + "\n\n"
                "Conversation so far:\n" + transcript + "\n\n"
            )
            if round_no == num_rounds - 1:
                full_user += CONVERGENCE_INSTRUCTION
            else:
                full_user += (
                    "Reply in character. Propose or refine the research question and feasibility. "
                    "Move toward one agreed question and a high-level approach."
                )

        reply = llm.generate(system_prompt=system, user_message=full_user)
        conversation.append((role, reply))

    # Parse agreement from the last reply (NLP expert ends with convergence block).
    last_reply = conversation[-1][1]
    parsed = _parse_agreement(last_reply)
    if parsed:
        question, rationale, approach = parsed
    else:
        question = "See conversation below."
        rationale = ""
        approach = last_reply[:800] + ("..." if len(last_reply) > 800 else "")

    return IdeationResult(
        agreed_question=question,
        rationale=rationale,
        approach=approach,
        full_conversation=conversation,
    )


def format_result(result: IdeationResult) -> str:
    """Human-readable summary of the ideation result."""
    lines = [
        "--- Ideation result ---",
        "",
        "Agreed question:",
        result.agreed_question,
        "",
        "Rationale:",
        result.rationale,
        "",
        "Approach:",
        result.approach,
        "",
        "--- Conversation (excerpt) ---",
    ]
    for role, msg in result.full_conversation[-4:]:  # last 2 exchanges
        lines.append(f"\n{role}:\n{msg[:1200]}{'...' if len(msg) > 1200 else ''}")
    return "\n".join(lines)


# Default output dir relative to project root (Docker: /app, local: cwd when run from project root).
_IDEATION_OUTPUT_DIR = "ideation"


def save_result(
    result: IdeationResult,
    output_dir: str | None = None,
    base_path: str | None = None,
) -> list[str]:
    """
    Write the ideation result to the repo for downstream use.
    - output_dir/<timestamp>.json and .md: timestamped copy (filename-safe, for history).
    - output_dir/latest.json and latest.md: same content, overwritten each run (most recent).
    Returns the list of absolute paths written.
    """
    base = (base_path or os.environ.get("AGENT_PROJECT_ROOT") or os.getcwd())
    base = os.path.abspath(base)
    out = os.path.join(base, output_dir or _IDEATION_OUTPUT_DIR)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # Filename-safe timestamp (no colons for Windows)
    safe_ts = timestamp.replace(":", "-")

    payload = {
        "agreed_question": result.agreed_question,
        "rationale": result.rationale,
        "approach": result.approach,
        "timestamp": timestamp,
    }
    md_content = format_result(result) + f"\n\n---\nSaved at {timestamp} UTC.\n"

    # Timestamped files (for history; manually pick one to use)
    ts_json = os.path.join(out, f"{safe_ts}.json")
    ts_md = os.path.join(out, f"{safe_ts}.md")
    with open(ts_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    written.append(os.path.abspath(ts_json))
    with open(ts_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    written.append(os.path.abspath(ts_md))

    # Latest (overwritten each run)
    latest_json = os.path.join(out, "latest.json")
    latest_md = os.path.join(out, "latest.md")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    written.append(os.path.abspath(latest_json))
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    written.append(os.path.abspath(latest_md))

    return written


if __name__ == "__main__":
    from src.main import GPTAgent

    llm = GPTAgent()
    result = run_ideation(llm, num_rounds=4)
    paths = save_result(result)
    print(format_result(result))
    print("\nSaved to:", ", ".join(paths))
    print("Timestamped copy in ideation/ for history; pick one manually when ready.")
