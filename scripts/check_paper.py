"""Pre-delivery checks on the technical document and the README. No data needed.

    python scripts/check_paper.py                  # paper/PompeuDataTeam.md, README.md
    python scripts/check_paper.py --strict         # exit 1 on any open item (use after the freeze)

Reports counts only: open TODO markers, the SUBMISSION.md sections present, whether the sha256
written in the README and in the document agree, words and forbidden mentions (agents, Claude).
"""

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA = re.compile(r"\b[0-9a-f]{64}\b")
# SUBMISSION.md: approach, assets, training data, possession rules, limitations, reproduction.
SECTIONS = {
    "approach": r"^##\s*\d*\.?\s*(summary|approach)",
    "assets": r"^##\s*\d*\.?\s*assets",
    "training data": r"^##\s*\d*\.?\s*training",
    "possession rules": r"^##\s*\d*\.?\s*possession rules",
    "limitations": r"^##\s*\d*\.?\s*limitations",
    "reproduction": r"^##\s*\d*\.?\s*reproduction",
}
FORBIDDEN = re.compile(r"\b(claude|agent|agents|anthropic|llm|llms|gpt|chatgpt|copilot|codex)\b",
                       re.I)
# Text that only makes sense in a draft (DATA.md asks for "to verify" on unchecked claims).
WORK_IN_PROGRESS = re.compile(r"to verify|in progress|being fitted|does not exist yet|not final yet",
                              re.I)


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def check(paper_text, readme_text):
    """List of (name, ok, detail)."""
    out = []
    todo = len(re.findall(r"\bTODO\b", paper_text))
    out.append(("no open TODO", todo == 0, "{} TODO markers".format(todo)))
    wip = sorted(set(m.group(0).lower() for m in WORK_IN_PROGRESS.finditer(paper_text)))
    out.append(("no work-in-progress wording", not wip, "found: {}".format(wip or "none")))
    draft = bool(re.search(r"^>\s*DRAFT", paper_text, re.M))
    out.append(("no DRAFT banner", not draft, "banner present" if draft else "none"))
    missing = [k for k, rx in SECTIONS.items() if not re.search(rx, paper_text, re.I | re.M)]
    out.append(("six SUBMISSION.md sections", not missing, "missing: {}".format(missing or "none")))
    ps, rs = set(SHA.findall(paper_text)), set(SHA.findall(readme_text))
    out.append(("sha256 in document", len(ps) == 1, "{} distinct".format(len(ps))))
    out.append(("sha256 document == README", bool(ps) and ps <= rs,
                "document {} / README {}".format(sorted(s[:8] for s in ps), sorted(s[:8] for s in rs))))
    bad = sorted(set(m.group(0).lower() for m in FORBIDDEN.finditer(paper_text)))
    out.append(("no mention of agents or LLMs", not bad, "found: {}".format(bad or "none")))
    words = len(paper_text.split())
    out.append(("length (informational, SUBMISSION.md: a few pages)", True, "{} words".format(words)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paper", default="paper/PompeuDataTeam.md")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    results = check(read(args.paper), read(args.readme))
    for name, ok, detail in results:
        print("{}  {:<52} {}".format("ok  " if ok else "OPEN", name, detail))
    if args.strict and not all(ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
