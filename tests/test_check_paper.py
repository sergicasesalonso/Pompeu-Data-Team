"""scripts/check_paper.py on synthetic text."""

import importlib.util
import os

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "check_paper.py")
_spec = importlib.util.spec_from_file_location("check_paper", _PATH)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

SHA = "ab" * 32
GOOD = """# T

## 1. Summary
x
## 2. Assets used and dropped
x
## 3. Training data
x
## 4. Possession rules
x
## 5. Limitations
x
## 6. Reproduction
sha {}
""".format(SHA)


def _by_name(results):
    return {n: ok for n, ok, _ in results}


def test_clean_document_passes_everything():
    assert all(ok for _, ok, _ in cp.check(GOOD, "readme " + SHA))


def test_open_items_are_reported():
    bad = "> DRAFT\n" + GOOD.replace("x\n## 5.", "TODO\n## 5.") + "\nwritten with Claude agents"
    r = _by_name(cp.check(bad, "readme " + "cd" * 32))
    assert not r["no open TODO"]
    assert not r["no DRAFT banner"]
    assert not r["sha256 document == README"]
    assert not r["no mention of agents or LLMs"]


def test_work_in_progress_wording_is_reported():
    r = _by_name(cp.check(GOOD.replace("x\n## 3.", "the fit is in progress (to verify)\n## 3."),
                          "readme " + SHA))
    assert not r["no work-in-progress wording"]


def test_missing_section_is_reported():
    r = _by_name(cp.check(GOOD.replace("## 5. Limitations", "## 5. Notes"), "readme " + SHA))
    assert not r["six SUBMISSION.md sections"]
