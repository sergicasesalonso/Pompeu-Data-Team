"""Run pytest on the given test files (tower helper: torre.py only launches repo scripts).

    python scripts/experiments/torre_pytest.py tests/test_hsmm_tcn.py tests/test_tcn_inputs.py

Writes passed / failed / skipped counts to TORRE_METRICS if set.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Counter:
    def __init__(self):
        self.counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
        self.failures = {}

    def pytest_runtest_logreport(self, report):
        if report.failed:                         # synthetic tests only: name + last line
            import re
            found = re.findall(r"^E\s+([A-Za-z_.]*(?:Error|Exception|Failed))", report.longreprtext,
                               re.M)
            self.failures[report.nodeid.split("::")[-1][:60]] = (found or ["unknown"])[0][:60]
            print("FAILED {} :: {}".format(report.nodeid, (found or ["?"])[0]))
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            key = "error" if report.when == "setup" and report.failed else report.outcome
            self.counts[key] += 1


def main():
    counter = Counter()
    os.chdir(ROOT)
    code = pytest.main(["-q", "-p", "no:cacheprovider"] + sys.argv[1:], plugins=[counter])
    print(json.dumps(counter.counts))
    if os.environ.get("TORRE_METRICS"):
        with open(os.environ["TORRE_METRICS"], "w", encoding="utf-8") as handle:
            json.dump(dict(counter.counts, exit=int(code), failures=counter.failures), handle)
    sys.exit(int(code))


if __name__ == "__main__":
    main()
