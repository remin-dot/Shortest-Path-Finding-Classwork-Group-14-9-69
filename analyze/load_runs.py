"""Load every saved cost-map run (analyze/runs/*.ipynb) back into the dict run_notebook.save() wrote."""

import json
import re
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_runs(folder=RUNS_DIR):
    runs = []
    for path in sorted(Path(folder).glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8-sig"))
        for cell in nb["cells"]:
            match = re.search(r'json\.loads\(r"""(.*)"""\)', "".join(cell["source"]), re.S)
            if match:
                runs.append(json.loads(match.group(1)))
                break
    return runs


if __name__ == "__main__":
    for run in load_runs():
        s = run["samples"]
        print("=====", run["name"], run["algorithm"], run["mode"], run["status"], run["saved_at"], "samples", len(s))
        print(" result", run["result"])
        print(" cal", run["calibration"])
        if s:
            tofs = [x["tof"] for x in s if x["tof"] is not None]
            print(" first", s[0])
            print(" last", s[-1])
            print(" tof None", len(s) - len(tofs), "min tof", min(tofs) if tofs else None)
