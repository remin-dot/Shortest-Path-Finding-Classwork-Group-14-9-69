"""Save a cost-map run (plans + recorded pose) as a self-contained, pre-executed Jupyter notebook."""

import contextlib
import io
import json
import base64
import warnings
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "analyze" / "runs"

DATA_CELL = '''import json

run = json.loads(r"""{data}""")
print(run["name"], "|", run["algorithm"], "|", run["mode"], "|", run["status"])'''

SUMMARY_CELL = '''print("{:<5} {:>5} {:>5} {:>9}  path".format("algo", "steps", "cost", "expanded"))
for name, plan in run["plans"].items():
    path = " -> ".join("({},{})".format(*c) for c in plan["path"]) if plan["path"] else "no path"
    print("{:<5} {:>5} {:>5} {:>9}  {}".format(name, plan["steps"], plan["cost"], len(plan["expanded"]), path))
print()
for key, value in run["result"].items():
    print("{:<18} {}".format(key, value))
print("{:<18} {}".format("samples", len(run["samples"])))'''

MAP_CELL = '''import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

m = run["map"]
cost = {tuple(int(v) for v in k.split(",")): c for k, c in m["cost"].items()}
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for x in range(1, m["width"] + 1):
    for y in range(1, m["height"] + 1):
        cell = [x, y]
        face = ("#3c4048" if cell in m["obstacles"] else "#c8ebc8" if cell == m["start"]
                else "#f5cdcd" if cell == m["goal"] else "#e2e9f2")
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor="black"))
        label = "OBSTACLE" if cell in m["obstacles"] else "({},{}) cost {}".format(x, y, cost[(x, y)])
        ax.text(x, y + 0.35, label, ha="center", fontsize=8, color="red" if cell in m["obstacles"] else "black")
colors = {"BFS": "tab:blue", "A*": "tab:orange"}
offset = {"BFS": -0.06, "A*": 0.06}
for name, plan in run["plans"].items():
    if plan["path"]:
        xs = [c[0] + offset.get(name, 0) for c in plan["path"]]
        ys = [c[1] + offset.get(name, 0) for c in plan["path"]]
        ax.plot(xs, ys, "--", lw=3 if name == run["algorithm"] else 1.5, color=colors.get(name),
                label="{} plan (steps {}, cost {})".format(name, plan["steps"], plan["cost"]))
if run["samples"]:
    ax.plot([s["mx"] for s in run["samples"]], [s["my"] for s in run["samples"]],
            color="gold", lw=2.5, label="actual trail")
    ax.plot(run["samples"][-1]["mx"], run["samples"][-1]["my"], "o", color="black")
ax.set_xlim(0.5, m["width"] + 0.5)
ax.set_ylim(0.5, m["height"] + 0.5)
ax.set_aspect("equal")
ax.set_title("{} - {} ({})".format(run["name"], run["status"], run["mode"]))
ax.legend(loc="lower left", fontsize=8)
plt.show()'''

TIMELINE_CELL = '''if run["samples"]:
    t = [s["t"] for s in run["samples"]]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, [s["mx"] for s in run["samples"]], label="map x")
    axes[0].plot(t, [s["my"] for s in run["samples"]], label="map y")
    axes[0].set_ylabel("cell")
    axes[0].legend()
    axes[1].plot(t, [s["heading"] for s in run["samples"]], color="tab:green")
    axes[1].set_ylabel("heading (deg)")
    axes[2].plot(t, [s["tof"] if s["tof"] is not None else float("nan") for s in run["samples"]], color="tab:cyan")
    axes[2].set_ylabel("ToF (mm)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.show()
else:
    print("Plan only - no motion recorded")'''


def next_name(algorithm, folder=None):
    folder = folder or RUNS_DIR
    prefix = algorithm.replace("*", "_star")  # "A*" -> "A_star"
    n = 1
    while (Path(folder) / "{}_{}.ipynb".format(prefix, n)).exists():
        n += 1
    return "{}_{}".format(prefix, n)


def _execute(source, namespace, count):
    """Runs one cell in-process so the saved notebook already shows text and plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(source, namespace)
    outputs = []
    if stdout.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(True)})
    for num in plt.get_fignums():
        png = io.BytesIO()
        plt.figure(num).savefig(png, format="png", dpi=100, bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"),
                                 "text/plain": ["<Figure>"]}})
    plt.close("all")
    return {"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
            "source": source.splitlines(True)}


def save(run, folder=None):
    folder = Path(folder or RUNS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    r = run["result"]
    title = ("# {name}\n\n| field | value |\n| --- | --- |\n| algorithm | {algorithm} |\n| mode | {mode} |\n"
             "| status | {status} |\n| saved | {saved_at} |\n").format(**run)
    title += ("| planned steps | {planned_steps} |\n| actual steps | {actual_steps} |\n| time (s) | {time_s} |\n"
              "| hit obstacle | {hit_obstacle} |\n| reach goal | {reach_goal} |\n| path cost | {path_cost} |\n").format(**r)
    namespace = {}
    sources = [DATA_CELL.format(data=json.dumps(run, indent=1)), SUMMARY_CELL, MAP_CELL, TIMELINE_CELL]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": title.splitlines(True)}]
    cells += [_execute(src, namespace, i + 1) for i, src in enumerate(sources)]
    notebook = {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
                "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                             "language_info": {"name": "python"}}}
    path = folder / (run["name"] + ".ipynb")
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return path
