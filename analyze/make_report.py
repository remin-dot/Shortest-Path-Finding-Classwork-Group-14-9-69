#!/usr/bin/env python3
"""Build the BFS vs A* comparison report (Thai, A4 PDF) from every run in analyze/runs/.

    python analyze/make_report.py

Figures and the HTML source go to analyze/report/; the PDF is printed with Microsoft Edge
(headless) so Thai text is shaped correctly, then printed again with real page numbers in the TOC.
"""

import html
import re
import itertools
import math
import os
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import cost_map_panel as cm  # noqa: E402
from load_runs import load_runs  # noqa: E402

OUT = HERE / "report"
FIG = OUT / "figures"
PDF_NAME = "รายงาน-Classwork-14-9-69-BFS-vs-Astar-RoboMaster.pdf"

DOCX_NAME = PDF_NAME.replace(".pdf", ".docx")

MEMBERS = [
    ("นาย ดาราชัย เดชะพันธุ์", "6810110109"),
    ("นาย ศุภกิตติ์ เชี่ยวหมอน", "6810110354"),
    ("นาย ทวีรัตน์ ยอดเสถียร", "6810110594"),
    ("นาย อาณัส อาเก๊ะ", "6810110755"),
]
GROUP = "พบลาบไม่ทรายชื่อ (Phob_Labb_Mai_sarb_cheu)"

ALGOS = ("BFS", "A*")
GREY = {"BFS": "#9a9a9a", "A*": "#111111"}
DIR_NAME = {(1, 0): "E", (0, -1): "S", (-1, 0): "W", (0, 1): "N"}


# ---------------------------------------------------------------- metrics
def seg_dist(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / float(dx * dx + dy * dy)))
    return math.hypot(px - a[0] - t * dx, py - a[1] - t * dy)


def metrics(run):
    s, cal = run["samples"], run["calibration"]
    path = [tuple(c) for c in run["plans"][run["algorithm"]]["path"]]
    cell_cm = cal["cell_m"] * 100.0
    xte = [min(seg_dist(p["mx"], p["my"], a, b) for a, b in zip(path, path[1:])) * cell_cm for p in s]
    entry = [0.0]
    for c in path[1:]:  # first time the robot centre is within 6 cm of the next cell centre
        entry.append(next(p["t"] for p in s if p["t"] > entry[-1]
                          and abs(p["mx"] - c[0]) < 0.1 and abs(p["my"] - c[1]) < 0.1))
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    h0 = math.radians(cal["start_heading"])
    start_dir = (int(round(math.cos(h0))), int(round(math.sin(h0))))
    turned = [a != b for a, b in zip([start_dir] + dirs, dirs)]
    last = s[-1]
    tof_min = min(s, key=lambda p: p["tof"] if p["tof"] is not None else 1e9)
    return {
        "name": run["name"], "algo": run["algorithm"], "run": run, "path": path, "dirs": dirs, "turned": turned,
        "turns": sum(turned), "time": run["result"]["time_s"], "entry": entry,
        "seg": [b - a for a, b in zip(entry, entry[1:])],
        "xte_max": max(xte), "xte_rms": math.sqrt(st.mean(x * x for x in xte)), "xte": xte,
        "goal_cm": run["result"]["goal_error_m"] * 100.0,
        "head_err": abs(cm.wrap(last["heading"] - round(last["heading"] / 90.0) * 90.0)),
        "travel": run["result"]["travelled_m"], "tof_min": tof_min, "tof_end": last["tof"],
        "tof_start": s[0]["tof"], "samples": len(s),
        "dt": st.mean(b["t"] - a["t"] for a, b in zip(s, s[1:])),
    }


def mean_sd(values, fmt="{:.2f}"):
    values = list(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return (fmt + " ± " + fmt).format(st.mean(values), sd)


def bfs_order_costs():
    """BFS path cost for every neighbour order - shows BFS cost is a tie-break accident."""
    saved, out = cm.MOVES, {}
    try:
        for order in itertools.permutations(saved):
            cm.MOVES = order
            path, _ = cm.bfs()
            out["".join(DIR_NAME[d] for d in order)] = (cm.path_cost(path), path)
    finally:
        cm.MOVES = saved
    return out


def astar_trace():
    """Re-runs A* and records g, h, f of every expanded node in pop order."""
    goal, cheapest = cm.GOAL, min(cm.COST.values())
    h = lambda c: (abs(c[0] - goal[0]) + abs(c[1] - goal[1])) * cheapest  # noqa: E731
    import heapq
    g, parent, done, rows = {cm.START: 0}, {cm.START: None}, set(), []
    heap = [(h(cm.START), 0, cm.START)]
    while heap:
        f, gc, cell = heapq.heappop(heap)
        if cell in done:
            continue
        done.add(cell)
        rows.append((cell, gc, h(cell), f, parent[cell]))
        if cell == goal:
            break
        for nxt in cm.neighbors(cell):
            ng = gc + cm.COST[nxt]
            if ng < g.get(nxt, math.inf):
                g[nxt], parent[nxt] = ng, cell
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))
    return rows, g


# ---------------------------------------------------------------- figures
def figures(ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": ["Leelawadee UI", "Tahoma", "DejaVu Sans"], "font.size": 10,
                         "axes.edgecolor": "#888", "axes.labelcolor": "#555", "xtick.color": "#555",
                         "ytick.color": "#555", "axes.grid": True, "grid.color": "#ddd"})
    FIG.mkdir(parents=True, exist_ok=True)
    plans = {n: fn() for n, fn in cm.ALGORITHMS.items()}

    def field(ax, show_cost=True):
        for x in range(1, cm.W + 1):
            for y in range(1, cm.H + 1):
                c = (x, y)
                face = "#5f5f5f" if c in cm.OBSTACLES else "#e6e6e6" if c in (cm.START, cm.GOAL) else "white"
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face, edgecolor="#333", lw=1))
                if c in cm.OBSTACLES:
                    ax.text(x, y, "X", ha="center", va="center", color="white", fontsize=12, weight="bold")
                elif show_cost:
                    tag = "S " if c == cm.START else "G " if c == cm.GOAL else ""
                    ax.text(x - 0.42, y + 0.33, "{}{}".format(tag, cm.COST[c]), fontsize=8, color="#555")
        ax.set_xlim(0.5, cm.W + 0.5)
        ax.set_ylim(0.5, cm.H + 0.5)
        ax.set_xticks(range(1, cm.W + 1))
        ax.set_yticks(range(1, cm.H + 1))
        ax.set_aspect("equal")
        ax.grid(False)

    # รูปที่ 2 planned paths
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    field(ax)
    for name, style, off in (("BFS", "--", -0.07), ("A*", "-", 0.07)):
        p = plans[name][0]
        ax.plot([c[0] + off for c in p], [c[1] + off for c in p], style, color=GREY[name], lw=2.6,
                label="{} (6 ก้าว, ต้นทุน {})".format(name, cm.path_cost(p)))
    ax.legend(loc="lower left", fontsize=8, framealpha=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_plans.png", dpi=200)
    plt.close(fig)

    # รูปที่ 3 actual trajectories
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax, name in zip(axes, ALGOS):
        field(ax, show_cost=False)
        p = plans[name][0]
        ax.plot([c[0] for c in p], [c[1] for c in p], ":", color="#bbb", lw=5, label="เส้นทางที่วางแผน")
        for m, st_ in zip([m for m in ms if m["algo"] == name], ("-", "--", "-.")):
            s = m["run"]["samples"]
            ax.plot([q["mx"] for q in s], [q["my"] for q in s], st_, color="#111", lw=1.2, label=m["name"])
        ax.set_title(name, fontsize=11)
        ax.legend(loc="lower left", fontsize=7, framealpha=1)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_trajectories.png", dpi=200)
    plt.close(fig)

    # รูปที่ 4 cross-track error
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True)
    for ax, name in zip(axes, ALGOS):
        for m, st_ in zip([m for m in ms if m["algo"] == name], ("-", "--", "-.")):
            ax.plot([q["t"] for q in m["run"]["samples"]], m["xte"], st_, color=GREY[name] if name == "BFS" else "#111",
                    lw=1.1, label=m["name"])
        ax.set_ylabel("ระยะห่างจากเส้นทาง (ซม.)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[1].set_xlabel("เวลา (วินาที)")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_xte.png", dpi=200)
    plt.close(fig)

    # รูปที่ 5 heading timeline (run 1 of each)
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for name, st_ in (("BFS", "--"), ("A*", "-")):
        m = next(m for m in ms if m["algo"] == name)
        s = m["run"]["samples"]
        unwrapped = np.degrees(np.unwrap(np.radians([q["heading"] for q in s])))  # no 360 -> 0 jumps
        ax.plot([q["t"] for q in s], unwrapped, st_, color=GREY[name] if name == "BFS" else "#111", lw=1.5,
                label=m["name"])
    ticks = list(range(-90, 451, 90))
    ax.set_yticks(ticks)
    ax.set_yticklabels(["{} ({})".format(cm.HEADINGS[t % 360], t) for t in ticks])
    ax.set_ylim(-120, 400)
    ax.set_xlabel("เวลา (วินาที)")
    ax.set_ylabel("ทิศบนแผนที่", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_heading.png", dpi=200)
    plt.close(fig)

    # รูปที่ 6 time per run
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    names = [m["name"] for m in ms]
    ax.bar(names, [m["time"] for m in ms], color=[GREY["BFS"] if m["algo"] == "BFS" else "#333" for m in ms])
    for i, m in enumerate(ms):
        ax.text(i, m["time"] + 0.15, "{:.2f}".format(m["time"]), ha="center", fontsize=8)
    ax.set_ylim(0, max(m["time"] for m in ms) * 1.15)
    ax.set_ylabel("เวลา (วินาที)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_time.png", dpi=200)
    plt.close(fig)

    # รูปที่ 7 ToF
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.2), sharex=True)
    for ax, name in zip(axes, ALGOS):
        for m, st_ in zip([m for m in ms if m["algo"] == name], ("-", "--", "-.")):
            s = m["run"]["samples"]
            ax.plot([q["t"] for q in s], [q["tof"] for q in s], st_, color=GREY[name] if name == "BFS" else "#111",
                    lw=1.1, label=m["name"])
        ax.axhline(cm.HIT_TOF_MM, color="#777", lw=0.8, ls=":")
        ax.set_ylabel("ToF (มม.)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[1].set_xlabel("เวลา (วินาที)")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_tof.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- html helpers
def table(caption, head, rows, num_cols=(), widths=None):
    cols = "".join('<col style="width:{}">'.format(w) for w in widths) if widths else ""
    out = ['<p class="cap">{}</p><table><colgroup>{}</colgroup><tr>'.format(caption, cols)]
    out += ["<th>{}</th>".format(h) for h in head] + ["</tr>"]
    for r in rows:
        out.append("<tr>" + "".join('<td class="{}">{}</td>'.format("num" if i in num_cols else "", v)
                                    for i, v in enumerate(r)) + "</tr>")
    return "".join(out) + "</table>"


def figure(src, caption, width="80%"):
    return '<div class="fig"><img src="figures/{}" style="width:{}"><p class="figcap">{}</p></div>'.format(
        src, width, caption)


def cell(c):
    return "({}, {})".format(*c)


def build_html(ms, toc_pages=None):
    by = {a: [m for m in ms if m["algo"] == a] for a in ALGOS}
    plans = {n: fn() for n, fn in cm.ALGORITHMS.items()}
    bfs_path, bfs_exp = plans["BFS"]
    ast_path, ast_exp = plans["A*"]
    trace, g_final = astar_trace()
    orders = bfs_order_costs()
    order_costs = sorted(set(c for c, _ in orders.values()))
    n_runs = len(ms)
    all_ok = all(m["run"]["result"]["reach_goal"] == "YES" for m in ms)
    no_hit = all(m["run"]["result"]["hit_obstacle"] == "NO" for m in ms)
    t_mean = {a: st.mean(m["time"] for m in by[a]) for a in ALGOS}
    seg_mean = {a: [st.mean(m["seg"][i] for m in by[a]) for i in range(6)] for a in ALGOS}
    turn_seg = [x for a in ALGOS for m in by[a] for x, t in zip(m["seg"], m["turned"]) if t]
    straight_seg = [x for a in ALGOS for m in by[a] for i, (x, t) in enumerate(zip(m["seg"], m["turned"])) if not t and i]
    first_seg = [m["seg"][0] for m in ms]
    dates = sorted(m["run"]["saved_at"] for m in ms)
    cal = ms[0]["run"]["calibration"]
    headings = {a: cm.HEADINGS[by[a][0]["run"]["calibration"]["start_heading"] % 360] for a in ALGOS}

    sections = []  # (id, level, title)

    def h(level, title):
        sid = "h{:02d}".format(len(sections))
        sections.append((sid, level, title))
        tag = "h1" if level == 1 else "h2"
        return '<{0} id="{1}"><span class="mark">@@{1}@@</span>{2}</{0}>'.format(tag, sid, title)

    body = []
    add = body.append

    # ---------------- cover
    members = "".join("<tr><td>{}</td><td>รหัสนักศึกษา {}</td></tr>".format(n, i) for n, i in MEMBERS)
    add("""<section class="cover">
<div class="title">รายงานผลการทดลอง Classwork 14-9-69<br>การค้นหาเส้นทางที่สั้นที่สุดด้วย BFS และ A*<br>บน DJI RoboMaster EP</div>
<p class="sub">Shortest Path Finding with BFS and A* on a Weighted 4×4 Grid using DJI RoboMaster EP</p>
<p class="abstract">การทดลองวางแผนเส้นทางบนสนามกริด 4×4 ที่มีสิ่งกีดขวางและต้นทุนต่อช่อง
เปรียบเทียบอัลกอริทึม BFS กับ A* ทั้งในเชิงแผน (จำนวนก้าว ต้นทุนรวม จำนวนโหนดที่ขยาย)
และในเชิงการเคลื่อนที่จริงของหุ่นยนต์ (เวลา ความคลาดเคลื่อน การชนสิ่งกีดขวาง และการถึงเป้าหมาย)
จากการรันบนหุ่นยนต์จริง {n} รอบ</p>
<p class="by">จัดทำโดย</p><table class="members">{members}</table>
<p class="foot">การทดลอง — Graph Search: Breadth-First Search และ A* Search</p>
<p class="foot2">สาขาวิชาวิศวกรรมปัญญาประดิษฐ์ (AI Engineering)<br>คณะวิศวกรรมศาสตร์ มหาวิทยาลัยสงขลานครินทร์<br>
กลุ่มโครงงาน: {group}</p></section>""".format(n=n_runs, members=members, group=GROUP))

    add("@@TOC@@")

    # ---------------- 1
    add('<div class="pb"></div>')
    add(h(1, "1. บทนำและวัตถุประสงค์ (Introduction and Objectives)"))
    add(h(2, "1.1 ที่มาของการทดลอง"))
    add("""<p>หุ่นยนต์เคลื่อนที่ที่ทราบแผนที่ล่วงหน้าต้องเลือกเส้นทางจากจุดเริ่มต้นไปยังเป้าหมายก่อนเริ่มเคลื่อนที่
คำว่า “เส้นทางที่สั้นที่สุด” มีได้สองความหมาย คือเส้นทางที่ใช้จำนวนก้าวน้อยที่สุด และเส้นทางที่มีต้นทุนรวมน้อยที่สุด
เมื่อทุกช่องมีต้นทุนเท่ากันสองความหมายนี้ให้ผลเหมือนกัน แต่เมื่อแต่ละช่องมีต้นทุนต่างกัน สองความหมายนี้อาจให้เส้นทางที่ต่างกันได้</p>
<p>การทดลองนี้ใช้อัลกอริทึม Breadth-First Search (BFS) ซึ่งรับประกันจำนวนก้าวน้อยที่สุด
และอัลกอริทึม A* ซึ่งรับประกันต้นทุนรวมน้อยที่สุดเมื่อใช้ heuristic ที่ไม่ประเมินเกินจริง
บนสนามกริด 4×4 ขนาดช่องละ {cm_:.2f} เมตร ที่มีสิ่งกีดขวาง 3 ช่องและมีต้นทุนกำกับทุกช่องตามใบงาน
จากนั้นสั่ง DJI RoboMaster EP ให้เดินตามเส้นทางของแต่ละอัลกอริทึมจริง
เพื่อดูว่าความแตกต่างในเชิงแผนส่งผลต่อการเคลื่อนที่จริงของหุ่นยนต์อย่างไร</p>""".format(cm_=cal["cell_m"]))
    add(h(2, "1.2 วัตถุประสงค์"))
    add("""<ol>
<li>เพื่อนิยามสนามกริดที่มีสิ่งกีดขวางและต้นทุนต่อช่องให้อยู่ในรูปของกราฟสำหรับการค้นหาเส้นทาง</li>
<li>เพื่อคำนวณเส้นทางด้วย BFS และ A* แล้วเปรียบเทียบจำนวนก้าว ต้นทุนรวม และจำนวนโหนดที่ขยาย</li>
<li>เพื่อตรวจสอบความถูกต้องของผลการค้นหาด้วยการคำนวณมือ ทั้งต้นทุนของเส้นทาง และลำดับการขยายโหนดของ A* จากค่า f = g + h</li>
<li>เพื่อสั่งหุ่นยนต์จริงเดินตามเส้นทางของแต่ละอัลกอริทึม และบันทึกจำนวนก้าวที่วางแผน จำนวนก้าวจริง เวลา
การชนสิ่งกีดขวาง และการถึงเป้าหมาย</li>
<li>เพื่อประเมินความสม่ำเสมอของการเคลื่อนที่จากการรันซ้ำอัลกอริทึมละ 3 รอบ</li></ol>""")
    add(h(2, "1.3 ขอบเขตของการทดลอง"))
    add("""<p>รายงานฉบับนี้นำเสนอผลการรันบนหุ่นยนต์จริง (โหมด REAL ROBOT) จำนวน {n} รอบ แบ่งเป็น BFS 3 รอบ
(BFS_1 ถึง BFS_3) และ A* 3 รอบ (A_star_1 ถึง A_star_3) บันทึกระหว่างเวลา {d0} ถึง {d1} น. ของวันที่ 14 กันยายน 2569
ข้อมูลทุกรอบถูกบันทึกเป็นไฟล์ Jupyter notebook ในโฟลเดอร์ <code>analyze/runs/</code> ด้วยปุ่ม SAVE RESULT ของโปรแกรม</p>
<p>ตำแหน่งของหุ่นยนต์ในรายงานมาจาก odometry ของล้อร่วมกับมุม yaw จาก IMU ของหุ่นยนต์เอง
ไม่ได้วัดด้วยระบบอ้างอิงภายนอก เช่น กล้องเหนือสนามหรือตลับเมตร ค่าความคลาดเคลื่อนเชิงตำแหน่งทั้งหมดในรายงานจึงเป็น
ค่าที่หุ่นยนต์ประมาณได้เอง ไม่ใช่ความคลาดเคลื่อนเทียบกับพื้นสนามจริง</p>""".format(
        n=n_runs, d0=dates[0][11:16], d1=dates[-1][11:16]))

    # ---------------- 2
    add(h(1, "2. อุปกรณ์และผังสนามทดลอง (Equipment and Field Layout)"))
    add(h(2, "2.1 อุปกรณ์ที่ใช้"))
    add(table("ตารางที่ 1&nbsp; รายการอุปกรณ์และซอฟต์แวร์ที่ใช้ในการทดลอง", ["รายการ", "รายละเอียดและหน้าที่"], [
        ["DJI RoboMaster EP", "หุ่นยนต์ฐานล้อ mecanum ถอดชุด gripper ออกและติดตั้ง gimbal แทน ใช้เป็นตัวกระทำในสนามจริง"],
        ["เซนเซอร์ ToF บน gimbal", "วัดระยะด้านหน้า ตั้ง gimbal เป็นโหมด chassis-lead และ recenter ก่อนเริ่ม เพื่อให้ ToF หันตามตัวรถ"],
        ["คอมพิวเตอร์โน้ตบุ๊ก", "Python 3.8 เชื่อมต่อ Wi-Fi ของหุ่นยนต์ (โหมด AP) ใช้รันอัลกอริทึม ควบคุม และบันทึกข้อมูล"],
        ["ซอฟต์แวร์", "RoboMaster Python SDK, pygame 2.6.1 (หน้าจอควบคุม), numpy และ matplotlib (บันทึกและพล็อตผล)"],
        ["เทปกระดาษและตลับเมตร", "ตีเส้นตารางกริด 4×4 ช่องละ {:.2f} เมตร".format(cal["cell_m"])],
        ["ป้ายสัญลักษณ์และกล่องสิ่งกีดขวาง", "ระบุจุดเริ่มต้น S เป้าหมาย G และช่องสิ่งกีดขวางบนสนาม"],
        ["พื้นที่ราบโล่ง", "ขนาดอย่างน้อย 2.4×2.4 เมตร ปราศจากคนในสนามระหว่างการรัน"],
    ], widths=("34%", "66%")))
    add(h(2, "2.2 ผังสนามกริด 4×4"))
    grid_rows = []
    for y in range(cm.H, 0, -1):
        tds = []
        for x in range(1, cm.W + 1):
            c = (x, y)
            if c in cm.OBSTACLES:
                tds.append('<td class="g-x">X</td>')
            else:
                tag = "S<br>" if c == cm.START else "G<br>" if c == cm.GOAL else ""
                tds.append('<td class="{}"><b>{}</b>{}</td>'.format(
                    "g-sg" if tag else "", tag, cm.COST[c]))
        grid_rows.append('<tr><th class="g-ax">{}</th>{}</tr>'.format(y, "".join(tds)))
    grid_rows.append('<tr><th></th>' + "".join('<th class="g-ax">{}</th>'.format(x) for x in range(1, cm.W + 1)) + "</tr>")
    add("""<p>สนามทดลองเป็นตารางกริด 4×4 อ้างอิงพิกัดในรูป (x, y) ตามใบงาน โดย x เพิ่มไปทางขวาและ y เพิ่มขึ้นด้านบน
หุ่นยนต์เริ่มที่ S = (1, 4) เป้าหมายคือ G = (4, 1) และมีสิ่งกีดขวางที่ (4, 4), (2, 3) และ (3, 2)
ตัวเลขในแต่ละช่องคือต้นทุนของการเข้าสู่ช่องนั้น หุ่นยนต์เคลื่อนที่ได้ 4 ทิศ คือ E, S, W และ N ครั้งละหนึ่งช่อง
และหมุนตัวให้หันไปทางทิศที่จะเคลื่อนที่ก่อนเดินหน้าทุกครั้ง</p>
<table class="grid">{}</table>
<p class="figcap">รูปที่ 1&nbsp; ผังสนามกริด 4×4 พร้อมต้นทุนต่อช่อง จุดเริ่มต้น S = (1, 4) เป้าหมาย G = (4, 1)<br>
และสิ่งกีดขวาง X ที่ (4, 4), (2, 3) และ (3, 2)</p>""".format("".join(grid_rows)))
    add(h(2, "2.3 การกำหนดต้นทุน (Cost Design)"))
    add(table("ตารางที่ 2&nbsp; โครงสร้างต้นทุนและเหตุผลเชิงออกแบบ", ["เหตุการณ์", "ต้นทุน", "เหตุผลเชิงออกแบบ"], [
        ["เข้าสู่ช่องว่างทั่วไป", "1 – 4", "จ่ายต้นทุนตามตัวเลขที่กำกับไว้ในช่องปลายทาง"],
        ["อยู่ที่จุดเริ่มต้น S", "ไม่นับ", "ช่อง S กำกับค่า 2 แต่หุ่นยนต์อยู่ในช่องนี้ตั้งแต่ต้น จึงไม่มีการเข้าสู่ช่อง"],
        ["เข้าสู่เป้าหมาย G", "1", "จ่ายต้นทุนของช่อง G และจบการค้นหาทันที"],
        ["เข้าสู่ช่องสิ่งกีดขวางหรือออกนอกสนาม", "—", "ไม่มีเส้นเชื่อมในกราฟ อัลกอริทึมจึงเลือกไม่ได้"],
    ], num_cols=(1,), widths=("32%", "12%", "56%")))
    add("""<p>ต้นทุนของเส้นทางคือผลรวมของต้นทุนช่องที่เข้าไปหลังออกจาก S ระยะ Manhattan จาก S ไป G เท่ากับ 6 ก้าว
และสิ่งกีดขวางไม่ได้บังคับให้อ้อม เส้นทางที่ใช้จำนวนก้าวน้อยที่สุดจึงยาว 6 ก้าว
ส่วนต้นทุนต่ำสุดที่เป็นไปได้คือ {} ซึ่งได้จากเส้นทางที่ A* เลือก สิ่งกีดขวางที่ (2, 3) และ (3, 2)
แบ่งสนามออกเป็นทางเลือกด้านบนขวาและด้านล่างซ้าย ซึ่งยาวเท่ากันแต่มีต้นทุนต่างกัน</p>""".format(cm.path_cost(ast_path)))

    # ---------------- 3
    add(h(1, "3. ทฤษฎีที่เกี่ยวข้อง (Theoretical Background)"))
    add(h(2, "3.1 นิยามปัญหาในรูปกราฟ"))
    add(table("ตารางที่ 3&nbsp; องค์ประกอบของกราฟในการทดลองนี้", ["องค์ประกอบ", "นิยามในการทดลองนี้"], [
        ["โหนด (Node)", "ช่องที่ไม่ใช่สิ่งกีดขวาง รวม {} โหนด (16 ช่อง หักสิ่งกีดขวาง 3 ช่อง)".format(len(cm.COST))],
        ["เส้นเชื่อม (Edge)", "เชื่อมช่องที่ติดกันใน 4 ทิศ E, S, W, N โดยไม่มีการเดินแนวทแยง"],
        ["น้ำหนักเส้นเชื่อม", "w(u, v) = cost(v) คือต้นทุนของช่องปลายทาง กราฟจึงเป็นกราฟมีทิศทาง เพราะ w(u, v) ≠ w(v, u) ได้"],
        ["จุดเริ่มต้นและเป้าหมาย", "S = (1, 4) และ G = (4, 1)"],
    ], widths=("30%", "70%")))
    add(h(2, "3.2 การค้นหาแบบ Breadth-First Search"))
    add("""<p>BFS ขยายโหนดทีละระดับความลึกด้วยคิวแบบเข้าก่อนออกก่อน (FIFO) โหนดที่อยู่ห่างจาก S หนึ่งก้าวจะถูกขยายทั้งหมด
ก่อนโหนดที่ห่างสองก้าว เมื่อพบ G ครั้งแรกจึงรับประกันว่าเป็นเส้นทางที่มีจำนวนเส้นเชื่อมน้อยที่สุด
แต่ BFS ไม่ใช้น้ำหนักเส้นเชื่อมเลย เมื่อมีหลายเส้นทางที่ยาวเท่ากัน เส้นทางที่ได้จึงขึ้นกับลำดับการเพิ่มเพื่อนบ้านเข้าคิว
ซึ่งโปรแกรมกำหนดเป็นลำดับคงที่ E, S, W, N</p>""")
    add(h(2, "3.3 การค้นหาแบบ A*"))
    add("""<p>A* เลือกขยายโหนดที่มีค่าประเมิน f(n) ต่ำที่สุดจาก priority queue โดย</p>
<pre class="eq">f(n) = g(n) + h(n)
h(n) = ( |x − x<sub>G</sub>| + |y − y<sub>G</sub>| ) × c<sub>min</sub>,   c<sub>min</sub> = min cost = 1</pre>
<p>g(n) คือต้นทุนจริงสะสมจาก S ถึง n และ h(n) คือค่าประมาณต้นทุนที่เหลือถึง G ทุกก้าวมีต้นทุนอย่างน้อย c<sub>min</sub>
และต้องใช้อย่างน้อยเท่ากับระยะ Manhattan จึงไม่มีทางไปถึง G ด้วยต้นทุนต่ำกว่า h(n) heuristic นี้จึงไม่ประเมินเกินจริง (admissible)
และสอดคล้อง (consistent) เพราะหนึ่งก้าวลดค่า h ได้ไม่เกิน c<sub>min</sub> ขณะที่จ่ายต้นทุนอย่างน้อย c<sub>min</sub>
ผลคือเมื่อ G ถูกดึงออกจากคิวครั้งแรก เส้นทางนั้นมีต้นทุนต่ำสุดแน่นอน
ในกรณีที่ค่า f เท่ากัน โปรแกรมเลือกโหนดที่มีค่า g ต่ำกว่าก่อน</p>""")
    add(h(2, "3.4 ความแตกต่างระหว่าง BFS กับ A*"))
    add("""<p>BFS เหมาะสมที่สุดเฉพาะเมื่อทุกเส้นเชื่อมมีน้ำหนักเท่ากัน ส่วน A* ใช้ทั้งต้นทุนจริง g และค่าประมาณ h
ถ้ากำหนด h(n) = 0 A* จะกลายเป็น Dijkstra ซึ่งยังให้ต้นทุนต่ำสุดแต่ขยายโหนดมากกว่า
ค่า h ที่ดีช่วยให้ A* ข้ามโหนดที่มี f สูงกว่าต้นทุนของคำตอบได้ บนสนามนี้ทั้งสองอัลกอริทึมให้เส้นทาง 6 ก้าวเท่ากัน
แต่ต้นทุนต่างกัน จึงเป็นกรณีที่แยกความหมายของ “สั้นที่สุด” สองแบบออกจากกันได้ชัดเจน</p>""")
    add(h(2, "3.5 ความหมายของค่าพารามิเตอร์ต่อผลลัพธ์"))
    add(table("ตารางที่ 4&nbsp; ผลของพารามิเตอร์แต่ละตัวต่อพฤติกรรมของระบบ", ["พารามิเตอร์", "ค่าที่ใช้", "ผลต่อผลลัพธ์"], [
        ["c<sub>min</sub> ใน h(n)", "1", "ค่าที่มากกว่าต้นทุนต่ำสุดจริงทำให้ h ประเมินเกินจริง A* อาจไม่ได้ต้นทุนต่ำสุด"],
        ["ลำดับเพื่อนบ้าน", "E, S, W, N", "กำหนดว่า BFS จะเลือกเส้นทางใดเมื่อมีหลายเส้นทางยาวเท่ากัน"],
        ["cell_m", "{:.2f} ม.".format(cal["cell_m"]), "ระยะจริงของหนึ่งช่อง ใช้แปลงตำแหน่ง odometry เป็นพิกัดช่อง"],
        ["max_speed", "{:.2f} ม./วินาที".format(cal["max_speed"]), "ความเร็วเดินหน้าสูงสุด กำหนดเวลาที่ใช้ต่อหนึ่งช่อง"],
        ["turn_speed", "{:.0f} °/วินาที".format(cal["turn_speed"]), "ความเร็วเชิงมุมสูงสุดขณะหมุนตัว"],
        ["TURN_TOL / POS_TOL", "{:.0f}° / {:.0f} ซม.".format(cm.TURN_TOL_DEG, cm.POS_TOL_M * 100),
         "เกณฑ์ว่าการหมุนหรือการเดินหนึ่งช่องเสร็จแล้ว ค่าเล็กแม่นกว่าแต่ใช้เวลานานกว่า"],
        ["HIT_TOF_MM", "{:.0f} มม.".format(cm.HIT_TOF_MM), "ระยะ ToF ที่ต่ำกว่านี้ถือว่าชนสิ่งกีดขวาง"],
    ], num_cols=(1,), widths=("24%", "20%", "56%")))

    # ---------------- 4
    add(h(1, "4. การออกแบบโปรแกรม (Program Design)"))
    add(h(2, "4.1 โครงสร้างของโปรแกรม"))
    add("<p>ส่วนค้นหาเส้นทาง ส่วนควบคุมหุ่นยนต์ และหน้าจอ pygame อยู่ในไฟล์ <code>src/cost_map_panel.py</code> "
        "ส่วนบันทึกผลเป็น notebook อยู่ในไฟล์ <code>src/run_notebook.py</code> องค์ประกอบหลักแบ่งได้ดังตารางที่ 5</p>")
    add(table("ตารางที่ 5&nbsp; องค์ประกอบหลักของโปรแกรมและหน้าที่", ["องค์ประกอบ", "หน้าที่"], [
        ["<code>bfs</code> / <code>astar</code>", "ค้นหาเส้นทางบนแผนที่ คืนค่าเส้นทางและลำดับโหนดที่ขยาย"],
        ["<code>path_cost</code>", "รวมต้นทุนของช่องที่เข้าไปหลังออกจาก S"],
        ["<code>SimRobot</code>", "หุ่นยนต์จำลองเชิงจลนศาสตร์ ใช้ตรวจสอบโปรแกรมก่อนสั่งหุ่นยนต์จริง"],
        ["<code>RealRobot</code>", "เชื่อมกับ SDK จริง อ่านตำแหน่ง มุม yaw มุม gimbal และ ToF แล้วสั่ง chassis.drive_speed"],
        ["<code>PathRunner</code>", "เดินตามเส้นทางทีละช่องแบบ closed-loop หมุนตัวเข้าทิศก่อนแล้วจึงเดินหน้า"],
        ["<code>App</code>", "หน้าจอ pygame แสดงแผนที่ เส้นทาง ตำแหน่งสด แท็บ CALIBRATE และปุ่ม SAVE RESULT"],
        ["<code>build_run</code>", "รวมข้อมูลรอบ คำนวณ planned steps, actual steps, time, hit obstacle และ reach goal"],
        ["<code>run_notebook.save</code>", "บันทึกรอบเป็น notebook ตั้งชื่ออัตโนมัติ BFS_n หรือ A_star_n"],
    ], widths=("30%", "70%")))
    add(h(2, "4.2 การแยกชั้นควบคุมหุ่นยนต์ออกจากอัลกอริทึม"))
    add("""<p>ตัวควบคุมทั้งสองชนิดมีเมท็อดเหมือนกัน คือ pose, drive, stop และ set_zero ทำให้ PathRunner
ทำงานได้โดยไม่ต้องทราบว่ากำลังสั่งหุ่นยนต์จริงหรือตัวจำลอง การเดินแต่ละช่องแบ่งเป็นสองขั้น
ขั้นแรกหมุนตัวจนทิศผิดพลาดน้อยกว่า {tt:.0f} องศา ขั้นที่สองเดินหน้าด้วยตัวควบคุมแบบสัดส่วนจากระยะที่เหลือ
พร้อมแก้ระยะด้านข้างและทิศทางไปพร้อมกัน จนระยะที่เหลือน้อยกว่า {pt:.0f} เซนติเมตร
คำสั่งความเร็วทุกครั้งกำหนด timeout 0.5 วินาที หากลูปควบคุมหยุดทำงาน หุ่นยนต์จะหยุดเอง</p>
<pre class="code">along   = (dx·cos h + dy·sin h) · cell_m        # ระยะที่เหลือตามทิศเดิน
lateral = (dx·sin h − dy·cos h) · cell_m        # ระยะเบี่ยงด้านข้าง
robot.drive(clamp(1.5·along, 0.05, max_speed),
            clamp(1.5·lateral, −0.1, 0.1),
            clamp(2.0·wrap(h − heading), −30, 30))
# RealRobot: chassis.drive_speed(x, y·y_sign, z·z_sign, timeout=0.5)</pre>
<p class="cap">ตัวอย่างที่ 1&nbsp; การคำนวณคำสั่งความเร็วขณะเดินหนึ่งช่องใน PathRunner._forward</p>""".format(
        tt=cm.TURN_TOL_DEG, pt=cm.POS_TOL_M * 100))
    add(h(2, "4.3 พารามิเตอร์ที่ตั้งค่าได้"))
    add("<p>ค่าทั้งหมดปรับได้จากแท็บ CALIBRATE ของหน้าจอ และบันทึกลงไฟล์ "
        "<code>calibration_output/motion_calibration.json</code> ค่าที่ใช้ในทั้ง {} รอบถูกบันทึกไว้ในไฟล์ผลลัพธ์ของแต่ละรอบ</p>".format(n_runs))
    add(table("ตารางที่ 6&nbsp; พารามิเตอร์ของโปรแกรมและค่าที่ใช้ในการทดลองนี้", ["ตัวเลือก", "ค่าที่ใช้", "ความหมาย"], [
        ["cell_m", "{:.2f}".format(cal["cell_m"]), "ความกว้างของช่องกริดเป็นเมตร"],
        ["start_heading", "BFS: {} / A*: {}".format(headings["BFS"], headings["A*"]),
         "ทิศที่หุ่นยนต์หันที่จุด S ตั้งให้ตรงกับทิศของก้าวแรกในแต่ละเส้นทาง"],
        ["forward_scale", "{:.3f}".format(cal["forward_scale"]), "อัตราส่วนระยะ odometry ต่อระยะจริง"],
        ["max_speed", "{:.2f}".format(cal["max_speed"]), "ความเร็วเดินหน้าสูงสุด (ม./วินาที)"],
        ["turn_speed", "{:.0f}".format(cal["turn_speed"]), "ความเร็วหมุนสูงสุด (°/วินาที)"],
        ["z_sign", "{:+d}".format(int(cal["z_sign"])), "เครื่องหมายของคำสั่งหมุน ปรับจากการทดสอบ TEST LEFT 90 บนหุ่นยนต์จริง"],
        ["yaw_sign / y_sign", "{:+d} / {:+d}".format(int(cal["yaw_sign"]), int(cal["y_sign"])), "เครื่องหมายของมุม yaw และแกน y"],
        ["tof_offset_mm", "{:.0f}".format(cal["tof_offset_mm"]), "ระยะเลนส์ ToF ห่างจากจุดศูนย์กลางตัวรถ"],
    ], num_cols=(1,), widths=("26%", "22%", "52%")))
    add(h(2, "4.4 ข้อมูลที่บันทึกในแต่ละรอบ"))
    add("""<p>ระหว่างการเดิน โปรแกรมบันทึกตำแหน่งทุก 0.1 วินาที (เฉลี่ยจริง {dt:.3f} วินาทีต่อแถว) หนึ่งรอบมีประมาณ
{ns} แถว เมื่อจบรอบ โปรแกรมคำนวณตัวชี้วัดตามนิยามในตารางที่ 7 แล้วบันทึกทั้งหมดลงใน notebook</p>
<pre class="code">samples: t, x, y, yaw, mx, my, heading, tof
result : planned_steps, actual_steps, time_s, hit_obstacle, reach_goal,
         path_cost, final_cell, goal_error_m, travelled_m</pre>
<p class="cap">ตัวอย่างที่ 2&nbsp; คอลัมน์ข้อมูลรายแถวและตัวชี้วัดสรุปที่บันทึกในแต่ละรอบ</p>""".format(
        dt=st.mean(m["dt"] for m in ms), ns=int(round(st.mean(m["samples"] for m in ms)))))
    add(table("ตารางที่ 7&nbsp; นิยามของตัวชี้วัดที่บันทึก", ["ตัวชี้วัด", "นิยาม"], [
        ["planned steps", "จำนวนก้าวของเส้นทางที่อัลกอริทึมวางแผน"],
        ["actual steps", "จำนวนครั้งที่หุ่นยนต์เข้าสู่ช่องใหม่ นับเมื่อจุดศูนย์กลางอยู่ห่างกึ่งกลางช่องไม่เกิน 0.35 ช่อง (21 ซม.) ทั้งสองแกน"],
        ["time (s)", "เวลาจากการกด RUN จนจบการเดิน"],
        ["hit obstacle", "YES เมื่อจุดศูนย์กลางเข้าช่องสิ่งกีดขวางหรือออกนอกสนาม หรือ ToF อ่านได้ต่ำกว่า {:.0f} มม.".format(cm.HIT_TOF_MM)],
        ["reach goal", "YES เมื่อ PathRunner จบด้วยสถานะ Done และช่องสุดท้ายที่เข้าคือ G"],
        ["goal_error_m", "ระยะจากตำแหน่งสุดท้ายถึงกึ่งกลางช่อง G ตาม odometry"],
    ], widths=("22%", "78%")))

    # ---------------- 5
    add(h(1, "5. วิธีดำเนินการทดลอง (Experimental Procedure)"))
    add(h(2, "5.1 ขั้นตอนการทดลอง"))
    add("""<ol>
<li>ตีเส้นตารางกริด 4×4 ช่องละ {c:.2f} เมตร วางป้าย S และ G และวางสิ่งกีดขวางตามรูปที่ 1</li>
<li>ทดสอบโปรแกรมในโหมด SIMULATION ว่า BFS และ A* ให้เส้นทางตามที่คำนวณ และหุ่นยนต์จำลองเดินถึงเป้าหมาย</li>
<li>เปิดโปรแกรมในโหมดหุ่นยนต์จริง กด CONNECT REAL แล้วตรวจทิศการหมุนด้วยปุ่ม TEST LEFT 90
และปรับ Z SIGN จนหุ่นยนต์หมุนซ้ายจริง จากนั้นกด SAVE CALIBRATION</li>
<li>วางหุ่นยนต์ที่กึ่งกลางช่อง S ให้หันไปทางทิศของก้าวแรก (BFS หันทิศ {hb} และ A* หันทิศ {ha})
ตั้งค่า start heading ให้ตรงกัน แล้วกด SET START POSE</li>
<li>เลือก FOLLOW เป็นอัลกอริทึมที่ต้องการ กด RUN และยืนยันในหน้าต่าง CONFIRM</li>
<li>เมื่อหุ่นยนต์หยุด กด SAVE RESULT เพื่อบันทึกรอบเป็น notebook</li>
<li>นำหุ่นยนต์กลับจุด S กด SET START POSE แล้วทำซ้ำจนครบอัลกอริทึมละ 3 รอบ</li></ol>""".format(
        c=cal["cell_m"], hb=headings["BFS"], ha=headings["A*"]))
    add(h(2, "5.2 คำสั่งที่ใช้และไฟล์ผลลัพธ์"))
    add("""<pre class="code">python main.py costmap --mode real
python analyze/make_report.py</pre>
<p class="cap">ตัวอย่างที่ 3&nbsp; คำสั่งเปิดหน้าจอควบคุมหุ่นยนต์จริง และคำสั่งสร้างรายงานฉบับนี้</p>""")
    add(table("ตารางที่ 8&nbsp; รอบการทดลองที่นำมาวิเคราะห์", ["รอบ", "อัลกอริทึม", "โหมด", "เวลาบันทึก", "จำนวนแถวข้อมูล"],
              [[m["name"], m["algo"], m["run"]["mode"].upper(), m["run"]["saved_at"], m["samples"]] for m in ms],
              num_cols=(4,), widths=("18%", "16%", "14%", "30%", "22%")))
    add(h(2, "5.3 มาตรการความปลอดภัย"))
    add("""<ul>
<li>กำหนดผู้ดูแลความปลอดภัยหนึ่งคน ประจำอยู่ใกล้สวิตช์ปิดของหุ่นยนต์ตลอดการรัน</li>
<li>ห้ามมีผู้ใดยืนอยู่ในสนามขณะโปรแกรมทำงาน</li>
<li>การรันบนหุ่นยนต์จริงต้องกดยืนยันในหน้าต่าง CONFIRM ทุกครั้ง และกด Space หรือ F1 เพื่อหยุดได้ตลอดเวลา</li>
<li>คำสั่ง drive_speed ทุกครั้งกำหนด timeout 0.5 วินาที และโปรแกรมสั่งหยุดล้อเมื่อปิดหน้าต่าง</li>
<li>หากหุ่นยนต์หมุนผิดทิศจนมุมผิดพลาดเพิ่มขึ้นเกิน 30 องศา โปรแกรมยกเลิกการเดินและแจ้งให้ปรับ Z SIGN หรือ YAW SIGN</li>
<li>เปิดโปรแกรมจาก terminal ของผู้ทดลองเอง เพื่อไม่ให้โปรเซสถูกปิดจากเครื่องมืออื่นขณะหุ่นยนต์กำลังเคลื่อนที่</li></ul>""")

    # ---------------- 6
    add('<div class="pb"></div>')
    add(h(1, "6. ผลการทดลอง (Results)"))
    add(h(2, "6.1 ผลสรุปรายรอบ"))
    rows = []
    for m in ms:
        r = m["run"]["result"]
        rows.append([m["name"], m["algo"], r["planned_steps"], r["actual_steps"], r["path_cost"],
                     "{:.2f}".format(r["time_s"]), r["hit_obstacle"], r["reach_goal"]])
    add(table("ตารางที่ 9&nbsp; ผลการรันบนหุ่นยนต์จริงทั้ง {} รอบ".format(n_runs),
              ["รอบ", "อัลกอริทึม", "planned steps", "actual steps", "ต้นทุน", "time (s)", "hit obstacle", "reach goal"],
              rows, num_cols=(2, 3, 4, 5), widths=("14%", "12%", "13%", "13%", "10%", "12%", "13%", "13%")))
    add("""<p>หุ่นยนต์{ok}ใน {n} รอบ{hit} จำนวนก้าวจริงเท่ากับจำนวนก้าวที่วางแผนไว้ 6 ก้าวในทุกรอบ
เวลาที่ใช้ของทุกรอบอยู่ในช่วง {tmin:.2f} ถึง {tmax:.2f} วินาที</p>""".format(
        ok="เดินถึงเป้าหมายครบทุกรอบ" if all_ok else "เดินถึงเป้าหมาย {} รอบ".format(
            sum(m["run"]["result"]["reach_goal"] == "YES" for m in ms)),
        n=n_runs, hit=" และไม่มีรอบใดตรวจพบการชนสิ่งกีดขวาง" if no_hit else "",
        tmin=min(m["time"] for m in ms), tmax=max(m["time"] for m in ms)))
    add(h(2, "6.2 เปรียบเทียบ BFS กับ A*"))

    def both(fn, fmt="{:.2f}"):
        return [mean_sd((fn(m) for m in by[a]), fmt) for a in ALGOS]

    add(table("ตารางที่ 10&nbsp; เปรียบเทียบตัวชี้วัดของสองอัลกอริทึม (ค่าเฉลี่ย ± ส่วนเบี่ยงเบนมาตรฐาน จาก 3 รอบ)",
              ["ตัวชี้วัด", "BFS", "A*", "ความหมาย"], [
        ["จำนวนก้าวที่วางแผน", len(bfs_path) - 1, len(ast_path) - 1, "เท่ากัน"],
        ["ต้นทุนรวมของเส้นทาง", cm.path_cost(bfs_path), cm.path_cost(ast_path),
         "A* ต่ำกว่า {} ({:.0%})".format(cm.path_cost(bfs_path) - cm.path_cost(ast_path),
                                        1 - cm.path_cost(ast_path) / float(cm.path_cost(bfs_path)))],
        ["จำนวนโหนดที่ขยาย", len(bfs_exp), len(ast_exp), "จากโหนดทั้งหมด {} โหนด".format(len(cm.COST))],
        ["จำนวนครั้งที่หมุนตัว", by["BFS"][0]["turns"], by["A*"][0]["turns"], "นับจากทิศเริ่มต้นที่ S"],
        ["อัตราการถึงเป้าหมาย", "{}/3".format(sum(m["run"]["result"]["reach_goal"] == "YES" for m in by["BFS"])),
         "{}/3".format(sum(m["run"]["result"]["reach_goal"] == "YES" for m in by["A*"])), ""],
        ["เวลา (วินาที)"] + both(lambda m: m["time"]) + ["ต่างกัน {:.2f} วินาที".format(abs(t_mean["A*"] - t_mean["BFS"]))],
        ["ระยะทางที่เคลื่อนที่ (ม.)"] + both(lambda m: m["travel"], "{:.3f}") + ["เส้นทางอุดมคติ {:.2f} ม.".format(6 * cal["cell_m"])],
        ["ความคลาดเคลื่อนที่เป้าหมาย (ซม.)"] + both(lambda m: m["goal_cm"]) + ["ตาม odometry"],
        ["ระยะห่างจากเส้นทางสูงสุด (ซม.)"] + both(lambda m: m["xte_max"]) + ["ตาม odometry"],
        ["ระยะห่างจากเส้นทาง RMS (ซม.)"] + both(lambda m: m["xte_rms"]) + ["ตาม odometry"],
        ["ทิศผิดพลาดที่เป้าหมาย (องศา)"] + both(lambda m: m["head_err"]) + ["เทียบทิศหลักที่ใกล้ที่สุด"],
    ], num_cols=(1, 2), widths=("30%", "19%", "19%", "32%")))
    add("""<p>ความแตกต่างเชิงแผนระหว่างสองอัลกอริทึมมีเพียงต้นทุนรวมและจำนวนโหนดที่ขยาย
ส่วนตัวชี้วัดของการเคลื่อนที่จริงทุกตัวใกล้เคียงกันมาก เวลาเฉลี่ยของ A* ({ta:.2f} วินาที) ต่างจาก BFS ({tb:.2f} วินาที)
เพียง {dd:.2f} วินาที ซึ่งอยู่ในระดับเดียวกับความแปรปรวนระหว่างรอบของอัลกอริทึมเดียวกัน</p>
{fig}""".format(ta=t_mean["A*"], tb=t_mean["BFS"], dd=abs(t_mean["A*"] - t_mean["BFS"]),
                 fig=figure("fig6_time.png", "รูปที่ 2&nbsp; เวลาที่ใช้ของแต่ละรอบ (สีเทา BFS สีเข้ม A*)", "92%")))
    add(h(2, "6.3 เส้นทางที่วางแผนและลำดับการขยายโหนด"))
    add(figure("fig2_plans.png", "รูปที่ 3&nbsp; เส้นทางที่วางแผนโดย BFS (เส้นประ) และ A* (เส้นทึบ) ตัวเลขมุมช่องคือต้นทุน", "58%"))
    exp_rows = []
    for i in range(max(len(bfs_exp), len(ast_exp))):
        exp_rows.append([i + 1, cell(bfs_exp[i]) if i < len(bfs_exp) else "", cell(ast_exp[i]) if i < len(ast_exp) else ""])
    add(table("ตารางที่ 11&nbsp; ลำดับการขยายโหนดของ BFS และ A*", ["ลำดับ", "BFS", "A*"], exp_rows,
              num_cols=(0,), widths=("20%", "40%", "40%")))
    add("""<p>BFS ใช้เส้นทางด้านบนขวา {bp} เพราะลำดับเพื่อนบ้านเริ่มจากทิศ E ทำให้ช่อง (2, 4) เข้าคิวก่อน (1, 3)
และสายของ (2, 4) ไปถึง G ก่อนในระดับความลึกที่ 6 ส่วน A* ใช้เส้นทางด้านล่างซ้าย {ap}
ซึ่งผ่านช่องต้นทุน 1 ที่ (2, 2) และหลีกเลี่ยงช่องต้นทุน 4 ที่ (3, 3) และ (4, 2) BFS ขยายโหนดครบทั้ง {nb} โหนด
ส่วน A* ขยาย {na} โหนด โดยไม่ขยาย (4, 2) เพราะ f(4, 2) = {f42} สูงกว่าต้นทุนของคำตอบ {c}</p>""".format(
        bp=" → ".join(cell(c) for c in bfs_path), ap=" → ".join(cell(c) for c in ast_path),
        nb=len(bfs_exp), na=len(ast_exp), f42=g_final.get((4, 2), 0) + 1, c=cm.path_cost(ast_path)))
    add(h(2, "6.4 เส้นทางจริงของหุ่นยนต์"))
    add(figure("fig3_trajectories.png", "รูปที่ 4&nbsp; ตำแหน่งจริงของหุ่นยนต์ตาม odometry ทั้ง 6 รอบ ซ้อนทับเส้นทางที่วางแผน", "96%"))
    add(figure("fig4_xte.png", "รูปที่ 5&nbsp; ระยะห่างของจุดศูนย์กลางหุ่นยนต์จากเส้นทางที่วางแผนตลอดการเดิน", "94%"))
    xmax = max(ms, key=lambda m: m["xte_max"])
    add("""<p>จากรูปที่ 4 เส้นทางจริงทั้ง 6 รอบซ้อนทับเส้นทางที่วางแผนจนแยกไม่ออกที่มาตราส่วนของสนาม
รูปที่ 5 แสดงว่าระยะห่างจากเส้นทางสูงสุดของทุกรอบไม่เกิน {xm:.2f} เซนติเมตร (รอบ {xn})
ค่าสูงสุดเกิดช่วงสั้น ๆ ขณะหุ่นยนต์หมุนตัวที่มุมเลี้ยว ซึ่งจุดศูนย์กลางของหุ่นยนต์ขยับเล็กน้อยระหว่างหมุน</p>""".format(
        xm=xmax["xte_max"], xn=xmax["name"]))
    add(h(2, "6.5 เวลาการเคลื่อนที่รายช่อง"))
    seg_rows = []
    for i in range(6):
        row = [i + 1]
        for a in ALGOS:
            m0 = by[a][0]
            row += ["{} → {} ({}{})".format(cell(m0["path"][i]), cell(m0["path"][i + 1]), DIR_NAME[m0["dirs"][i]],
                                             ", หมุน" if m0["turned"][i] else ""),
                    mean_sd((m["seg"][i] for m in by[a]))]
        seg_rows.append(row)
    add(table("ตารางที่ 12&nbsp; เวลาที่ใช้ต่อหนึ่งก้าว นับจากถึงกึ่งกลางช่องก่อนหน้าถึงกึ่งกลางช่องถัดไป (วินาที, 3 รอบ)",
              ["ก้าว", "BFS: การเคลื่อนที่", "BFS: เวลา", "A*: การเคลื่อนที่", "A*: เวลา"], seg_rows,
              num_cols=(0, 2, 4), widths=("8%", "30%", "16%", "30%", "16%")))
    add(figure("fig5_heading.png", "รูปที่ 6&nbsp; ทิศของหุ่นยนต์บนแผนที่ตลอดการเดิน รอบ BFS_1 และ A_star_1", "94%"))
    add("""<p>ก้าวที่ต้องหมุนตัวก่อนเดินใช้เวลาเฉลี่ย {ts:.2f} วินาที ส่วนก้าวที่เดินตรงโดยไม่หมุนใช้เวลาเฉลี่ย {ss:.2f} วินาที
การหมุนตัวหนึ่งครั้งจึงเพิ่มเวลาประมาณ {dt:.1f} วินาที ทั้งสองเส้นทางมีการหมุนตัว 3 ครั้งเท่ากัน (รูปที่ 6)
และมีจำนวนก้าวเท่ากัน เวลารวมจึงใกล้เคียงกัน ก้าวแรกใช้เวลาเฉลี่ยเพียง {fs:.2f} วินาที เพราะเริ่มนับจากกึ่งกลางช่อง S
ขณะที่ก้าวอื่นเริ่มนับตั้งแต่หุ่นยนต์เข้าใกล้กึ่งกลางช่องก่อนหน้าในระยะ 6 ซม. จึงรวมช่วงที่ตัวควบคุมลดความเร็วเข้ากึ่งกลางช่องไว้ด้วย</p>""".format(
        ts=st.mean(turn_seg), ss=st.mean(straight_seg), dt=st.mean(turn_seg) - st.mean(straight_seg),
        fs=st.mean(first_seg)))
    add(h(2, "6.6 ค่าที่อ่านได้จากเซนเซอร์ ToF"))
    add(figure("fig7_tof.png", "รูปที่ 7&nbsp; ระยะที่อ่านได้จาก ToF บน gimbal ตลอดการเดิน เส้นประแนวนอนคือเกณฑ์การชน {:.0f} มม.".format(
        cm.HIT_TOF_MM), "94%"))
    tof_rows = [[m["name"], m["tof_start"], "{} (t = {:.1f} s ที่ {})".format(
        m["tof_min"]["tof"], m["tof_min"]["t"], cell((round(m["tof_min"]["mx"]), round(m["tof_min"]["my"])))),
        m["tof_end"], "YES" if m["tof_min"]["tof"] < cm.HIT_TOF_MM else "NO"] for m in ms]
    add(table("ตารางที่ 13&nbsp; ค่า ToF ที่จุดเริ่มต้น ค่าต่ำสุดระหว่างทาง และค่าที่เป้าหมาย (มม.)",
              ["รอบ", "เริ่มต้น", "ต่ำสุด", "ที่เป้าหมาย", "ต่ำกว่าเกณฑ์"], tof_rows,
              num_cols=(1, 3), widths=("18%", "14%", "36%", "16%", "16%")))
    add("""<p>ค่า ToF ต่ำสุดของทุกรอบอยู่ระหว่าง {lo} ถึง {hi} มม. สูงกว่าเกณฑ์การชน {th:.0f} มม. ทุกรอบ
ที่เป้าหมาย รอบ A* ทั้งสามรอบหันทิศ E และอ่านได้ {ae} มม. ส่วนรอบ BFS หันทิศ S และอ่านได้ {be} มม.
ค่าที่เป้าหมายของ BFS_2 และ BFS_3 ต่ำกว่า BFS_1 อย่างชัดเจน แสดงว่ามีวัตถุอยู่ด้านหน้าหุ่นยนต์ใกล้กว่าในรอบนั้น
ข้อมูลที่บันทึกไม่ระบุว่าวัตถุนั้นคืออะไร</p>""".format(
        lo=min(m["tof_min"]["tof"] for m in ms), hi=max(m["tof_min"]["tof"] for m in ms), th=cm.HIT_TOF_MM,
        ae=", ".join(str(m["tof_end"]) for m in by["A*"]), be=", ".join(str(m["tof_end"]) for m in by["BFS"])))

    # ---------------- 7
    add(h(1, "7. การตรวจสอบความถูกต้อง (Verification)"))
    add("<p>เพื่อยืนยันว่าโปรแกรมคำนวณตามทฤษฎีในหัวข้อที่ 3 จริง จึงคำนวณต้นทุนเส้นทาง "
        "และลำดับการขยายโหนดของ A* ด้วยมือ แล้วเปรียบเทียบกับค่าที่โปรแกรมบันทึกไว้ในไฟล์ผลลัพธ์</p>")
    add(h(2, "7.1 การคำนวณต้นทุนเส้นทางด้วยมือ"))
    add("""<pre class="eq">BFS : cost(2,4) + cost(3,4) + cost(3,3) + cost(4,3) + cost(4,2) + cost(4,1)
    = {b}
A*  : cost(1,3) + cost(1,2) + cost(2,2) + cost(2,1) + cost(3,1) + cost(4,1)
    = {a}</pre>""".format(
        b=" + ".join(str(cm.COST[c]) for c in bfs_path[1:]) + " = " + str(cm.path_cost(bfs_path)),
        a=" + ".join(str(cm.COST[c]) for c in ast_path[1:]) + " = " + str(cm.path_cost(ast_path))))
    add(h(2, "7.2 ลำดับการขยายโหนดของ A*"))
    add("<p>ค่า h ของแต่ละโหนดคือระยะ Manhattan ถึง (4, 1) คูณ 1 ค่า g คือผลรวมต้นทุนตามเส้นทางที่ดีที่สุดที่พบ ณ ขณะนั้น "
        "และโหนดถูกดึงออกจากคิวตามค่า f จากน้อยไปมาก หากค่า f เท่ากันจะเลือกค่า g ที่น้อยกว่าก่อน</p>")
    tr_rows = []
    for i, (c, g, hh, f, par) in enumerate(trace):
        tr_rows.append([i + 1, cell(c), "—" if par is None else cell(par),
                        "0" if par is None else "{} + {} = {}".format(g - cm.COST[c], cm.COST[c], g),
                        hh, f, cell(ast_exp[i]), "ตรงกัน" if ast_exp[i] == c else "ไม่ตรง"])
    add(table("ตารางที่ 14&nbsp; ค่าที่คำนวณด้วยมือ เทียบกับลำดับการขยายที่โปรแกรมบันทึก",
              ["ลำดับ", "โหนด", "มาจาก", "g", "h", "f", "โปรแกรม", "ผล"], tr_rows,
              num_cols=(0, 3, 4, 5), widths=("8%", "13%", "13%", "18%", "7%", "7%", "15%", "12%")))
    add(table("ตารางที่ 15&nbsp; เปรียบเทียบผลการคำนวณด้วยมือกับค่าที่โปรแกรมบันทึก",
              ["ปริมาณ", "คำนวณด้วยมือ", "ค่าในไฟล์ผลลัพธ์", "ผลการตรวจสอบ"], [
        ["ต้นทุนเส้นทาง BFS", cm.path_cost(bfs_path), by["BFS"][0]["run"]["plans"]["BFS"]["cost"],
         "ตรงกัน" if cm.path_cost(bfs_path) == by["BFS"][0]["run"]["plans"]["BFS"]["cost"] else "ไม่ตรง"],
        ["ต้นทุนเส้นทาง A*", cm.path_cost(ast_path), by["A*"][0]["run"]["plans"]["A*"]["cost"],
         "ตรงกัน" if cm.path_cost(ast_path) == by["A*"][0]["run"]["plans"]["A*"]["cost"] else "ไม่ตรง"],
        ["จำนวนโหนดที่ A* ขยาย", len(trace), len(by["A*"][0]["run"]["plans"]["A*"]["expanded"]),
         "ตรงกัน" if len(trace) == len(by["A*"][0]["run"]["plans"]["A*"]["expanded"]) else "ไม่ตรง"],
    ], num_cols=(1, 2), widths=("34%", "22%", "22%", "22%")))
    last = by["A*"][0]["run"]["samples"][-1]
    add(h(2, "7.3 การตรวจสอบตัวชี้วัดการถึงเป้าหมาย"))
    add("""<p>แถวสุดท้ายของรอบ A_star_1 บันทึกตำแหน่งบนแผนที่ (mx, my) = ({mx:.4f}, {my:.4f})
เมื่อปัดเป็นช่องได้ (4, 1) ซึ่งคือ G ความคลาดเคลื่อนที่เป้าหมายคำนวณได้ดังนี้</p>
<pre class="eq">goal_error = √((4 − {mx:.4f})² + (1 − {my:.4f})²) × {c:.2f}
           = {ge:.4f} m  ≈  {gr:.3f} m</pre>
<p>ค่าที่โปรแกรมบันทึกคือ {rec:.3f} เมตร ตรงกันเมื่อปัดเป็นทศนิยมสามตำแหน่ง</p>""".format(
        mx=last["mx"], my=last["my"], c=cal["cell_m"],
        ge=math.hypot(4 - last["mx"], 1 - last["my"]) * cal["cell_m"],
        gr=round(math.hypot(4 - last["mx"], 1 - last["my"]) * cal["cell_m"], 3),
        rec=by["A*"][0]["run"]["result"]["goal_error_m"]))

    # ---------------- 8
    add(h(1, "8. อภิปรายผล (Discussion)"))
    add(h(2, "8.1 “สั้นที่สุด” ในเชิงจำนวนก้าวกับเชิงต้นทุน"))
    add("""<p>ทั้งสองอัลกอริทึมให้เส้นทาง 6 ก้าวซึ่งสั้นที่สุดในเชิงจำนวนก้าว แต่ต้นทุนของ BFS ({cb}) สูงกว่า A* ({ca})
ต้นทุนของ BFS เป็นผลจากลำดับการเพิ่มเพื่อนบ้าน ไม่ใช่ผลของการพิจารณาต้นทุน เมื่อทดลองสลับลำดับเพื่อนบ้านครบทั้ง
24 แบบ BFS ให้ต้นทุนเส้นทางที่เป็นไปได้ {oc} ขึ้นกับลำดับที่ใช้ ในขณะที่ A* ให้ต้นทุน {ca} เสมอ
ผลนี้ยืนยันว่าถ้าต้องการต้นทุนต่ำสุด ต้องใช้อัลกอริทึมที่นำต้นทุนมาคำนวณ เช่น A* หรือ Dijkstra</p>""".format(
        cb=cm.path_cost(bfs_path), ca=cm.path_cost(ast_path), oc=", ".join(str(c) for c in order_costs)))
    add(h(2, "8.2 ต้นทุนบนแผนที่ไม่ได้แปลงเป็นเวลาจริง"))
    add("""<p>แม้ A* จะมีต้นทุนต่ำกว่า BFS ร้อยละ {pct:.0f} แต่เวลาจริงแทบไม่ต่างกัน สาเหตุคือตัวเลขต้นทุนในใบงานเป็นค่าเชิงนามธรรม
พื้นสนามทุกช่องเหมือนกัน หุ่นยนต์จึงใช้เวลาต่อช่องเท่ากันไม่ว่าต้นทุนจะเป็นเท่าใด
เวลารวมขึ้นกับจำนวนก้าวและจำนวนครั้งที่หมุนตัว ซึ่งทั้งสองเส้นทางเท่ากันที่ 6 ก้าวและ 3 ครั้ง
หากต้นทุนสะท้อนสิ่งที่วัดได้จริง เช่น พื้นผิวที่ต้องลดความเร็ว ความแตกต่างของต้นทุนจึงจะปรากฏในเวลาหรือพลังงานที่ใช้</p>""".format(
        pct=(1 - cm.path_cost(ast_path) / float(cm.path_cost(bfs_path))) * 100))
    add(h(2, "8.3 ความสม่ำเสมอของการเคลื่อนที่"))
    add("""<p>ส่วนเบี่ยงเบนมาตรฐานของเวลาภายในอัลกอริทึมเดียวกันไม่เกิน {sd:.2f} วินาที ความคลาดเคลื่อนที่เป้าหมายไม่เกิน {ge:.1f} ซม.
และทิศผิดพลาดที่เป้าหมายไม่เกิน {he:.2f} องศา ความสม่ำเสมอนี้มาจากการควบคุมแบบ closed-loop
ที่ใช้ตำแหน่งและมุมที่วัดได้ในการตัดสินว่าแต่ละขั้นเสร็จแล้ว แทนการสั่งระยะทางแบบเปิดวง
อย่างไรก็ตาม ค่าเหล่านี้วัดด้วย odometry เดียวกับที่ใช้ควบคุม จึงแสดงว่าตัวควบคุมพาหุ่นยนต์ไปถึงตำแหน่งที่ตัวเองเชื่อว่าเป็นเป้าหมาย
แต่ไม่ได้พิสูจน์ว่าตำแหน่งจริงบนพื้นแม่นยำในระดับเดียวกัน</p>""".format(
        sd=max(st.stdev([m["time"] for m in by[a]]) for a in ALGOS),
        ge=max(m["goal_cm"] for m in ms), he=max(m["head_err"] for m in ms)))
    add(h(2, "8.4 จำนวนโหนดที่ขยาย"))
    add("""<p>A* ขยายโหนดน้อยกว่า BFS เพียง {d} โหนด เพราะสนามมีเพียง {n} โหนด และค่า h ที่ใช้ c<sub>min</sub> = 1
ต่ำกว่าต้นทุนเฉลี่ยจริงของแต่ละก้าว ({avg:.2f}) ค่อนข้างมาก h จึงแยกโหนดที่ดีออกจากโหนดที่ไม่ดีได้ไม่ชัด
ข้อได้เปรียบด้านจำนวนโหนดของ A* จะชัดขึ้นเมื่อสนามใหญ่ขึ้น หรือเมื่อ heuristic ใกล้เคียงต้นทุนจริงมากขึ้นโดยยังไม่ประเมินเกินจริง</p>""".format(
        d=len(bfs_exp) - len(ast_exp), n=len(cm.COST), avg=st.mean(cm.COST.values())))

    # ---------------- 9
    add(h(1, "9. ข้อจำกัดและข้อเสนอแนะ (Limitations and Future Work)"))
    add(h(2, "9.1 ข้อจำกัดของการทดลอง"))
    add("""<ul>
<li><b>ตำแหน่งมาจาก odometry ของหุ่นยนต์เอง</b> ไม่มีการวัดตำแหน่งจริงบนพื้นสนาม ความคลาดเคลื่อนและระยะห่างจากเส้นทางในรายงานจึงเป็นค่าที่หุ่นยนต์ประมาณได้เอง</li>
<li><b>จำนวนรอบน้อย</b> อัลกอริทึมละ 3 รอบ ค่าส่วนเบี่ยงเบนมาตรฐานจึงยังไม่เสถียรพอสำหรับการทดสอบทางสถิติ</li>
<li><b>ทิศเริ่มต้นต่างกัน</b> BFS เริ่มหันทิศ {hb} และ A* เริ่มหันทิศ {ha} เพื่อให้ตรงกับก้าวแรก หากเริ่มทิศเดียวกัน อัลกอริทึมหนึ่งจะต้องหมุนตัวเพิ่ม</li>
<li><b>ต้นทุนเป็นค่านามธรรม</b> ไม่มีความสัมพันธ์กับพื้นผิวหรือความยากของการเคลื่อนที่จริง จึงเปรียบเทียบผลของต้นทุนต่อเวลาจริงไม่ได้</li>
<li><b>การตรวจจับการชนเป็นแบบทางอ้อม</b> ใช้ตำแหน่ง odometry และ ToF ด้านหน้าเพียงตัวเดียว ไม่ครอบคลุมการเฉี่ยวด้านข้างหรือด้านหลัง</li>
<li><b>แผนที่คงที่</b> ไม่มีสิ่งกีดขวางเคลื่อนที่ และไม่มีการวางแผนใหม่ระหว่างทาง</li></ul>""".format(
        hb=headings["BFS"], ha=headings["A*"]))
    add(h(2, "9.2 ข้อเสนอแนะสำหรับการพัฒนาต่อ"))
    add("""<ol>
<li>วัดตำแหน่งสุดท้ายจริงด้วยตลับเมตร หรือกล้องเหนือสนาม แล้วเทียบกับค่าจาก odometry เพื่อหาความคลาดเคลื่อนจริงและปรับค่า forward_scale</li>
<li>เพิ่มจำนวนรอบเป็นอย่างน้อย 10 รอบต่ออัลกอริทึม และเริ่มทุกรอบด้วยทิศเดียวกัน เพื่อให้เปรียบเทียบเวลาได้เป็นธรรม</li>
<li>ให้ต้นทุนมีความหมายทางกายภาพ เช่น จำกัดความเร็วในช่องต้นทุนสูงตามสัดส่วน แล้ววัดว่า A* ประหยัดเวลาจริงได้เท่าใด</li>
<li>เพิ่มต้นทุนของการหมุนตัวลงในการค้นหา เพราะผลในหัวข้อที่ 6.5 แสดงว่าการหมุนหนึ่งครั้งใช้เวลาเพิ่มขึ้นอย่างมีนัย</li>
<li>เปรียบเทียบกับ Dijkstra และ Greedy Best-First Search บนสนามเดียวกัน เพื่อแสดงบทบาทของ g และ h แยกจากกัน</li></ol>""")

    # ---------------- 10
    add(h(1, "10. สรุปผลการทดลอง (Conclusion)"))
    add("""<p>การทดลองนี้แสดงให้เห็นว่าบนสนามกริด 4×4 ที่มีสิ่งกีดขวาง 3 ช่องและต้นทุนต่อช่อง
BFS และ A* ให้เส้นทางที่สั้นที่สุดในเชิงจำนวนก้าวเท่ากันที่ 6 ก้าว แต่ A* ให้ต้นทุนรวม {ca} ต่ำกว่า BFS ที่ {cb}
และขยายโหนดน้อยกว่า ({na} เทียบกับ {nb} โหนด) การคำนวณด้วยมือทั้งต้นทุนเส้นทางและลำดับการขยายโหนดของ A*
ตรงกับค่าที่โปรแกรมบันทึกทุกตำแหน่ง ยืนยันความถูกต้องของการนำอัลกอริทึมไปใช้</p>
<p>บนหุ่นยนต์จริงทั้ง {n} รอบ หุ่นยนต์{ok} จำนวนก้าวจริงเท่ากับจำนวนก้าวที่วางแผนทุกรอบ
เวลาเฉลี่ยของ BFS คือ {tb:.2f} วินาที และของ A* คือ {ta:.2f} วินาที ซึ่งแทบไม่ต่างกัน
เพราะทั้งสองเส้นทางมีจำนวนก้าวและจำนวนการหมุนตัวเท่ากัน ขณะที่ต้นทุนในใบงานไม่ได้สะท้อนความยากของพื้นสนามจริง</p>
<p>ข้อค้นพบสำคัญคือ การเลือกอัลกอริทึมต้องสอดคล้องกับความหมายของต้นทุน หากต้นทุนเป็นเพียงจำนวนก้าว BFS เพียงพอ
แต่หากต้นทุนแทนสิ่งที่มีผลจริง เช่น เวลา พลังงาน หรือความเสี่ยง ต้องใช้ A* จึงจะรับประกันผลที่ดีที่สุด
และควรรวมต้นทุนของการหมุนตัวไว้ในการค้นหาด้วย</p>""".format(
        ca=cm.path_cost(ast_path), cb=cm.path_cost(bfs_path), na=len(ast_exp), nb=len(bfs_exp), n=n_runs,
        ok="เดินถึงเป้าหมายครบทุกรอบโดยไม่ตรวจพบการชนสิ่งกีดขวาง" if all_ok and no_hit else "เดินถึงเป้าหมายตามตารางที่ 9",
        tb=t_mean["BFS"], ta=t_mean["A*"]))

    # ---------------- 11
    add(h(1, "11. ภาคผนวก: ไฟล์ผลลัพธ์และวิธีทำซ้ำ (Appendix)"))
    add(table("ตารางที่ 16&nbsp; ไฟล์ที่เกี่ยวข้องกับการทดลอง", ["ไฟล์", "เนื้อหา"], [
        ["<code>analyze/runs/BFS_1–3.ipynb</code>", "ข้อมูลดิบ ตัวชี้วัด และกราฟของรอบ BFS ทั้ง 3 รอบ"],
        ["<code>analyze/runs/A_star_1–3.ipynb</code>", "ข้อมูลดิบ ตัวชี้วัด และกราฟของรอบ A* ทั้ง 3 รอบ"],
        ["<code>analyze/load_runs.py</code>", "อ่านข้อมูลทุกรอบกลับจาก notebook"],
        ["<code>analyze/make_report.py</code>", "คำนวณตัวชี้วัด สร้างกราฟ และสร้างรายงานฉบับนี้"],
        ["<code>src/cost_map_panel.py</code>", "BFS, A*, ตัวควบคุมหุ่นยนต์ และหน้าจอ pygame"],
        ["<code>src/run_notebook.py</code>", "บันทึกรอบการทดลองเป็น notebook"],
        ["<code>calibration_output/motion_calibration.json</code>", "ค่าที่ตั้งจากแท็บ CALIBRATE"],
    ], widths=("46%", "54%")))
    add("""<p>ตัวเลขทุกค่าในรายงานคำนวณจากไฟล์ใน <code>analyze/runs/</code> การสร้างรายงานซ้ำใช้คำสั่งเดียวในตัวอย่างที่ 3
และหากมีรอบใหม่ถูกบันทึกเพิ่ม รายงานจะรวมรอบนั้นโดยอัตโนมัติ</p>
<pre class="code">python main.py costmap              # ทดสอบในโหมดจำลอง
python tests/test_cost_map.py       # ทดสอบ BFS, A*, การบันทึกผล</pre>
<p class="cap">ตัวอย่างที่ 4&nbsp; คำสั่งทดสอบโปรแกรมก่อนรันบนหุ่นยนต์จริง</p>""")

    # ---------------- TOC
    toc = ['<div class="pb"></div><h1 class="toc-title">สารบัญ</h1><div class="toc">']
    for sid, level, title in sections:
        page = toc_pages.get(sid, "") if isinstance(toc_pages, dict) else ""
        toc.append('<div class="toc-row l{}"><span class="t">{}</span><span class="dots"></span>'
                   '<span class="p">{}</span></div>'.format(level, title, page))
    toc.append("</div>")
    if toc_pages == "docx":  # Word builds its own TOC field from the headings
        toc = ['<div class="pb"></div><p class="toc-title">สารบัญ</p><tocfield></tocfield>']
    content = "\n".join(body).replace("@@TOC@@", "".join(toc))
    return PAGE.replace("@@BODY@@", content), sections


PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>BFS vs A* RoboMaster</title>
<style>
@page { size: A4; margin: 24mm 20mm 22mm 22mm;
  @top-left { content: "หน้า " counter(page); font-family: "Leelawadee UI", Tahoma, sans-serif; font-size: 10pt; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Leelawadee UI", Tahoma, sans-serif; font-size: 11pt; line-height: 1.75; color: #000; margin: 0; }
h1 { font-size: 16pt; font-weight: 700; margin: 16pt 0 6pt; break-after: avoid; }
h2 { font-size: 13pt; font-weight: 700; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; } ol, ul { margin: 0 0 6pt; padding-left: 20pt; } li { margin-bottom: 2pt; }
.pb { break-before: page; }
.mark { position: absolute; color: #fff; font-size: 1px; }
code, pre { font-family: Consolas, monospace; }
code { font-size: 10pt; }
pre.code, pre.eq { font-size: 10pt; line-height: 1.5; margin: 4pt 0 6pt 12pt; white-space: pre-wrap; break-inside: avoid; }
pre.eq { font-size: 10.5pt; }
table { border-collapse: collapse; width: 100%; margin: 0 0 8pt; font-size: 10pt; line-height: 1.45; break-inside: auto; }
tr { break-inside: avoid; }
th { background: #e8e8e8; font-weight: 700; text-align: left; border: 1px solid #9a9a9a; padding: 3pt 6pt; }
td { border: 1px solid #9a9a9a; padding: 3pt 6pt; vertical-align: middle; }
td.num { text-align: right; font-family: Consolas, monospace; white-space: nowrap; }
.cap { font-size: 10pt; margin: 8pt 0 3pt; break-after: avoid; }
.fig { text-align: center; margin: 6pt 0 8pt; break-inside: avoid; }
.fig img { display: block; margin: 0 auto; }
.figcap { font-size: 10pt; text-align: center; margin: 3pt 0 8pt; }
table.grid { width: auto; margin: 8pt auto 2pt; font-size: 11pt; }
table.grid td { width: 46pt; height: 40pt; text-align: center; border: 1px solid #333; line-height: 1.25; }
table.grid td.g-x { background: #5f5f5f; color: #fff; font-weight: 700; }
table.grid td.g-sg { background: #d9d9d9; }
table.grid th.g-ax { background: none; border: none; text-align: center; font-weight: 400; color: #555; }
table.grid th { background: none; border: none; }
.cover { text-align: center; padding-top: 80pt; }
.cover .title { display: block; font-size: 21pt; font-weight: 700; line-height: 1.55; }
.cover .sub { font-size: 13pt; margin: 18pt 0 22pt; }
.cover .abstract { font-size: 10.5pt; margin: 0 40pt 40pt; line-height: 1.8; }
.cover .by { font-weight: 700; font-size: 13pt; margin-bottom: 4pt; }
table.members { width: auto; margin: 0 auto 40pt; font-size: 11pt; }
table.members td { border: none; padding: 0 8pt; text-align: left; }
.cover .foot { margin-bottom: 26pt; }
.toc-title { margin-top: 0; }
.toc-row { display: flex; align-items: baseline; font-size: 10.5pt; line-height: 1.9; }
.toc-row.l2 { padding-left: 16pt; }
.toc-row .dots { flex: 1; border-bottom: 1.5px dotted #555; margin: 0 4pt; transform: translateY(-4pt); }
.toc-row .p { min-width: 16pt; text-align: right; }
.toc + h1, .toc-row.l1 .t { }
</style></head><body>@@BODY@@</body></html>"""


# ---------------------------------------------------------------- pdf
def edge():
    for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", "")):
        exe = Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
        if exe.exists():
            return str(exe)
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def print_pdf(html_path, pdf_path):
    browser = edge()
    if not browser:
        raise RuntimeError("Microsoft Edge / Chrome not found - open {} and print to PDF manually".format(html_path))
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
                    "--print-to-pdf={}".format(pdf_path), html_path.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def section_pages(pdf_path, sections):
    import fitz  # PyMuPDF
    pages = {}
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            for sid, _, _ in sections:
                if sid not in pages and "@@{}@@".format(sid) in text:
                    pages[sid] = i + 1
    return pages


# ---------------------------------------------------------------- docx
class _Node(object):
    def __init__(self, tag, attrs, parent=None):
        self.tag, self.attrs, self.parent, self.kids = tag, dict(attrs), parent, []

    @property
    def cls(self):
        return self.attrs.get("class", "")


def _parse(markup):
    from html.parser import HTMLParser

    root = _Node("root", {})

    class P(HTMLParser):
        cur = root

        def handle_starttag(self, tag, attrs):
            node = _Node(tag, attrs, self.cur)
            self.cur.kids.append(node)
            if tag not in ("br", "img", "col"):
                self.cur = node

        def handle_endtag(self, tag):
            node = self.cur
            while node is not root and node.tag != tag:
                node = node.parent
            if node is not root:
                self.cur = node.parent

        def handle_data(self, data):
            self.cur.kids.append(data)

    P(convert_charrefs=True).feed(markup)
    return root


def _runs(node, fmt=None, keep_space=False):
    """Inline content -> [{text, b, code, sub}] ; {"br": true} for line breaks."""
    fmt, out = dict(fmt or {}), []
    for kid in node.kids:
        if isinstance(kid, str):
            text = kid if keep_space else " ".join(kid.split())
            if not keep_space and kid[:1].isspace() and out and not out[-1].get("br"):
                text = " " + text
            if not keep_space and kid[-1:].isspace() and text.strip():
                text += " "
            if text:
                out.append(dict(fmt, text=text.replace("\xa0", " ")))
        elif kid.tag == "br":
            out.append({"br": True})
        elif kid.cls == "mark":
            continue
        else:
            sub = dict(fmt)
            sub.update({"b": True} if kid.tag in ("b", "strong", "th") else
                       {"code": True} if kid.tag == "code" else {"sub": True} if kid.tag == "sub" else {})
            out += _runs(kid, sub, keep_space)
    if out and not keep_space and "text" in out[0]:
        out[0]["text"] = out[0]["text"].lstrip()
    if out and not keep_space and "text" in out[-1]:
        out[-1]["text"] = out[-1]["text"].rstrip()
    return out


def html_to_blocks(markup):
    import struct

    body = _parse(markup.split("<body>", 1)[1].rsplit("</body>", 1)[0])
    blocks = []

    def walk(nodes):
        for n in nodes:
            if isinstance(n, str):
                continue
            if n.tag == "section":
                blocks.append({"t": "cover_start"})
                walk(n.kids)
                blocks.append({"t": "cover_end"})
            elif n.tag == "div" and n.cls == "pb":
                blocks.append({"t": "pagebreak"})
            elif n.tag == "div" and n.cls == "fig":
                walk(n.kids)
            elif n.tag == "div" and n.cls == "title":
                blocks.append({"t": "p", "style": "title", "runs": _runs(n)})
            elif n.tag == "tocfield":
                blocks.append({"t": "toc"})
            elif n.tag in ("h1", "h2"):
                blocks.append({"t": "h", "level": int(n.tag[1]), "runs": _runs(n)})
            elif n.tag == "p":
                blocks.append({"t": "p", "style": n.cls or None, "runs": _runs(n)})
            elif n.tag in ("ol", "ul"):
                blocks.append({"t": n.tag, "items": [_runs(li) for li in n.kids if not isinstance(li, str)]})
            elif n.tag == "pre":
                lines, line = [], []
                for r in _runs(n, keep_space=True):
                    parts = r.get("text", "").split("\n") if "text" in r else [None]
                    for i, part in enumerate(parts):
                        if i:
                            lines.append(line)
                            line = []
                        if part:
                            line.append(dict(r, text=part))
                lines.append(line)
                blocks.append({"t": "pre", "style": n.cls, "lines": [l for l in lines if l or True]})
            elif n.tag == "img":
                path = OUT / n.attrs["src"]
                w, h = struct.unpack(">II", path.read_bytes()[16:24])  # PNG IHDR
                pct = float(n.attrs.get("style", "width:80%").split(":")[1].rstrip("%")) / 100.0
                blocks.append({"t": "img", "src": str(path), "w": w, "h": h, "pct": pct})
            elif n.tag == "table":
                rows = [r for r in n.kids if not isinstance(r, str) and r.tag == "tr"]
                cols = [c for g in n.kids if not isinstance(g, str) and g.tag == "colgroup"
                        for c in g.kids if not isinstance(c, str)]
                grid = []
                for r in rows:
                    grid.append([{"runs": _runs(c), "head": c.tag == "th", "cls": c.cls}
                                 for c in r.kids if not isinstance(c, str)])
                blocks.append({"t": "table", "style": n.cls or None, "rows": grid,
                               "widths": [float(c.attrs["style"].split(":")[1].rstrip("%")) for c in cols]})
            else:
                walk(n.kids)

    walk(body.kids)
    return blocks


THAI_RE = re.compile("[฀-๿]+")
# a break may not come before a following vowel / tone mark, nor after a leading vowel
NO_BREAK_BEFORE = set("ะัาำิีึืฺุูๅ็่้๊๋์ํ๎")
NO_BREAK_AFTER = set("เแโใไ")


def add_thai_breaks(blocks):
    """Word only wraps Thai at spaces unless Thai proofing is installed, so mark word boundaries with U+200B.

    Segmentation uses the Windows Thai word breaker (Windows.Data.Text.WordsSegmenter); without it the
    document is still valid, lines just wrap at spaces.
    """
    import json
    import tempfile

    texts = []

    def collect(node):
        if isinstance(node, dict):
            if isinstance(node.get("text"), str) and THAI_RE.search(node["text"]):
                texts.append(node)
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(blocks)
    if not texts:
        return
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
        src.write_text(json.dumps([t["text"] for t in texts], ensure_ascii=False), encoding="utf-8")
        script = (
            "$null = [Windows.Data.Text.WordsSegmenter, Windows.Data.Text, ContentType = WindowsRuntime];"
            "$seg = New-Object Windows.Data.Text.WordsSegmenter 'th';"
            "$items = [IO.File]::ReadAllText('{src}', [Text.Encoding]::UTF8) | ConvertFrom-Json;"
            "$out = @(foreach ($s in $items) {{ ,@($seg.GetTokens($s) | ForEach-Object {{ [int]$_.SourceTextSegment.StartPosition }}) }});"
            "[IO.File]::WriteAllText('{dst}', (ConvertTo-Json -InputObject $out -Depth 3 -Compress), [Text.Encoding]::UTF8)"
        ).format(src=src, dst=dst)
        try:
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                           check=True, timeout=120, stdout=subprocess.DEVNULL)
            starts = json.loads(dst.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            print("note: Thai word breaker unavailable ({}); Word will wrap Thai at spaces only".format(exc))
            return
    if len(texts) == 1 and starts and not isinstance(starts[0], list):
        starts = [starts]
    for node, pos in zip(texts, starts):
        s, out, cuts = node["text"], [], set(p for p in (pos if isinstance(pos, list) else [pos]) if p)
        for i, ch in enumerate(s):
            if (i in cuts and THAI_RE.match(s[i - 1]) and THAI_RE.match(ch)
                    and ch not in NO_BREAK_BEFORE and s[i - 1] not in NO_BREAK_AFTER):
                out.append("​")
            out.append(ch)
        node["text"] = "".join(out)


def word_finalize(docx_path, pdf_check=None):
    """Updates the TOC field (page numbers) in Microsoft Word and optionally exports a PDF for review."""
    script = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0;"
        "try {{ $d = $w.Documents.Open('{p}'); foreach ($t in $d.TablesOfContents) {{ $t.Update() }};"
        "$d.Fields.Update() | Out-Null; foreach ($t in $d.TablesOfContents) {{ $t.Update() }}; $d.Save();"
        "{pdf} $d.Close() }} finally {{ $w.Quit() }}"
    ).format(p=str(docx_path).replace("'", "''"),
             pdf="$d.ExportAsFixedFormat('{}', 17);".format(str(pdf_check).replace("'", "''")) if pdf_check else "")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                   check=True, timeout=300)


def build_docx(ms, pdf_check=None):
    import json

    doc, _ = build_html(ms, "docx")
    model = OUT / "report_docx.json"
    blocks = html_to_blocks(doc)
    add_thai_breaks(blocks)
    model.write_text(json.dumps({"blocks": blocks}, ensure_ascii=False), encoding="utf-8")
    docx_path = OUT / DOCX_NAME
    node = shutil.which("node")
    if not node or not (HERE / "node_modules" / "docx").exists():
        raise RuntimeError("needs Node.js and `npm install` in analyze/ to build the .docx")
    subprocess.run([node, str(HERE / "report_docx.js"), str(model), str(docx_path)], check=True, timeout=180)
    try:
        word_finalize(docx_path, pdf_check)
    except Exception as exc:  # no Word: the TOC fills in when the user presses F9 / accepts the update prompt
        print("note: Microsoft Word not available to refresh the TOC ({})".format(exc))
    return docx_path


def main():
    if "--docx-only" in sys.argv:
        runs = load_runs()
        order = {"BFS": 0, "A*": 1}
        ms = sorted((metrics(r) for r in runs), key=lambda m: (order[m["algo"]], m["name"]))
        figures(ms)
        check = sys.argv[sys.argv.index("--check-pdf") + 1] if "--check-pdf" in sys.argv else None
        print("wrote {}".format(build_docx(ms, check).relative_to(ROOT)))
        return 0
    runs = load_runs()
    if not runs:
        print("no runs in analyze/runs/")
        return 1
    order = {"BFS": 0, "A*": 1}
    ms = sorted((metrics(r) for r in runs), key=lambda m: (order[m["algo"]], m["name"]))
    figures(ms)
    html_path = OUT / "report.html"
    pdf_path = OUT / PDF_NAME
    doc, sections = build_html(ms)
    html_path.write_text(doc, encoding="utf-8")
    print_pdf(html_path, pdf_path)
    pages = section_pages(pdf_path, sections)
    doc, _ = build_html(ms, pages)  # second pass: real page numbers in the TOC
    html_path.write_text(doc, encoding="utf-8")
    print_pdf(html_path, pdf_path)
    if section_pages(pdf_path, sections) != pages:
        print("warning: TOC page numbers shifted on the second pass")
    print("wrote {}".format(pdf_path.relative_to(ROOT)))
    print("wrote {}".format(build_docx(ms).relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
