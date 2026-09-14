#!/usr/bin/env python3
"""4x4 cost-map Mission Control for the classwork field.

* BFS and A* (multi-select) on the fixed 4x4 map with per-cell costs.
* Planned paths (dashed) vs. real/simulated trail, live robot pose.
* SIM backend (no hardware) and REAL backend (RoboMaster EP over Wi-Fi).
  The robot carries a gimbal (no gripper); the ToF sensor rides on the gimbal,
  so the gimbal is put in chassis-lead mode and recentered to look ahead.
* CALIBRATE tab: cell size, start heading, odometry scale, axis signs,
  1-cell / 90-degree tests, and ToF / Sharp sample capture + polynomial fit.

    python main.py costmap                # simulation
    python main.py costmap --mode real    # physical robot (AP mode)
"""

import atexit
import heapq
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # calibration plots are saved, never shown

ROOT = Path(__file__).resolve().parent.parent
CAL_FILE = ROOT / "calibration_output" / "motion_calibration.json"
SENSOR_CSV = ROOT / "data" / "calibration_measurements.csv"

# ---------------------------------------------------------------- map
# (x, y): x grows right, y grows up, exactly as printed on the field sheet.
W = H = 4
START, GOAL = (1, 4), (4, 1)
OBSTACLES = {(4, 4), (2, 3), (3, 2)}
COST = {  # every free cell; entering a cell pays its cost
    (1, 4): 2, (2, 4): 3, (3, 4): 1,
    (1, 3): 2, (3, 3): 4, (4, 3): 2,
    (1, 2): 3, (2, 2): 1, (4, 2): 4,
    (1, 1): 2, (2, 1): 3, (3, 1): 2, (4, 1): 1,
}
MOVES = ((1, 0), (0, -1), (-1, 0), (0, 1))  # E, S, W, N - BFS tie-break order


def neighbors(cell):
    for dx, dy in MOVES:
        nxt = (cell[0] + dx, cell[1] + dy)
        if nxt in COST:
            yield nxt


def path_cost(path):
    return sum(COST[c] for c in path[1:])


def _trace(parent, goal):
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def bfs(start=START, goal=GOAL):
    """Fewest moves; ignores cell cost. Returns (path or None, expansion order)."""
    parent, queue, expanded = {start: None}, deque([start]), []
    while queue:
        cell = queue.popleft()
        expanded.append(cell)
        if cell == goal:
            return _trace(parent, goal), expanded
        for nxt in neighbors(cell):
            if nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    return None, expanded


def astar(start=START, goal=GOAL):
    """Cheapest total cost. Manhattan x cheapest cell cost keeps h admissible."""
    cheapest = min(COST.values())

    def h(c):
        return (abs(c[0] - goal[0]) + abs(c[1] - goal[1])) * cheapest

    g, parent, done, expanded = {start: 0}, {start: None}, set(), []
    heap = [(h(start), 0, start)]
    while heap:
        _, gc, cell = heapq.heappop(heap)
        if cell in done:
            continue
        done.add(cell)
        expanded.append(cell)
        if cell == goal:
            return _trace(parent, goal), expanded
        for nxt in neighbors(cell):
            ng = gc + COST[nxt]
            if ng < g.get(nxt, math.inf):
                g[nxt], parent[nxt] = ng, cell
                heapq.heappush(heap, (ng + h(nxt), ng, nxt))
    return None, expanded


ALGORITHMS = {"BFS": bfs, "A*": astar}

# ---------------------------------------------------------------- calibration
DEFAULT_CAL = {
    "cell_m": 0.60,          # field grid size
    "start_heading": 0,      # map direction the robot faces at START: 0=E 90=N 180=W 270=S
    "forward_scale": 1.0,    # odometry metres per real metre (wheel slip)
    "max_speed": 0.35,       # m/s
    "turn_speed": 90.0,      # deg/s
    "z_sign": -1,            # -1: drive_speed z>0 turns CW on this EP (measured 2026-09-14)
    "yaw_sign": 1,           # +1: attitude yaw grows clockwise
    "y_sign": 1,             # +1: position / drive y points to the robot's right
    "tof_offset_mm": 0.0,    # ToF lens distance ahead of chassis centre (on the gimbal)
}
TURN_TOL_DEG, POS_TOL_M, STEP_TIMEOUT_S = 2.0, 0.01, 20.0
HIT_TOF_MM = 100.0  # ToF closer than this = touching something


def build_run(name, algorithm, mode, results, samples, status, cal):
    """Everything a saved run needs: map, both plans, recorded pose samples and the scored result."""
    path = results[algorithm][0]
    cells = []
    for s in samples:  # a cell counts once the robot is well inside it (no boundary jitter)
        c = (round(s["mx"]), round(s["my"]))
        if abs(s["mx"] - c[0]) < 0.35 and abs(s["my"] - c[1]) < 0.35 and (not cells or cells[-1] != c):
            cells.append(c)
    hit = (any((round(s["mx"]), round(s["my"])) not in COST for s in samples)
           or any(s["tof"] is not None and s["tof"] < HIT_TOF_MM for s in samples))
    last = samples[-1] if samples else None
    travelled = sum(math.hypot(b["mx"] - a["mx"], b["my"] - a["my"]) for a, b in zip(samples, samples[1:]))
    return {
        "name": name, "algorithm": algorithm, "mode": mode, "status": status,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "map": {"width": W, "height": H, "start": list(START), "goal": list(GOAL),
                "obstacles": sorted(list(c) for c in OBSTACLES),
                "cost": {"{},{}".format(*c): v for c, v in COST.items()}},
        "plans": {n: {"path": [list(c) for c in p] if p else None,
                      "steps": len(p) - 1 if p else None,
                      "cost": path_cost(p) if p else None,
                      "expanded": [list(c) for c in e]} for n, (p, e) in results.items()},
        "result": {
            "planned_steps": len(path) - 1 if path else None,
            "actual_steps": max(len(cells) - 1, 0),
            "time_s": round(last["t"], 2) if last else 0.0,
            "hit_obstacle": "YES" if hit else "NO",
            "reach_goal": "YES" if cells and cells[-1] == GOAL and status == "Done" else "NO",
            "path_cost": path_cost(path) if path else None,
            "final_cell": list(cells[-1]) if cells else None,
            "goal_error_m": round(math.hypot(last["mx"] - GOAL[0], last["my"] - GOAL[1]) * cal["cell_m"], 3)
            if last else None,
            "travelled_m": round(travelled * cal["cell_m"], 3),
        },
        "calibration": dict(cal),
        "samples": samples,
    }


def load_cal():
    cal = dict(DEFAULT_CAL)
    try:
        cal.update(json.loads(CAL_FILE.read_text(encoding="utf-8-sig")))  # -sig: tolerate editor BOMs
    except (OSError, ValueError):
        pass
    return cal


def save_cal(cal):
    CAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAL_FILE.write_text(json.dumps(cal, indent=2) + "\n", encoding="utf-8")


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def odom_to_map(x, y, yaw, cal):
    """Robot frame (x fwd, y right in metres, yaw CW deg, zeroed at START) -> map (mx, my, heading CCW deg)."""
    h = math.radians(cal["start_heading"])
    mx = START[0] + (x * math.cos(h) + y * math.sin(h)) / cal["cell_m"]
    my = START[1] + (x * math.sin(h) - y * math.cos(h)) / cal["cell_m"]
    return mx, my, (cal["start_heading"] - yaw) % 360.0


def sim_tof_mm(mx, my, heading, cal):
    """Distance from robot centre to the first obstacle cell / field edge."""
    dx, dy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    for i in range(1, 400):
        d = i * 0.02
        if (round(mx + dx * d), round(my + dy * d)) not in COST:
            return d * cal["cell_m"] * 1000.0
    return None


# ---------------------------------------------------------------- backends
class SimRobot(object):
    """Ideal kinematic robot: same pose/drive surface as RealRobot."""

    is_real = False

    def __init__(self):
        self.speed = 1.0
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.x = self.y = self.yaw = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.t = time.monotonic()
        self.gimbal_yaw, self.tof, self.adc = 0.0, None, {}

    def _step(self):
        now = time.monotonic()
        dt, self.t = min(now - self.t, 0.1) * self.speed, now
        vx, vy, wz = self.cmd
        r = math.radians(self.yaw)
        self.x += (vx * math.cos(r) - vy * math.sin(r)) * dt
        self.y += (vx * math.sin(r) + vy * math.cos(r)) * dt
        self.yaw = wrap(self.yaw - wz * dt)

    def pose(self):
        with self._lock:
            self._step()
            return self.x, self.y, self.yaw

    def drive(self, vx, vy, wz):
        with self._lock:
            self._step()
            self.cmd = (vx, vy, wz)

    def stop(self):
        self.drive(0.0, 0.0, 0.0)

    def set_zero(self):
        with self._lock:
            self.reset()

    def close(self):
        pass


class RealRobot(object):
    """RoboMaster EP: odometry + IMU yaw for pose, gimbal-mounted ToF."""

    is_real = True

    def __init__(self, cal, conn_type="ap"):
        from calibrate import load_robot_sdk

        sdk = load_robot_sdk()
        self.cal = cal
        self.raw = [0.0, 0.0, 0.0]
        self.zero = (0.0, 0.0, 0.0)
        self.gimbal_yaw, self.tof, self.adc = 0.0, None, {}
        self.poll_adc = False
        self.ep = sdk.Robot()
        self.ep.initialize(conn_type=conn_type)
        atexit.register(self.stop)  # never leave the wheels latched on exit
        # Gimbal follows the chassis so the ToF always looks where the robot drives.
        self.ep.set_robot_mode(mode="chassis_lead")
        self.ep.gimbal.recenter().wait_for_completed()
        self.ep.chassis.sub_position(cs=0, freq=20, callback=self._on_pos)
        self.ep.chassis.sub_attitude(freq=20, callback=self._on_att)
        self.ep.gimbal.sub_angle(freq=20, callback=self._on_gimbal)
        self.ep.sensor.sub_distance(freq=20, callback=self._on_tof)
        self._alive = True
        threading.Thread(target=self._adc_loop, daemon=True).start()
        time.sleep(0.5)
        self.set_zero()

    def _on_pos(self, p):
        self.raw[0], self.raw[1] = p[0], p[1]

    def _on_att(self, a):
        self.raw[2] = a[0]

    def _on_gimbal(self, a):
        self.gimbal_yaw = a[1]

    def _on_tof(self, d):
        self.tof = d[0]

    def _adc_loop(self):
        # ponytail: polls only while the CALIBRATE tab is open; the adapter call blocks.
        while self._alive:
            if self.poll_adc:
                for sensor, sid in (("sharp_left", 1), ("sharp_right", 2)):
                    try:
                        self.adc[sensor] = self.ep.sensor_adaptor.get_adc(id=sid, port=sid)
                    except Exception:
                        self.adc[sensor] = None
            time.sleep(0.5)

    def set_zero(self):
        self.zero = tuple(self.raw)

    def pose(self):
        c = self.cal
        x0, y0, a0 = self.zero
        dx, dy = self.raw[0] - x0, (self.raw[1] - y0) * c["y_sign"]
        r = math.radians(a0 * c["yaw_sign"])
        f = c["forward_scale"]
        return ((dx * math.cos(r) + dy * math.sin(r)) / f,
                (-dx * math.sin(r) + dy * math.cos(r)) / f,
                wrap((self.raw[2] - a0) * c["yaw_sign"]))

    def drive(self, vx, vy, wz):
        # timeout = SDK watchdog: wheels stop if the control loop stalls.
        self.ep.chassis.drive_speed(x=vx, y=vy * self.cal["y_sign"], z=wz * self.cal["z_sign"], timeout=0.5)

    def stop(self):
        try:
            self.ep.chassis.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass

    def recenter_gimbal(self):
        self.ep.gimbal.recenter()

    def close(self):
        self._alive = False
        self.stop()
        atexit.unregister(self.stop)  # connection is closing; a later stop would only log SDK errors
        for unsub in (self.ep.chassis.unsub_position, self.ep.chassis.unsub_attitude,
                      self.ep.gimbal.unsub_angle, self.ep.sensor.unsub_distance):
            try:
                unsub()
            except Exception:
                pass
        self.ep.close()


def _clamp(v, lo, hi):
    return math.copysign(min(max(abs(v), lo), hi), v)


class PathRunner(threading.Thread):
    """Closed-loop executor. plan = [("cell", (x, y)) | ("turn", heading_deg), ...]."""

    def __init__(self, robot, cal, plan, log, period=0.05):
        super(PathRunner, self).__init__(daemon=True)
        self.robot, self.cal, self.plan, self.log, self.period = robot, cal, plan, log, period
        self.stop_event = threading.Event()
        self.index, self.ok, self.done, self.status = 0, False, False, "Running"

    def map_pose(self):
        return odom_to_map(*self.robot.pose(), cal=self.cal)

    def run(self):
        try:
            for self.index, (kind, arg) in enumerate(self.plan):
                if kind == "turn":
                    ok = self._turn(arg)
                else:
                    mx, my, _ = self.map_pose()
                    heading = round(math.degrees(math.atan2(arg[1] - my, arg[0] - mx)) / 90.0) * 90.0
                    ok = self._turn(heading) and self._forward(arg, heading)
                if not ok:
                    self.status = "Stopped"
                    self.log(self.status)
                    return
            self.ok, self.status = True, "Done"
            self.log(self.status)
        except Exception as exc:
            self.status = "Aborted: {}".format(exc)
            self.log(self.status)
        finally:
            self.robot.stop()
            self.done = True

    def _turn(self, target):
        t0, first, flipped = time.monotonic(), None, False
        while not self.stop_event.is_set():
            err = wrap(target - self.map_pose()[2])
            if abs(err) < TURN_TOL_DEG:
                self.robot.stop()
                return True
            first = abs(err) if first is None else first
            if abs(err) > first + 8:
                if flipped:
                    raise RuntimeError("turning the wrong way - check YAW SIGN in CALIBRATE")
                # Command and IMU disagree on rotation direction: fix the command sign once.
                self.cal["z_sign"], flipped, first = -self.cal["z_sign"], True, None
                self.robot.stop()
                self.log("Turned wrong way: Z SIGN -> {:+d} (SAVE CALIBRATION to keep)".format(self.cal["z_sign"]))
                time.sleep(0.3)
                continue
            if time.monotonic() - t0 > STEP_TIMEOUT_S:
                raise RuntimeError("turn timed out")
            self.robot.drive(0.0, 0.0, _clamp(3.0 * err, 10.0, self.cal["turn_speed"]))
            time.sleep(self.period)
        return False

    def _forward(self, cell, heading):
        t0, cm, h = time.monotonic(), self.cal["cell_m"], math.radians(heading)
        while not self.stop_event.is_set():
            mx, my, hd = self.map_pose()
            dx, dy = cell[0] - mx, cell[1] - my
            along = (dx * math.cos(h) + dy * math.sin(h)) * cm
            lateral = (dx * math.sin(h) - dy * math.cos(h)) * cm  # + = target on the right
            if along < POS_TOL_M:
                self.robot.stop()
                return True
            if time.monotonic() - t0 > STEP_TIMEOUT_S:
                raise RuntimeError("move timed out")
            self.robot.drive(_clamp(1.5 * along, 0.05, self.cal["max_speed"]),
                             max(-0.1, min(0.1, 1.5 * lateral)),
                             max(-30.0, min(30.0, 2.0 * wrap(heading - hd))))
            time.sleep(self.period)
        return False


# ---------------------------------------------------------------- UI
BG, PANEL, PANEL_HI = (18, 22, 30), (34, 40, 54), (52, 60, 78)
TEXT, DIM, ACCENT = (222, 228, 236), (130, 140, 156), (0, 140, 210)
CELL_BG, CELL_TXT = (226, 233, 242), (40, 48, 60)
GREEN, RED, YELLOW, CYAN = (70, 190, 110), (225, 75, 75), (245, 205, 60), (60, 220, 230)
ALGO_COLOR = {"BFS": (70, 150, 255), "A*": (255, 140, 40)}
ALGO_OFFSET = {"BFS": -9, "A*": 9}
HEADINGS = {0: "E", 90: "N", 180: "W", 270: "S"}
CELL, MX0, MY0 = 165, 60, 80
WIN_W, WIN_H = 1300, 840


class App(object):
    def __init__(self, mode="sim", conn_type="ap"):
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("RoboMaster 4x4 Cost Map - BFS / A*")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont("consolas", 15)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self.cal = load_cal()
        self.sim, self.real, self.robot = SimRobot(), None, None
        self.robot = self.sim
        self.conn_type, self.connecting = conn_type, False
        self.results = {name: fn() for name, fn in ALGORITHMS.items()}
        self.selected = {"BFS": True, "A*": True}
        self.follow = "A*"
        self.tab, self.runner, self.confirm = "PLAN", None, None
        self.trail, self.logs = [], deque(maxlen=9)
        self.ref_mm, self.measured_cm = 100.0, 60.0
        self.record, self.last_result, self.saving = None, None, False
        self.click = None
        self.log("Map loaded: START {} GOAL {}".format(START, GOAL))
        if mode == "real":
            self.connect()

    # ------------------------------------------------------------ helpers
    def log(self, msg):
        self.logs.append(time.strftime("%H:%M:%S ") + msg)
        print("[costmap] " + msg)

    def text(self, s, x, y, color=TEXT, font=None, center=False):
        img = (font or self.font).render(str(s), True, color)
        rect = img.get_rect(center=(x, y)) if center else img.get_rect(topleft=(x, y))
        self.screen.blit(img, rect)

    def button(self, x, y, w, h, label, on=False, enabled=True):
        pg = self.pg
        r = pg.Rect(x, y, w, h)
        hover = r.collidepoint(pg.mouse.get_pos())
        color = ACCENT if on else (PANEL_HI if hover and enabled else PANEL)
        pg.draw.rect(self.screen, color, r, border_radius=5)
        pg.draw.rect(self.screen, PANEL_HI, r, 1, border_radius=5)
        self.text(label, r.centerx, r.centery, TEXT if enabled else DIM, center=True)
        return enabled and self.click is not None and r.collidepoint(self.click)

    def stepper(self, x, y, label, key, step, fmt="{:.2f}"):
        self.text(label, x, y + 5)
        self.text(fmt.format(self.cal[key]), x + 200, y + 5, YELLOW)
        if self.button(x + 290, y, 36, 26, "-"):
            self.cal[key] = round(self.cal[key] - step, 4)
        if self.button(x + 332, y, 36, 26, "+"):
            self.cal[key] = round(self.cal[key] + step, 4)

    def busy(self):
        return self.runner is not None and not self.runner.done

    def map_pose(self):
        return odom_to_map(*self.robot.pose(), cal=self.cal)

    def tof_mm(self, mx, my, heading):
        if self.robot.is_real:
            return self.robot.tof
        return sim_tof_mm(mx, my, heading, self.cal)

    def to_screen(self, mx, my):
        return int(MX0 + (mx - 0.5) * CELL), int(MY0 + (H + 0.5 - my) * CELL)

    # ------------------------------------------------------------ actions
    def connect(self):
        if self.connecting:
            return
        self.connecting = True
        self.log("Connecting to robot ({})...".format(self.conn_type.upper()))

        def work():
            try:
                self.real = RealRobot(self.cal, self.conn_type)
                self.robot, self.trail = self.real, []
                self.log("Connected. Pose zeroed at START {} facing {}".format(
                    START, HEADINGS[self.cal["start_heading"] % 360]))
            except Exception as exc:  # no silent mock fallback
                self.log("Connect failed: {}".format(exc))
            finally:
                self.connecting = False

        threading.Thread(target=work, daemon=True).start()

    def disconnect(self):
        if self.real:
            self.real.close()
        self.real, self.robot, self.trail = None, self.sim, []
        self.log("Disconnected - SIM mode")

    def estop(self):
        if self.runner:
            self.runner.stop_event.set()
        self.robot.stop()
        self.log("STOP")

    def start_plan(self, plan, what, record=None):
        if self.busy():
            return

        def go():
            self.runner = PathRunner(self.robot, self.cal, plan, self.log)
            if record:
                self.record = {"algorithm": record, "mode": "real" if self.robot.is_real else "sim",
                               "t0": time.monotonic(), "last": -1.0, "samples": [], "status": "Running"}
            self.runner.start()
            self.log("Running " + what)

        if self.robot.is_real:
            self.confirm = ("REAL ROBOT will drive: {}. Clear the field.".format(what), go)
        else:
            go()

    def run_path(self):
        path = self.results[self.follow][0]
        mx, my, _ = self.map_pose()
        if (round(mx), round(my)) != path[0]:
            self.log("Robot is not on START {} - RESET / SET START POSE first".format(path[0]))
            return
        self.start_plan([("cell", c) for c in path[1:]], self.follow + " path", record=self.follow)

    def sample(self):
        """Records pose at 10 Hz while a path run is active; scores it when the run ends."""
        rec = self.record
        if rec is None or rec["status"] != "Running":
            return
        # Robot time: a 5x simulation must not report a 5x shorter run.
        t = (time.monotonic() - rec["t0"]) * (1.0 if self.robot.is_real else self.sim.speed)
        if t - rec["last"] >= 0.1 or self.runner.done:
            ox, oy, yaw = self.robot.pose()
            mx, my, heading = odom_to_map(ox, oy, yaw, self.cal)
            tof = self.tof_mm(mx, my, heading - self.robot.gimbal_yaw)
            rec["samples"].append({"t": round(t, 3), "x": round(ox, 4), "y": round(oy, 4), "yaw": round(yaw, 2),
                                   "mx": round(mx, 4), "my": round(my, 4), "heading": round(heading, 2),
                                   "tof": round(tof, 1) if tof is not None else None})
            rec["last"] = t
        if self.runner.done:
            rec["status"] = self.runner.status
            self.last_result = build_run("", rec["algorithm"], rec["mode"], self.results,
                                         rec["samples"], rec["status"], self.cal)["result"]

    def save_result(self):
        import run_notebook

        rec = self.record if self.record and self.record["status"] != "Running" else None
        algo = rec["algorithm"] if rec else self.follow
        run = build_run(run_notebook.next_name(algo), algo,
                        rec["mode"] if rec else ("real" if self.robot.is_real else "sim"), self.results,
                        rec["samples"] if rec else [], rec["status"] if rec else "Planned", self.cal)
        self.record, self.saving = None, True  # each save is its own run

        def work():
            try:
                path = run_notebook.save(run)
                self.log("Saved " + str(path.relative_to(ROOT)))
            except Exception as exc:
                self.log("Save failed: {}".format(exc))
            finally:
                self.saving = False

        threading.Thread(target=work, daemon=True).start()

    def zero_here(self):
        self.robot.set_zero()
        self.trail = []
        self.log("Pose zeroed: robot at START facing {}".format(HEADINGS[self.cal["start_heading"] % 360]))

    # ------------------------------------------------------------ drawing
    def dashed(self, color, points, width=4, dash=14):
        pg = self.pg
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            for i in range(0, int(length), dash * 2):
                a, b = i / length, min(i + dash, length) / length
                pg.draw.line(self.screen, color, (x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                             (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), width)

    def draw_map(self):
        pg = self.pg
        self.text("4 x 4 COST MAP", MX0, 30, TEXT, self.big)
        explored = self.results[self.follow][1] if self.selected[self.follow] else []
        for x in range(1, W + 1):
            for y in range(1, H + 1):
                c = (x, y)
                r = pg.Rect(MX0 + (x - 1) * CELL, MY0 + (H - y) * CELL, CELL, CELL)
                fill = (200, 235, 200) if c == START else (245, 205, 205) if c == GOAL else CELL_BG
                pg.draw.rect(self.screen, (60, 64, 72) if c in OBSTACLES else fill, r)
                pg.draw.rect(self.screen, (20, 20, 20), r, 2)
                self.text("({},{})".format(x, y), r.x + 8, r.y + 6, DIM if c in OBSTACLES else CELL_TXT, self.small)
                if c in OBSTACLES:
                    self.text("OBSTACLE", r.centerx, r.centery, RED, center=True)
                    continue
                self.text("Cost = {}".format(COST[c]), r.centerx, r.centery - 32, CELL_TXT, center=True)
                if c in (START, GOAL):
                    self.text("START" if c == START else "GOAL", r.centerx, r.bottom - 22,
                              (30, 130, 60) if c == START else RED, center=True)
                if c in explored:
                    self.text("#{}".format(explored.index(c) + 1), r.right - 30, r.y + 6,
                              ALGO_COLOR[self.follow], self.small)
        for i in range(1, W + 1):
            self.text(i, MX0 + (i - 0.5) * CELL, MY0 + H * CELL + 18, TEXT, center=True)
            self.text(i, MX0 - 22, MY0 + (H - i + 0.5) * CELL, TEXT, center=True)
        self.text("x", MX0 + W * CELL + 15, MY0 + H * CELL + 18, TEXT, center=True)
        self.text("y", MX0 - 22, MY0 - 16, TEXT, center=True)

        for name in ALGORITHMS:
            path = self.results[name][0]
            if self.selected[name] and path:
                off = ALGO_OFFSET[name]
                pts = [(sx + off, sy + off) for sx, sy in (self.to_screen(*c) for c in path)]
                self.dashed(ALGO_COLOR[name], pts, 6 if name == self.follow else 3)

        if len(self.trail) > 1:
            pg.draw.lines(self.screen, YELLOW, False, [self.to_screen(*p) for p in self.trail], 3)

        mx, my, heading = self.map_pose()
        cx, cy = self.to_screen(mx, my)
        tof_heading = heading - self.robot.gimbal_yaw  # gimbal yaw grows clockwise
        tof = self.tof_mm(mx, my, tof_heading)
        if tof:
            reach = (tof + self.cal["tof_offset_mm"]) / 1000.0 / self.cal["cell_m"] * CELL
            ex = cx + math.cos(math.radians(tof_heading)) * reach
            ey = cy - math.sin(math.radians(tof_heading)) * reach
            pg.draw.line(self.screen, CYAN, (cx, cy), (ex, ey), 2)
            pg.draw.circle(self.screen, CYAN, (int(ex), int(ey)), 5)
        rad = int(CELL * 0.2)
        pg.draw.circle(self.screen, (30, 30, 36), (cx, cy), rad)
        pg.draw.circle(self.screen, GREEN if not self.robot.is_real else ACCENT, (cx, cy), rad, 4)
        hx = cx + math.cos(math.radians(heading)) * rad * 1.4
        hy = cy - math.sin(math.radians(heading)) * rad * 1.4
        pg.draw.line(self.screen, TEXT, (cx, cy), (hx, hy), 4)

    def draw_header(self):
        x = 740
        mode = "REAL ROBOT" if self.robot.is_real else "SIMULATION"
        state = "CONNECTING" if self.connecting else "RUNNING" if self.busy() else "IDLE"
        self.text("{}  |  {}".format(mode, state), x, 18, YELLOW if self.robot.is_real else GREEN, self.big)
        if self.button(WIN_W - 170, 12, 150, 36, "STOP (Space)"):
            self.estop()
        if self.button(x, 56, 120, 30, "PLAN", self.tab == "PLAN"):
            self.tab = "PLAN"
        if self.button(x + 130, 56, 120, 30, "CALIBRATE", self.tab == "CALIBRATE"):
            self.tab = "CALIBRATE"
        if self.real:
            self.real.poll_adc = self.tab == "CALIBRATE"

    def draw_plan(self):
        x, y = 740, 105
        self.text("ALGORITHMS (multi-select)", x, y, DIM)
        for i, name in enumerate(ALGORITHMS):
            if self.button(x + i * 130, y + 22, 120, 30, name, self.selected[name]):
                self.selected[name] = not self.selected[name]
                if not self.selected[self.follow]:
                    self.follow = next((n for n, on in self.selected.items() if on), self.follow)
        y += 65
        for name in ALGORITHMS:
            path, expanded = self.results[name]
            color = ALGO_COLOR[name] if self.selected[name] else DIM
            self.pg.draw.rect(self.screen, color, (x, y + 4, 14, 14))
            if path:
                self.text("{:<4} steps {}  cost {}  expanded {}".format(
                    name, len(path) - 1, path_cost(path), len(expanded)), x + 22, y, color)
                self.text(" -> ".join("({},{})".format(*c) for c in path), x + 22, y + 20, color, self.small)
            else:
                self.text("{}: no path".format(name), x + 22, y, color)
            y += 46
        self.text("FOLLOW", x, y + 6, DIM)
        for i, name in enumerate(ALGORITHMS):
            if self.button(x + 80 + i * 110, y, 100, 30, name, self.follow == name, self.selected[name]):
                self.follow = name
        y += 50
        self.text("ROBOT", x, y + 6, DIM)
        if self.real:
            if self.button(x + 80, y, 150, 30, "DISCONNECT", enabled=not self.busy()):
                self.disconnect()
        elif self.button(x + 80, y, 150, 30, "CONNECT REAL", enabled=not self.connecting):
            self.connect()
        if not self.robot.is_real:
            self.text("sim speed", x + 250, y + 6, DIM)
            for i, s in enumerate((1, 2, 5)):
                if self.button(x + 345 + i * 55, y, 48, 30, "{}x".format(s), self.sim.speed == s):
                    self.sim.speed = float(s)
        y += 45
        if self.button(x, y, 170, 36, "RUN " + self.follow, enabled=not self.busy() and self.selected[self.follow]):
            self.run_path()
        if self.button(x + 180, y, 170, 36, "RESET TO START", enabled=not self.busy()):
            self.zero_here()
        if self.button(x + 360, y, 170, 36, "CLEAR TRAIL"):
            self.trail = []
        y += 46
        finished = self.record is not None and self.record["status"] != "Running"
        if self.button(x, y, 170, 36, "SAVING..." if self.saving else "SAVE RESULT",
                       enabled=not self.busy() and not self.saving):
            self.save_result()
        self.text("{} -> analyze/runs/".format(
            "last run" if finished else "plan only"), x + 180, y + 10, DIM)
        y += 46
        r = self.last_result
        self.text("LAST RUN", x, y, DIM)
        if r:
            self.text("planned {}  actual {}  time {:.1f} s".format(
                r["planned_steps"], r["actual_steps"], r["time_s"]), x, y + 22)
            self.text("hit obstacle {}".format(r["hit_obstacle"]), x, y + 44,
                      RED if r["hit_obstacle"] == "YES" else GREEN)
            self.text("reach goal {}".format(r["reach_goal"]), x + 200, y + 44,
                      GREEN if r["reach_goal"] == "YES" else RED)
        else:
            self.text("--", x, y + 22, DIM)
        y += 76
        self.draw_pose(x, y)

    def draw_pose(self, x, y):
        ox, oy, yaw = self.robot.pose()
        mx, my, heading = self.map_pose()
        self.text("LIVE POSE", x, y, DIM)
        self.text("odom  x {:+.3f} m  y {:+.3f} m  yaw {:+.1f}".format(ox, oy, yaw), x, y + 22)
        self.text("map   ({:.2f}, {:.2f})  cell ({},{})  heading {:.0f}".format(
            mx, my, round(mx), round(my), heading), x, y + 44)
        tof = self.tof_mm(mx, my, heading - self.robot.gimbal_yaw)
        self.text("ToF   {}  gimbal yaw {:+.1f}".format(
            "{:.0f} mm".format(tof) if tof else "--", self.robot.gimbal_yaw), x, y + 66, CYAN)
        if self.busy():
            kind, arg = self.runner.plan[self.runner.index]
            self.text("step {}/{} -> {} {}".format(self.runner.index + 1, len(self.runner.plan), kind, arg),
                      x, y + 88, YELLOW)

    def draw_calibrate(self):
        from calibrate import append_measurement, fit_command

        x, y = 740, 105
        self.text("MOTION", x, y, DIM)
        y += 22
        self.stepper(x, y, "cell size (m)", "cell_m", 0.01)
        self.text("start heading", x, y + 37)
        if self.button(x + 200, y + 32, 80, 26, HEADINGS[self.cal["start_heading"] % 360], enabled=not self.busy()):
            self.cal["start_heading"] = (self.cal["start_heading"] + 90) % 360
        self.stepper(x, y + 64, "forward scale", "forward_scale", 0.01, "{:.3f}")
        self.stepper(x, y + 96, "max speed (m/s)", "max_speed", 0.05)
        self.stepper(x, y + 128, "turn speed (deg/s)", "turn_speed", 10, "{:.0f}")
        self.stepper(x, y + 160, "ToF offset (mm)", "tof_offset_mm", 5, "{:.0f}")
        y += 196
        for i, key in enumerate(("z_sign", "yaw_sign", "y_sign")):
            label = "{} {:+d}".format(key.split("_")[0].upper(), int(self.cal[key]))
            if self.button(x + i * 125, y, 115, 28, label + " SIGN", enabled=not self.busy()):
                self.cal[key] = -self.cal[key]
        y += 38
        if self.button(x, y, 170, 30, "SET START POSE", enabled=not self.busy()):
            self.zero_here()
        mx, my, heading = self.map_pose()
        cell = (round(mx), round(my))
        snapped = round(heading / 90.0) * 90.0
        ahead = (cell[0] + int(round(math.cos(math.radians(snapped)))),
                 cell[1] + int(round(math.sin(math.radians(snapped)))))
        if self.button(x + 180, y, 130, 30, "TEST 1 CELL", enabled=not self.busy()):
            self.start_plan([("cell", ahead)], "1-cell test")
        if self.button(x + 320, y, 130, 30, "TEST LEFT 90", enabled=not self.busy()):
            self.start_plan([("turn", (snapped + 90) % 360)], "left 90 test")
        y += 40
        self.text("measured {:.1f} cm".format(self.measured_cm), x, y + 6)
        for i, d in enumerate((-5, -1, 1, 5)):
            if self.button(x + 165 + i * 46, y, 42, 28, "{:+d}".format(d)):
                self.measured_cm += d
        if self.button(x + 355, y, 130, 28, "APPLY DIST"):
            # Robot stopped when odometry said cell_m; it really went measured_cm.
            self.cal["forward_scale"] = round(
                self.cal["forward_scale"] * self.cal["cell_m"] * 100.0 / max(self.measured_cm, 1.0), 4)
            self.log("forward scale -> {:.4f}".format(self.cal["forward_scale"]))
        y += 36
        if self.button(x, y, 200, 30, "SAVE CALIBRATION"):
            save_cal(self.cal)
            self.log("Saved " + str(CAL_FILE.relative_to(ROOT)))

        y += 48
        self.text("SENSORS  (ToF on gimbal, Sharp on adapter)", x, y, DIM)
        adc = self.robot.adc
        tof = self.robot.tof if self.robot.is_real else None
        self.text("ToF {}   Sharp L {}   Sharp R {}".format(
            tof if tof is not None else "--", adc.get("sharp_left", "--"), adc.get("sharp_right", "--")),
            x, y + 22, CYAN)
        y += 46
        self.text("reference {:.0f} mm".format(self.ref_mm), x, y + 6)
        for i, d in enumerate((-50, -10, 10, 50)):
            if self.button(x + 180 + i * 50, y, 46, 28, "{:+d}".format(d)):
                self.ref_mm = max(10.0, self.ref_mm + d)
        y += 36
        for i, (label, sensor, raw) in enumerate((("REC ToF", "tof", tof),
                                                  ("REC SHARP L", "sharp_left", adc.get("sharp_left")),
                                                  ("REC SHARP R", "sharp_right", adc.get("sharp_right")))):
            if self.button(x + i * 150, y, 140, 30, label, enabled=raw is not None):
                append_measurement(SENSOR_CSV, sensor, raw, self.ref_mm, int(time.time()))
                self.log("{} raw {} @ {:.0f} mm".format(sensor, raw, self.ref_mm))
        y += 40
        if self.button(x, y, 150, 30, "RECENTER GIMBAL", enabled=self.robot.is_real):
            self.real.recenter_gimbal()
        if self.button(x + 160, y, 150, 30, "FIT & SAVE"):
            def fit():
                try:
                    fit_command(SENSOR_CSV, ROOT / "calibration_output")
                    self.log("Sensor fit saved to calibration_output/")
                except Exception as exc:
                    self.log("Fit failed: {}".format(exc))
            threading.Thread(target=fit, daemon=True).start()

    def draw_logs(self):
        x, y = 740, WIN_H - 20 * len(self.logs) - 12
        for i, line in enumerate(self.logs):
            self.text(line[:70], x, y + i * 20, DIM, self.small)

    def draw_confirm(self):
        pg = self.pg
        shade = pg.Surface((WIN_W, WIN_H), pg.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        self.screen.blit(shade, (0, 0))
        r = pg.Rect(WIN_W // 2 - 300, WIN_H // 2 - 80, 600, 160)
        pg.draw.rect(self.screen, PANEL, r, border_radius=8)
        pg.draw.rect(self.screen, YELLOW, r, 2, border_radius=8)
        self.text(self.confirm[0], r.centerx, r.y + 45, YELLOW, center=True)
        if self.button(r.x + 90, r.bottom - 60, 180, 38, "CONFIRM (Y)"):
            action, self.confirm = self.confirm[1], None
            action()
        elif self.button(r.right - 270, r.bottom - 60, 180, 38, "CANCEL (N)"):
            self.confirm = None

    # ------------------------------------------------------------ loop
    def run(self):
        pg = self.pg
        clock = pg.time.Clock()
        try:
            while True:
                self.click = None
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        return 0
                    if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                        self.click = event.pos
                    if event.type == pg.KEYDOWN:
                        if event.key in (pg.K_SPACE, pg.K_F1):
                            self.estop()
                        elif self.confirm and event.key == pg.K_y:
                            action, self.confirm = self.confirm[1], None
                            action()
                        elif event.key in (pg.K_n, pg.K_ESCAPE):
                            self.confirm = None
                modal_click, self.click = self.click, None if self.confirm else self.click

                mx, my, _ = self.map_pose()
                if not self.trail or math.hypot(mx - self.trail[-1][0], my - self.trail[-1][1]) > 0.02:
                    self.trail.append((mx, my))
                self.sample()

                self.screen.fill(BG)
                self.draw_map()
                self.draw_header()
                self.draw_plan() if self.tab == "PLAN" else self.draw_calibrate()
                self.draw_logs()
                if self.confirm:
                    self.click = modal_click
                    self.draw_confirm()
                pg.display.flip()
                clock.tick(60)
        finally:
            if self.runner:
                self.runner.stop_event.set()
            self.robot.stop()
            if self.real:
                self.real.close()
            pg.quit()


def run(mode="sim", conn_type="ap"):
    return App(mode, conn_type).run()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(run("real" if "--real" in sys.argv else "sim"))
