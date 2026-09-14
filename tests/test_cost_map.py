import math
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cost_map_panel as cm


class CostMapTest(unittest.TestCase):
    def test_bfs_fewest_moves(self):
        path, _ = cm.bfs()
        self.assertEqual((path[0], path[-1]), (cm.START, cm.GOAL))
        self.assertEqual(len(path) - 1, 6)
        self.assertEqual(cm.path_cost(path), 15)  # E-first tie-break takes the top route

    def test_astar_cheapest(self):
        path, _ = cm.astar()
        self.assertEqual(cm.path_cost(path), 12)
        self.assertEqual(path, [(1, 4), (1, 3), (1, 2), (2, 2), (2, 1), (3, 1), (4, 1)])
        self.assertFalse(set(path) & cm.OBSTACLES)

    def test_odom_to_map(self):
        cal = dict(cm.DEFAULT_CAL, start_heading=270)  # facing S
        mx, my, h = cm.odom_to_map(0.6, 0.6, 90.0, cal)  # 1 cell fwd, 1 cell right, turned right
        self.assertAlmostEqual(mx, 0.0)
        self.assertAlmostEqual(my, 3.0)
        self.assertAlmostEqual(h, 180.0)

    def test_sim_drives_astar_path(self):
        cal = dict(cm.DEFAULT_CAL)
        sim = cm.SimRobot()
        sim.speed = 4.0
        path, _ = cm.astar()
        runner = cm.PathRunner(sim, cal, [("cell", c) for c in path[1:]], lambda m: None)
        runner.start()
        runner.join(30)
        self.assertTrue(runner.ok)
        mx, my, _ = cm.odom_to_map(*sim.pose(), cal=cal)
        self.assertLess(math.hypot(mx - cm.GOAL[0], my - cm.GOAL[1]), 0.05)


    def test_record_score_and_notebook(self):
        import json
        import tempfile
        import run_notebook

        cal = dict(cm.DEFAULT_CAL)
        sim = cm.SimRobot()
        sim.speed = 4.0
        results = {n: fn() for n, fn in cm.ALGORITHMS.items()}
        runner = cm.PathRunner(sim, cal, [("cell", c) for c in results["A*"][0][1:]], lambda m: None)
        runner.start()
        samples, t0 = [], time.monotonic()
        while not runner.done:
            mx, my, h = cm.odom_to_map(*sim.pose(), cal=cal)
            samples.append({"t": time.monotonic() - t0, "mx": mx, "my": my, "heading": h,
                            "tof": cm.sim_tof_mm(mx, my, h, cal)})
            time.sleep(0.02)
        run = cm.build_run("A_star_1", "A*", "sim", results, samples, runner.status, cal)
        r = run["result"]
        self.assertEqual((r["planned_steps"], r["actual_steps"]), (6, 6))
        self.assertEqual((r["hit_obstacle"], r["reach_goal"]), ("NO", "YES"))
        self.assertGreater(r["time_s"], 0)

        crash = [{"t": 0, "mx": 1, "my": 4, "heading": 0, "tof": None},
                 {"t": 1, "mx": 2, "my": 3, "heading": 0, "tof": None}]  # (2,3) is an obstacle
        self.assertEqual(cm.build_run("x", "BFS", "sim", results, crash, "Done", cal)["result"]["hit_obstacle"], "YES")

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_notebook.next_name("A*", tmp), "A_star_1")
            path = run_notebook.save(run, tmp)
            nb = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            self.assertTrue(any(o["output_type"] == "display_data" for c in nb["cells"] for o in c.get("outputs", [])))
            self.assertEqual(run_notebook.next_name("A*", tmp), "A_star_2")
            self.assertEqual(run_notebook.next_name("BFS", tmp), "BFS_1")

    def test_turn_fixes_wrong_z_sign(self):
        cal = dict(cm.DEFAULT_CAL, z_sign=1)  # wrong for this EP: hardware z>0 turns CW

        class EP(cm.SimRobot):
            def drive(self, vx, vy, wz):
                cm.SimRobot.drive(self, vx, vy, -wz * cal["z_sign"])

        robot = EP()
        robot.speed = 4.0
        runner = cm.PathRunner(robot, cal, [("turn", 90.0)], lambda m: None)
        runner.start()
        runner.join(30)
        self.assertTrue(runner.ok)
        self.assertEqual(cal["z_sign"], -1)
        self.assertAlmostEqual(cm.odom_to_map(*robot.pose(), cal=cal)[2], 90.0, delta=cm.TURN_TOL_DEG)


if __name__ == "__main__":
    unittest.main()
