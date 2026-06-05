"""Tests for src/voxel_nav.py (port of reference NodeManager + Pathfinding)."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from src.voxel_nav import (
    CELL_SIZE,
    Graph,
    Node,
    NodeType,
    Serial,
    VoxelNavManager,
    find_path_astar,
    serial_to_center,
    world_to_serial,
)


def _line(graph: Graph, points: list[Serial]) -> None:
    for s in points:
        graph.add_node(Node(serial=s, node_type=NodeType.REACHABLE))


class TestSerialMath(unittest.TestCase):
    def test_world_to_serial_quarter_meter(self):
        # cell size 0.25 -> each meter = 4 voxels
        self.assertEqual(world_to_serial(0.0, 0.0, 0.0), (0, 0, 0))
        self.assertEqual(world_to_serial(0.25, 0.0, 0.0), (1, 0, 0))
        self.assertEqual(world_to_serial(0.24, 0.0, 0.0), (0, 0, 0))
        self.assertEqual(world_to_serial(-0.01, 0.0, 0.0), (-1, 0, 0))
        self.assertEqual(world_to_serial(1.0, 2.0, -3.5), (4, 8, -14))

    def test_serial_to_center(self):
        # center is corner + half cell
        cx, cy, cz = serial_to_center((4, 8, -14))
        self.assertAlmostEqual(cx, 1.125, places=4)
        self.assertAlmostEqual(cy, 2.125, places=4)
        self.assertAlmostEqual(cz, -3.375, places=4)


class TestGraphBasics(unittest.TestCase):
    def test_add_find_contains(self):
        g = Graph()
        n = Node(serial=(1, 0, 1))
        g.add_node(n)
        self.assertIn((1, 0, 1), g)
        self.assertIs(g.get((1, 0, 1)), n)
        self.assertEqual(len(g), 1)

    def test_find_closest_ignores_unreachable(self):
        g = Graph()
        g.add_node(Node(serial=(0, 0, 0), node_type=NodeType.UNREACHABLE))
        g.add_node(Node(serial=(10, 0, 0), node_type=NodeType.REACHABLE))
        closest = g.find_closest(0.0, 0.0, 0.0)
        self.assertIsNotNone(closest)
        self.assertEqual(closest.serial, (10, 0, 0))

    def test_round_trip_json(self):
        g = Graph()
        g.add_node(Node(serial=(1, 2, 3), node_type=NodeType.REACHABLE))
        g.add_node(Node(serial=(4, 5, 6), node_type=NodeType.IFFY, label="hub"))
        restored = Graph.from_dict(json.loads(json.dumps(g.to_dict())))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored.get((4, 5, 6)).node_type, NodeType.IFFY)
        self.assertEqual(restored.get((4, 5, 6)).label, "hub")


class TestPathfinding(unittest.TestCase):
    def test_straight_corridor(self):
        g = Graph()
        # 1D corridor along X
        _line(g, [(i, 0, 0) for i in range(0, 6)])
        result = find_path_astar(g, (0, 0, 0), (5, 0, 0))
        self.assertTrue(result.found)
        # filter keeps only the endpoint for a straight run
        self.assertEqual(result.serials, [(5, 0, 0)])
        # full path is every cell
        self.assertEqual(result.full_serials,
                         [(i, 0, 0) for i in range(0, 6)])
        self.assertAlmostEqual(result.cost, 5.0, places=4)

    def test_turn_in_path(self):
        # L-shape: walk +x then +z
        g = Graph()
        _line(g, [(i, 0, 0) for i in range(0, 4)])
        _line(g, [(3, 0, j) for j in range(1, 4)])
        result = find_path_astar(g, (0, 0, 0), (3, 0, 3))
        self.assertTrue(result.found)
        # A* prefers the diagonal cut so filtered path has at least one
        # turn before reaching the goal
        self.assertGreaterEqual(len(result.serials), 2)
        self.assertEqual(result.serials[-1], (3, 0, 3))

    def test_no_path_when_disconnected(self):
        g = Graph()
        g.add_node(Node(serial=(0, 0, 0)))
        g.add_node(Node(serial=(10, 0, 10)))
        self.assertFalse(find_path_astar(g, (0, 0, 0), (10, 0, 10)).found)

    def test_diagonal_blocked_when_both_orthogonal_missing(self):
        # only start and a diagonal neighbor exist -> corner check blocks it
        g = Graph()
        g.add_node(Node(serial=(0, 0, 0)))
        g.add_node(Node(serial=(1, 0, 1)))
        result = find_path_astar(g, (0, 0, 0), (1, 0, 1))
        self.assertFalse(result.found)

    def test_diagonal_allowed_when_one_orthogonal_present(self):
        g = Graph()
        g.add_node(Node(serial=(0, 0, 0)))
        g.add_node(Node(serial=(1, 0, 0)))  # one orthogonal neighbor exists
        g.add_node(Node(serial=(1, 0, 1)))
        result = find_path_astar(g, (0, 0, 0), (1, 0, 1))
        self.assertTrue(result.found)

    def test_vertical_step_costs_extra(self):
        # straight horizontal vs. one Y up + horizontal
        g = Graph()
        g.add_node(Node(serial=(0, 0, 0)))
        g.add_node(Node(serial=(1, 0, 0)))
        g.add_node(Node(serial=(1, 1, 0)))
        r_flat = find_path_astar(g, (0, 0, 0), (1, 0, 0))
        r_up = find_path_astar(g, (0, 0, 0), (1, 1, 0))
        self.assertTrue(r_flat.found and r_up.found)
        # vertical move uses sqrt2 + 0.4142
        self.assertGreater(r_up.cost, r_flat.cost)

    def test_skips_unreachable_nodes(self):
        g = Graph()
        _line(g, [(0, 0, 0), (2, 0, 0)])
        g.add_node(Node(serial=(1, 0, 0), node_type=NodeType.UNREACHABLE))
        result = find_path_astar(g, (0, 0, 0), (2, 0, 0))
        # only path goes through the unreachable cell, so no path
        self.assertFalse(result.found)


class TestManagerLearningAndPersistence(unittest.TestCase):
    def test_observe_adds_nodes_in_learning_mode(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_test")
        mgr.observe(0.0, 0.0, 0.0)
        mgr.observe(0.30, 0.0, 0.0)  # next voxel over (since cell=0.25)
        self.assertEqual(len(mgr.graph), 2)
        self.assertIsNotNone(mgr.current)
        self.assertIsNotNone(mgr.previous)

    def test_observe_skips_when_not_grounded(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_x")
        mgr.observe(0.0, 0.0, 0.0, grounded=False)
        self.assertEqual(len(mgr.graph), 0)

    def test_save_and_reload_per_world(self):
        tmp = Path(tempfile.mkdtemp())
        mgr = VoxelNavManager(data_dir=tmp, learning_mode=True)
        mgr.load_world("wrld_a")
        for i in range(5):
            mgr.observe(i * 0.25, 0.0, 0.0)
        mgr.flush()
        self.assertTrue((tmp / "wrld_a.json").exists())

        mgr2 = VoxelNavManager(data_dir=tmp)
        mgr2.load_world("wrld_a")
        self.assertEqual(len(mgr2.graph), 5)

    def test_plan_to_uses_current_as_start(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_plan")
        for i in range(6):
            mgr.observe(i * 0.25, 0.0, 0.0)
        # now ask to go from current (which is last observed) backwards
        result = mgr.plan_to((0.0, 0.0, 0.0))
        self.assertTrue(result.found)
        self.assertEqual(result.serials[-1], (0, 0, 0))


class TestFillInteriorGaps(unittest.TestCase):
    def test_fills_single_cell_hole(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap")
        # 3x3 floor patch with the center cell missing
        for x in range(3):
            for z in range(3):
                if (x, z) == (1, 1):
                    continue
                mgr.graph.add_node(Node(serial=(x, 0, z),
                                        node_type=NodeType.REACHABLE))
        filled = mgr.fill_interior_gaps()
        self.assertEqual(filled, 1)
        self.assertIn((1, 0, 1), mgr.graph)

    def test_skips_edge_and_walls(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap2")
        # straight line of floor: the empty cells beside it only have 1
        # cardinal reachable neighbor, so nothing should be filled
        for x in range(4):
            mgr.graph.add_node(Node(serial=(x, 0, 0),
                                    node_type=NodeType.REACHABLE))
        # an explicit wall hole should never be promoted
        mgr.graph.add_node(Node(serial=(1, 0, 1),
                                node_type=NodeType.UNREACHABLE))
        before = len(mgr.graph)
        filled = mgr.fill_interior_gaps()
        self.assertEqual(filled, 0)
        self.assertEqual(len(mgr.graph), before)
        self.assertEqual(mgr.graph.get((1, 0, 1)).node_type,
                         NodeType.UNREACHABLE)

    def test_does_not_stack_double_layer(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap_dl")
        # solid 3x3 floor at y=0, plus a single floor cell up at y=1 in the
        # center. the hole beside that upper cell must NOT be filled, since
        # filling it would stack a second floor right on top of y=0 floor.
        for x in range(3):
            for z in range(3):
                mgr.graph.add_node(Node(serial=(x, 0, z),
                                        node_type=NodeType.REACHABLE))
        mgr.graph.add_node(Node(serial=(1, 1, 1),
                                node_type=NodeType.REACHABLE))
        before = len(mgr.graph)
        mgr.fill_interior_gaps()
        # nothing at y=1 next to the lone upper cell should appear above the
        # y=0 floor
        for x in range(3):
            for z in range(3):
                if (x, z) == (1, 1):
                    continue
                self.assertNotIn((x, 1, z), mgr.graph)
        self.assertEqual(len(mgr.graph), before)

    def test_does_not_climb_stairs(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap_stairs")
        # a staircase: each step one cell up and one cell along +x. the open
        # wedge under/beside the steps must stay empty.
        for i in range(5):
            mgr.graph.add_node(Node(serial=(i, i, 0),
                                    node_type=NodeType.REACHABLE))
        filled = mgr.fill_interior_gaps()
        self.assertEqual(filled, 0)

    def test_skips_large_void(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap_void")
        # an 8x8 ring of floor with a big 6x6 empty middle. that middle is a
        # void (balcony opening / pit), not a speckle hole, so leave it empty.
        for x in range(8):
            for z in range(8):
                if 1 <= x <= 6 and 1 <= z <= 6:
                    continue
                mgr.graph.add_node(Node(serial=(x, 0, z),
                                        node_type=NodeType.REACHABLE))
        filled = mgr.fill_interior_gaps()
        self.assertEqual(filled, 0)
        self.assertNotIn((3, 0, 3), mgr.graph)

    def test_fills_small_pocket(self):
        mgr = VoxelNavManager(data_dir=tempfile.mkdtemp(), learning_mode=True)
        mgr.load_world("wrld_gap_pocket")
        # 4x4 floor with a 2x2 hole in the middle -> small enclosed pocket,
        # should be fully closed.
        hole = {(1, 1), (2, 1), (1, 2), (2, 2)}
        for x in range(4):
            for z in range(4):
                if (x, z) in hole:
                    continue
                mgr.graph.add_node(Node(serial=(x, 0, z),
                                        node_type=NodeType.REACHABLE))
        filled = mgr.fill_interior_gaps()
        self.assertEqual(filled, 4)
        for x, z in hole:
            self.assertIn((x, 0, z), mgr.graph)


if __name__ == "__main__":
    unittest.main()

