"""Unit tests for SpatialDepthGate and AdaptiveBoundaryRefiner modules."""
import unittest
import numpy as np
from fastreasonseg_core.spatial_depth_gate import compute_3d_spatial_graph, SpatialDepthGate
from fastreasonseg_core.boundary_refiner import (
    AdaptiveBoundaryRefiner,
    calculate_boundary_f_score,
    synthesize_corrective_points,
)

class TestSpatialDepthGate(unittest.TestCase):
    def setUp(self):
        self.objects = [
            {
                "object_id": "car_front",
                "label": "car",
                "detection_confidence": 0.95,
                "bounding_box_xyxy": [50, 100, 250, 300],
                "relative_depth": {"mean": 1.2, "median": 1.1},
            },
            {
                "object_id": "truck_back",
                "label": "truck",
                "detection_confidence": 0.88,
                "bounding_box_xyxy": [180, 80, 420, 320],
                "relative_depth": {"mean": 4.5, "median": 4.4},
            },
        ]
        self.w, self.h = 640, 480

    def test_spatial_graph_depth_and_occlusion(self):
        graph = compute_3d_spatial_graph(self.objects, self.w, self.h)
        self.assertEqual(graph["num_objects"], 2)
        # Verify car_front has rank 1 in depth
        self.assertEqual(graph["depth_ordering"][0]["object_id"], "car_front")
        # Check relations
        rel = graph["pairwise_relations"][0]
        self.assertIn("foreground_of", rel["depth_relation"])
        self.assertTrue(rel["is_occluding"])

    def test_gating_reranker(self):
        gate = SpatialDepthGate()
        graph = compute_3d_spatial_graph(self.objects, self.w, self.h)
        # Query asking for vehicle behind
        ranked = gate.score_candidates("find the vehicle behind", graph)
        self.assertEqual(ranked[0]["object_id"], "truck_back")

        # Query asking for vehicle in front
        ranked_front = gate.score_candidates("the vehicle in front closest to camera", graph)
        self.assertEqual(ranked_front[0]["object_id"], "car_front")


class TestAdaptiveBoundaryRefiner(unittest.TestCase):
    def setUp(self):
        # Create a synthetic circle mask with some ragged edges (bleed)
        self.h, self.w = 100, 100
        y, x = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((x - 50)**2 + (y - 50)**2)
        self.gt_mask = dist_from_center <= 30
        
        # Ragged mask with edge bleed
        self.pred_mask = (dist_from_center <= 32).copy()
        self.pred_mask[40:60, 80:85] = True  # Bleed peninsula

        # Depth map: foreground is close (depth=2), background is far (depth=8)
        self.depth_map = np.full((self.h, self.w), 8.0, dtype=np.float32)
        self.depth_map[dist_from_center <= 30] = 2.0

    def test_boundary_f_score(self):
        f_eval = calculate_boundary_f_score(self.pred_mask, self.gt_mask, tolerance=2.0)
        self.assertGreater(f_eval["f_score"], 0.65)
        self.assertLessEqual(f_eval["f_score"], 1.0)

    def test_corrective_point_synthesis(self):
        pts, lbls, diag = synthesize_corrective_points(self.pred_mask, self.depth_map)
        self.assertIn(1, lbls)  # Must include positive prompt
        self.assertGreater(len(pts), 0)

    def test_refiner_execution(self):
        refiner = AdaptiveBoundaryRefiner()
        out = refiner.refine(self.pred_mask, self.depth_map)
        self.assertTrue(out["refined_mask"].any())
        self.assertIn("boundary_metrics", out)
        # Refined mask should remove the bleed peninsula
        self.assertFalse(out["refined_mask"][50, 83])


if __name__ == "__main__":
    unittest.main()
