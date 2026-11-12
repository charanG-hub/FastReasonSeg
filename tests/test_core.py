import unittest
import numpy as np
from fastreasonseg_core.digital_twin import build_digital_twin
from fastreasonseg_core.tracking import IoUTracker, box_iou

class DigitalTwinTests(unittest.TestCase):
    def test_builds_object_and_depth_stats(self):
        mask = np.zeros((10, 12), dtype=bool); mask[2:6, 3:8] = True
        depth = np.linspace(0, 1, 120, dtype=np.float32).reshape(10, 12)
        twin = build_digital_twin((12, 10), [{"box": [2, 1, 9, 8], "mask": mask,
            "label": "dog", "confidence": .91}], depth)
        self.assertEqual(twin["object_count"], 1)
        self.assertEqual(twin["objects"][0]["label"], "dog")
        self.assertEqual(twin["objects"][0]["mask"]["area_pixels"], 20)
        self.assertIsNotNone(twin["objects"][0]["relative_depth"])

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaises(ValueError):
            build_digital_twin((12, 10), [{"box": [1, 1, 5, 5], "mask": np.zeros((2,2))}])

class TrackerTests(unittest.TestCase):
    def test_iou_and_stable_id(self):
        self.assertGreater(box_iou([0,0,10,10], [1,1,11,11]), .6)
        tracker = IoUTracker()
        first = tracker.update([{"box": [0,0,10,10], "label": "car", "confidence": .9}], 0, (100,100))
        second = tracker.update([{"box": [1,0,11,10], "label": "car", "confidence": .8}], 1, (100,100))
        self.assertEqual(first[0]["track_id"], second[0]["track_id"])

if __name__ == "__main__":
    unittest.main()
