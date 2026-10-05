import unittest
import numpy as np
from metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_bpp,
    calculate_compression_ratio,
    calculate_rate_distortion,
    calculate_bd_rate,
    evaluate_compression_suite,
)


class TestCompressionMetrics(unittest.TestCase):
    def setUp(self):
        # Create synthetic 64x64 test images
        self.orig = np.ones((64, 64, 3), dtype=np.uint8) * 128
        self.comp = self.orig.copy()

    def test_identical_images(self):
        self.assertEqual(calculate_psnr(self.orig, self.comp), float("inf"))
        self.assertAlmostEqual(calculate_ssim(self.orig, self.comp), 1.0, places=2)
        self.assertEqual(calculate_mae(self.orig, self.comp), 0.0)

    def test_distorted_images(self):
        distorted = self.orig.copy()
        distorted[0:10, 0:10] = 200
        psnr = calculate_psnr(self.orig, distorted)
        self.assertGreater(psnr, 0.0)
        self.assertLess(psnr, 100.0)
        self.assertGreater(calculate_mae(self.orig, distorted), 0.0)

    def test_compression_ratio_and_bpp(self):
        orig_bytes = 64 * 64 * 3
        comp_bytes = int(orig_bytes / 4)
        ratio = calculate_compression_ratio(orig_bytes, comp_bytes)
        self.assertAlmostEqual(ratio, 4.0, places=1)
        bpp = calculate_bpp(comp_bytes, 64, 64)
        self.assertAlmostEqual(bpp, 6.0, places=1)

    def test_bd_rate_calculation(self):
        # Curve 1: higher bitrate
        r1 = [0.1, 0.2, 0.4, 0.8]
        p1 = [28.0, 31.0, 34.0, 37.0]

        # Curve 2: lower bitrate for identical PSNR (more efficient)
        r2 = [0.08, 0.16, 0.32, 0.64]
        p2 = [28.0, 31.0, 34.0, 37.0]

        bd_rate = calculate_bd_rate(r1, p1, r2, p2)
        # Curve 2 uses ~20% lower bitrate -> BD-rate should be negative ~ -20%
        self.assertLess(bd_rate, 0.0)
        self.assertAlmostEqual(bd_rate, -20.0, delta=2.0)

    def test_evaluate_compression_suite(self):
        suite = evaluate_compression_suite(
            self.orig, self.comp, compressed_bytes=1024, lambda_param=0.01
        )
        self.assertEqual(suite["psnr_db"], float("inf"))
        self.assertEqual(suite["perceptual_grade"], "EXCELLENT")
        self.assertIn("rd_cost", suite)
        self.assertIn("bpp", suite)


if __name__ == "__main__":
    unittest.main()
