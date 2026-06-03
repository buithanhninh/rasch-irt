"""
IRT Unit Tests for rasch-irt library
Verifies JMLE estimation, fit statistics, SE computation, assumptions, Auto-Fit and high-level E2E run_irt.
"""
import unittest
import numpy as np
from rasch_irt.core import prob_3pl
from rasch_irt.irt import (
    JMLEConfig,
    run_jmle,
    run_irt,
    run_auto_fit,
    compute_fit_statistics,
    compute_true_scores,
    check_unidimensionality,
    check_local_independence,
)

class TestIrt(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N, self.M = 10, 80
        self.theta = np.random.normal(0, 1, self.M)
        self.a = np.random.uniform(0.8, 1.5, self.N)
        self.b = np.random.normal(0, 1, self.N)
        self.c = np.zeros(self.N)
        self.P = prob_3pl(self.theta, self.a, self.b, self.c)
        self.U = (np.random.rand(self.N, self.M) < self.P).astype(int)

    def test_fit_statistics(self):
        """Kiểm tra tính tương hợp Infit và Outfit MNSQ"""
        fit = compute_fit_statistics(self.U, self.P)
        Q = 1.0 - self.P
        variance = self.P * Q
        variance = np.maximum(variance, 1e-10)
        residual_sq = (self.U - self.P) ** 2
        
        # OUTFIT item
        outfit_item_manual = np.mean(residual_sq / variance, axis=1)
        np.testing.assert_allclose(fit['outfit_item'], outfit_item_manual, rtol=1e-5, atol=1e-8)
        
        # INFIT item
        infit_item_manual = np.sum(residual_sq, axis=1) / np.sum(variance, axis=1)
        np.testing.assert_allclose(fit['infit_item'], infit_item_manual, rtol=1e-5, atol=1e-8)
        
        # OUTFIT person
        outfit_person_manual = np.mean(residual_sq / variance, axis=0)
        np.testing.assert_allclose(fit['outfit_person'], outfit_person_manual, rtol=1e-5, atol=1e-8)

    def test_true_scores(self):
        """Kiểm tra quy đổi điểm thực True Score trên thang 10"""
        ts = compute_true_scores(self.theta, self.a, self.b, self.c)
        ts_manual = (np.sum(self.P, axis=0) / self.N) * 10.0
        np.testing.assert_allclose(ts, np.clip(ts_manual, 0.0, 10.0), rtol=1e-5, atol=1e-8)

    def test_unidimensionality(self):
        """Kiểm tra tiên đề đơn hướng (PCA)"""
        pca = check_unidimensionality(self.U)
        self.assertTrue(pca["first_eigenvalue"] >= 0)
        self.assertTrue(pca["ratio_explained"] > 0)

    def test_local_independence(self):
        """Kiểm tra tiên đề độc lập cục bộ (Yen's Q3)"""
        q3 = check_local_independence(self.U, self.theta, self.a, self.b, self.c)
        self.assertTrue(q3["max_q3"] >= 0.0)

    def test_run_jmle_1pl(self):
        """Kiểm tra chạy thuật toán JMLE cho mô hình 1PL (Rasch)"""
        config = JMLEConfig(model_type=1, max_iter=20, tol=0.01)
        res = run_jmle(self.U, config)
        self.assertEqual(len(res.theta), self.M)
        self.assertEqual(len(res.b), self.N)
        np.testing.assert_allclose(res.a, np.ones(self.N))

    def test_run_jmle_2pl(self):
        """Kiểm tra chạy thuật toán JMLE cho mô hình 2PL"""
        config = JMLEConfig(model_type=2, max_iter=20, tol=0.01)
        res = run_jmle(self.U, config)
        self.assertEqual(len(res.theta), self.M)
        self.assertEqual(len(res.a), self.N)
        self.assertEqual(len(res.b), self.N)

    def test_run_irt_e2e(self):
        """Kiểm tra E2E hàm run_irt bao gồm đầy đủ dữ liệu vẽ đồ thị và tiên đề"""
        config = JMLEConfig(model_type=2, max_iter=25, tol=0.005)
        res = run_irt(self.U, config)
        self.assertTrue(len(res.items) == self.N)
        self.assertTrue(len(res.persons) == self.M)
        self.assertIsNotNone(res.assumptions)
        self.assertTrue(len(res.theta_density) > 0)
        
        # Verify specific formats
        self.assertIn("theta", res.items[0].icc_points[0])
        self.assertIn("prob", res.items[0].icc_points[0])
        self.assertIn("info", res.items[0].iif_points[0])

    def test_auto_fit(self):
        """Kiểm tra chạy Auto-Fit so sánh mô hình tự động"""
        auto_res = run_auto_fit(self.U, num_options=4, max_iter=15, epsilon=0.01)
        self.assertTrue(len(auto_res.models) > 0)
        self.assertIn(auto_res.recommended_model, [1, 2, 3])
        self.assertIsNotNone(auto_res.recommendation_reason)

if __name__ == "__main__":
    unittest.main()
