"""
CTT Unit Tests for rasch-irt library
Verifies difficulty, discrimination, Point-Biserial, reliability, and full CTT pipeline.
"""
import unittest
import numpy as np
from rasch_irt.ctt import (
    run_ctt,
    compute_difficulty,
    compute_discrimination_d,
    compute_point_biserial,
    compute_reliability,
)
from rasch_irt.core import prob_3pl

class TestCtt(unittest.TestCase):
    def setUp(self):
        # Thiết lập ma trận dữ liệu mẫu cố định
        np.random.seed(42)
        self.N, self.M = 20, 300
        self.theta = np.random.normal(0, 1, self.M)
        self.b_true = np.random.normal(0, 1, self.N)
        self.a_true = np.random.uniform(0.5, 2.0, self.N)
        self.P = prob_3pl(self.theta, self.a_true, self.b_true, np.zeros(self.N))
        self.U = (np.random.rand(self.N, self.M) < self.P).astype(int)

    def test_difficulty(self):
        """Kiểm tra tính toán độ khó p"""
        p = compute_difficulty(self.U)
        p_manual = np.array([self.U[i].sum() / self.M for i in range(self.N)])
        np.testing.assert_allclose(p, p_manual, rtol=1e-5, atol=1e-8)

    def test_discrimination_d(self):
        """Kiểm tra tính toán độ phân biệt D (27%)"""
        D = compute_discrimination_d(self.U)
        total_scores = self.U.sum(axis=0)
        sorted_indices = np.argsort(total_scores)
        n_group = max(int(self.M * 0.27), 1)
        lower_group = sorted_indices[:n_group]
        upper_group = sorted_indices[-n_group:]
        
        D_manual = np.array([self.U[i, upper_group].mean() - self.U[i, lower_group].mean() for i in range(self.N)])
        np.testing.assert_allclose(D, D_manual, rtol=1e-5, atol=1e-8)

    def test_point_biserial(self):
        """Kiểm tra Point-Biserial hiệu chỉnh (Corrected Item-Total Correlation)"""
        rpb = compute_point_biserial(self.U)
        total_scores = self.U.sum(axis=0).astype(float)
        
        # Ground truth: loại bỏ câu 0 khỏi tổng điểm, rồi tính tương quan Pearson r với U[0]
        corrected_scores_0 = total_scores - self.U[0]
        rpb_manual_0 = np.corrcoef(self.U[0], corrected_scores_0)[0, 1]
        self.assertAlmostEqual(rpb[0], rpb_manual_0, places=10)

    def test_reliability(self):
        """Kiểm tra tính toán KR-20, Cronbach Alpha và SEM"""
        rel = compute_reliability(self.U)
        p = compute_difficulty(self.U)
        pq = p * (1.0 - p)
        total_scores = self.U.sum(axis=0).astype(float)
        variance_total = total_scores.var() # Population variance
        
        kr20_manual = (self.N / (self.N - 1)) * (1.0 - pq.sum() / variance_total)
        self.assertAlmostEqual(rel["kr20"], kr20_manual, delta=1e-3)
        self.assertAlmostEqual(rel["kr20"], rel["cronbach_alpha"], delta=1e-3)
        
        sem_manual = total_scores.std() * np.sqrt(1.0 - max(rel["kr20"], 0.0))
        self.assertAlmostEqual(rel["sem"], sem_manual, delta=0.01)

    def test_full_pipeline(self):
        """Kiểm tra Pipeline chạy CTT toàn bộ bao gồm loại câu hỏi xấu"""
        ctt_result = run_ctt(self.U)
        self.assertEqual(len(ctt_result.items), self.N)
        
        for item in ctt_result.items:
            if item.discrimination_d < 0:
                self.assertTrue(item.excluded_from_irt)
                self.assertIn(item.item_number, ctt_result.bad_items)

if __name__ == "__main__":
    unittest.main()
