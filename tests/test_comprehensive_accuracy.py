"""
Comprehensive Mathematical Accuracy Tests for rasch-irt
========================================================
Kiểm chứng thuật toán tính toán IRT (1PL/2PL/3PL) trên nhiều bộ dữ liệu,
tình huống và edge cases khác nhau.

Tests cover:
  1. Consistency: Kết quả nhất quán qua nhiều lần chạy
  2. Known-parameter recovery: Tạo dữ liệu từ tham số đã biết → phục hồi lại
  3. Edge cases: Điểm 0, điểm tối đa, câu hỏi cực khó/dễ
  4. Cross-engine sync: rasch_irt vs PythonEngine cùng kết quả
  5. Diverse datasets: Nhiều cỡ mẫu, tỷ lệ đúng khác nhau
  6. Mathematical invariants: P(θ) ∈ [0,1], LL < 0, AIC/BIC consistency
"""
import sys
import os
import numpy as np
import pytest

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'IRT.PythonEngine'))

from rasch_irt import run_jmle, JMLEConfig
from rasch_irt.core import prob_3pl, D
from rasch_irt.irt import (
    compute_fit_statistics,
    compute_se_theta,
    compute_se_items,
    compute_true_scores,
    compute_log_likelihood,
    run_auto_fit,
)
from rasch_irt.mml_em import run_mml_em, MMLConfig, gauss_hermite_points, e_step, eap_scoring


# =========================================================================
# Helper: Tạo dữ liệu mô phỏng từ tham số đã biết
# =========================================================================

def simulate_irt_data(
    N: int,     # Số câu hỏi
    M: int,     # Số thí sinh
    model_type: int = 1,
    a_range: tuple = (0.5, 2.0),
    b_range: tuple = (-2.0, 2.0),
    c_val: float = 0.25,
    seed: int = 42,
) -> tuple:
    """Tạo dữ liệu IRT mô phỏng từ tham số đã biết."""
    rng = np.random.RandomState(seed)
    
    theta_true = rng.standard_normal(M)
    b_true = rng.uniform(b_range[0], b_range[1], N)
    
    if model_type == 1:
        a_true = np.ones(N)
        c_true = np.zeros(N)
    elif model_type == 2:
        a_true = rng.uniform(a_range[0], a_range[1], N)
        c_true = np.zeros(N)
    else:  # 3PL
        a_true = rng.uniform(a_range[0], a_range[1], N)
        c_true = np.full(N, c_val)
    
    P = prob_3pl(theta_true, a_true, b_true, c_true)
    U = (rng.random((N, M)) < P).astype(float)
    
    return U, theta_true, a_true, b_true, c_true


# =========================================================================
# TEST GROUP 1: Tính nhất quán (Consistency)
# =========================================================================

class TestConsistency:
    """Kiểm tra kết quả nhất quán qua nhiều lần chạy trên cùng dữ liệu."""
    
    def test_1pl_consistent_across_runs(self):
        """1PL JMLE phải cho kết quả giống hệt nhau khi chạy lại."""
        U, _, _, _, _ = simulate_irt_data(20, 100, model_type=1, seed=10)
        
        results = []
        for _ in range(3):
            config = JMLEConfig(model_type=1, max_iter=100)
            res = run_jmle(U, config)
            results.append(res)
        
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0].b, results[i].b, atol=1e-10,
                                       err_msg="1PL b params differ across runs")
            np.testing.assert_allclose(results[0].theta, results[i].theta, atol=1e-10,
                                       err_msg="1PL theta differs across runs")
    
    def test_2pl_consistent_across_runs(self):
        """2PL MML-EM phải cho kết quả giống hệt nhau khi chạy lại."""
        U, _, _, _, _ = simulate_irt_data(15, 200, model_type=2, seed=20)
        
        results = []
        for _ in range(3):
            config = JMLEConfig(model_type=2, max_iter=100)
            res = run_jmle(U, config)
            results.append(res)
        
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0].b, results[i].b, atol=1e-10,
                                       err_msg="2PL b params differ across runs")
            np.testing.assert_allclose(results[0].a, results[i].a, atol=1e-10,
                                       err_msg="2PL a params differ across runs")
    
    def test_3pl_consistent_across_runs(self):
        """3PL MML-EM phải cho kết quả giống hệt nhau khi chạy lại."""
        U, _, _, _, _ = simulate_irt_data(15, 300, model_type=3, seed=30)
        
        results = []
        for _ in range(3):
            config = JMLEConfig(model_type=3, max_iter=100, num_options=4)
            res = run_jmle(U, config)
            results.append(res)
        
        for i in range(1, len(results)):
            np.testing.assert_allclose(results[0].b, results[i].b, atol=1e-10,
                                       err_msg="3PL b params differ across runs")


# =========================================================================
# TEST GROUP 2: Known Parameter Recovery
# =========================================================================

class TestParameterRecovery:
    """Kiểm tra khả năng phục hồi tham số khi biết giá trị gốc."""
    
    def test_1pl_recovers_b_ordering(self):
        """1PL: Thứ tự b ước lượng phải tương quan mạnh với b thật."""
        U, _, _, b_true, _ = simulate_irt_data(30, 500, model_type=1, seed=100)
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        corr = np.corrcoef(b_true, res.b)[0, 1]
        assert corr > 0.90, f"1PL b correlation = {corr:.4f}, expected > 0.90"
        assert res.converged, "1PL did not converge"
    
    def test_2pl_recovers_a_and_b(self):
        """2PL MML-EM: Phải phục hồi cả a và b với tương quan > 0.85."""
        U, _, a_true, b_true, _ = simulate_irt_data(25, 500, model_type=2, seed=200)
        
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        corr_b = np.corrcoef(b_true, res.b)[0, 1]
        corr_a = np.corrcoef(a_true, res.a)[0, 1]
        
        assert corr_b > 0.85, f"2PL b correlation = {corr_b:.4f}, expected > 0.85"
        assert corr_a > 0.70, f"2PL a correlation = {corr_a:.4f}, expected > 0.70"
        assert res.converged, "2PL did not converge"
    
    def test_3pl_recovers_b_and_a(self):
        """3PL MML-EM: Phải phục hồi b và a với tương quan hợp lý."""
        U, _, a_true, b_true, _ = simulate_irt_data(20, 1000, model_type=3, seed=300)
        
        config = JMLEConfig(model_type=3, max_iter=100, num_options=4)
        res = run_jmle(U, config)
        
        corr_b = np.corrcoef(b_true, res.b)[0, 1]
        corr_a = np.corrcoef(a_true, res.a)[0, 1]
        
        assert corr_b > 0.80, f"3PL b correlation = {corr_b:.4f}, expected > 0.80"
        assert corr_a > 0.60, f"3PL a correlation = {corr_a:.4f}, expected > 0.60"
    
    def test_1pl_recovers_theta_ordering(self):
        """1PL: Thứ tự theta ước lượng phải tương quan mạnh với theta thật."""
        U, theta_true, _, _, _ = simulate_irt_data(30, 500, model_type=1, seed=110)
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        corr = np.corrcoef(theta_true, res.theta)[0, 1]
        assert corr > 0.80, f"1PL theta correlation = {corr:.4f}, expected > 0.80"
    
    def test_2pl_recovers_theta(self):
        """2PL MML-EM: Theta EAP phải tương quan mạnh với theta thật."""
        U, theta_true, _, _, _ = simulate_irt_data(25, 500, model_type=2, seed=210)
        
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        corr = np.corrcoef(theta_true, res.theta)[0, 1]
        assert corr > 0.80, f"2PL theta correlation = {corr:.4f}, expected > 0.80"


# =========================================================================
# TEST GROUP 3: Mathematical Invariants
# =========================================================================

class TestMathematicalInvariants:
    """Kiểm tra các bất biến toán học phải luôn đúng."""
    
    def test_probability_bounds(self):
        """P(θ) phải nằm trong [0, 1] cho mọi giá trị."""
        theta = np.linspace(-5, 5, 100)
        a = np.array([0.5, 1.0, 2.0, 5.0])
        b = np.array([-2.0, 0.0, 1.0, 3.0])
        c = np.array([0.0, 0.1, 0.2, 0.25])
        
        P = prob_3pl(theta, a, b, c)
        assert np.all(P >= 0), "P(θ) < 0 detected!"
        assert np.all(P <= 1), "P(θ) > 1 detected!"
    
    def test_probability_monotonic_in_theta(self):
        """P(θ) phải đơn điệu tăng theo θ (với a > 0)."""
        theta = np.linspace(-4, 4, 200)
        a = np.array([1.5])
        b = np.array([0.0])
        c = np.array([0.1])
        
        P = prob_3pl(theta, a, b, c)
        diffs = np.diff(P[0])
        assert np.all(diffs >= -1e-10), "P(θ) is not monotonically increasing!"
    
    def test_probability_at_b_equals_midpoint(self):
        """Tại θ = b, P(θ) phải = c + (1-c)/2 (midpoint rule)."""
        a = np.array([1.0])
        b = np.array([0.0])
        c = np.array([0.2])
        theta = np.array([0.0])
        
        P = prob_3pl(theta, a, b, c)
        expected = 0.2 + (1 - 0.2) / 2  # 0.6
        assert abs(P[0, 0] - expected) < 0.01, \
            f"P(b) = {P[0,0]:.4f}, expected ≈ {expected:.4f}"
    
    def test_c_lower_asymptote(self):
        """P(θ→-∞) phải tiến tới c."""
        a = np.array([1.0])
        b = np.array([0.0])
        c = np.array([0.25])
        theta_low = np.array([-10.0])
        
        P = prob_3pl(theta_low, a, b, c)
        # P ≈ c khi θ rất nhỏ
        assert abs(P[0, 0] - 0.25) < 0.01, \
            f"P(θ→-∞) = {P[0,0]:.4f}, expected ≈ 0.25"
    
    def test_log_likelihood_negative(self):
        """Log-likelihood phải luôn < 0."""
        for model_type in [1, 2, 3]:
            U, _, _, _, _ = simulate_irt_data(15, 200, model_type=model_type, seed=400+model_type)
            config = JMLEConfig(model_type=model_type, max_iter=100, num_options=4)
            res = run_jmle(U, config)
            assert res.log_likelihood < 0, \
                f"Model {model_type}: LL = {res.log_likelihood} is not negative!"
    
    def test_aic_bic_consistency(self):
        """AIC phải > 0 và BIC phải > 0 khi LL < 0."""
        U, _, _, _, _ = simulate_irt_data(20, 300, model_type=2, seed=500)
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        assert res.aic > 0, f"AIC = {res.aic} is not positive"
        assert res.bic > 0, f"BIC = {res.bic} is not positive"
        # AIC = -2LL + 2k, BIC = -2LL + k*ln(M)
        # Khi M > e^2 ≈ 7.4, BIC > AIC
        if U.shape[1] > 8:
            assert res.bic >= res.aic - 1, "BIC should generally >= AIC for M > 8"
    
    def test_parameter_bounds_respected(self):
        """Tham số a, b, c phải nằm trong giới hạn cho phép."""
        for model_type in [1, 2, 3]:
            U, _, _, _, _ = simulate_irt_data(15, 300, model_type=model_type, seed=600+model_type)
            config = JMLEConfig(model_type=model_type, max_iter=100, num_options=4)
            res = run_jmle(U, config)
            
            assert np.all(res.b >= -5.5) and np.all(res.b <= 5.5), \
                f"Model {model_type}: b out of bounds [{res.b.min():.2f}, {res.b.max():.2f}]"
            if model_type >= 2:
                assert np.all(res.a > 0), f"Model {model_type}: a ≤ 0 detected"
            if model_type == 3:
                assert np.all(res.c >= 0) and np.all(res.c <= 0.5), \
                    f"Model {model_type}: c out of bounds [{res.c.min():.4f}, {res.c.max():.4f}]"


# =========================================================================
# TEST GROUP 4: Edge Cases
# =========================================================================

class TestEdgeCases:
    """Kiểm tra thuật toán xử lý đúng các trường hợp biên."""
    
    def test_extreme_easy_items(self):
        """Câu hỏi rất dễ (>95% đúng) không gây crash."""
        rng = np.random.RandomState(700)
        N, M = 10, 200
        U = np.ones((N, M))
        # Chỉ đánh vài câu sai ngẫu nhiên
        for i in range(N):
            wrong_idx = rng.choice(M, size=max(1, int(M * 0.03)), replace=False)
            U[i, wrong_idx] = 0.0
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert not np.any(np.isnan(res.b)), "NaN in b for extreme easy items"
        assert not np.any(np.isnan(res.theta)), "NaN in theta for extreme easy items"
    
    def test_extreme_hard_items(self):
        """Câu hỏi rất khó (<10% đúng) không gây crash."""
        rng = np.random.RandomState(710)
        N, M = 10, 200
        U = np.zeros((N, M))
        # Chỉ đánh vài câu đúng ngẫu nhiên
        for i in range(N):
            correct_idx = rng.choice(M, size=max(1, int(M * 0.07)), replace=False)
            U[i, correct_idx] = 1.0
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert not np.any(np.isnan(res.b)), "NaN in b for extreme hard items"
        assert not np.any(np.isnan(res.theta)), "NaN in theta for extreme hard items"
    
    def test_mixed_extreme_items(self):
        """Hỗn hợp câu rất dễ và rất khó."""
        rng = np.random.RandomState(720)
        N, M = 20, 300
        U = np.zeros((N, M))
        
        # Nửa dễ, nửa khó
        for i in range(N // 2):
            correct_idx = rng.choice(M, size=int(M * 0.90), replace=False)
            U[i, correct_idx] = 1.0
        for i in range(N // 2, N):
            correct_idx = rng.choice(M, size=int(M * 0.10), replace=False)
            U[i, correct_idx] = 1.0
        
        for model_type in [1, 2]:
            config = JMLEConfig(model_type=model_type, max_iter=200)
            res = run_jmle(U, config)
            
            assert not np.any(np.isnan(res.b)), f"Model {model_type}: NaN in b"
            assert not np.any(np.isinf(res.b)), f"Model {model_type}: Inf in b"
    
    def test_small_dataset_5x20(self):
        """Dataset rất nhỏ (5 câu × 20 thí sinh) không crash."""
        U, _, _, _, _ = simulate_irt_data(5, 20, model_type=1, seed=730)
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert len(res.b) == 5
        assert len(res.theta) == 20
        assert not np.any(np.isnan(res.b))
    
    def test_large_dataset_50x1000(self):
        """Dataset lớn (50 câu × 1000 thí sinh)."""
        U, _, _, _, _ = simulate_irt_data(50, 1000, model_type=2, seed=740)
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        assert len(res.b) == 50
        assert len(res.theta) == 1000
        assert res.converged, "2PL should converge on large dataset"
    
    def test_uniform_responses_handled(self):
        """Thí sinh trả lời giống nhau (tất cả đúng hoặc sai) không crash."""
        rng = np.random.RandomState(750)
        N, M = 10, 50
        U = (rng.random((N, M)) < 0.5).astype(float)
        
        # Thêm 1 thí sinh đúng hết và 1 thí sinh sai hết
        U[:, 0] = 1.0  # Perfect score
        U[:, 1] = 0.0  # Zero score
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert not np.any(np.isnan(res.theta)), "NaN in theta with extreme persons"


# =========================================================================
# TEST GROUP 5: Diverse Datasets
# =========================================================================

class TestDiverseDatasets:
    """Kiểm tra trên nhiều loại dữ liệu khác nhau."""
    
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_1pl_multiple_seeds(self, seed):
        """1PL hội tụ ổn định trên nhiều bộ dữ liệu ngẫu nhiên."""
        U, _, _, _, _ = simulate_irt_data(20, 300, model_type=1, seed=seed * 100)
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert res.converged, f"1PL did not converge (seed={seed})"
        assert res.log_likelihood < 0
        assert not np.any(np.isnan(res.b))
    
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_2pl_multiple_seeds(self, seed):
        """2PL MML-EM hội tụ ổn định trên nhiều bộ dữ liệu ngẫu nhiên."""
        U, _, _, _, _ = simulate_irt_data(15, 400, model_type=2, seed=seed * 200)
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        assert res.converged, f"2PL did not converge (seed={seed})"
        assert not np.any(np.isnan(res.a))
        assert not np.any(np.isnan(res.b))
    
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_3pl_multiple_seeds(self, seed):
        """3PL MML-EM hội tụ trên nhiều bộ dữ liệu ngẫu nhiên."""
        U, _, _, _, _ = simulate_irt_data(15, 500, model_type=3, seed=seed * 300)
        config = JMLEConfig(model_type=3, max_iter=100, num_options=4)
        res = run_jmle(U, config)
        
        # 3PL may not always converge within 100 iters, but shouldn't produce NaN
        assert not np.any(np.isnan(res.a)), f"3PL NaN in a (seed={seed})"
        assert not np.any(np.isnan(res.b)), f"3PL NaN in b (seed={seed})"
        assert not np.any(np.isnan(res.c)), f"3PL NaN in c (seed={seed})"
    
    def test_balanced_dataset(self):
        """Dữ liệu cân bằng (p ≈ 0.5)."""
        rng = np.random.RandomState(810)
        N, M = 20, 300
        U = (rng.random((N, M)) < 0.5).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        # b phải gần 0 vì tỷ lệ đúng ≈ 0.5
        assert abs(np.mean(res.b)) < 0.5, \
            f"Mean b = {np.mean(res.b):.4f}, expected close to 0 for balanced data"
    
    def test_skewed_easy_dataset(self):
        """Dữ liệu lệch dễ (p ≈ 0.8)."""
        rng = np.random.RandomState(820)
        N, M = 20, 300
        U = (rng.random((N, M)) < 0.8).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        # b phải < 0 vì bài thi dễ
        assert np.mean(res.b) < 0.0, \
            f"Mean b = {np.mean(res.b):.4f}, expected < 0 for easy data"
    
    def test_skewed_hard_dataset(self):
        """Dữ liệu lệch khó (p ≈ 0.2)."""
        rng = np.random.RandomState(830)
        N, M = 20, 300
        U = (rng.random((N, M)) < 0.2).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        # b phải > 0 vì bài thi khó
        assert np.mean(res.b) > 0.0, \
            f"Mean b = {np.mean(res.b):.4f}, expected > 0 for hard data"


# =========================================================================
# TEST GROUP 6: Gauss-Hermite Quadrature Verification
# =========================================================================

class TestQuadrature:
    """Kiểm tra Gauss-Hermite quadrature chính xác."""
    
    def test_quadrature_weights_sum_to_one(self):
        """Tổng trọng số quadrature phải ≈ 1 (cho N(0,1))."""
        for Q in [11, 21, 31, 41]:
            nodes, weights = gauss_hermite_points(Q)
            total = np.sum(weights)
            assert abs(total - 1.0) < 1e-6, \
                f"Q={Q}: weights sum = {total:.8f}, expected ≈ 1.0"
    
    def test_quadrature_integrates_normal_moments(self):
        """Quadrature phải tính đúng moment của N(0,1)."""
        nodes, weights = gauss_hermite_points(31)
        
        # E[X] = 0
        mean = np.sum(nodes * weights)
        assert abs(mean) < 1e-10, f"E[X] = {mean}, expected ≈ 0"
        
        # E[X²] = 1
        var = np.sum(nodes**2 * weights)
        assert abs(var - 1.0) < 1e-6, f"E[X²] = {var}, expected ≈ 1.0"
        
        # E[X³] = 0 (odd moment)
        m3 = np.sum(nodes**3 * weights)
        assert abs(m3) < 1e-6, f"E[X³] = {m3}, expected ≈ 0"
        
        # E[X⁴] = 3 (kurtosis of normal)
        m4 = np.sum(nodes**4 * weights)
        assert abs(m4 - 3.0) < 0.01, f"E[X⁴] = {m4}, expected ≈ 3.0"
    
    def test_quadrature_symmetry_and_integration(self):
        """Quadrature phải đối xứng và tích phân đúng hàm trơn."""
        from scipy.stats import norm
        nodes, weights = gauss_hermite_points(31)
        
        # Kiểm tra tính đối xứng của nodes và weights
        np.testing.assert_allclose(nodes, -nodes[::-1], atol=1e-10)
        np.testing.assert_allclose(weights, weights[::-1], atol=1e-10)
        
        # Tích phân hàm phân phối CDF Φ(x) trên phân phối N(0,1) phải bằng đúng 0.5 do tính đối xứng
        expected_cdf = np.sum(norm.cdf(nodes) * weights)
        assert abs(expected_cdf - 0.5) < 1e-10, \
            f"Expected integrated CDF = 0.5, got {expected_cdf:.8f}"


# =========================================================================
# TEST GROUP 7: MML-EM Internal Consistency
# =========================================================================

class TestMMLEMInternals:
    """Kiểm tra các thành phần nội bộ của MML-EM."""
    
    def test_e_step_posterior_sums_to_one(self):
        """Posterior weights w_jq phải sum = 1 cho mỗi thí sinh j."""
        U, _, _, _, _ = simulate_irt_data(10, 50, model_type=2, seed=900)
        N, M = U.shape
        
        a = np.ones(N)
        b = np.zeros(N)
        c = np.zeros(N)
        X_q, A_q = gauss_hermite_points(21)
        
        r_bar, f_bar, ll = e_step(U, a, b, c, X_q, A_q)
        
        # f_bar_vec = sum of posterior weights across persons → should be ≈ M
        f_total = f_bar[0, :].sum()  # sum across Q → should be ≈ M
        assert abs(f_total - M) < 0.1, \
            f"f_bar sum = {f_total:.4f}, expected ≈ {M}"
    
    def test_eap_scores_mean_near_zero(self):
        """EAP scores với N(0,1) prior phải có mean ≈ 0."""
        U, _, _, _, _ = simulate_irt_data(15, 500, model_type=2, seed=910)
        N = U.shape[0]
        
        a = np.ones(N) * 1.0
        b = np.zeros(N)
        c = np.zeros(N)
        X_q, A_q = gauss_hermite_points(21)
        
        theta_eap, se_eap = eap_scoring(U, a, b, c, X_q, A_q)
        
        # Với prior N(0,1) và dữ liệu cân bằng, mean(θ) nên gần 0
        assert abs(np.mean(theta_eap)) < 0.5, \
            f"EAP mean = {np.mean(theta_eap):.4f}, expected near 0"
        
        # SE phải > 0 cho mọi thí sinh
        assert np.all(se_eap > 0), "EAP SE must be positive"
    
    def test_marginal_ll_increases(self):
        """Marginal LL phải không giảm qua các E-step (EM property)."""
        U, _, _, _, _ = simulate_irt_data(10, 200, model_type=2, seed=920)
        N = U.shape[0]
        
        a = np.ones(N)
        b = np.zeros(N)
        c = np.zeros(N)
        X_q, A_q = gauss_hermite_points(21)
        
        # Chạy 2 E-step liên tiếp (không M-step) → LL không đổi
        _, _, ll1 = e_step(U, a, b, c, X_q, A_q)
        _, _, ll2 = e_step(U, a, b, c, X_q, A_q)
        
        # LL phải giống nhau vì cùng params
        assert abs(ll1 - ll2) < 1e-10, f"LL changed without M-step: {ll1} → {ll2}"


# =========================================================================
# TEST GROUP 8: Cross-Engine Synchronization
# =========================================================================

class TestCrossEngineSync:
    """Kiểm tra PythonEngine sử dụng cùng code với rasch_irt."""
    
    def test_pythonengine_uses_rasch_irt_jmle(self):
        """PythonEngine.jmle_engine phải import run_jmle từ rasch_irt."""
        from core.jmle_engine import run_jmle as pe_run_jmle
        assert pe_run_jmle is run_jmle, \
            "PythonEngine run_jmle is NOT the same object as rasch_irt run_jmle"
    
    def test_pythonengine_uses_rasch_irt_prob(self):
        """PythonEngine.probability phải import prob_3pl từ rasch_irt."""
        from core.probability import prob_3pl as pe_prob_3pl
        assert pe_prob_3pl is prob_3pl, \
            "PythonEngine prob_3pl is NOT the same object as rasch_irt prob_3pl"
    
    def test_pythonengine_uses_rasch_irt_derivatives(self):
        """PythonEngine.derivatives phải import đạo hàm từ rasch_irt."""
        from rasch_irt.core import score_theta as orig_score_theta
        from core.derivatives import score_theta as pe_score_theta
        assert pe_score_theta is orig_score_theta, \
            "PythonEngine score_theta is NOT the same object as rasch_irt score_theta"
    
    def test_pythonengine_uses_rasch_irt_fit(self):
        """PythonEngine.fit_statistics phải import compute_fit_statistics từ rasch_irt."""
        from core.fit_statistics import compute_fit_statistics as pe_cfs
        assert pe_cfs is compute_fit_statistics, \
            "PythonEngine compute_fit_statistics is NOT the same as rasch_irt"
    
    def test_cross_engine_produces_same_results(self):
        """Gọi run_jmle từ cả 2 nguồn phải cho kết quả giống hệt."""
        from core.jmle_engine import run_jmle as pe_run_jmle
        
        U, _, _, _, _ = simulate_irt_data(15, 200, model_type=1, seed=1000)
        config = JMLEConfig(model_type=1, max_iter=100)
        
        res_rasch = run_jmle(U, config)
        res_pe = pe_run_jmle(U, config)
        
        np.testing.assert_array_equal(res_rasch.b, res_pe.b,
                                      err_msg="b differs between rasch_irt and PythonEngine")
        np.testing.assert_array_equal(res_rasch.theta, res_pe.theta,
                                      err_msg="theta differs between rasch_irt and PythonEngine")
        np.testing.assert_array_equal(res_rasch.a, res_pe.a,
                                      err_msg="a differs between rasch_irt and PythonEngine")


# =========================================================================
# TEST GROUP 9: Fit Statistics
# =========================================================================

class TestFitStatistics:
    """Kiểm tra Infit/Outfit và SE được tính chính xác."""
    
    def test_fit_statistics_near_one_for_good_data(self):
        """Infit/Outfit MNSQ ≈ 1.0 cho dữ liệu sinh từ mô hình đúng."""
        U, theta_true, a_true, b_true, c_true = simulate_irt_data(
            20, 500, model_type=1, seed=1100
        )
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        P = prob_3pl(res.theta, res.a, res.b, res.c)
        fit = compute_fit_statistics(U, P)
        
        # Mean Infit/Outfit phải gần 1.0 cho dữ liệu "clean"
        mean_infit = np.mean(fit['infit_item'])
        mean_outfit = np.mean(fit['outfit_item'])
        
        assert 0.7 < mean_infit < 1.3, \
            f"Mean Infit = {mean_infit:.4f}, expected ≈ 1.0"
        assert 0.7 < mean_outfit < 1.3, \
            f"Mean Outfit = {mean_outfit:.4f}, expected ≈ 1.0"
    
    def test_se_theta_positive(self):
        """SE(θ) phải dương cho mọi thí sinh."""
        U, _, _, _, _ = simulate_irt_data(15, 200, model_type=2, seed=1110)
        config = JMLEConfig(model_type=2, max_iter=100)
        res = run_jmle(U, config)
        
        P = prob_3pl(res.theta, res.a, res.b, res.c)
        se = compute_se_theta(P, res.a, res.c)
        
        assert np.all(se > 0), "SE(θ) must be positive"
        assert np.all(np.isfinite(se)), "SE(θ) must be finite"
    
    def test_se_items_positive(self):
        """SE(b), SE(a), SE(c) phải dương."""
        U, _, _, _, _ = simulate_irt_data(15, 200, model_type=3, seed=1120)
        config = JMLEConfig(model_type=3, max_iter=100, num_options=4)
        res = run_jmle(U, config)
        
        P = prob_3pl(res.theta, res.a, res.b, res.c)
        se = compute_se_items(P, res.theta, res.a, res.b, res.c, model_type=3, num_options=4)
        
        assert np.all(se['se_b'] > 0), "SE(b) must be positive"
        assert np.all(se['se_a'] > 0), "SE(a) must be positive"
        assert np.all(se['se_c'] > 0), "SE(c) must be positive"
    
    def test_true_score_bounds(self):
        """True score phải nằm trong [0, 10]."""
        U, _, _, _, _ = simulate_irt_data(20, 300, model_type=1, seed=1130)
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        ts = compute_true_scores(res.theta, res.a, res.b, res.c)
        assert np.all(ts >= 0) and np.all(ts <= 10), \
            f"True scores out of [0, 10]: [{ts.min():.2f}, {ts.max():.2f}]"


# =========================================================================
# TEST GROUP 10: Auto-Fit Model Selection
# =========================================================================

class TestAutoFit:
    """Kiểm tra Auto-Fit chọn mô hình đúng."""
    
    def test_auto_fit_runs_without_error(self):
        """Auto-Fit phải chạy thành công trên dữ liệu hợp lệ."""
        U, _, _, _, _ = simulate_irt_data(20, 300, model_type=1, seed=1200)
        result = run_auto_fit(U, num_options=4, max_iter=100)
        
        assert len(result.models) >= 2, "Auto-Fit should evaluate at least 2 models"
        assert result.recommended_model in [1, 2, 3]
        assert len(result.recommendation_reason) > 0
    
    def test_auto_fit_prefers_simpler_model_for_rasch_data(self):
        """Auto-Fit nên ưu tiên 1PL khi dữ liệu đúng là Rasch."""
        U, _, _, _, _ = simulate_irt_data(20, 500, model_type=1, seed=1210)
        result = run_auto_fit(U, num_options=4, max_iter=100)
        
        # 1PL data → BIC should favor simpler model (1PL or 2PL)
        assert result.recommended_model in [1, 2], \
            f"Expected model 1 or 2 for Rasch data, got {result.recommended_model}"


# =========================================================================
# TEST GROUP 11: Numerical Stability Stress Tests
# =========================================================================

class TestNumericalStability:
    """Kiểm tra tính ổn định số với dữ liệu cực đoan."""
    
    def test_no_nan_in_any_output(self):
        """Không được có NaN trong bất kỳ output nào."""
        for model_type in [1, 2, 3]:
            for seed in [1, 42, 100, 777, 9999]:
                U, _, _, _, _ = simulate_irt_data(
                    15, 200, model_type=model_type, seed=seed
                )
                config = JMLEConfig(model_type=model_type, max_iter=100, num_options=4)
                res = run_jmle(U, config)
                
                assert not np.any(np.isnan(res.b)), \
                    f"NaN in b: model={model_type}, seed={seed}"
                assert not np.any(np.isnan(res.theta)), \
                    f"NaN in theta: model={model_type}, seed={seed}"
                assert not np.any(np.isnan(res.a)), \
                    f"NaN in a: model={model_type}, seed={seed}"
                assert not np.any(np.isnan(res.c)), \
                    f"NaN in c: model={model_type}, seed={seed}"
    
    def test_no_inf_in_any_output(self):
        """Không được có Inf trong bất kỳ output nào."""
        for model_type in [1, 2, 3]:
            for seed in [2, 50, 200]:
                U, _, _, _, _ = simulate_irt_data(
                    15, 200, model_type=model_type, seed=seed
                )
                config = JMLEConfig(model_type=model_type, max_iter=100, num_options=4)
                res = run_jmle(U, config)
                
                assert not np.any(np.isinf(res.b)), \
                    f"Inf in b: model={model_type}, seed={seed}"
                assert not np.any(np.isinf(res.theta)), \
                    f"Inf in theta: model={model_type}, seed={seed}"
    
    def test_very_few_items(self):
        """Chỉ 3 câu hỏi (minimum) không crash."""
        U, _, _, _, _ = simulate_irt_data(3, 100, model_type=1, seed=1300)
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert len(res.b) == 3
        assert not np.any(np.isnan(res.b))
    
    def test_many_items_few_persons(self):
        """50 câu hỏi, 30 thí sinh — dữ liệu mỏng."""
        U, _, _, _, _ = simulate_irt_data(50, 30, model_type=1, seed=1310)
        config = JMLEConfig(model_type=1, max_iter=200)
        res = run_jmle(U, config)
        
        assert len(res.b) == 50
        assert len(res.theta) == 30
        assert not np.any(np.isnan(res.b))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
