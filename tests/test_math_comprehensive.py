"""
Bộ Kiểm tra Toán học Toàn diện cho thư viện rasch-irt
=====================================================
Kiểm tra từng công thức toán học bằng:
1. Finite difference so sánh với analytical gradient
2. Cross-check với công thức tham chiếu
3. Edge cases & numerical stability
4. Parameter recovery trên synthetic data
"""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
from scipy.stats import norm

# Import from rasch_irt
from rasch_irt.core import (
    D, prob_3pl,
    score_theta, info_theta,
    score_b, info_b,
    score_a, info_a,
    get_beta_params, score_c_map, info_c_map,
    newton_raphson_batch,
)
from rasch_irt.ctt import (
    compute_difficulty,
    compute_discrimination_d,
    compute_point_biserial,
    compute_biserial,
    compute_reliability,
    sanity_check,
    run_ctt,
)
from rasch_irt.irt import (
    compute_fit_statistics,
    compute_log_likelihood,
    _count_params,
    _compute_aic_bic,
    compute_iif_points,
    compute_se_theta,
    compute_se_items,
    compute_true_scores,
    run_jmle,
    JMLEConfig,
)


class TestProb3PL(unittest.TestCase):
    """Phase 2A: Kiểm tra hàm xác suất 3PL"""
    
    def test_manual_single_point(self):
        """Tính tay P cho 1 câu, 1 thí sinh và đối chiếu"""
        theta = np.array([0.0])
        a = np.array([1.0])
        b = np.array([0.0])
        c = np.array([0.0])
        
        # P = 0 + (1-0)/(1 + exp(-1.702*1*(0-0))) = 1/(1+1) = 0.5
        P = prob_3pl(theta, a, b, c)
        self.assertAlmostEqual(P[0, 0], 0.5, places=5,
            msg="Khi theta=b và c=0, P phải bằng 0.5")
    
    def test_3pl_with_guessing(self):
        """Khi theta=b với c>0, P = (1+c)/2"""
        theta = np.array([1.0])
        a = np.array([1.5])
        b = np.array([1.0])
        c = np.array([0.25])
        
        # P = 0.25 + 0.75/(1+exp(0)) = 0.25 + 0.375 = 0.625
        P = prob_3pl(theta, a, b, c)
        expected = 0.25 + 0.75 / 2
        self.assertAlmostEqual(P[0, 0], expected, places=5,
            msg="Khi theta=b, P = (1+c)/2")
    
    def test_1pl_rasch_degenerate(self):
        """1PL (a=1, c=0) phải degenerates thành Rasch model"""
        theta = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        a = np.ones(3)
        b = np.array([-1.0, 0.0, 1.0])
        c = np.zeros(3)
        
        P = prob_3pl(theta, a, b, c)
        
        # Rasch: P = 1/(1 + exp(-D*(theta-b)))
        for i in range(3):
            for j in range(5):
                expected = 1.0 / (1.0 + np.exp(-D * (theta[j] - b[i])))
                self.assertAlmostEqual(P[i, j], expected, places=8)
    
    def test_output_shape(self):
        """Output phải là [N × M]"""
        theta = np.random.randn(100)
        a = np.ones(20)
        b = np.random.randn(20)
        c = np.zeros(20)
        P = prob_3pl(theta, a, b, c)
        self.assertEqual(P.shape, (20, 100))
    
    def test_numerical_stability_extreme_theta(self):
        """Không có NaN/Inf khi theta cực đại"""
        theta = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        a = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        b = np.zeros(5)
        c = np.zeros(5)
        
        P = prob_3pl(theta, a, b, c)
        self.assertFalse(np.any(np.isnan(P)), "Không được có NaN")
        self.assertFalse(np.any(np.isinf(P)), "Không được có Inf")
        self.assertTrue(np.all(P > 0), "P phải > 0")
        self.assertTrue(np.all(P < 1), "P phải < 1")
    
    def test_monotonicity(self):
        """P phải tăng theo theta (tính đơn điệu)"""
        theta = np.linspace(-4, 4, 100)
        a = np.array([1.0])
        b = np.array([0.0])
        c = np.array([0.0])
        
        P = prob_3pl(theta, a, b, c)[0]
        diffs = np.diff(P)
        self.assertTrue(np.all(diffs > 0), 
            "P(theta) phải đơn điệu tăng")


class TestFiniteDifferenceGradients(unittest.TestCase):
    """Phase 2B-2E: Kiểm tra gradient bằng finite difference"""
    
    def setUp(self):
        np.random.seed(42)
        self.N, self.M = 10, 50
        self.theta = np.random.randn(self.M)
        self.a = np.random.uniform(0.5, 2.0, self.N)
        self.b = np.random.randn(self.N) * 0.5
        self.c = np.random.uniform(0.05, 0.25, self.N)
        
        # Sinh dữ liệu nhị phân
        P_true = prob_3pl(self.theta, self.a, self.b, self.c)
        self.U = (np.random.rand(self.N, self.M) < P_true).astype(float)
        
        self.h = 1e-5  # Step size cho finite difference
    
    def _log_likelihood(self, U, theta, a, b, c):
        P = prob_3pl(theta, a, b, c)
        return compute_log_likelihood(U, P)
    
    def test_score_theta_finite_diff(self):
        """score_theta phải khớp finite difference ∂ℓ/∂θ"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        analytical = score_theta(self.U, P, self.a, self.c)
        
        numerical = np.zeros(self.M)
        for j in range(self.M):
            theta_plus = self.theta.copy()
            theta_minus = self.theta.copy()
            theta_plus[j] += self.h
            theta_minus[j] -= self.h
            
            ll_plus = self._log_likelihood(self.U, theta_plus, self.a, self.b, self.c)
            ll_minus = self._log_likelihood(self.U, theta_minus, self.a, self.b, self.c)
            numerical[j] = (ll_plus - ll_minus) / (2 * self.h)
        
        assert_allclose(analytical, numerical, atol=1e-4,
            err_msg="score_theta không khớp finite difference")
    
    def test_score_b_finite_diff(self):
        """score_b phải khớp finite difference ∂ℓ/∂b"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        analytical = score_b(self.U, P, self.a, self.c)
        
        numerical = np.zeros(self.N)
        for i in range(self.N):
            b_plus = self.b.copy()
            b_minus = self.b.copy()
            b_plus[i] += self.h
            b_minus[i] -= self.h
            
            ll_plus = self._log_likelihood(self.U, self.theta, self.a, b_plus, self.c)
            ll_minus = self._log_likelihood(self.U, self.theta, self.a, b_minus, self.c)
            numerical[i] = (ll_plus - ll_minus) / (2 * self.h)
        
        assert_allclose(analytical, numerical, atol=1e-4,
            err_msg="score_b không khớp finite difference")
    
    def test_score_a_finite_diff(self):
        """score_a phải khớp finite difference ∂ℓ/∂a"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        analytical = score_a(self.U, P, self.theta, self.a, self.b, self.c)
        
        numerical = np.zeros(self.N)
        for i in range(self.N):
            a_plus = self.a.copy()
            a_minus = self.a.copy()
            a_plus[i] += self.h
            a_minus[i] -= self.h
            
            ll_plus = self._log_likelihood(self.U, self.theta, a_plus, self.b, self.c)
            ll_minus = self._log_likelihood(self.U, self.theta, a_minus, self.b, self.c)
            numerical[i] = (ll_plus - ll_minus) / (2 * self.h)
        
        assert_allclose(analytical, numerical, atol=1e-4,
            err_msg="score_a không khớp finite difference")
    
    def test_score_c_likelihood_finite_diff(self):
        """Phần likelihood của score_c phải khớp finite difference ∂ℓ/∂c"""
        # score_c_map = likelihood_part + prior_part
        # Kiểm tra riêng likelihood part
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        
        numerical = np.zeros(self.N)
        for i in range(self.N):
            c_plus = self.c.copy()
            c_minus = self.c.copy()
            c_plus[i] += self.h
            c_minus[i] -= self.h
            
            ll_plus = self._log_likelihood(self.U, self.theta, self.a, self.b, c_plus)
            ll_minus = self._log_likelihood(self.U, self.theta, self.a, self.b, c_minus)
            numerical[i] = (ll_plus - ll_minus) / (2 * self.h)
        
        # score_c_map với flat prior (alpha=1, beta=1) = chỉ likelihood part
        analytical = score_c_map(self.U, P, self.c, alpha_prior=1.0, beta_prior=1.0)
        
        assert_allclose(analytical, numerical, atol=1e-3,
            err_msg="score_c (likelihood part) không khớp finite difference")
    
    def test_score_c_map_with_prior_finite_diff(self):
        """score_c_map toàn phần (likelihood + beta prior) khớp finite difference"""
        alpha_p, beta_p = get_beta_params(4)  # alpha=5, beta=15
        
        def map_objective(c):
            """log-likelihood + log-beta-prior"""
            from scipy.stats import beta as beta_dist
            P = prob_3pl(self.theta, self.a, self.b, c)
            ll = compute_log_likelihood(self.U, P)
            # Log Beta prior
            safe_c = np.clip(c, 1e-10, 1 - 1e-10)
            log_prior = np.sum((alpha_p - 1) * np.log(safe_c) + (beta_p - 1) * np.log(1 - safe_c))
            return ll + log_prior
        
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        analytical = score_c_map(self.U, P, self.c, alpha_p, beta_p)
        
        numerical = np.zeros(self.N)
        for i in range(self.N):
            c_plus = self.c.copy()
            c_minus = self.c.copy()
            c_plus[i] += self.h
            c_minus[i] -= self.h
            numerical[i] = (map_objective(c_plus) - map_objective(c_minus)) / (2 * self.h)
        
        assert_allclose(analytical, numerical, atol=1e-3,
            err_msg="score_c_map (full MAP) không khớp finite difference")
    
    def test_info_theta_positive(self):
        """Fisher Information phải > 0 cho mọi thí sinh"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        I = info_theta(P, self.a, self.c)
        self.assertTrue(np.all(I > 0), "Fisher Information theta phải > 0")
    
    def test_info_b_positive(self):
        """Fisher Information cho b phải > 0"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        I = info_b(P, self.a, self.c)
        self.assertTrue(np.all(I > 0), "Fisher Information b phải > 0")
    
    def test_info_a_positive(self):
        """Fisher Information cho a phải > 0"""
        P = prob_3pl(self.theta, self.a, self.b, self.c)
        I = info_a(P, self.theta, self.a, self.b, self.c)
        self.assertTrue(np.all(I > 0), "Fisher Information a phải > 0")


class TestBetaPrior(unittest.TestCase):
    """Phase 2E: Kiểm tra Beta Prior cho c"""
    
    def test_expected_values(self):
        """E[c] = alpha / (alpha + beta) phải khớp 1/num_options"""
        for num_opt, expected_ec in [(2, 0.5), (3, 0.33), (4, 0.25), (5, 0.20)]:
            alpha, beta = get_beta_params(num_opt)
            ec = alpha / (alpha + beta)
            self.assertAlmostEqual(ec, expected_ec, places=2,
                msg=f"E[c] cho {num_opt} phương án phải ≈ {expected_ec}")
    
    def test_prior_gradient_sign_stabilizes_c(self):
        """Prior gradient phải kéo c về phía E[c]"""
        alpha_p, beta_p = get_beta_params(4)  # E[c] = 0.25
        
        # c rất nhỏ (< E[c]) → prior gradient phải > 0 (kéo lên)
        c_low = np.array([0.05])
        safe_c = max(c_low[0], 1e-5)
        safe_1c = max(1 - c_low[0], 1e-5)
        prior_grad_low = (alpha_p - 1) / safe_c - (beta_p - 1) / safe_1c
        self.assertGreater(prior_grad_low, 0,
            "Prior gradient khi c < E[c] phải > 0")
        
        # c rất lớn (> E[c]) → prior gradient phải < 0 (kéo xuống)
        c_high = np.array([0.35])
        safe_c = max(c_high[0], 1e-5)
        safe_1c = max(1 - c_high[0], 1e-5)
        prior_grad_high = (alpha_p - 1) / safe_c - (beta_p - 1) / safe_1c
        self.assertLess(prior_grad_high, 0,
            "Prior gradient khi c > E[c] phải < 0")


class TestNewtonRaphson(unittest.TestCase):
    """Phase 2F: Kiểm tra Newton-Raphson"""
    
    def test_update_direction(self):
        """NR phải cập nhật theo hướng TĂNG likelihood (maximize)"""
        # score > 0 và info > 0 → delta > 0 → theta tăng
        scores = np.array([1.0, -1.0])
        infos = np.array([2.0, 2.0])
        current = np.array([0.0, 0.0])
        
        new_vals = newton_raphson_batch(scores, infos, current)
        self.assertGreater(new_vals[0], current[0],
            "Khi score > 0, theta phải tăng")
        self.assertLess(new_vals[1], current[1],
            "Khi score < 0, theta phải giảm")
    
    def test_damping_clip(self):
        """Delta phải bị clip trong [-3, 3]"""
        scores = np.array([1000.0])
        infos = np.array([1.0])
        current = np.array([0.0])
        
        new_vals = newton_raphson_batch(scores, infos, current)
        delta = new_vals[0] - current[0]
        self.assertLessEqual(abs(delta), 3.0 + 1e-10,
            "Delta phải bị clip trong [-3, 3]")
    
    def test_safe_division_zero_info(self):
        """Khi info ≈ 0, không được có NaN/Inf"""
        scores = np.array([1.0])
        infos = np.array([1e-20])
        current = np.array([0.0])
        
        new_vals = newton_raphson_batch(scores, infos, current)
        self.assertFalse(np.isnan(new_vals[0]))
        self.assertFalse(np.isinf(new_vals[0]))
    
    def test_clamping_bounds(self):
        """Output phải nằm trong [val_min, val_max]"""
        scores = np.array([100.0])
        infos = np.array([1.0])
        current = np.array([9.5])
        
        new_vals = newton_raphson_batch(scores, infos, current, val_min=-5.0, val_max=5.0)
        self.assertLessEqual(new_vals[0], 5.0)


class TestCTTFormulas(unittest.TestCase):
    """Phase 1: Kiểm tra công thức CTT"""
    
    def setUp(self):
        np.random.seed(123)
        self.N, self.M = 15, 200
        # Tạo dữ liệu có cấu trúc
        theta = np.random.randn(self.M)
        b = np.random.randn(self.N) * 0.8
        P_true = 1.0 / (1.0 + np.exp(-(theta[np.newaxis, :] - b[:, np.newaxis])))
        self.U = (np.random.rand(self.N, self.M) < P_true).astype(float)
    
    def test_difficulty_formula(self):
        """p = mean of each row"""
        p = compute_difficulty(self.U)
        for i in range(self.N):
            expected = self.U[i].sum() / self.M
            self.assertAlmostEqual(p[i], expected, places=10)
    
    def test_difficulty_range(self):
        """p phải nằm trong [0, 1]"""
        p = compute_difficulty(self.U)
        self.assertTrue(np.all(p >= 0) and np.all(p <= 1))
    
    def test_discrimination_27pct_groups(self):
        """D sử dụng đúng 27% nhóm cao/thấp"""
        D_vals = compute_discrimination_d(self.U)
        
        # Tính manual
        total_scores = self.U.sum(axis=0)
        sorted_idx = np.argsort(total_scores)
        n_group = max(int(self.M * 0.27), 1)
        lower = sorted_idx[:n_group]
        upper = sorted_idx[-n_group:]
        
        for i in range(self.N):
            p_upper = self.U[i, upper].mean()
            p_lower = self.U[i, lower].mean()
            expected_d = p_upper - p_lower
            self.assertAlmostEqual(D_vals[i], expected_d, places=10)
    
    def test_point_biserial_corrected(self):
        """rpb hiệu chỉnh loại bỏ câu i khỏi tổng điểm"""
        rpb = compute_point_biserial(self.U)
        
        total_scores = self.U.sum(axis=0).astype(float)
        
        # Tính manual cho câu 0
        i = 0
        corrected_scores = total_scores - self.U[i].astype(float)
        correct_mask = self.U[i] == 1
        incorrect_mask = self.U[i] == 0
        
        n1 = correct_mask.sum()
        n0 = incorrect_mask.sum()
        
        if n1 > 0 and n0 > 0:
            corrected_std = corrected_scores.std()
            mean_correct = corrected_scores[correct_mask].mean()
            mean_incorrect = corrected_scores[incorrect_mask].mean()
            p = n1 / (n1 + n0)
            q = 1 - p
            
            expected_rpb = (mean_correct - mean_incorrect) / corrected_std * np.sqrt(p * q)
            self.assertAlmostEqual(rpb[i], expected_rpb, places=10,
                msg="rpb corrected phải loại bỏ câu i")
    
    def test_biserial_formula(self):
        """Kiểm tra biserial = rpb * sqrt(pq) / phi(z_p)"""
        bis = compute_biserial(self.U)
        rpb = compute_point_biserial(self.U)
        p = self.U.mean(axis=1)
        p = np.clip(p, 0.01, 0.99)
        
        z = norm.ppf(p)
        ordinate = norm.pdf(z)
        
        for i in range(self.N):
            if ordinate[i] > 1e-10:
                expected = rpb[i] * np.sqrt(p[i] * (1 - p[i])) / ordinate[i]
                expected = np.clip(expected, -1.0, 1.0)
                self.assertAlmostEqual(bis[i], expected, places=8,
                    msg=f"Biserial formula cho câu {i} không khớp")
    
    def test_kr20_equals_alpha_binary(self):
        """Cho dữ liệu nhị phân, KR-20 = Cronbach Alpha"""
        reliability = compute_reliability(self.U)
        # Với dữ liệu nhị phân, cả hai công thức phải cho cùng kết quả
        self.assertAlmostEqual(reliability['kr20'], reliability['cronbach_alpha'], places=10,
            msg="KR-20 phải bằng Cronbach Alpha cho dữ liệu nhị phân")
    
    def test_kr20_manual(self):
        """KR-20 = (N/(N-1)) * (1 - sum(pq)/var_total)"""
        reliability = compute_reliability(self.U)
        
        total_scores = self.U.sum(axis=0).astype(float)
        var_total = total_scores.var(ddof=0)
        p = self.U.mean(axis=1)
        sum_pq = np.sum(p * (1 - p))
        
        expected_kr20 = (self.N / (self.N - 1)) * (1 - sum_pq / var_total)
        self.assertAlmostEqual(reliability['kr20'], expected_kr20, places=4)
    
    def test_sem_formula(self):
        """SEM = SD * sqrt(1 - KR20)"""
        reliability = compute_reliability(self.U)
        
        total_scores = self.U.sum(axis=0).astype(float)
        sd = total_scores.std(ddof=0)
        kr20 = reliability['kr20']
        
        expected_sem = sd * np.sqrt(1 - max(kr20, 0))
        self.assertAlmostEqual(reliability['sem'], round(expected_sem, 4), places=3)
    
    def test_sanity_check_all_correct(self):
        """Sanity check phải loại câu p=1"""
        U = np.ones((3, 10))
        U[1] = np.random.binomial(1, 0.5, 10)
        _, removed, reasons = sanity_check(U)
        
        self.assertIn(0, removed, "Câu toàn đúng phải bị loại")
        self.assertIn(2, removed, "Câu toàn đúng phải bị loại")
    
    def test_sanity_check_all_wrong(self):
        """Sanity check phải loại câu p=0"""
        U = np.zeros((3, 10))
        U[1] = np.random.binomial(1, 0.5, 10)
        _, removed, reasons = sanity_check(U)
        
        self.assertIn(0, removed)
        self.assertIn(2, removed)


class TestFitStatistics(unittest.TestCase):
    """Phase 3C: Kiểm tra Infit/Outfit MNSQ"""
    
    def setUp(self):
        np.random.seed(42)
        self.N, self.M = 10, 100
        theta = np.random.randn(self.M)
        a = np.ones(self.N)
        b = np.random.randn(self.N) * 0.5
        c = np.zeros(self.N)
        self.P = prob_3pl(theta, a, b, c)
        self.U = (np.random.rand(self.N, self.M) < self.P).astype(float)
    
    def test_fit_stats_manual(self):
        """Infit/Outfit phải khớp tính tay theo Wright & Masters"""
        fit = compute_fit_statistics(self.U, self.P)
        
        # Tính manual
        Q = 1.0 - self.P
        W = self.P * Q
        residual_sq = (self.U - self.P) ** 2
        standardized_sq = residual_sq / np.maximum(W, 1e-10)
        
        # Item Outfit = mean of standardized_sq across persons
        expected_outfit = standardized_sq.mean(axis=1)
        assert_allclose(fit['outfit_item'], expected_outfit, atol=1e-10)
        
        # Item Infit = sum(residual_sq) / sum(variance)
        expected_infit = residual_sq.sum(axis=1) / np.maximum(W.sum(axis=1), 1e-10)
        assert_allclose(fit['infit_item'], expected_infit, atol=1e-10)
    
    def test_perfect_fit_mnsq_near_one(self):
        """Khi data match model perfectly, MNSQ ≈ 1"""
        # Tạo data sinh từ model
        np.random.seed(99)
        N, M = 20, 500
        theta = np.random.randn(M)
        a = np.ones(N)
        b = np.random.randn(N)
        c = np.zeros(N)
        P = prob_3pl(theta, a, b, c)
        U = (np.random.rand(N, M) < P).astype(float)
        
        fit = compute_fit_statistics(U, P)
        
        # Mean infit/outfit should be close to 1 for well-fitting data
        mean_infit = fit['infit_item'].mean()
        mean_outfit = fit['outfit_item'].mean()
        
        self.assertAlmostEqual(mean_infit, 1.0, delta=0.15,
            msg=f"Mean Infit = {mean_infit}, kỳ vọng ≈ 1.0")
        self.assertAlmostEqual(mean_outfit, 1.0, delta=0.15,
            msg=f"Mean Outfit = {mean_outfit}, kỳ vọng ≈ 1.0")
    
    def test_person_fit_keys_exist(self):
        """Phải có cả item fit và person fit"""
        fit = compute_fit_statistics(self.U, self.P)
        self.assertIn('infit_item', fit)
        self.assertIn('outfit_item', fit)
        self.assertIn('infit_person', fit)
        self.assertIn('outfit_person', fit)
        self.assertEqual(len(fit['infit_item']), self.N)
        self.assertEqual(len(fit['infit_person']), self.M)


class TestDegreesOfFreedom(unittest.TestCase):
    """Phase 3D: Kiểm tra _count_params"""
    
    def test_1pl_params(self):
        """1PL: k = N + M - 1 (N difficulties + M abilities - 1 constraint)"""
        N, M = 20, 100
        k = _count_params(N, M, 1)
        # N b-params + M theta-params - 1 location constraint
        self.assertEqual(k, N + M - 1)
    
    def test_2pl_params(self):
        """2PL: k = 2N + M - 2 (N*a + N*b + M*theta - 2 constraints)"""
        N, M = 20, 100
        k = _count_params(N, M, 2)
        # N a-params + N b-params + M theta - 2 (mean=0, sd=1)
        self.assertEqual(k, 2 * N + M - 2)
    
    def test_3pl_params(self):
        """3PL: k = 3N + M - 2"""
        N, M = 20, 100
        k = _count_params(N, M, 3)
        self.assertEqual(k, 3 * N + M - 2)


class TestAICBIC(unittest.TestCase):
    """Phase 3E: Kiểm tra AIC/BIC"""
    
    def test_aic_formula(self):
        """AIC = -2*ll + 2*k"""
        ll = -500.0
        k = 30
        aic, _ = _compute_aic_bic(ll, k, 100)
        self.assertAlmostEqual(aic, -2 * ll + 2 * k)
    
    def test_bic_formula(self):
        """BIC = -2*ll + k*ln(M)"""
        ll = -500.0
        k = 30
        M = 100
        _, bic = _compute_aic_bic(ll, k, M)
        self.assertAlmostEqual(bic, -2 * ll + k * np.log(M))
    
    def test_bic_uses_persons_not_observations(self):
        """BIC phải dùng M (thí sinh), không phải N*M (tổng ô)"""
        ll = -500.0
        k = 30
        M = 100
        _, bic = _compute_aic_bic(ll, k, M)
        
        # Nếu dùng N*M thay vì M, BIC sẽ lớn hơn nhiều
        bic_wrong = -2 * ll + k * np.log(20 * M)
        self.assertLess(bic, bic_wrong,
            msg="BIC phải dùng M, không phải N*M")


class TestIIFFormula(unittest.TestCase):
    """Phase 3F: Kiểm tra Item Information Function"""
    
    def test_iif_1pl(self):
        """Cho 1PL (c=0): I(theta) = D^2 * a^2 * P * Q"""
        a, b, c = 1.0, 0.0, 0.0
        points = compute_iif_points(a, b, c)
        
        for pt in points:
            theta = pt['theta']
            info = pt['info']
            
            # Manual calculation
            exponent = -D * a * (theta - b)
            P = 1.0 / (1.0 + np.exp(np.clip(exponent, -30, 30)))
            Q = 1.0 - P
            expected_info = D**2 * a**2 * P * Q
            
            self.assertAlmostEqual(info, expected_info, places=4,
                msg=f"IIF tại theta={theta} không khớp cho 1PL")
    
    def test_iif_3pl(self):
        """Cho 3PL: I = D^2 * a^2 * (P-c)^2 * Q / ((1-c)^2 * P)"""
        a, b, c = 1.5, -0.5, 0.2
        points = compute_iif_points(a, b, c)
        
        for pt in points:
            theta = pt['theta']
            info = pt['info']
            
            # Manual
            exponent = -D * a * (theta - b)
            P = c + (1 - c) / (1 + np.exp(np.clip(exponent, -30, 30)))
            P = np.clip(P, 1e-10, 1 - 1e-10)
            Pstar = (P - c) / (1 - c)
            Q = 1.0 - P
            expected_info = D**2 * a**2 * Pstar**2 * Q / max(P, 1e-15)
            
            self.assertAlmostEqual(info, expected_info, delta=max(abs(expected_info) * 1e-4, 1e-8),
                msg=f"IIF 3PL tại theta={theta}")


class TestSEComputation(unittest.TestCase):
    """Phase 3G: Kiểm tra Standard Error"""
    
    def test_se_theta_inverse_sqrt_info(self):
        """SE(theta) = 1/sqrt(I(theta))"""
        np.random.seed(42)
        theta = np.random.randn(50)
        a = np.random.uniform(0.5, 2, 10)
        b = np.random.randn(10)
        c = np.zeros(10)
        P = prob_3pl(theta, a, b, c)
        
        se = compute_se_theta(P, a, c)
        I = info_theta(P, a, c)
        expected_se = 1.0 / np.sqrt(np.maximum(I, 1e-10))
        
        assert_allclose(se, expected_se, atol=1e-10)
    
    def test_se_decreases_with_more_items(self):
        """SE(theta) phải giảm khi có nhiều câu hỏi hơn"""
        np.random.seed(42)
        theta = np.array([0.0])
        
        # 5 items
        a5 = np.ones(5)
        b5 = np.zeros(5)
        c5 = np.zeros(5)
        P5 = prob_3pl(theta, a5, b5, c5)
        se5 = compute_se_theta(P5, a5, c5)[0]
        
        # 20 items
        a20 = np.ones(20)
        b20 = np.zeros(20)
        c20 = np.zeros(20)
        P20 = prob_3pl(theta, a20, b20, c20)
        se20 = compute_se_theta(P20, a20, c20)[0]
        
        self.assertLess(se20, se5,
            msg="SE phải nhỏ hơn khi có nhiều câu hỏi hơn")


class TestTrueScore(unittest.TestCase):
    """Phase 3H: Kiểm tra True Score mapping"""
    
    def test_true_score_range(self):
        """True score phải nằm trong [0, 10]"""
        theta = np.linspace(-4, 4, 100)
        a = np.ones(20)
        b = np.random.randn(20)
        c = np.zeros(20)
        
        ts = compute_true_scores(theta, a, b, c)
        self.assertTrue(np.all(ts >= 0))
        self.assertTrue(np.all(ts <= 10))
    
    def test_true_score_monotone(self):
        """True score phải tăng theo theta"""
        theta = np.linspace(-4, 4, 100)
        a = np.ones(10)
        b = np.zeros(10)
        c = np.zeros(10)
        
        ts = compute_true_scores(theta, a, b, c)
        diffs = np.diff(ts)
        self.assertTrue(np.all(diffs >= 0),
            msg="True score phải đơn điệu tăng theo theta")


class TestJMLEParameterRecovery(unittest.TestCase):
    """Phase 3A: Kiểm tra JMLE convergence & parameter recovery"""
    
    def test_1pl_convergence(self):
        """1PL phải hội tụ trên data vừa"""
        np.random.seed(42)
        N, M = 20, 300
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, np.ones(N), b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        self.assertTrue(result.converged, "1PL phải hội tụ")
    
    def test_1pl_parameter_recovery_b(self):
        """1PL b recovered phải tương quan cao với b_true"""
        np.random.seed(42)
        N, M = 20, 500
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, np.ones(N), b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        # Chuẩn hóa b_true về cùng scale
        b_true_std = (b_true - b_true.mean()) / max(b_true.std(), 1e-5)
        b_est_std = (result.b - result.b.mean()) / max(result.b.std(), 1e-5)
        
        r = np.corrcoef(b_true_std, b_est_std)[0, 1]
        self.assertGreater(r, 0.95,
            msg=f"Tương quan b_true vs b_est = {r:.4f}, cần > 0.95")
    
    def test_1pl_standardization(self):
        """Sau 1PL, mean(theta) phải ≈ 0"""
        np.random.seed(42)
        N, M = 20, 300
        b_true = np.random.randn(N)
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, np.ones(N), b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        self.assertAlmostEqual(result.theta.mean(), 0.0, delta=0.01,
            msg="Mean(theta) phải ≈ 0 sau standardization")
    
    def test_2pl_theta_mean_near_zero(self):
        """Sau 2PL (MML-EM), mean(theta_EAP) ≈ 0 (do prior N(0,1))"""
        np.random.seed(42)
        N, M = 20, 500
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        self.assertAlmostEqual(result.theta.mean(), 0.0, delta=0.15,
            msg="Mean(theta_EAP) phải ≈ 0")
    
    def test_log_likelihood_negative(self):
        """Log-likelihood phải < 0 (log of probabilities < 1)"""
        np.random.seed(42)
        N, M = 15, 200
        b_true = np.random.randn(N)
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, np.ones(N), b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=1)
        result = run_jmle(U, config)
        
        self.assertLess(result.log_likelihood, 0,
            msg="Log-likelihood phải < 0")


class TestMMLEM(unittest.TestCase):
    """Kiểm tra thuật toán MML-EM cho 2PL/3PL"""
    
    def test_gauss_hermite_accuracy(self):
        """GH quadrature phải tích phân chính xác Gaussian moments"""
        from rasch_irt.mml_em import gauss_hermite_points
        
        X_q, A_q = gauss_hermite_points(21)
        
        # ∫ φ(x) dx = 1
        integral_0 = np.sum(A_q)
        self.assertAlmostEqual(integral_0, 1.0, places=10,
            msg="∫ φ(x) dx phải = 1")
        
        # ∫ x φ(x) dx = 0 (odd function)
        integral_1 = np.sum(X_q * A_q)
        self.assertAlmostEqual(integral_1, 0.0, places=10,
            msg="∫ x φ(x) dx phải = 0")
        
        # ∫ x² φ(x) dx = 1 (variance of N(0,1))
        integral_2 = np.sum(X_q**2 * A_q)
        self.assertAlmostEqual(integral_2, 1.0, places=6,
            msg="∫ x² φ(x) dx phải = 1")
    
    def test_e_step_posterior_sum_to_one(self):
        """E-step posterior weights phải sum to 1 cho mỗi thí sinh"""
        from rasch_irt.mml_em import gauss_hermite_points, e_step
        
        np.random.seed(42)
        N, M = 10, 50
        a = np.ones(N)
        b = np.random.randn(N)
        c = np.zeros(N)
        
        P = prob_3pl(np.random.randn(M), a, b, c)
        U = (np.random.rand(N, M) < P).astype(float)
        
        X_q, A_q = gauss_hermite_points(21)
        r_bar, f_bar, ll = e_step(U, a, b, c, X_q, A_q)
        
        # f_bar[i, :].sum() should equal M (total persons)
        # because f_bar_q = sum_j w_jq and sum_q w_jq = 1 for each j
        self.assertAlmostEqual(f_bar[0, :].sum(), M, delta=0.01,
            msg="Sum of f_bar across quadrature should ≈ M")
    
    def test_marginal_ll_increases(self):
        """Marginal LL phải tăng đơn điệu qua các iteration"""
        from rasch_irt.mml_em import (
            gauss_hermite_points, e_step, _m_step_item_nr,
            _initialize_mml, MMLConfig, B_MIN, B_MAX
        )
        from rasch_irt.core import get_beta_params
        
        np.random.seed(42)
        N, M = 15, 200
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.5
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        a, b, c = _initialize_mml(U, 2, 4)
        X_q, A_q = gauss_hermite_points(21)
        alpha_p, beta_p = get_beta_params(4)
        
        lls = []
        for _ in range(10):
            r_bar, f_bar, ll = e_step(U, a, b, c, X_q, A_q)
            lls.append(ll)
            a, b, c = _m_step_item_nr(
                r_bar, f_bar, X_q, a, b, c, 2, alpha_p, beta_p
            )
        
        # LL should be non-decreasing (EM property)
        for i in range(1, len(lls)):
            self.assertGreaterEqual(lls[i], lls[i-1] - 1e-6,
                msg=f"LL decreased at iteration {i}: {lls[i]:.2f} < {lls[i-1]:.2f}")
    
    def test_2pl_parameter_recovery_a(self):
        """MML-EM phải recover a chính xác (r > 0.70 vs ground truth)"""
        np.random.seed(2024)
        N, M = 30, 500
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        # Standardize for comparison
        a_true_std = (a_true - a_true.mean()) / max(a_true.std(), 1e-5)
        a_est_std = (result.a - result.a.mean()) / max(result.a.std(), 1e-5)
        r = np.corrcoef(a_true_std, a_est_std)[0, 1]
        
        self.assertGreater(r, 0.70,
            msg=f"a recovery r = {r:.4f}, cần > 0.70. "
                f"a_est range: [{result.a.min():.2f}, {result.a.max():.2f}]")
    
    def test_2pl_parameter_recovery_b(self):
        """MML-EM phải recover b chính xác (r > 0.95 vs ground truth)"""
        np.random.seed(2024)
        N, M = 30, 500
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        b_true_std = (b_true - b_true.mean()) / max(b_true.std(), 1e-5)
        b_est_std = (result.b - result.b.mean()) / max(result.b.std(), 1e-5)
        r = np.corrcoef(b_true_std, b_est_std)[0, 1]
        
        self.assertGreater(r, 0.95,
            msg=f"b recovery r = {r:.4f}, cần > 0.95")
    
    def test_2pl_a_not_inflated(self):
        """MML-EM a phải nằm trong phạm vi hợp lý, KHÔNG bị phóng đại"""
        np.random.seed(2024)
        N, M = 30, 500
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        # OLD JMLE: a ∈ [5.7, 10.0] — bị phóng đại nghiêm trọng
        # MML-EM: a phải nằm trong phạm vi hợp lý
        self.assertLess(result.a.max(), 5.0,
            msg=f"a_max = {result.a.max():.2f}, không được > 5.0 (JMLE cũ: 10.0)")
        self.assertGreater(result.a.mean(), 0.3,
            msg=f"a_mean = {result.a.mean():.2f}, phải > 0.3")
    
    def test_eap_theta_recovery(self):
        """EAP theta phải tương quan cao với ground truth (r > 0.80)"""
        np.random.seed(2024)
        N, M = 30, 500
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        r = np.corrcoef(theta_true, result.theta)[0, 1]
        self.assertGreater(r, 0.80,
            msg=f"theta recovery r = {r:.4f}, cần > 0.80")
    
    def test_1pl_still_uses_jmle(self):
        """1PL phải vẫn dùng JMLE (kết quả phải a=1.0)"""
        np.random.seed(42)
        N, M = 15, 200
        b_true = np.random.randn(N)
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, np.ones(N), b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=1, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        # 1PL JMLE forces a = 1.0
        assert_allclose(result.a, np.ones(N), atol=1e-10,
            err_msg="1PL phải có a = 1.0 (JMLE)")
    
    def test_run_jmle_routes_to_mml(self):
        """run_jmle(model_type=2) phải tự động dùng MML-EM (a hợp lý)"""
        np.random.seed(42)
        N, M = 15, 300
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.5
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=50, tol=0.001)
        result = run_jmle(U, config)
        
        # Nếu routing đúng → MML-EM → a hợp lý
        # Nếu vẫn dùng JMLE → a ∈ [5, 10]
        self.assertLess(result.a.max(), 5.0,
            msg="Routing failed: a vẫn bị phóng đại → vẫn dùng JMLE")
    
    def test_convergence_2pl(self):
        """MML-EM 2PL phải hội tụ trong giới hạn iteration"""
        np.random.seed(42)
        N, M = 20, 300
        a_true = np.random.uniform(0.5, 2.0, N)
        b_true = np.random.randn(N) * 0.8
        theta_true = np.random.randn(M)
        P = prob_3pl(theta_true, a_true, b_true, np.zeros(N))
        U = (np.random.rand(N, M) < P).astype(float)
        
        config = JMLEConfig(model_type=2, max_iter=100, tol=0.001)
        result = run_jmle(U, config)
        
        self.assertTrue(result.converged,
            msg=f"MML-EM 2PL không hội tụ sau {result.iterations} iterations")


class TestLogLikelihood(unittest.TestCase):
    """Kiểm tra compute_log_likelihood"""
    
    def test_perfect_prediction(self):
        """Khi P match U perfectly, ll gần 0"""
        U = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
        P = np.array([[0.999, 0.001, 0.999], [0.001, 0.999, 0.001]])
        ll = compute_log_likelihood(U, P)
        self.assertGreater(ll, -0.1, "LL gần 0 khi prediction hoàn hảo")
    
    def test_random_prediction(self):
        """Khi P = 0.5, ll = N*M*log(0.5)"""
        N, M = 10, 50
        U = np.random.binomial(1, 0.5, (N, M)).astype(float)
        P = 0.5 * np.ones((N, M))
        ll = compute_log_likelihood(U, P)
        expected = N * M * np.log(0.5)
        self.assertAlmostEqual(ll, expected, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

