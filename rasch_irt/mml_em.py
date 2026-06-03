"""
MML-EM Module — Marginal Maximum Likelihood via Expectation-Maximization
=========================================================================
Thuật toán ước lượng tham số IRT 2PL/3PL chuẩn ngành.

Tham chiếu:
  - Bock & Aitkin (1981). Marginal Maximum Likelihood Estimation of Item Parameters.
  - Baker & Kim (2004). Item Response Theory: Parameter Estimation Techniques, Ch. 8-9.
  - R package `ltm` (Rizopoulos, 2006) — sử dụng cùng phương pháp MML-EM.

MML-EM tránh hoàn toàn vấn đề Neyman-Scott Inconsistency của JMLE
bằng cách tích phân (integrate out) θ khỏi hàm likelihood.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .core import (
    prob_3pl,
    D,
    get_beta_params,
)


# =========================================================================
# 1. Cấu hình MML-EM
# =========================================================================

@dataclass
class MMLConfig:
    """Cấu hình cho thuật toán MML-EM

    Attributes:
        model_type: Loại mô hình IRT (2 = 2PL, 3 = 3PL)
        num_options: Số phương án trả lời (dùng cho Beta Prior c trong 3PL)
        max_iter: Số vòng lặp EM tối đa
        tol: Ngưỡng hội tụ trên thay đổi marginal log-likelihood
        num_quadrature: Số điểm Gauss-Hermite quadrature (mặc định 21, chuẩn R ltm)
    """
    model_type: int = 2
    num_options: int = 4
    max_iter: int = 500
    tol: float = 1e-4
    num_quadrature: int = 21


# Hạn chế biên tham số
B_MIN, B_MAX = -5.0, 5.0
A_MIN, A_MAX = 0.2, 5.0    # Hẹp hơn JMLE vì MML ổn định hơn
C_MIN, C_MAX = 0.0001, 0.35

# Số bước NR tối đa cho mỗi M-step
MAX_M_STEP_ITER = 20


# =========================================================================
# 2. Gauss-Hermite Quadrature
# =========================================================================

def gauss_hermite_points(Q: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính các điểm và trọng số Gauss-Hermite quadrature cho phân phối N(0,1).

    Sử dụng biến đổi: ∫ f(x) φ(x) dx ≈ Σ_q f(X_q) * A_q
    với X_q là nodes và A_q là weights đã chuẩn hóa cho N(0,1).

    Args:
        Q: Số điểm quadrature (khuyến nghị 21 hoặc 31)

    Returns:
        (nodes, weights): Tuple gồm nodes [Q] và weights [Q] cho N(0,1)
    """
    # numpy.polynomial.hermite_e (probabilist's Hermite)
    # cung cấp quadrature cho ∫ f(x) exp(-x²/2) dx
    nodes, weights = np.polynomial.hermite_e.hermegauss(Q)
    # Chuẩn hóa weights để tích phân = 1 cho phân phối N(0,1)
    weights = weights / np.sqrt(2 * np.pi)
    return nodes, weights


# =========================================================================
# 3. E-Step: Expectation
# =========================================================================

def e_step(
    U: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    X_q: np.ndarray,
    A_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    E-Step: Tính posterior weights và expected sufficient statistics.

    Với mỗi thí sinh j, tính posterior probability tại mỗi quadrature point q:
        w_jq = L_j(X_q) * A_q / Σ_q' L_j(X_q') * A_q'
    Sau đó tính expected statistics:
        r̄_iq = Σ_j u_ij * w_jq   (expected correct tại quad point q)
        f̄_iq = Σ_j w_jq          (expected total tại quad point q)

    Args:
        U: Ma trận nhị phân [N × M]
        a, b, c: Tham số câu hỏi [N]
        X_q: Quadrature nodes [Q]
        A_q: Quadrature weights [Q]

    Returns:
        r_bar: [N × Q] expected number correct
        f_bar: [N × Q] expected number responses (= expected sample size)
        marginal_ll: Scalar marginal log-likelihood
    """
    N, M = U.shape
    Q = len(X_q)

    # Tính P(X_q) cho tất cả items và quad points: [N × Q]
    P_q = prob_3pl(X_q, a, b, c)  # [N × Q]

    # Tính log-likelihood cho mỗi thí sinh tại mỗi quad point
    # log L_j(X_q) = Σ_i [ u_ij log P_iq + (1-u_ij) log(1-P_iq) ]
    P_safe = np.clip(P_q, 1e-15, 1.0 - 1e-15)
    log_P = np.log(P_safe)      # [N × Q]
    log_Q = np.log(1 - P_safe)  # [N × Q]

    # log_L_jq = U^T @ log_P + (1-U)^T @ log_Q  → [M × Q]
    # U is [N × M], log_P is [N × Q]
    log_L = U.T @ log_P + (1.0 - U).T @ log_Q  # [M × Q]

    # Thêm log prior weights: log(A_q)
    log_A = np.log(np.maximum(A_q, 1e-300))  # [Q]
    log_numerator = log_L + log_A[np.newaxis, :]  # [M × Q]

    # Log-sum-exp trick cho numerical stability
    log_max = np.max(log_numerator, axis=1, keepdims=True)  # [M × 1]
    log_denominator = log_max + np.log(
        np.sum(np.exp(log_numerator - log_max), axis=1, keepdims=True)
    )  # [M × 1]

    # Posterior weights: w_jq = exp(log_numerator - log_denominator)
    log_w = log_numerator - log_denominator  # [M × Q]
    w = np.exp(log_w)  # [M × Q]

    # Expected sufficient statistics
    # r̄_iq = Σ_j u_ij * w_jq → U @ w = [N × M] @ [M × Q] = [N × Q]
    r_bar = U @ w       # [N × Q]
    # f̄_q = Σ_j w_jq → [Q] (same for all items)
    f_bar_vec = w.sum(axis=0)  # [Q]
    # Expand to [N × Q] for convenience
    f_bar = np.tile(f_bar_vec, (N, 1))  # [N × Q]

    # Marginal log-likelihood = Σ_j log[ Σ_q L_j(X_q) * A_q ]
    marginal_ll = float(np.sum(log_denominator))

    return r_bar, f_bar, marginal_ll


# =========================================================================
# 4. M-Step: Maximization
# =========================================================================

def _m_step_item_nr(
    r_bar: np.ndarray,
    f_bar: np.ndarray,
    X_q: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    model_type: int,
    alpha_prior: float,
    beta_prior: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-Step: Newton-Raphson update cho item parameters trên expected statistics.

    Sử dụng marginal score và information thay vì individual observations.
    Score function trên expected data:
        ∂ℓ_M/∂ξ_i = Σ_q [r̄_iq - f̄_iq * P_iq] * ∂P_iq/∂ξ_i / [P_iq * Q_iq]

    Args:
        r_bar, f_bar: Expected statistics từ E-step [N × Q]
        X_q: Quadrature nodes [Q]
        a, b, c: Current item parameters [N]
        model_type: 2 hoặc 3
        alpha_prior, beta_prior: Beta prior parameters cho c

    Returns:
        (a_new, b_new, c_new): Updated parameters
    """
    N = len(a)
    Q = len(X_q)

    for _nr_iter in range(MAX_M_STEP_ITER):
        # Tính P tại quadrature points: [N × Q]
        P_q = prob_3pl(X_q, a, b, c)
        P_safe = np.clip(P_q, 1e-15, 1 - 1e-15)

        # Common terms
        # P* = (P - c) / (1 - c): phần logistic thuần
        c_mat = c[:, np.newaxis]       # [N × 1]
        P_star = (P_safe - c_mat) / np.maximum(1.0 - c_mat, 1e-10)
        P_star = np.clip(P_star, 1e-15, 1.0)
        Q_val = 1.0 - P_safe           # [N × Q]

        # Residual: r̄_iq - f̄_iq * P_iq
        residual = r_bar - f_bar * P_safe  # [N × Q]

        # --- Update b ---
        # Score_b = Σ_q -D * a_i * P*_iq / P_iq * residual_iq
        a_mat = a[:, np.newaxis]  # [N × 1]
        score_b = np.sum(
            -D * a_mat * P_star / P_safe * residual,
            axis=1
        )  # [N]

        # Info_b = Σ_q D² * a_i² * P*² * Q / P * f̄_iq
        info_b = np.sum(
            D**2 * a_mat**2 * P_star**2 * Q_val / P_safe * f_bar,
            axis=1
        )  # [N]
        info_b = np.maximum(info_b, 1e-10)

        delta_b = score_b / info_b
        delta_b = np.clip(delta_b, -1.0, 1.0)  # Conservative damping
        b = b + delta_b
        b = np.clip(b, B_MIN, B_MAX)

        # --- Update a (2PL, 3PL) ---
        # Recompute P with updated b
        P_q = prob_3pl(X_q, a, b, c)
        P_safe = np.clip(P_q, 1e-15, 1 - 1e-15)
        P_star = (P_safe - c_mat) / np.maximum(1.0 - c_mat, 1e-10)
        P_star = np.clip(P_star, 1e-15, 1.0)
        Q_val = 1.0 - P_safe
        residual = r_bar - f_bar * P_safe

        # X_q matrix: [N × Q] (broadcast)
        X_mat = np.tile(X_q, (N, 1))  # [N × Q]
        b_mat = b[:, np.newaxis]       # [N × 1]

        # Score_a = Σ_q D * (X_q - b_i) * P*_iq / P_iq * residual_iq
        score_a = np.sum(
            D * (X_mat - b_mat) * P_star / P_safe * residual,
            axis=1
        )  # [N]

        # Log-normal prior: score += -1/a, info += 1/a²
        score_a += -1.0 / np.maximum(a, 1e-5)

        # Info_a = Σ_q D² * (X_q - b_i)² * P*² * Q / P * f̄_iq
        info_a = np.sum(
            D**2 * (X_mat - b_mat)**2 * P_star**2 * Q_val / P_safe * f_bar,
            axis=1
        )  # [N]
        info_a += 1.0 / np.maximum(a**2, 1e-5)  # Log-normal prior info
        info_a = np.maximum(info_a, 1e-10)

        delta_a = score_a / info_a
        delta_a = np.clip(delta_a, -0.5, 0.5)  # Conservative damping
        a = a + delta_a
        a = np.clip(a, A_MIN, A_MAX)

        # --- Update c (3PL only) ---
        if model_type == 3:
            P_q = prob_3pl(X_q, a, b, c)
            P_safe = np.clip(P_q, 1e-15, 1 - 1e-15)
            Q_val = 1.0 - P_safe
            residual = r_bar - f_bar * P_safe

            # Score_c = Σ_q residual / [(1-c)*P] + prior
            score_c = np.sum(
                residual / (np.maximum(1.0 - c_mat, 1e-10) * P_safe),
                axis=1
            )  # [N]

            # Beta prior gradient
            safe_c = np.maximum(c, 1e-5)
            safe_1c = np.maximum(1.0 - c, 1e-5)
            score_c += (alpha_prior - 1.0) / safe_c - (beta_prior - 1.0) / safe_1c

            # Info_c = Σ_q Q / [(1-c)² * P] * f̄_iq + prior
            info_c = np.sum(
                Q_val / (np.maximum(1.0 - c_mat, 1e-10)**2 * P_safe) * f_bar,
                axis=1
            )  # [N]
            info_c += (alpha_prior - 1.0) / safe_c**2 + (beta_prior - 1.0) / safe_1c**2
            info_c = np.maximum(info_c, 1e-10)

            delta_c = score_c / info_c
            delta_c = np.clip(delta_c, -0.05, 0.05)  # Very conservative for c
            c = c + delta_c
            c = np.clip(c, C_MIN, C_MAX)

        # Check NR inner convergence
        max_delta = max(
            np.max(np.abs(delta_b)),
            np.max(np.abs(delta_a)),
        )
        if model_type == 3:
            max_delta = max(max_delta, np.max(np.abs(delta_c)))
        if max_delta < 1e-4:
            break

    return a, b, c


# =========================================================================
# 5. EAP Scoring
# =========================================================================

def eap_scoring(
    U: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    X_q: np.ndarray,
    A_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expected A Posteriori (EAP) scoring cho năng lực thí sinh.

    θ̂_j^EAP = Σ_q X_q * w_jq
    SE_j = sqrt( Σ_q (X_q - θ̂_j)² * w_jq )

    Args:
        U: Ma trận nhị phân [N × M]
        a, b, c: Tham số câu hỏi (đã ước lượng) [N]
        X_q: Quadrature nodes [Q]
        A_q: Quadrature weights [Q]

    Returns:
        (theta_eap, se_eap): Điểm năng lực [M] và sai số chuẩn [M]
    """
    N, M = U.shape
    Q = len(X_q)

    # P tại quadrature points: [N × Q]
    P_q = prob_3pl(X_q, a, b, c)
    P_safe = np.clip(P_q, 1e-15, 1.0 - 1e-15)

    log_P = np.log(P_safe)      # [N × Q]
    log_Q = np.log(1 - P_safe)  # [N × Q]

    # log L_j(X_q) = U^T @ log_P + (1-U)^T @ log_Q → [M × Q]
    log_L = U.T @ log_P + (1.0 - U).T @ log_Q

    # Posterior weights
    log_A = np.log(np.maximum(A_q, 1e-300))
    log_numerator = log_L + log_A[np.newaxis, :]  # [M × Q]

    log_max = np.max(log_numerator, axis=1, keepdims=True)
    log_denom = log_max + np.log(
        np.sum(np.exp(log_numerator - log_max), axis=1, keepdims=True)
    )
    w = np.exp(log_numerator - log_denom)  # [M × Q]

    # EAP estimate
    theta_eap = np.sum(w * X_q[np.newaxis, :], axis=1)  # [M]

    # EAP standard error
    deviations_sq = (X_q[np.newaxis, :] - theta_eap[:, np.newaxis])**2  # [M × Q]
    var_eap = np.sum(w * deviations_sq, axis=1)  # [M]
    se_eap = np.sqrt(np.maximum(var_eap, 1e-10))  # [M]

    return theta_eap, se_eap


# =========================================================================
# 6. Main MML-EM Loop
# =========================================================================

def _initialize_mml(
    U: np.ndarray,
    model_type: int,
    num_options: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Khởi tạo tham số câu hỏi cho MML-EM"""
    N, M = U.shape

    # b: từ logit tỷ lệ đúng
    p = np.clip(U.mean(axis=1), 0.01, 0.99)
    b = -np.log(p / (1.0 - p))
    b = np.clip(b, B_MIN, B_MAX)

    # a: tất cả = 1.0
    a = np.ones(N)

    # c: 1/num_options cho 3PL, 0 cho 2PL
    c = np.full(N, 1.0 / num_options) if model_type == 3 else np.zeros(N)

    return a, b, c


def _count_params_mml(N: int, model_type: int) -> int:
    """Đếm số tham số cho MML (không tính θ vì đã integrate out)"""
    if model_type == 2:
        return 2 * N  # N params a + N params b
    else:
        return 3 * N  # N params a + N params b + N params c


def run_mml_em(U: np.ndarray, config: MMLConfig) -> "JMLEResult":
    """
    Ước lượng tham số IRT 2PL/3PL bằng thuật toán MML-EM.

    Quy trình:
        1. Khởi tạo tham số (a, b, c)
        2. Lặp EM:
            E-Step: Tính posterior weights và expected statistics
            M-Step: Newton-Raphson update item parameters
            Kiểm tra hội tụ trên marginal log-likelihood
        3. EAP scoring: Ước lượng θ cho từng thí sinh
        4. Tính AIC/BIC và trả về JMLEResult

    Args:
        U: Ma trận nhị phân [N × M]
        config: Cấu hình MMLConfig

    Returns:
        JMLEResult: Kết quả ước lượng (cùng format với JMLE để backward compatible)
    """
    # Import ở đây để tránh circular dependency
    from .irt import JMLEResult

    N, M = U.shape

    # 1. Khởi tạo
    a, b, c = _initialize_mml(U, config.model_type, config.num_options)
    X_q, A_q = gauss_hermite_points(config.num_quadrature)

    alpha_prior, beta_prior = get_beta_params(config.num_options)

    # 2. EM Loop
    converged = False
    iterations = 0
    prev_ll = -np.inf

    for em_iter in range(config.max_iter):
        # E-Step
        r_bar, f_bar, marginal_ll = e_step(U, a, b, c, X_q, A_q)

        # M-Step
        a, b, c = _m_step_item_nr(
            r_bar, f_bar, X_q, a, b, c,
            config.model_type, alpha_prior, beta_prior
        )

        iterations = em_iter + 1

        # Kiểm tra hội tụ trên marginal LL
        delta_ll = marginal_ll - prev_ll
        if iterations > 1 and abs(delta_ll) < config.tol:
            converged = True
            break

        prev_ll = marginal_ll

    # 3. EAP Scoring
    theta, se_theta = eap_scoring(U, a, b, c, X_q, A_q)

    # 4. Tính AIC/BIC
    num_params = _count_params_mml(N, config.model_type)
    aic = -2.0 * marginal_ll + 2.0 * num_params
    bic = -2.0 * marginal_ll + num_params * np.log(M)

    return JMLEResult(
        converged=converged,
        iterations=iterations,
        theta=theta,
        b=b,
        a=a,
        c=c,
        log_likelihood=marginal_ll,
        aic=float(aic),
        bic=float(bic),
        num_params=num_params,
    )
