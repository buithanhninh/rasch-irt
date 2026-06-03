"""
IRT Module — Item Response Theory (1PL, 2PL, 3PL)
Chứa thuật toán hiệu chỉnh JMLE Newton-Raphson, Auto-Fit so sánh mô hình,
đánh giá các tiên đề IRT (Unidimensionality, Local Independence) và phân tích ICC/IIF.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from scipy.stats import norm

from .core import (
    prob_3pl,
    D,
    score_theta,
    info_theta,
    score_b,
    info_b,
    score_a,
    info_a,
    get_beta_params,
    score_c_map,
    info_c_map,
    newton_raphson_batch,
)
from .exceptions import InvalidMatrixError, ConvergenceError


# =========================================================================
# 1. Định nghĩa Cấu hình và Kết quả đầu ra (Configs & Dataclasses)
# =========================================================================

@dataclass
class JMLEConfig:
    """Cấu hình cho thuật toán hiệu chỉnh JMLE"""
    model_type: int = 3           # 1 = 1PL (Rasch), 2 = 2PL, 3 = 3PL
    num_options: int = 4          # Số phương án lựa chọn (để tính Beta Prior)
    max_iter: int = 100           # Số vòng lặp tối đa
    tol: float = 0.001            # Ngưỡng hội tụ (max_change < tol)
    epsilon: float = 0.001        # Đồng bộ ngược với phiên bản cũ


@dataclass
class JMLEResult:
    """Kết quả thô thu được từ vòng lặp JMLE"""
    converged: bool
    iterations: int
    theta: np.ndarray
    b: np.ndarray
    a: np.ndarray
    c: np.ndarray
    log_likelihood: float
    aic: float = 0.0
    bic: float = 0.0
    num_params: int = 0


@dataclass
class IrtItemResult:
    """Kết quả IRT chi tiết cho từng câu hỏi"""
    item_number: int
    param_a: float              # Độ phân biệt a (Cố định = 1.0 trong 1PL)
    param_b: float              # Độ khó b
    param_c: float              # Xác suất đoán mò c (Cố định = 0.0 trong 1PL/2PL)
    se_a: float                 # Sai số chuẩn của a
    se_b: float                 # Sai số chuẩn của b
    se_c: float                 # Sai số chuẩn của c
    infit_mnsq: float           # Chỉ số Infit
    outfit_mnsq: float          # Chỉ số Outfit
    fit_flag: str               # Phân loại độ tương hợp ("Tốt", "Chấp nhận", v.v.)
    icc_points: List[Dict[str, float]] = field(default_factory=list)  # Các điểm ICC để vẽ biểu đồ
    iif_points: List[Dict[str, float]] = field(default_factory=list)  # Các điểm IIF để vẽ biểu đồ
    empirical_icc: List[Dict[str, Any]] = field(default_factory=list) # Dữ liệu thực nghiệm ICC


@dataclass
class IrtPersonResult:
    """Kết quả năng lực thí sinh"""
    index: int
    raw_score: int              # Điểm thô
    theta_mle: float            # Điểm năng lực thực sự theta
    standard_error: float       # Sai số tiêu chuẩn của theta
    true_score_10: float        # Quy đổi năng lực sang thang điểm 10
    infit_mnsq: float           # Infit của thí sinh
    outfit_mnsq: float          # Outfit của thí sinh


@dataclass
class AssumptionResult:
    """Kết quả kiểm tra tiên đề IRT"""
    unidimensionality_passed: bool = True
    pca_first_eigenvalue: float = 0.0
    pca_second_eigenvalue: float = 0.0
    pca_ratio_explained: float = 0.0
    pca_eigenvalue_ratio: float = 0.0
    
    local_independence_passed: bool = True
    max_q3_value: float = 0.0
    q3_flagged_pairs: List[Dict[str, Any]] = field(default_factory=list)
    
    warnings: List[str] = field(default_factory=list)


@dataclass
class IrtResult:
    """Kết quả phân tích IRT toàn diện"""
    converged: bool
    iterations: int
    log_likelihood: float
    aic: float
    bic: float
    num_params: int
    items: List[IrtItemResult]
    persons: List[IrtPersonResult]
    theta_density: List[Dict[str, float]] = field(default_factory=list)
    assumptions: Optional[AssumptionResult] = None


@dataclass
class ModelFitResult:
    """Kết quả tương hợp của từng mô hình"""
    model_type: int
    model_name: str
    aic: float = 0.0
    bic: float = 0.0
    log_likelihood: float = 0.0
    converged: bool = False
    iterations: int = 0
    num_params: int = 0


@dataclass
class AutoFitResult:
    """Kết quả so sánh và chọn mô hình tối ưu tự động"""
    models: List[ModelFitResult] = field(default_factory=list)
    recommended_model: int = 1
    recommendation_reason: str = ""
    sample_size: int = 0


MODEL_NAMES = {
    1: "1PL (Rasch)",
    2: "2PL",
    3: "3PL",
}

# Hạn chế biên tham số để chống phân kỳ toán học
THETA_MIN, THETA_MAX = -10.0, 10.0
B_MIN, B_MAX = -5.0, 5.0
A_MIN, A_MAX = 0.01, 10.0
C_MIN, C_MAX = 0.0001, 0.4

MAX_INNER_THETA = 50
MAX_INNER_ITEM = 5


# =========================================================================
# 2. Hàm Khởi tạo và Tiện ích Lõi (Core Utilities)
# =========================================================================

def initialize_parameters(
    U: np.ndarray, 
    model_type: int = 3, 
    num_options: int = 4,
    theta_min: float = -10.0, 
    theta_max: float = 10.0,
    b_min: float = -5.0, 
    b_max: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Khởi tạo tham số ban đầu dựa trên điểm thô và độ khó thô.
    """
    N, M = U.shape
    
    # Khởi tạo theta từ điểm thô chuẩn hóa
    raw_scores = U.sum(axis=0)
    mean_score = raw_scores.mean()
    std_score = max(raw_scores.std(), 1e-10)
    theta = (raw_scores - mean_score) / std_score
    theta = np.clip(theta, theta_min, theta_max)
    
    # Khởi tạo b từ tỷ lệ đúng logit
    p = U.mean(axis=1)
    p = np.clip(p, 0.01, 0.99)
    b = -np.log(p / (1.0 - p))
    b = np.clip(b, b_min, b_max)
    
    # Khởi tạo a = 1.0
    a = np.ones(N)
    
    # Khởi tạo c = 1/num_options cho 3PL hoặc 0 cho 1PL/2PL
    c = np.full(N, 1.0 / num_options) if model_type == 3 else np.zeros(N)
    
    return theta, a, b, c


def handle_extreme_scores(theta: np.ndarray, U: np.ndarray, correction: float = 0.3) -> np.ndarray:
    """
    Hiệu chỉnh Bayesian cho các thí sinh đạt điểm tuyệt đối (0 hoặc tối đa).
    """
    N = U.shape[0]
    raw_scores = U.sum(axis=0)
    
    extreme_low = (raw_scores == 0)
    if extreme_low.any():
        adj = correction / N
        theta[extreme_low] = np.log(adj / (1.0 - adj))
        
    extreme_high = (raw_scores == N)
    if extreme_high.any():
        adj = (N - correction) / N
        theta[extreme_high] = np.log(adj / (1.0 - adj))
        
    return theta


def compute_log_likelihood(U: np.ndarray, P: np.ndarray) -> float:
    """Tính log-likelihood của toàn bộ mô hình"""
    P_safe = np.clip(P, 1e-15, 1.0 - 1e-15)
    ll = np.sum(U * np.log(P_safe) + (1.0 - U) * np.log(1.0 - P_safe))
    return float(ll)


def _count_params(N: int, M: int, model_type: int) -> int:
    """Đếm số lượng tham số tự do"""
    if model_type == 1:
        return N + M - 1
    elif model_type == 2:
        return 2 * N + M - 2
    else:
        return 3 * N + M - 2


def _compute_aic_bic(ll: float, num_params: int, M: int) -> tuple[float, float]:
    """Tính toán chỉ số AIC/BIC dựa trên log-likelihood"""
    aic = -2.0 * ll + 2.0 * num_params
    bic = -2.0 * ll + num_params * np.log(M)
    return float(aic), float(bic)


# =========================================================================
# 3. Thuật toán Hiệu chỉnh JMLE (Joint Maximum Likelihood Estimation)
# =========================================================================

def run_jmle(U: np.ndarray, config: JMLEConfig) -> JMLEResult:
    """
    Ước lượng JMLE Newton-Raphson đồng thời cho các tham số câu hỏi và thí sinh.
    """
    N, M = U.shape
    theta, a, b, c = initialize_parameters(U, config.model_type, config.num_options)
    
    alpha_prior, beta_prior = get_beta_params(config.num_options)
    convergence_tol = min(config.tol, config.epsilon)
    
    converged = False
    iterations = 0
    
    for outer in range(config.max_iter):
        theta_old = theta.copy()
        
        # 1. E-STEP: Ước lượng năng lực theta
        for _inner in range(MAX_INNER_THETA):
            P = prob_3pl(theta, a, b, c)
            st = score_theta(U, P, a, c)
            it = info_theta(P, a, c)
            
            safe_it = np.where(np.abs(it) < 1e-15, 1.0, it)
            deltas = st / safe_it
            deltas = np.clip(deltas, -3.0, 3.0)
            
            theta = theta + deltas
            theta = np.clip(theta, THETA_MIN, THETA_MAX)
            
            if np.max(np.abs(deltas)) < convergence_tol:
                break
        
        # Chuẩn hóa để tránh trôi dạt tham số
        mean_theta = theta.mean()
        if config.model_type == 1:
            theta = theta - mean_theta
            b = b - mean_theta
        else:
            std_theta = theta.std()
            if std_theta > 1e-5:
                theta = (theta - mean_theta) / std_theta
                b = (b - mean_theta) / std_theta
                a = a * std_theta
        
        # 2. M-STEP: Ước lượng tham số câu hỏi (b, a, c)
        for _inner in range(MAX_INNER_ITEM):
            # Cập nhật b
            P = prob_3pl(theta, a, b, c)
            sb = score_b(U, P, a, c)
            ib = info_b(P, a, c)
            b = newton_raphson_batch(sb, ib, b, val_min=B_MIN, val_max=B_MAX)
            
            # Cập nhật a (2PL, 3PL)
            if config.model_type >= 2:
                P = prob_3pl(theta, a, b, c)
                sa = score_a(U, P, theta, a, b, c)
                ia = info_a(P, theta, a, b, c)
                
                # Bổ sung Log-Normal Prior cho a để ổn định toán học
                sa += -1.0 / np.maximum(a, 1e-5)
                ia += 1.0 / np.maximum(a**2, 1e-5)
                
                a = newton_raphson_batch(sa, ia, a, val_min=A_MIN, val_max=A_MAX)
                
            # Cập nhật c (3PL)
            if config.model_type == 3:
                P = prob_3pl(theta, a, b, c)
                sc = score_c_map(U, P, c, alpha_prior, beta_prior)
                ic = info_c_map(P, c, alpha_prior, beta_prior)
                c = newton_raphson_batch(sc, ic, c, val_min=C_MIN, val_max=C_MAX)
                
        # Kiểm tra hội tụ ngoài
        max_change = np.max(np.abs(theta - theta_old))
        iterations = outer + 1
        
        if max_change < convergence_tol:
            converged = True
            break
            
    # Hậu xử lý chuẩn hóa lần cuối
    if config.model_type == 1:
        a = np.ones(N)
        mean_t = theta.mean()
        theta = theta - mean_t
        b = b - mean_t
    else:
        mean_t = theta.mean()
        std_t = max(theta.std(), 1e-5)
        theta = (theta - mean_t) / std_t
        b = (b - mean_t) / std_t
        a = a * std_t
        
    P_final = prob_3pl(theta, a, b, c)
    ll = compute_log_likelihood(U, P_final)
    
    num_params = _count_params(N, M, config.model_type)
    aic, bic = _compute_aic_bic(ll, num_params, M)
    
    return JMLEResult(
        converged=converged,
        iterations=iterations,
        theta=theta,
        b=b,
        a=a,
        c=c,
        log_likelihood=ll,
        aic=aic,
        bic=bic,
        num_params=num_params
    )


# =========================================================================
# 4. Tính toán Chỉ số Khớp và Sai số (Fit Statistics & Standard Errors)
# =========================================================================

def compute_fit_statistics(U: np.ndarray, P: np.ndarray) -> dict[str, np.ndarray]:
    """
    Tính toán chỉ số INFIT và OUTFIT MNSQ cho từng câu hỏi (items) và thí sinh (persons).
    """
    Q = 1.0 - P
    variance = P * Q
    variance = np.maximum(variance, 1e-10)
    
    residual_sq = (U - P) ** 2
    standardized_sq = residual_sq / variance
    
    N, M = U.shape
    
    # ITEM FIT (theo hàng)
    outfit_item = np.mean(standardized_sq, axis=1)
    infit_item = np.sum(residual_sq, axis=1) / np.maximum(np.sum(variance, axis=1), 1e-10)
    
    # PERSON FIT (theo cột)
    outfit_person = np.mean(standardized_sq, axis=0)
    infit_person = np.sum(residual_sq, axis=0) / np.maximum(np.sum(variance, axis=0), 1e-10)
    
    return {
        'infit_item': infit_item,
        'outfit_item': outfit_item,
        'infit_person': infit_person,
        'outfit_person': outfit_person
    }


def classify_fit(mnsq: float) -> str:
    """Phân loại mức độ tương hợp (Fit) dựa trên MNSQ"""
    if mnsq < 0.5:
        return "Quá phù hợp"
    elif mnsq < 0.7:
        return "Chấp nhận"
    elif mnsq <= 1.3:
        return "Tốt"
    elif mnsq <= 1.5:
        return "Hơi lệch"
    else:
        return "Không phù hợp"


def compute_se_theta(P: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính sai số tiêu chuẩn SE(theta) cho mỗi thí sinh"""
    info = info_theta(P, a, c)
    return 1.0 / np.sqrt(np.maximum(info, 1e-10))


def compute_se_items(
    P: np.ndarray, 
    theta: np.ndarray, 
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray, 
    model_type: int, 
    num_options: int = 4
) -> dict[str, np.ndarray]:
    """Tính sai số tiêu chuẩn cho tham số các câu hỏi"""
    N = len(a)
    
    # SE(b)
    i_b = info_b(P, a, c)
    se_b = 1.0 / np.sqrt(np.maximum(i_b, 1e-10))
    
    # SE(a)
    if model_type >= 2:
        i_a = info_a(P, theta, a, b, c)
        se_a = 1.0 / np.sqrt(np.maximum(i_a, 1e-10))
    else:
        se_a = np.zeros(N)
        
    # SE(c)
    if model_type == 3:
        alpha, beta = get_beta_params(num_options)
        i_c = info_c_map(P, c, alpha, beta)
        se_c = 1.0 / np.sqrt(np.maximum(i_c, 1e-10))
    else:
        se_c = np.zeros(N)
        
    return {'se_a': se_a, 'se_b': se_b, 'se_c': se_c}


def compute_true_scores(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Quy đổi theta sang thang điểm thực 10"""
    P = prob_3pl(theta, a, b, c)
    N = len(a)
    raw_true_score = np.sum(P, axis=0)
    true_score_10 = (raw_true_score / N) * 10.0
    return np.clip(true_score_10, 0.0, 10.0)


# =========================================================================
# 5. Phân tích Dữ liệu Thực nghiệm (Empirical & Curves Data)
# =========================================================================

def compute_empirical_icc(U: np.ndarray, theta: np.ndarray, num_bins: int = 15) -> list[list[dict]]:
    """Tính xác suất đúng thực tế để đối chiếu với đường cong ICC"""
    N, M = U.shape
    sort_idx = np.argsort(theta)
    sorted_theta = theta[sort_idx]
    sorted_U = U[:, sort_idx]
    
    bin_edges = np.array_split(np.arange(M), num_bins)
    
    results = []
    for i in range(N):
        item_bins = []
        for b_idx in bin_edges:
            if len(b_idx) == 0:
                continue
            bin_theta = sorted_theta[b_idx].mean()
            bin_prob = sorted_U[i, b_idx].mean()
            item_bins.append({
                "theta": round(float(bin_theta), 2),
                "prob": round(float(bin_prob), 4),
                "count": len(b_idx)
            })
        results.append(item_bins)
        
    return results


def compute_theta_density(theta: np.ndarray, num_bins: int = 50) -> list[dict]:
    """Tính phổ phân bố điểm năng lực thí sinh"""
    if len(np.unique(theta)) < 2:
        return []
    counts, bin_edges = np.histogram(theta, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    density_data = [
        {"theta": round(float(c), 2), "density": round(float(d), 4)}
        for c, d in zip(bin_centers, counts)
    ]
    return density_data


def compute_icc_points(a: float, b: float, c: float) -> list[dict[str, float]]:
    """Tính 81 điểm ICC trên dải năng lực [-4, 4]"""
    theta_range = np.linspace(-4, 4, 81)
    P = prob_3pl(theta_range, np.array([a]), np.array([b]), np.array([c]))[0]
    return [{"theta": float(th), "prob": float(p)} for th, p in zip(theta_range, P)]


def compute_iif_points(a: float, b: float, c: float) -> list[dict[str, float]]:
    """Tính 81 điểm thông tin IIF trên dải năng lực [-4, 4]"""
    theta_range = np.linspace(-4, 4, 81)
    P = prob_3pl(theta_range, np.array([a]), np.array([b]), np.array([c]))[0]
    Pstar = np.maximum((P - c) / (1 - c), 1e-15)
    Q = 1.0 - P
    info = (D**2) * (a**2) * (Pstar**2) * Q / np.maximum(P, 1e-15)
    return [{"theta": float(th), "info": float(inf)} for th, inf in zip(theta_range, info)]


# =========================================================================
# 6. Kiểm tra Tiên đề Hệ thống (Assumption Checking)
# =========================================================================

def check_unidimensionality(U: np.ndarray, threshold_ratio: float = 3.0) -> dict:
    """Kiểm tra đơn hướng PCA"""
    variance = U.var(axis=1)
    valid_mask = variance > 1e-10
    U_valid = U[valid_mask]
    
    if U_valid.shape[0] < 3:
        return {
            "passed": True,
            "first_eigenvalue": 0.0,
            "second_eigenvalue": 0.0,
            "ratio": 0.0,
            "ratio_explained": 0.0
        }
        
    corr_matrix = np.corrcoef(U_valid)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    first_ev = float(eigenvalues[0])
    second_ev = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    
    total_variance = float(eigenvalues.sum())
    ratio_explained = first_ev / max(total_variance, 1e-10) * 100
    
    ratio = first_ev / max(second_ev, 1e-10)
    passed = ratio >= threshold_ratio
    
    return {
        "passed": passed,
        "first_eigenvalue": round(first_ev, 4),
        "second_eigenvalue": round(second_ev, 4),
        "ratio": round(ratio, 2),
        "ratio_explained": round(ratio_explained, 2)
    }


def check_local_independence(
    U: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    threshold_q3: float = 0.2
) -> dict:
    """Kiểm tra độc lập cục bộ dùng chỉ số Yen's Q3"""
    N, M = U.shape
    P = prob_3pl(theta, a, b, c)
    residuals = U - P
    
    flagged_pairs = []
    max_q3 = 0.0
    
    for i in range(N):
        for j in range(i + 1, N):
            r_i = residuals[i]
            r_j = residuals[j]
            
            std_i = np.std(r_i)
            std_j = np.std(r_j)
            if std_i < 1e-10 or std_j < 1e-10:
                continue
                
            q3 = float(np.corrcoef(r_i, r_j)[0, 1])
            if np.isnan(q3):
                continue
                
            abs_q3 = abs(q3)
            if abs_q3 > max_q3:
                max_q3 = abs_q3
                
            if abs_q3 > threshold_q3:
                flagged_pairs.append({
                    "i": int(i + 1),
                    "j": int(j + 1),
                    "q3": round(q3, 4)
                })
                
    passed = len(flagged_pairs) == 0
    return {
        "passed": passed,
        "max_q3": round(max_q3, 4),
        "flagged_pairs": flagged_pairs,
        "total_flagged": len(flagged_pairs)
    }


def run_assumption_checks(
    U: np.ndarray,
    theta: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
) -> AssumptionResult:
    """Đánh giá toàn diện các tiên đề khảo thí của mô hình"""
    result = AssumptionResult()
    
    pca = check_unidimensionality(U)
    result.unidimensionality_passed = pca["passed"]
    result.pca_first_eigenvalue = pca["first_eigenvalue"]
    result.pca_second_eigenvalue = pca["second_eigenvalue"]
    result.pca_ratio_explained = pca["ratio_explained"]
    result.pca_eigenvalue_ratio = pca["ratio"]
    
    if not pca["passed"]:
        result.warnings.append(
            f"Bài thi có thể đo lường nhiều cấu trúc năng lực. "
            f"Tỷ số eigenvalue = {pca['ratio']:.1f} (cần ≥ 3.0)."
        )
        
    if theta is not None and a is not None and b is not None and c is not None:
        q3 = check_local_independence(U, theta, a, b, c)
        result.local_independence_passed = q3["passed"]
        result.max_q3_value = q3["max_q3"]
        result.q3_flagged_pairs = q3["flagged_pairs"]
        
        if not q3["passed"]:
            result.warnings.append(
                f"Phát hiện {q3['total_flagged']} cặp câu hỏi vi phạm độc lập cục bộ (|Q3| > 0.2)."
            )
            
    return result


# =========================================================================
# 7. Ước lượng Tự động (Auto-Fit Engine)
# =========================================================================

def run_auto_fit(
    U: np.ndarray,
    num_options: int = 4,
    max_iter: int = 100,
    epsilon: float = 0.001
) -> AutoFitResult:
    """
    Chạy so sánh đồng thời các mô hình 1PL, 2PL, 3PL để chọn ra mô hình tối ưu.
    """
    N, M = U.shape
    result = AutoFitResult(sample_size=M)
    
    models_to_run = [1, 2, 3]
    if M < 200:
        models_to_run = [1, 2]  # Tránh overfit 3PL trên tập mẫu nhỏ
        
    for model_type in models_to_run:
        try:
            config = JMLEConfig(
                model_type=model_type,
                num_options=num_options,
                max_iter=max_iter,
                epsilon=epsilon
            )
            jmle_res = run_jmle(U, config)
            
            fit_res = ModelFitResult(
                model_type=model_type,
                model_name=MODEL_NAMES[model_type],
                aic=round(jmle_res.aic, 2),
                bic=round(jmle_res.bic, 2),
                log_likelihood=round(jmle_res.log_likelihood, 2),
                converged=jmle_res.converged,
                iterations=jmle_res.iterations,
                num_params=jmle_res.num_params
            )
            result.models.append(fit_res)
        except Exception:
            result.models.append(ModelFitResult(
                model_type=model_type,
                model_name=MODEL_NAMES[model_type],
                converged=False,
                iterations=0
            ))
            
    valid_models = [m for m in result.models if m.log_likelihood < 0]
    converged_models = [m for m in valid_models if m.converged]
    candidates = converged_models if len(converged_models) > 0 else valid_models
    
    if candidates:
        best = min(candidates, key=lambda m: m.bic)
        result.recommended_model = best.model_type
        
        if best.model_type == 1:
            result.recommendation_reason = (
                f"Hệ thống khuyến nghị sử dụng mô hình 1PL (Rasch) "
                f"vì có BIC = {best.bic:.1f} thấp nhất, cho thấy cấu trúc dữ liệu đơn giản."
            )
        elif best.model_type == 2:
            result.recommendation_reason = (
                f"Hệ thống khuyến nghị sử dụng mô hình 2PL "
                f"vì có BIC = {best.bic:.1f} thấp nhất. Phù hợp để đánh giá cả độ khó và phân biệt."
            )
        else:
            result.recommendation_reason = (
                f"Hệ thống khuyến nghị sử dụng mô hình 3PL "
                f"vì có BIC = {best.bic:.1f} thấp nhất, giúp khử nhiễu yếu tố đoán mò."
            )
    else:
        result.recommended_model = 1
        result.recommendation_reason = (
            "Không có mô hình nào hội tụ. Khuyến nghị sử dụng 1PL (Rasch) làm giải pháp fallback."
        )
        
    return result


def compute_model_fit(
    U: np.ndarray, 
    theta: np.ndarray, 
    a: np.ndarray, 
    b: np.ndarray, 
    c: np.ndarray,
    model_type: int
) -> dict[str, Any]:
    """
    Tính các tiêu chí lựa chọn mô hình: AIC, BIC, Log-Likelihood.
    """
    N, M = U.shape
    P = prob_3pl(theta, a, b, c)
    ll = compute_log_likelihood(U, P)
    k = _count_params(N, M, model_type)
    aic, bic = _compute_aic_bic(ll, k, M)
    
    return {
        "log_likelihood": ll,
        "aic": aic,
        "bic": bic,
        "num_params": k,
        "num_observations": M,
    }


# =========================================================================
# 8. Cổng Kết nối E2E Cao cấp (Unified High-Level Entrypoint API)
# =========================================================================

def run_irt(U: np.ndarray, config: JMLEConfig) -> IrtResult:
    """
    Thực hiện phân tích IRT toàn diện từ E2E, tự động tính Fit Statistics,
    Standard Errors, True Score, Phổ mật độ, và kiểm thử các Tiên đề.
    
    Args:
        U: Ma trận nhị phân 0/1 [N câu × M thí sinh]
        config: Cấu hình JMLEConfig
        
    Returns:
        IrtResult: Chứa đầy đủ các trường dữ liệu phân tích chuẩn hóa
    """
    N, M = U.shape
    
    # 1. Chạy thuật toán JMLE Newton-Raphson
    jmle_res = run_jmle(U, config)
    
    # 2. Hậu xử lý xác suất
    P = prob_3pl(jmle_res.theta, jmle_res.a, jmle_res.b, jmle_res.c)
    
    # 3. Tính Fit Statistics (Infit/Outfit MNSQ)
    fit_stats = compute_fit_statistics(U, P)
    
    # 4. Tính Standard Errors cho items và persons
    se_theta = compute_se_theta(P, jmle_res.a, jmle_res.c)
    se_items = compute_se_items(
        P, jmle_res.theta, jmle_res.a, jmle_res.b, jmle_res.c, 
        config.model_type, config.num_options
    )
    
    # 5. Quy đổi sang True Score thang 10
    ts_10 = compute_true_scores(jmle_res.theta, jmle_res.a, jmle_res.b, jmle_res.c)
    
    # 6. Tính Empirical ICC cho các câu hỏi
    emp_icc = compute_empirical_icc(U, jmle_res.theta)
    
    # 7. Đóng gói kết quả câu hỏi (Items)
    items_out = []
    for i in range(N):
        max_mnsq = max(float(fit_stats['infit_item'][i]), float(fit_stats['outfit_item'][i]))
        fit_flg = classify_fit(max_mnsq)
        
        items_out.append(IrtItemResult(
            item_number=i + 1,
            param_a=float(jmle_res.a[i]),
            param_b=float(jmle_res.b[i]),
            param_c=float(jmle_res.c[i]),
            se_a=float(se_items['se_a'][i]),
            se_b=float(se_items['se_b'][i]),
            se_c=float(se_items['se_c'][i]),
            infit_mnsq=round(float(fit_stats['infit_item'][i]), 4),
            outfit_mnsq=round(float(fit_stats['outfit_item'][i]), 4),
            fit_flag=fit_flg,
            icc_points=compute_icc_points(jmle_res.a[i], jmle_res.b[i], jmle_res.c[i]),
            iif_points=compute_iif_points(jmle_res.a[i], jmle_res.b[i], jmle_res.c[i]),
            empirical_icc=emp_icc[i]
        ))
        
    # 8. Đóng gói kết quả thí sinh (Persons)
    persons_out = []
    for j in range(M):
        persons_out.append(IrtPersonResult(
            index=j + 1,
            raw_score=int(U[:, j].sum()),
            theta_mle=round(float(jmle_res.theta[j]), 4),
            standard_error=round(float(se_theta[j]), 4),
            true_score_10=round(float(ts_10[j]), 2),
            infit_mnsq=round(float(fit_stats['infit_person'][j]), 4),
            outfit_mnsq=round(float(fit_stats['outfit_person'][j]), 4)
        ))
        
    # 9. Tính phổ mật độ năng lực (Wright Map density)
    theta_density = compute_theta_density(jmle_res.theta)
    
    # 10. Đánh giá các tiên đề giả định (Assumptions check)
    assumptions = run_assumption_checks(U, jmle_res.theta, jmle_res.a, jmle_res.b, jmle_res.c)
    
    return IrtResult(
        converged=jmle_res.converged,
        iterations=jmle_res.iterations,
        log_likelihood=round(jmle_res.log_likelihood, 2),
        aic=round(jmle_res.aic, 2),
        bic=round(jmle_res.bic, 2),
        num_params=jmle_res.num_params,
        items=items_out,
        persons=persons_out,
        theta_density=theta_density,
        assumptions=assumptions
    )
