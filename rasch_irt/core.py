"""
Core Mathematical Module — rasch-irt
Chứa toàn bộ các hàm toán học lõi cho mô hình IRT 1PL/2PL/3PL và hiệu chỉnh JMLE.
"""
import numpy as np
from scipy.stats import beta

D = 1.702  # Hằng số scaling tiệm cận Normal Ogive

# =========================================================================
# 1. Hàm xác suất và Đạo hàm (Probability & Derivatives)
# =========================================================================

def prob_3pl(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Tính ma trận xác suất làm đúng theo mô hình 3PL.
    P_ij = c_i + (1 - c_i) / (1 + e^(-D * a_i * (theta_j - b_i)))
    
    Args:
        theta: Năng lực thí sinh [M]
        a: Độ phân biệt [N]
        b: Độ khó [N]
        c: Đoán mò [N]
    Returns:
        Ma trận xác suất [N × M]
    """
    N = len(b)
    M = len(theta)
    
    # Phát triển chiều ma trận qua broadcasting
    theta_mat = np.tile(theta, (N, 1))  # [N × M]
    a_mat = a[:, np.newaxis]           # [N × 1]
    b_mat = b[:, np.newaxis]           # [N × 1]
    c_mat = c[:, np.newaxis]           # [N × 1]
    
    exponent = -D * a_mat * (theta_mat - b_mat)
    exponent = np.clip(exponent, -30.0, 30.0)  # Chống tràn số exponent
    
    P = c_mat + (1.0 - c_mat) / (1.0 + np.exp(exponent))
    return np.clip(P, 1e-10, 1.0 - 1e-10)     # Kẹp biên chống chia cho 0


def score_theta(U: np.ndarray, P: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính đạo hàm cấp 1 theo theta (Score function) cho mỗi thí sinh [M]"""
    a_mat = a[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    # Công thức: sum_i [ D * a_i * (U_ij - P_ij) * (P_ij - c_i) / ((1 - c_i) * P_ij) ]
    term = D * a_mat * (U - P) * (P - c_mat) / (np.maximum(1.0 - c_mat, 1e-5) * P)
    return term.sum(axis=0)


def info_theta(P: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính Fisher Information (đạo hàm cấp 2 đổi dấu) theo theta cho mỗi thí sinh [M]"""
    a_mat = a[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    # Công thức: sum_i [ D^2 * a_i^2 * (P_ij - c_i)^2 * (1 - P_ij) / ((1 - c_i)^2 * P_ij) ]
    denom = np.maximum(1.0 - c_mat, 1e-5) ** 2
    term = (D**2 * a_mat**2 * (P - c_mat)**2 * (1.0 - P)) / (denom * P)
    return term.sum(axis=0)


def score_b(U: np.ndarray, P: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính đạo hàm cấp 1 theo b (Score function) cho mỗi câu hỏi [N]"""
    a_mat = a[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    # Công thức: sum_j [ -D * a_i * (U_ij - P_ij) * (P_ij - c_i) / ((1 - c_i) * P_ij) ]
    term = -D * a_mat * (U - P) * (P - c_mat) / (np.maximum(1.0 - c_mat, 1e-5) * P)
    return term.sum(axis=1)


def info_b(P: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính Fisher Information theo b cho mỗi câu hỏi [N]"""
    a_mat = a[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    denom = np.maximum(1.0 - c_mat, 1e-5) ** 2
    term = (D**2 * a_mat**2 * (P - c_mat)**2 * (1.0 - P)) / (denom * P)
    return term.sum(axis=1)


def score_a(U: np.ndarray, P: np.ndarray, theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính đạo hàm cấp 1 theo a cho mỗi câu hỏi [N]"""
    N = len(b)
    theta_mat = np.tile(theta, (N, 1))
    b_mat = b[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    # Công thức: sum_j [ D * (theta_j - b_i) * (U_ij - P_ij) * (P_ij - c_i) / ((1 - c_i) * P_ij) ]
    term = D * (theta_mat - b_mat) * (U - P) * (P - c_mat) / (np.maximum(1.0 - c_mat, 1e-5) * P)
    return term.sum(axis=1)


def info_a(P: np.ndarray, theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tính Fisher Information theo a cho mỗi câu hỏi [N]"""
    N = len(b)
    theta_mat = np.tile(theta, (N, 1))
    b_mat = b[:, np.newaxis]
    c_mat = c[:, np.newaxis]
    
    denom = np.maximum(1.0 - c_mat, 1e-5) ** 2
    term = (D**2 * (theta_mat - b_mat)**2 * (P - c_mat)**2 * (1.0 - P)) / (denom * P)
    return term.sum(axis=1)

# =========================================================================
# 2. Cấu hình Beta Prior cho c (Guessing Parameter)
# =========================================================================

def get_beta_params(num_options: int) -> tuple[float, float]:
    """Trả về tham số alpha, beta cho prior của c dựa trên số phương án lựa chọn"""
    if num_options == 2:
        return 10.0, 10.0   # E[c] = 0.50
    elif num_options == 3:
        return 6.6, 13.4    # E[c] = 0.33
    elif num_options == 4:
        return 5.0, 15.0    # E[c] = 0.25 (Lý tưởng cho trắc nghiệm A,B,C,D)
    elif num_options == 5:
        return 4.0, 16.0    # E[c] = 0.20
    else:
        return 5.0, 15.0


def score_c_map(U: np.ndarray, P: np.ndarray, c: np.ndarray, alpha_prior: float, beta_prior: float) -> np.ndarray:
    """Tính đạo hàm cấp 1 MAP (bổ sung Beta Prior) cho tham số c [N]"""
    c_mat = c[:, np.newaxis]
    
    # Likelihood term: sum_j [ (U_ij - P_ij) / ((1 - c_i) * P_ij) ]
    likelihood_term = (U - P) / (np.maximum(1.0 - c_mat, 1e-5) * P)
    L_c = likelihood_term.sum(axis=1)
    
    # Beta Prior log derivative: (alpha - 1)/c - (beta - 1)/(1-c)
    safe_c = np.maximum(c, 1e-5)
    safe_1_c = np.maximum(1.0 - c, 1e-5)
    prior_term = (alpha_prior - 1.0) / safe_c - (beta_prior - 1.0) / safe_1_c
    
    return L_c + prior_term


def info_c_map(P: np.ndarray, c: np.ndarray, alpha_prior: float, beta_prior: float) -> np.ndarray:
    """Tính Fisher Information MAP cho c [N]"""
    c_mat = c[:, np.newaxis]
    
    # Likelihood term: sum_j [ (1 - P_ij) / ((1 - c_i)^2 * P_ij) ]
    denom = np.maximum(1.0 - c_mat, 1e-5) ** 2
    likelihood_term = (1.0 - P) / (denom * P)
    I_c = likelihood_term.sum(axis=1)
    
    # Beta Prior second derivative (negative): (alpha - 1)/c^2 + (beta - 1)/(1-c)^2
    safe_c = np.maximum(c, 1e-5)
    safe_1_c = np.maximum(1.0 - c, 1e-5)
    prior_term = (alpha_prior - 1.0) / (safe_c**2) + (beta_prior - 1.0) / (safe_1_c**2)
    
    return I_c + prior_term

# =========================================================================
# 3. Newton-Raphson Update Step
# =========================================================================

def newton_raphson_batch(
    scores: np.ndarray, 
    infos: np.ndarray,
    current_vals: np.ndarray,
    val_min: float = -10.0, 
    val_max: float = 10.0
) -> np.ndarray:
    """Batch Newton-Raphson nâng cao cho nhiều tham số cùng lúc"""
    safe_infos = np.where(np.abs(infos) < 1e-15, 1.0, infos)
    deltas = scores / safe_infos
    deltas = np.clip(deltas, -3.0, 3.0)  # Cực kỳ quan trọng: Damping tránh nhảy quá đà
    new_vals = current_vals + deltas
    return np.clip(new_vals, val_min, val_max)

# =========================================================================
# 4. Hậu xử lý (Fit statistics & Standard Errors & True Score)
# =========================================================================

def compute_fit_statistics(U: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính chỉ số Infit và Outfit MNSQ cho từng câu hỏi"""
    N, M = U.shape
    
    W = P * (1.0 - P)  # Variance matrix [N × M]
    safe_W = np.maximum(W, 1e-10)
    
    # Standardized residuals: z^2 = (U - P)^2 / W
    z_sq = (U - P)**2 / safe_W
    
    # Outfit MNSQ
    outfit = z_sq.mean(axis=1)
    
    # Infit MNSQ
    infit = (z_sq * W).sum(axis=1) / np.maximum(W.sum(axis=1), 1e-10)
    
    # Gán cờ Fit
    fit_flags = []
    for i in range(N):
        inf = infit[i]
        out = outfit[i]
        if inf > 1.3 or out > 1.3:
            fit_flags.append("Underfit")
        elif inf < 0.7 or out < 0.7:
            fit_flags.append("Overfit")
        else:
            fit_flags.append("Fit")
            
    return infit, outfit, np.array(fit_flags)


def compute_standard_errors(P: np.ndarray, a: np.ndarray, c: np.ndarray, theta: np.ndarray, model_type: int) -> tuple:
    """Tính sai số tiêu chuẩn SE cho theta và các tham số a, b, c từ Fisher Information"""
    N, M = P.shape
    
    # 1. SE of Theta [M]
    it = info_theta(P, a, c)
    se_theta = 1.0 / np.sqrt(np.maximum(it, 1e-5))
    
    # 2. SE of Parameters [N]
    ib = info_b(P, a, c)
    se_b = 1.0 / np.sqrt(np.maximum(ib, 1e-5))
    
    se_a = np.zeros(N)
    se_c = np.zeros(N)
    
    if model_type >= 2:
        ia = info_a(P, theta, a, b=np.zeros(N), c=c)  # b=0 do đã standardize
        se_a = 1.0 / np.sqrt(np.maximum(ia, 1e-5))
        
    if model_type == 3:
        # MAP Prior parameters
        alpha_p, beta_p = get_beta_params(4)
        ic = info_c_map(P, c, alpha_p, beta_p)
        se_c = 1.0 / np.sqrt(np.maximum(ic, 1e-5))
        
    return se_theta, se_a, se_b, se_c


def theta_to_true_score(theta: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Quy đổi năng lực theta sang thang điểm 10 thực tế dựa trên đường cong TIF"""
    N = len(b)
    P = prob_3pl(theta, a, b, c)
    
    # Điểm thực lý thuyết = sum_i P_ij
    expected_scores = P.sum(axis=0)  # [M]
    
    # Ánh xạ tuyến tính về thang 10
    true_scores = (expected_scores / N) * 10.0
    return np.clip(true_scores, 0.0, 10.0)
