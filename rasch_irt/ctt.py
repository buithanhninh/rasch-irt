"""
CTT Module — Classical Test Theory Analysis
Chạy phân tích khảo thí cổ điển độc lập hoặc trước IRT để sàng lọc câu hỏi.
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from scipy.stats import norm, skew, kurtosis
from .exceptions import InvalidMatrixError


@dataclass
class CttItemResult:
    """Kết quả CTT cho 1 câu hỏi"""
    item_number: int
    difficulty: float  # p — tỷ lệ trả lời đúng
    discrimination_d: float  # D — (upper - lower) / N_group
    point_biserial: float  # rpb (Corrected Point-Biserial)
    biserial: Optional[float] = None
    distractor_analysis: Optional[dict] = None
    quality_flag: str = "Tốt"
    excluded_from_irt: bool = False
    exclusion_reason: Optional[str] = None


@dataclass
class CttResult:
    """Kết quả CTT toàn bộ"""
    items: list[CttItemResult] = field(default_factory=list)
    # Test-level statistics
    kr20: float = 0.0
    cronbach_alpha: float = 0.0
    sem: float = 0.0
    mean_score: float = 0.0
    std_dev: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    total_students: int = 0
    total_items: int = 0
    # Sanity check
    sanity_removed_items: list[int] = field(default_factory=list)
    bad_items: list[int] = field(default_factory=list)  # D < 0 hoặc rpb < 0


def sanity_check(U: np.ndarray) -> tuple[np.ndarray, list[int], list[str]]:
    """
    Giai đoạn 1: Kiểm tra tính toàn vẹn dữ liệu.
    Loại bỏ câu hỏi có p=0%, p=100%, hoặc phương sai = 0.
    
    Args:
        U: Ma trận nhị phân [N câu × M thí sinh]
    
    Returns:
        (U_clean, removed_indices, reasons)
    """
    N, M = U.shape
    removed = []
    reasons = []
    
    p = U.mean(axis=1)  # [N]
    variance = U.var(axis=1)  # [N]
    
    for i in range(N):
        if p[i] == 0.0:
            removed.append(i)
            reasons.append("p=0")
        elif p[i] == 1.0:
            removed.append(i)
            reasons.append("p=1")
        elif variance[i] < 1e-10:
            removed.append(i)
            reasons.append("var=0")
    
    # Giữ lại các câu hợp lệ
    valid_mask = np.ones(N, dtype=bool)
    valid_mask[removed] = False
    U_clean = U[valid_mask]
    
    return U_clean, removed, reasons


def compute_difficulty(U: np.ndarray) -> np.ndarray:
    """Tính độ khó p = tỷ lệ trả lời đúng cho mỗi câu"""
    return U.mean(axis=1)


def compute_discrimination_d(U: np.ndarray) -> np.ndarray:
    """
    Tính độ phân biệt D bằng phương pháp 27% upper-lower.
    D = (P_upper - P_lower), giá trị [-1, 1]
    """
    N, M = U.shape
    total_scores = U.sum(axis=0)  # [M] — tổng điểm mỗi thí sinh
    
    # Sắp xếp thí sinh theo điểm
    sorted_indices = np.argsort(total_scores)
    
    # Lấy 27% cao nhất và thấp nhất
    n_group = max(int(M * 0.27), 1)
    lower_group = sorted_indices[:n_group]
    upper_group = sorted_indices[-n_group:]
    
    # P_upper và P_lower cho mỗi câu
    p_upper = U[:, upper_group].mean(axis=1)
    p_lower = U[:, lower_group].mean(axis=1)
    
    D = p_upper - p_lower
    return D


def compute_point_biserial(U: np.ndarray) -> np.ndarray:
    """
    Tính tương quan Point-biserial HIỆU CHỈNH (Corrected Item-Total Correlation)
    cho mỗi câu hỏi.

    Corrected rpb: loại bỏ chính câu i khỏi tổng điểm trước khi tính,
    tránh hiện tượng chồng lấn phần tử (part-whole contamination).
    Tham chiếu: Henrysson (1963), chuẩn SPSS Reliability Analysis.

    rpb_corrected = (M1_corr - M0_corr) / SD_corrected * sqrt(p * q)
    trong đó corrected_scores = total_scores - U[i]
    """
    N, M = U.shape
    total_scores = U.sum(axis=0).astype(float)  # [M]

    rpb = np.zeros(N)
    for i in range(N):
        correct_mask = U[i] == 1
        incorrect_mask = U[i] == 0

        n1 = correct_mask.sum()
        n0 = incorrect_mask.sum()

        if n1 == 0 or n0 == 0 or n1 + n0 < 3:
            rpb[i] = 0.0
            continue

        # Corrected: loại bỏ điểm của chính câu i khỏi tổng điểm
        corrected_scores = total_scores - U[i].astype(float)
        corrected_std = corrected_scores.std()
        if corrected_std < 1e-10:
            rpb[i] = 0.0
            continue

        mean_correct = corrected_scores[correct_mask].mean()
        mean_incorrect = corrected_scores[incorrect_mask].mean()
        p = n1 / (n1 + n0)
        q = 1 - p

        rpb[i] = (mean_correct - mean_incorrect) / corrected_std * np.sqrt(p * q)

    return rpb


def compute_biserial(U: np.ndarray) -> np.ndarray:
    """Tính tương quan Biserial từ Point-biserial"""
    N, M = U.shape
    p = U.mean(axis=1)
    p = np.clip(p, 0.01, 0.99)
    
    rpb = compute_point_biserial(U)
    
    # Biserial = rpb * sqrt(p*q) / ordinate(z)
    z = norm.ppf(p)
    ordinate = norm.pdf(z)
    ordinate = np.maximum(ordinate, 1e-10)
    
    biserial = rpb * np.sqrt(p * (1 - p)) / ordinate
    # Kẹp về [-1, 1] để tránh giá trị vượt ngưỡng vật lý khi p cực đoan
    return np.clip(biserial, -1.0, 1.0)


def compute_distractor_analysis(
    raw_responses: np.ndarray,
    answer_key: np.ndarray,
    num_options: int = 4
) -> list[dict]:
    """
    Phân tích phương án nhiễu cho mỗi câu.
    
    Args:
        raw_responses: Ma trận phản hồi gốc [N câu × M thí sinh] (giá trị 1-based: 1=A, 2=B, ...) hoặc (chữ: 'A', 'B', ...)
        answer_key: Đáp án đúng [N]
        num_options: Số phương án
    
    Returns:
        List of dicts, mỗi dict chứa thông tin phân tích cho 1 câu
    """
    N, M = raw_responses.shape
    total_scores = np.zeros(M)
    
    # Chuẩn hóa đáp án và phản hồi sang dạng viết hoa
    raw_upper = np.vectorize(lambda x: str(x).strip().upper())(raw_responses)
    key_upper = np.vectorize(lambda x: str(x).strip().upper())(answer_key)
    
    # Tính điểm tổng dựa trên đáp án
    for i in range(N):
        total_scores += (raw_upper[i] == key_upper[i]).astype(float)
    
    # 27% upper-lower groups
    sorted_idx = np.argsort(total_scores)
    n_group = max(int(M * 0.27), 1)
    lower_idx = sorted_idx[:n_group]
    upper_idx = sorted_idx[-n_group:]
    
    # Định nghĩa nhãn phương án
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:num_options]
    
    # Định nghĩa giá trị so khớp
    # Hỗ trợ cả định dạng chữ ('A', 'B') và số 1-based ('1', '2')
    results = []
    for i in range(N):
        item_analysis = {}
        for opt_idx, label in enumerate(option_labels):
            opt_val_num = str(opt_idx + 1)
            opt_val_char = label
            
            # Kiểm tra xem đáp án có khớp với opt_val_num hoặc opt_val_char không
            is_key = (key_upper[i] == opt_val_num) or (key_upper[i] == opt_val_char)
            
            upper_count = int(((raw_upper[i, upper_idx] == opt_val_num) | (raw_upper[i, upper_idx] == opt_val_char)).sum())
            lower_count = int(((raw_upper[i, lower_idx] == opt_val_num) | (raw_upper[i, lower_idx] == opt_val_char)).sum())
            total_count = int(((raw_upper[i] == opt_val_num) | (raw_upper[i] == opt_val_char)).sum())
            
            item_analysis[label] = {
                "upper": upper_count,
                "lower": lower_count,
                "total": total_count,
                "proportion": round(total_count / max(M, 1), 4),
                "is_key": is_key,
                "discrimination": round((upper_count - lower_count) / max(n_group, 1), 4)
            }
        results.append(item_analysis)
    
    return results


def compute_reliability(U: np.ndarray) -> dict:
    """
    Tính các chỉ số độ tin cậy: KR-20, Cronbach Alpha, SEM.
    """
    N, M = U.shape
    total_scores = U.sum(axis=0).astype(float)
    
    mean_score = float(total_scores.mean())
    std_dev = float(max(total_scores.std(ddof=0), 1e-10))  # Population SD
    variance_total = std_dev ** 2
    
    # KR-20
    p = U.mean(axis=1)
    sum_pq = float(np.sum(p * (1 - p)))
    kr20 = (N / max(N - 1, 1)) * (1 - sum_pq / max(variance_total, 1e-10))
    kr20 = float(np.clip(kr20, -1, 1))
    
    # Cronbach Alpha (tương đương KR-20 cho biến nhị phân)
    item_variances = U.var(axis=1, ddof=0)  # Population variance
    sum_item_var = float(item_variances.sum())
    alpha = (N / max(N - 1, 1)) * (1 - sum_item_var / max(variance_total, 1e-10))
    alpha = float(np.clip(alpha, -1, 1))
    
    # SEM
    sem = std_dev * np.sqrt(1 - max(kr20, 0))
    
    # Skewness & Kurtosis
    skewness_val = float(skew(total_scores))
    kurtosis_val = float(kurtosis(total_scores))
    
    return {
        "kr20": round(kr20, 4),
        "cronbach_alpha": round(alpha, 4),
        "sem": round(float(sem), 4),
        "mean_score": round(mean_score, 2),
        "std_dev": round(std_dev, 4),
        "skewness": round(skewness_val, 4),
        "kurtosis": round(kurtosis_val, 4),
    }


def classify_ctt_item(p: float, d: float, rpb: float) -> str:
    """Phân loại chất lượng câu hỏi dựa trên CTT"""
    if d < 0:
        return "Loại"
    if rpb < 0.1:
        return "Kém"
    if p < 0.1 or p > 0.9:
        return "Kém"
    if rpb < 0.2:
        return "Trung bình"
    if d < 0.2:
        return "Trung bình"
    return "Tốt"


def run_ctt(
    U: np.ndarray,
    raw_responses: Optional[np.ndarray] = None,
    answer_key: Optional[np.ndarray] = None,
    num_options: int = 4
) -> CttResult:
    """
    Chạy phân tích CTT toàn diện.
    
    Args:
        U: Ma trận nhị phân [N câu × M thí sinh] (đã chấm điểm)
        raw_responses: Ma trận phản hồi gốc (tùy chọn, cho distractor analysis)
        answer_key: Đáp án đúng (tùy chọn)
        num_options: Số phương án
    
    Returns:
        CttResult với đầy đủ các thống kê CTT
    """
    U = np.atleast_2d(U)
    N_orig, M = U.shape
    if N_orig == 0 or M == 0:
        raise InvalidMatrixError("Ma trận nhị phân đầu vào không được rỗng.")
        
    result = CttResult()
    result.total_students = M
    result.total_items = N_orig
    
    # Giai đoạn 1: Sanity Check
    U_clean, removed_indices, removed_reasons = sanity_check(U)
    result.sanity_removed_items = removed_indices
    
    N_clean = U_clean.shape[0]
    
    # Mapping: index trong U_clean → item_number gốc (1-based)
    valid_indices = [i for i in range(N_orig) if i not in removed_indices]
    
    # Giai đoạn 2: Tính CTT cho các câu hợp lệ
    p = compute_difficulty(U_clean)
    D = compute_discrimination_d(U_clean)
    rpb = compute_point_biserial(U_clean)
    
    try:
        biserial = compute_biserial(U_clean)
    except Exception:
        biserial = np.zeros(N_clean)
    
    # Distractor analysis (nếu có dữ liệu gốc)
    distractor_data = None
    if raw_responses is not None and answer_key is not None:
        raw_responses = np.atleast_2d(raw_responses)
        answer_key = np.atleast_1d(answer_key)
        # Lọc raw_responses theo valid_indices
        raw_clean = raw_responses[valid_indices]
        key_clean = answer_key[valid_indices]
        distractor_data = compute_distractor_analysis(raw_clean, key_clean, num_options)
    
    # Build item results
    bad_items = []
    for idx in range(N_clean):
        item_num = valid_indices[idx] + 1  # 1-based
        flag = classify_ctt_item(float(p[idx]), float(D[idx]), float(rpb[idx]))
        
        excluded = False
        exclusion_reason = None
        
        # Lọc nghiêm ngặt: D < 0 hoặc rpb < 0 đều vi phạm tiên đề đơn điệu của IRT
        if D[idx] < 0 and rpb[idx] < 0:
            excluded = True
            exclusion_reason = "D<0, rpb<0"
            bad_items.append(item_num)
        elif D[idx] < 0:
            excluded = True
            exclusion_reason = "D<0"
            bad_items.append(item_num)
        elif rpb[idx] < 0:
            excluded = True
            exclusion_reason = "rpb<0"
            bad_items.append(item_num)
        
        item_result = CttItemResult(
            item_number=item_num,
            difficulty=round(float(p[idx]), 4),
            discrimination_d=round(float(D[idx]), 4),
            point_biserial=round(float(rpb[idx]), 4),
            biserial=round(float(biserial[idx]), 4) if biserial is not None else None,
            distractor_analysis=distractor_data[idx] if distractor_data else None,
            quality_flag=flag,
            excluded_from_irt=excluded,
            exclusion_reason=exclusion_reason,
        )
        result.items.append(item_result)
    
    # Thêm các câu bị sanity-removed
    for rem_idx, reason in zip(removed_indices, removed_reasons):
        item_num = rem_idx + 1
        result.items.append(CttItemResult(
            item_number=item_num,
            difficulty=0.0 if reason == "p=0" else 1.0 if reason == "p=1" else 0.5,
            discrimination_d=0.0,
            point_biserial=0.0,
            quality_flag="Loại",
            excluded_from_irt=True,
            exclusion_reason=reason,
        ))
    
    # Sắp xếp lại theo item_number
    result.items.sort(key=lambda x: x.item_number)
    result.bad_items = bad_items
    
    # Giai đoạn 3: Reliability (trên dữ liệu sạch)
    if N_clean >= 3 and M >= 3:
        reliability = compute_reliability(U_clean)
        result.kr20 = reliability["kr20"]
        result.cronbach_alpha = reliability["cronbach_alpha"]
        result.sem = reliability["sem"]
        result.mean_score = reliability["mean_score"]
        result.std_dev = reliability["std_dev"]
        result.skewness = reliability["skewness"]
        result.kurtosis = reliability["kurtosis"]
    
    return result
