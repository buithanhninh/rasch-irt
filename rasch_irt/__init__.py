"""
rasch-irt — Thư viện Đo lường Khảo thí Cổ điển & Hiện đại (CTT & IRT)
Dành cho ứng dụng chấm thi và phân tích đánh giá chất lượng giáo dục Rasch.vn.
"""

from .exceptions import (
    RaschIrtError,
    InvalidMatrixError,
    ConvergenceError,
    ZeroVarianceError,
)

from .scoring import (
    score_responses,
)

from .ctt import (
    CttItemResult,
    CttResult,
    run_ctt,
)

from .mml_em import (
    MMLConfig,
    run_mml_em,
)

from .irt import (
    JMLEConfig,
    JMLEResult,
    IrtItemResult,
    IrtPersonResult,
    AssumptionResult,
    IrtResult,
    ModelFitResult,
    AutoFitResult,
    run_jmle,
    run_irt,
    run_auto_fit,
    compute_model_fit,
    classify_fit,
    compute_fit_statistics,
    run_assumption_checks,
    compute_se_theta,
    compute_se_items,
    compute_true_scores,
    compute_empirical_icc,
    compute_theta_density,
)

__version__ = "1.1.0"
__author__ = "Bùi Thành Ninh"

__all__ = [
    # Exceptions
    "RaschIrtError",
    "InvalidMatrixError",
    "ConvergenceError",
    "ZeroVarianceError",
    
    # Scoring
    "score_responses",
    
    # CTT
    "CttItemResult",
    "CttResult",
    "run_ctt",
    
    # IRT — MML-EM (2PL/3PL)
    "MMLConfig",
    "run_mml_em",
    
    # IRT — JMLE (1PL) & Unified API
    "JMLEConfig",
    "JMLEResult",
    "IrtItemResult",
    "IrtPersonResult",
    "AssumptionResult",
    "IrtResult",
    "ModelFitResult",
    "AutoFitResult",
    "run_jmle",
    "run_irt",
    "run_auto_fit",
    "compute_model_fit",
    "classify_fit",
    "compute_fit_statistics",
    "run_assumption_checks",
    "compute_se_theta",
    "compute_se_items",
    "compute_true_scores",
    "compute_empirical_icc",
    "compute_theta_density",
]
