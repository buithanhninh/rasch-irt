"""
Scoring Module — score_responses
Tự động đối chiếu phản hồi của thí sinh với đáp án đúng để tạo ma trận nhị phân 0/1.
"""
import numpy as np
from .exceptions import InvalidMatrixError

def score_responses(
    raw_responses: np.ndarray, 
    answer_key: np.ndarray,
    missing_value: str = ""
) -> np.ndarray:
    """
    Tự động đối chiếu phản hồi học sinh với đáp án để chấm điểm.
    
    Args:
        raw_responses: Ma trận phản hồi thô [N câu hỏi × M thí sinh] (ví dụ: 'A', 'B', 'C' hoặc '1', '2')
        answer_key: Mảng đáp án chuẩn [N câu hỏi]
        missing_value: Ký tự đại diện cho câu bỏ trống
        
    Returns:
        U: Ma trận nhị phân 0/1 [N câu hỏi × M thí sinh] (1 = Đúng, 0 = Sai hoặc bỏ trống)
    """
    raw_responses = np.atleast_2d(raw_responses)
    answer_key = np.atleast_1d(answer_key)
    
    N, M = raw_responses.shape
    if N != len(answer_key):
        raise InvalidMatrixError(
            f"Kích thước ma trận không khớp: Số câu hỏi trong ma trận phản hồi ({N}) "
            f"phải bằng số câu hỏi trong đáp án đúng ({len(answer_key)})."
        )
        
    U = np.zeros((N, M), dtype=np.int32)
    
    for i in range(N):
        key = str(answer_key[i]).strip().upper()
        for j in range(M):
            student_ans = str(raw_responses[i, j]).strip().upper()
            if student_ans == missing_value.strip().upper() or not student_ans:
                U[i, j] = 0
            elif student_ans == key:
                U[i, j] = 1
            else:
                U[i, j] = 0
                
    return U
