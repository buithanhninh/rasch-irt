"""
Custom Exceptions for rasch-irt library
Định nghĩa các ngoại lệ chuyên biệt cho rủi ro toán học và lỗi ma trận dữ liệu.
"""

class RaschIrtError(Exception):
    """Lớp ngoại lệ cha cho tất cả các lỗi trong thư viện rasch-irt"""
    pass

class InvalidMatrixError(RaschIrtError):
    """Ném ra khi ma trận phản hồi đầu vào không hợp lệ hoặc rỗng"""
    pass

class ConvergenceError(RaschIrtError):
    """Ném ra khi thuật toán JMLE không thể hội tụ sau số vòng lặp tối đa"""
    pass

class ZeroVarianceError(RaschIrtError):
    """Ném ra khi câu hỏi hoặc thí sinh có phương sai bằng 0 (làm đúng 100% hoặc sai 100%)"""
    pass
