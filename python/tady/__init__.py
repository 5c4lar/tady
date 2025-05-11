"""
Tady: A Neural Disassembler without Consistency Violations
"""
# Add version information
__version__ = "0.1.0"

try:
    # Import the C++ extension
    import tady_cpp as cpp
except ImportError as e:
    import warnings
    warnings.warn("C++ extension could not be imported. Some functionality may be limited.")
    warnings.warn(e.msg)
