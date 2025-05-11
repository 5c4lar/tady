
try:
    # Import the C++ extension
    import tady_cpp as cpp
except ImportError as e:
    import warnings
    warnings.warn("C++ extension could not be imported. Some functionality may be limited.")
    warnings.warn(e.msg)
