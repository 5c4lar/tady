
try:
    # Import the C++ extension
    import tady_cpp as cpp
except ImportError:
    import warnings
    warnings.warn("C++ extension could not be imported. Some functionality may be limited.")
