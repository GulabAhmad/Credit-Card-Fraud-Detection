import warnings

# Harmless joblib/loky warning on Windows about counting physical CPU cores.
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
