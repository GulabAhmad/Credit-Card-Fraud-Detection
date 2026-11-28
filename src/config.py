"""Project-wide settings: paths, seed and the business goal (SRS sections 2.4, 5.3)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET = "Class"
SCALE_COLS = ["Time", "Amount"]  # V1-V28 are already PCA-scaled

# Business goal (agreed): catch at least 80% of fraud, and at least 70% of
# alerts must be real fraud. Primary model-quality metric is PR-AUC; the final
# block/approve decision comes from the minimum-cost threshold.
TARGET_RECALL = 0.80
TARGET_PRECISION = 0.70
PRIMARY_METRIC = "average_precision"  # = PR-AUC

# Cost model (illustrative assumption, stated in the presentation):
# a missed fraud costs its full Amount; a false alarm costs C_FP.
C_FP = 5.0

# Stop rule: if 2-3 experiments improve PR-AUC by less than this, stop tuning.
MIN_PR_AUC_GAIN = 0.01
