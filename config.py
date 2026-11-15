STORE = "CA_3"
DEPT = "FOODS_3"
ITEMS = {"low": None, "medium": None, "high": None}  # filled in once picked

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

TRAIN_DAYS = 1800
ZERO_RUN_LEN = 7
DEMAND_PCTL = 99
TARGET_MEANS = {"low": 2, "medium": 8, "high": 20}

UNIT_COST_FRAC = 0.6
HOLD_FRAC = 0.02
ORDER_COST_FRAC = 2.0
LOST_FRAC = 0.2

GAMMA = 0.95
VI_TOL = 1e-6

Q_ALPHA = 0.1
Q_EPS_START, Q_EPS_END = 1.0, 0.05
Q_PASSES = 200

DATA_SIZES = [60, 180, 365, 900, 1800]