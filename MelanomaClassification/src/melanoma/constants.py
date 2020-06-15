import enum
from pathlib import Path

PROJECT_ROOT = Path.home() / "work" / "Kaggle" / "MelanomaClassification"
RESULT_DIR = PROJECT_ROOT / "results"


class Labels(enum.Enum):
    benign = 0
    malignant = 1
