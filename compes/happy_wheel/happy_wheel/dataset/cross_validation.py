"""cross_validation.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .. import logger


def create_cross_validation_dataset(train_csv: Path, num_fold: int, output_dir: Path):
    """Create CV dataset.
    Args:
        train_csv (Path): Path to csv which records filename and class label.
            Csv file must have column of name "image" (file basename) and "species" (label).
        num_fold (int) : Number of CV folds.
    """
    assert train_csv.is_file()

    df: pd.DataFrame = pd.read_csv(train_csv)

    skf = StratifiedKFold(n_splits=num_fold)
    for idx, (train, test) in enumerate(skf.split(df, df.species)):
        df.iloc[train].to_csv(output_dir / f"train_cv{idx}.csv", index=False, encoding="utf-8")
        df.iloc[test].to_csv(output_dir / f"valid_cv{idx}.csv", index=False, encoding="utf-8")
    logger.info(f"Save csvs to {output_dir}")
