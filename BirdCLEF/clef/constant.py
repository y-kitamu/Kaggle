from pathlib import Path

TEST_STRIDE_SEC = 5
AUDIO_HELTZ = 32000

PROJECT_ROOT_PATH = Path(__file__).absolute().parents[1]
DATA_PATH = PROJECT_ROOT_PATH / "data"

# 元データの場所
RAW_DATA_PATH = DATA_PATH / "raw"
TRAIN_METADATA_CSV_PATH = RAW_DATA_PATH / "train_metadata.csv"
TRAIN_SOUNDSCAPE_CSV_PATH = RAW_DATA_PATH / "train_soundscape_labels.csv"
TRAIN_SHORT_AUDIO_PATH = RAW_DATA_PATH / "train_short_audio"
TRAIN_SOUNDSCAPE_PATH = RAW_DATA_PATH / "train_soundscape"

# 処理済みデータの場所
PREPROC_DATA_PATH = DATA_PATH / "preprocessed"
PREPROC_DATA_PATH.mkdir(parents=True, exist_ok=True)
