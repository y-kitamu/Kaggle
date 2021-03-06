import glob
from typing import TYPE_CHECKING, List

import clef
from clef.constant import PREPROC_DATA_PATH

if TYPE_CHECKING:
    from pathlib import Path
    from clef.config.base_definitions import DataConfig


def get_datadir_path(config: "DataConfig") -> "Path":
    """`config`に設定されている値に従って、tfrecords保管用のディレクトリ名を取得する。
    Args:
        config (DataConfig) :
    Return:
        dirpath (Path) :
    """
    dirpath = PREPROC_DATA_PATH / config.name
    return dirpath


def get_tfrecords_files(config: "DataConfig") -> List[str]:
    """`config`に設定されている値に従って、tfrecordsファイルを取得する
    """
    dirpath = get_tfrecords_dirpath(config)
    file_list = list(glob.glob(str(dirpath / "{}*.tfrecords".format(config.tfrecords_basename))))
    for fname in file_list:
        clef.logger.debug("  {}".format(fname))
    return file_list
