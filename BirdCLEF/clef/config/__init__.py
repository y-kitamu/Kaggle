from clef.config import base_definitions
from clef.config import clef_definitions

import os
from typing import Any, Optional

from hydra.experimental import compose, initialize
from omegaconf import OmegaConf, DictConfig


def read_cfg(filename: str) -> DictConfig:
    """`filename`で指定したyamlファイルの設定を読み込む
    Args:
        filename (str) : 読み込む設定ファイル
    Return:
        cfg (omegaconf.DictConfig) :
    """
    cur_dir = os.getcwd()
    file_dir = os.path.dirname(filename)
    rel_file_dir = os.path.relpath(file_dir, cur_dir)
    with initialize(config_path=rel_file_dir):
        cfg = compose(config_name=os.path.basename(filename).split(".")[0])
        # print(OmegaConf.to_yaml(cfg))
    return cfg


def export_dataclass_to_yaml(config: Any, output_fname: Optional[str] = None) -> None:
    """dataclassオブジェクトをyaml形式に出力する
    Args:
        config (object) : dataclass形式のオブジェクト
        output (str)  : 出力ファイル名。Noneの場合、stdoutに出力
    Output:
        yaml file (output_fname) : `config`の設定を書き込んだyamlファイル
    """
    conf = OmegaConf.structured(config)

    if output_fname is not None:
        with open(output_fname, 'w') as f:
            print(OmegaConf.to_yaml(conf), file=f)
    else:
        print(OmegaConf.to_yaml(conf))
