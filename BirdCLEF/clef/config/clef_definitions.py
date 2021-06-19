from typing import Optional
import dataclasses

import clef
from clef.config import base_definitions

DATA_NUM_PER_TEST_CYCLE = clef.constant.TEST_STRIDE_SEC * clef.constant.AUDIO_HELTZ


@dataclasses.dataclass
class SpectrogramConfig:
    num_fft: int = 512
    fft_window: int = 1600
    fft_stride: int = 1600


class TaskConfig(base_definitions.TaskConfig):
    spectrogram: SpectrogramConfig = SpectrogramConfig()
    num_folds: int = 4
    random_state: Optional[int] = None
    input_shape = (int(spectrogram.num_fft / 2) + 1,
                   int(DATA_NUM_PER_TEST_CYCLE / spectrogram.fft_window), 1)
