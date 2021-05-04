import dataclasses

from clef.config import base_definitions


@dataclasses.dataclass
class SpectrogramConfig:
    num_fft: int = 512
    fft_window: int = 1600
    fft_stride: int = 1600


class TaskConfig(base_definitions.TaskConfig):
    spectrogram: SpectrogramConfig = SpectrogramConfig()
