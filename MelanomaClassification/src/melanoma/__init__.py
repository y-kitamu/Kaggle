VERSION = (0, 0, 0)
__version__ = ".".join(["{}".format(x) for x in VERSION])

from melanoma import constants
from melanoma import utility

from melanoma import trainer
from melanoma import models
from melanoma import dataset
from melanoma import evaluate
from melanoma import predictor
