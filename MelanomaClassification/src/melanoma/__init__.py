VERSION = (0, 0, 0)
__version__ = ".".join(["{}".format(x) for x in VERSION])

from melanoma import constants
from melanoma import functions

from melanoma import augmentations
from melanoma import extensions
from melanoma import evaluator
from melanoma import trainer
from melanoma import models
from melanoma import dataset
from melanoma import evaluate
from melanoma.models import predictor
from melanoma.models import loss
