##################
# LOGGING
##################

import logging
import sys

level = logging.WARN
# level = logging.DEBUG
# enabling debug logging causes lag.

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

##################
# DATA CLASSES
##################

from dataclasses import dataclass, fields, _MISSING_TYPE
from typing import Any

@dataclass
class NoneRefersDefault:
    def __post_init__(self):
        # Loop through the fields
        for field in fields(self):
            # If there is a default and the value of the field is none we can assign a value
            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                setattr(self, field.name, field.default)
                

##################
# TORCH
##################
def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

            