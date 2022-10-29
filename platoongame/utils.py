##################
# LOGGING
##################

from datetime import datetime
import logging

level = logging.WARN
# level = logging.DEBUG

# handler = logging.StreamHandler(sys.stdout)
# enabling debug logging causes lag using streamhandler
handler = logging.FileHandler("./logs.log")
handler.setLevel(level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def get_prefix() -> str:
    return datetime.now().strftime("%Y-%m-%d %H%M-%S %f")

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
import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://stackoverflow.com/a/63630138/11141271
def dec2bin(x: torch.Tensor, bits: int):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b: torch.Tensor, bits: int):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)
