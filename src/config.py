from dataclasses import dataclass
from pathlib import Path
import yaml 

@dataclass
class Config:
    seed: int
    data_dir: str
    num_workers: int
    epochs: int
    batch_size: int
    lr: float
    run_name: str
    out_dir: str
    