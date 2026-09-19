"""Scientific settings and execution options for the PRNN study."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np

SPLIT_SEED = 20260908
SEARCH_SEED = 42
SEEDS = (17, 42)
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
LEARNING_RATE = 1.0e-3
BATCH_SIZE = 64
T_REF_C = 121.1
Z_F_C = 10.0
INITIAL_TEMPERATURE_C = 20.0
F_SAFETY_THRESHOLD_MIN = 3.0
NEAR_THRESHOLD_FRACTION = 0.25
CONSISTENCY_SCORE_WEIGHT = 0.25
SAFETY_MARGIN_FLOOR_MIN = 0.30
CFD_CASES_PER_HOUR = 17.28
EXPECTED_DATASET_ROWS = 10526
EXPECTED_DATASET_SHA256 = "e7addf07b2a0e3df6480f62795715e14e89868ec51fc6b817943da4ba32b3ac1"
EXPECTED_SPLIT_SHA256 = {
 "train": "341eb1f8d04ccf7924544c4fa16b2b10aba98a3a915238ff444674cbedd3bcc2",
 "validation": "614049222d56fdfeb1416dbf883ce7044ebd493da883b467328c9f828a364b24",
 "test": "ee10ced8c90f7b861d66f14832f77c51d58ea0c06d90a79fbb86f81cf2b3952c",
}
INPUT_NAMES = ["CUT", "time_heat", "time_cool", "temp_retort"]
INPUT_LOWER_BOUNDS = np.array([1., 5., 1., 120.])
INPUT_UPPER_BOUNDS = np.array([5., 20., 5., 140.])
RESOLUTION_CANDIDATES = ((100,100), (200,200), (300,300))
UNIT_CANDIDATES = (128, 192)
DROPOUT_CANDIDATES = (0., 0.1)
PHYSICS_WEIGHT_CANDIDATES = (0., 0.1, 0.5, 1.)
SUPERVISED_WEIGHT_CANDIDATES = ((1.,1.,1.), (1.,2.,1.), (1.,1.,2.), (2.,1.,1.))
PROTOCOL_ORDER = ["random_interpolation", "retort_temperature_band", "high_temperature_short_heating_corner", "domain_boundary_shell"]
PROTOCOL_LABELS = dict(zip(PROTOCOL_ORDER, ["Random interpolation", "Withheld temperature band", "Withheld high-T/short-time corner", "Withheld domain boundaries"]))
EXPECTED_HOLDOUT_COUNTS = dict(zip(PROTOCOL_ORDER[1:], [2189,441,3707]))
ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class ModelConfig:
    model_kind: str = "lstm"  # lstm, gru, or mlp
    n_heat: int = 200
    n_cool: int = 200
    units: int = 128
    encoder_units: int = 128
    dropout: float = 0.10
    lambda_phys: float = 0.5
    learning_rate: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    temperature_weight: float = 1.0
    logF_weight: float = 1.0
    retention_weight: float = 1.0
    initial_condition_weight: float = 0.10
    stage_continuity_weight: float = 0.10

    def __post_init__(self):
        if self.model_kind not in {"lstm", "gru", "mlp"}:
            raise ValueError("Unsupported model kind.")
        for name in ("n_heat", "n_cool", "units", "encoder_units", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < (2 if name in {"n_heat", "n_cool"} else 1):
                raise ValueError(f"Invalid {name}: {value}")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must lie in [0,1).")
        for name in ("lambda_phys", "temperature_weight", "logF_weight", "retention_weight", "initial_condition_weight", "stage_continuity_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid loss weight {name}.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")

@dataclass
class Normalization:
    x_mean: List[float]
    x_std: List[float]
    temperature_mean: float
    temperature_std: float
    logF_mean: float
    logF_std: float
    retention_mean: float
    retention_std: float
    max_total_time_min: float
    wall_temperature_mean: float
    wall_temperature_std: float
    dt_mean: float
    dt_std: float

@dataclass
class RunOptions:
    mode: str = "full"
    output: Path = field(default_factory=lambda: ROOT / "outputs")
    data: Path | None = None
    epochs: int = 70
    seeds: tuple[int, ...] = SEEDS
    final_seed: int = 42
    precision: str = "mixed_float16"
    device: str = "gpu"
    strict_reference: bool = True
    resume: bool = True
    threads: int = 2
    resolution_tolerance: float = 0.01
    sample_cases: int = 256

    def validate(self) -> None:
        if self.mode not in {"full", "check"}:
            raise ValueError("Unknown run mode.")
        if self.mode == "full" and self.epochs != 70:
            raise ValueError("Full search candidates must receive 70 epochs.")
        if self.epochs < 1 or (self.mode == "check" and self.epochs > 10):
            raise ValueError("Software checks require between 1 and 10 epochs.")
        if self.threads < 1:
            raise ValueError("threads must be positive.")
        if self.mode == "check" and not 128 <= self.sample_cases <= 2048:
            raise ValueError("Real-data software checks require 128 to 2048 sample cases.")
        if self.precision not in {"float32", "mixed_float16"}:
            raise ValueError("Unsupported precision.")
        if self.final_seed not in self.seeds:
            raise ValueError("The final seed must be one of the predefined repeat seeds.")
        if len(set(self.seeds)) != len(self.seeds) or len(self.seeds) < 2:
            raise ValueError("At least two distinct repeat seeds are required.")
