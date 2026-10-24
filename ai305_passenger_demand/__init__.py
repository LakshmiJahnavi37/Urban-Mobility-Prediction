"""
AI305: Public Transport Passenger Demand Prediction Module
Owner: 24B11AI305 (O. Bheemeswari)
Part of Urban-Mobility-Prediction Platform (Team 4)
"""

import os
# Safeguard for Windows OpenBLAS memory allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__version__ = "1.0.0"
__author__ = "24B11AI305"

from ai305_passenger_demand.predict import (
    PassengerDemandPredictor,
    ModelNotTrainedError,
    HistoricalLookaheadError
)

__all__ = [
    "PassengerDemandPredictor",
    "ModelNotTrainedError",
    "HistoricalLookaheadError"
]
