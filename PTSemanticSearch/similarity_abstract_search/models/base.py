"""Model class to be extended by specific types of models """
from pathlib import Path
from typing import Callable, Dict, Optional

DIRNAME = Path(__file__).parents[1].resolve()/'weights'