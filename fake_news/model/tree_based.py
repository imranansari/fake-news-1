import logging
import os
import pickle
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


class RandomForestModel(object):
    def __init__(self, model_cache_path: Optional[str] = None):
        if model_cache_path and os.path.exists(model_cache_path):
            LOGGER.info("Loading model from scratch...")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            LOGGER.info("Initializing model from scratch...")
            self.model = RandomForestClassifier()
    
    def train(self, input: np.array, labels: List[bool]) -> None:
        self.model.fit(input, labels)
    
    def predict(self, input: np.array) -> np.array:
        return self.model.predict_proba(input)
    
    def predict_labels(self, input: np.array) -> np.array:
        proba = self.model.predict_proba(input)
        return np.argmax(proba, axis=1)
    
    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def save(self, model_cache_path: str):
        LOGGER.info("Saving model to disk...")
        # TODO (mihail): Save using joblib
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)
