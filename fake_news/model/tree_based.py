from typing import Dict
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel(object):
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def train(self, input: np.array, labels: List[bool]) -> None:
        self.model.fit(input, labels)
    
    def predict(self, input: np.array) -> np.array:
        return self.model.predict_proba(input)
        
    def get_params(self) -> Dict:
        return self.model.get_params()
