import json
from typing import Dict
from typing import List


def read_json_data(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        return json.load(f)
