import json
import logging
import os
import pickle
from copy import deepcopy
from functools import partial
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
# Mapping from speaker title variants seen in data to their canonical form
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

CANONICAL_SPEAKER_TITLES = {
    "u.s. house of representative": "u.s. house of representatives",
    "u.s. representativej": "u.s. representative",
    "talks show host": "talk show host",
    "u. s. congressman": "u.s. congressman",
    "politican action committee": "political action committee",
    "retired": "retiree",
    "restauranteur": "restaurateur"
}

SIX_WAY_LABEL_TO_BINARY = {
    "pants-fire": False,
    "barely-true": False,
    "false": False,
    "true": True,
    "half-true": True,
    "mostly-true": True
}

CANONICAL_STATE = {
    "tennesse": "tennessee",
    "district of columbia": "washington d.c.",
    "washington dc": "washington d.c.",
    "washington, d.c.": "washington d.c.",
    "washington d.c.": "washington d.c.",
    "tex": "texas",
    "texas": "texas",
    "washington state": "washington",
    "washington": "washington",
    "virgina": "virginia",
    "virgiia": "virginia",
    "virginia": "virginia",
    "pennsylvania": "pennsylvania",
    "pa - pennsylvania": "pennsylvania",
    "rhode island": "rhode island",
    "rhode island": "rhode island",
    "ohio": "ohio",
    "ohio": "ohio"
}

PARTY_AFFILIATIONS = {
    "republican", "democrat", "none", "organization", "independent",
    "columnist", "activist", "talk-show-host", "libertarian",
    "newsmaker", "journalist", "labor-leader", "state-official",
    "business-leader", "education-official", "tea-party-member",
    "green", "liberal-party-canada", "government-body", "Moderate",
    "democratic-farmer-labor", "ocean-state-tea-party-action",
    "constitution-party"
}


def extract_manual_features(datapoints: List[Dict], optimal_credit_bins_path: str) -> List[Dict]:
    all_features = []
    with open(optimal_credit_bins_path) as f:
        optimal_credit_bins = json.load(f)
    for datapoint in datapoints:
        features = {}
        features["speaker"] = datapoint["speaker"]
        features["speaker_title"] = datapoint["speaker_title"]
        features["state_info"] = datapoint["state_info"]
        features["party_affiliation"] = datapoint["party_affiliation"]
        # Compute credit score features
        for feat in ["barely_true_count", "false_count", "half_true_count", "mostly_true_count", "pants_fire_count"]:
            features[feat] = str(compute_bin_idx(datapoint[feat], optimal_credit_bins[feat]))
        all_features.append(features)
    return all_features


def extract_statements(datapoints: List[Dict]) -> List[str]:
    return [datapoint["statement"] for datapoint in datapoints]


def construct_datapoint(input: str) -> Dict:
    # TODO (mihail): What are reasonable defaults for other features
    return {
        "statement": input,
        "speaker": "",
        "subject": "",
        "speaker_title": "",
        "state_info": "",
        "party_affiliation": "",
        "barely_true_count": float("nan"),
        "false_count": float("nan"),
        "half_true_count": float("nan"),
        "mostly_true_count": float("nan"),
        "pants_fire_count": float("nan"),
        "context": "",
        "justification": ""
    }


class TreeFeaturizer(object):
    def __init__(self, featurizer_cache_path: str, config: Optional[Dict] = None):
        if os.path.exists(featurizer_cache_path):
            LOGGER.info("Loading featurizer from cache...")
            # TODO (mihail): Fill in here
            with open(featurizer_cache_path, "rb") as f:
                self.combined_featurizer = pickle.load(f)
        else:
            LOGGER.info("Creating featurizer from scratch...")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            dict_featurizer = DictVectorizer()
            tfidf_featurizer = TfidfVectorizer()
            
            statement_transformer = FunctionTransformer(extract_statements)
            manual_feature_transformer = FunctionTransformer(partial(extract_manual_features,
                                                                     optimal_credit_bins_path=
                                                                     os.path.join(base_dir,
                                                                                  config["credit_bins_path"])))
            
            manual_feature_pipeline = Pipeline([
                ("manual_features", manual_feature_transformer),
                ("manual_featurizer", dict_featurizer)
            ])
            
            ngram_feature_pipeline = Pipeline([
                ("statements", statement_transformer),
                ("ngram_featurizer", tfidf_featurizer)
            ])
            
            self.combined_featurizer = FeatureUnion([
                ("manual_feature_pipe", manual_feature_pipeline),
                ("ngram_feature_pipe", ngram_feature_pipeline)
            ])
    
    def get_all_feature_names(self) -> List[str]:
        all_feature_names = []
        for name, pipeline in self.combined_featurizer.transformer_list:
            final_pipe_name, final_pipe_transformer = pipeline.steps[-1]
            all_feature_names.extend(final_pipe_transformer.get_feature_names())
        return all_feature_names
    
    def fit(self, datapoints: List[Dict]) -> None:
        self.combined_featurizer.fit(datapoints)
    
    def featurize(self, datapoints: List[Dict]) -> np.array:
        return self.combined_featurizer.transform(datapoints)
    
    def save(self, featurizer_cache_path: str):
        LOGGER.info("Saving featurizer to disk...")
        with open(featurizer_cache_path, "wb") as f:
            pickle.dump(self.combined_featurizer, f)


def compute_bin_idx(val: float, bins: List[float]) -> int:
    for idx, bin_val in enumerate(bins):
        if val <= bin_val:
            return idx


# NOTE: Making sure that all normalization operations preserve immutability of inputs
def normalize_labels(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        normalized_datapoint["label"] = SIX_WAY_LABEL_TO_BINARY[datapoint["label".lower().strip()]]
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_speaker_title(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        old_speaker_title = normalized_datapoint["speaker_title"]
        old_speaker_title = old_speaker_title.lower().strip().replace("-", " ")
        # Then canonicalize
        if old_speaker_title in CANONICAL_SPEAKER_TITLES:
            old_speaker_title = CANONICAL_SPEAKER_TITLES[old_speaker_title]
        normalized_datapoint["speaker_title"] = old_speaker_title
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_party_affiliations(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        if normalized_datapoint["party_affiliation"] not in PARTY_AFFILIATIONS:
            normalized_datapoint["party_affiliation"] = "none"
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_state_info(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        old_state_info = normalized_datapoint["state_info"]
        old_state_info = old_state_info.lower().strip().replace("-", " ")
        if old_state_info in CANONICAL_STATE:
            old_state_info = CANONICAL_STATE[old_state_info]
        normalized_datapoint["state_info"] = old_state_info
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_counts(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for idx, datapoint in enumerate(datapoints):
        normalized_datapoint = deepcopy(datapoint)
        for count_col in ["barely_true_count",
                          "false_count",
                          "half_true_count",
                          "mostly_true_count",
                          "pants_fire_count"]:
            if count_col in normalized_datapoint:
                normalized_datapoint[count_col] = float(normalized_datapoint[count_col])
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean(datapoints: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker_title(
        normalize_and_clean_party_affiliations(
            normalize_and_clean_state_info(
                normalize_and_clean_counts(
                    normalize_labels(
                        datapoints
                    )
                )
            )
        )
    )
