import argparse
import csv
import json
import logging
import os
import pickle
import random
from typing import Dict
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from fake_news.utils.features import normalize_and_clean

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()


def read_datapoints(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=[
            "id",
            "statement_json",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_title",
            "state_info",
            "party_affiliation",
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
            "context",
            "justification"
        ])
        return [row for row in reader]


def extract_manual_features(datapoints: List[Dict]) -> List[Dict]:
    all_features = []
    for datapoint in datapoints:
        features = {}
        features["speaker_title"] = datapoint["speaker_title"]
        all_features.append(features)
    return all_features


def extract_statements(datapoints: List[Dict]) -> List[str]:
    return [datapoint["statement"] for datapoint in datapoints]


def set_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    # TODO (mihail): Add torch random seeds when we get to those models


# TODO (mihail): Define types for datapoint

if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)
    
    set_seed()
    
    if os.path.exists(config["train_cached_features_path"]) and \
        os.path.exists(config["val_cached_features_path"]) and \
        os.path.exists(config["test_cached_features_path"]):
        LOGGER.info("Loading up cached features from disk...")
        with open(config["train_cached_features_path"], "rb") as f:
            train_features, train_labels = pickle.load(f)
        with open(config["val_cached_features_path"], "rb") as f:
            val_features, val_labels = pickle.load(f)
        with open(config["test_cached_features_path"], "rb") as f:
            test_features, test_labels = pickle.load(f)
    else:
        LOGGER.info("Featurizing data from scratch...")
        # Read data
        train_datapoints = read_datapoints(config["train_data_path"])
        val_datapoints = read_datapoints(config["val_data_path"])
        test_datapoints = read_datapoints(config["test_data_path"])
        
        train_datapoints = normalize_and_clean(train_datapoints)
        val_datapoints = normalize_and_clean(val_datapoints)
        test_datapoints = normalize_and_clean(test_datapoints)
        
        # Featurize
        dict_featurizer = DictVectorizer()
        tfidf_featurizer = TfidfVectorizer()
        
        statement_transformer = FunctionTransformer(extract_statements)
        manual_feature_transformer = FunctionTransformer(extract_manual_features)
        
        manual_feature_pipeline = Pipeline([
            ("manual_features", manual_feature_transformer),
            ("manual_featurizer", dict_featurizer)
        ])
        
        tfidf_feature_pipeline = Pipeline([
            ("statements", statement_transformer),
            ("tfidf_featurizer", tfidf_featurizer)
        ])
        
        combined_featurizer = FeatureUnion([
            ("manual_feature_pipe", manual_feature_pipeline),
            ("tfidf_featurizer", tfidf_feature_pipeline)
        ])
        
        train_features = combined_featurizer.fit_transform(train_datapoints)
        val_features = combined_featurizer.transform(val_datapoints)
        test_features = combined_featurizer.transform(test_datapoints)
        
        train_labels = [datapoint["label"] for datapoint in train_datapoints]
        val_labels = [datapoint["label"] for datapoint in val_datapoints]
        test_labels = [datapoint["label"] for datapoint in test_datapoints]
        
        # Dump to cache
        with open(config["train_cached_features_path"], "wb") as f:
            pickle.dump([train_features, train_labels], f)
        with open(config["val_cached_features_path"], "wb") as f:
            pickle.dump([val_features, val_labels], f)
        with open(config["test_cached_features_path"], "wb") as f:
            pickle.dump([test_features, test_labels], f)
    
    if config["evaluate"]:
        LOGGER.info("Loading up previously saved model...")
        if not os.path.exists(config["model_output_path"]):
            raise ValueError("Model output path does not exist but in `evaluate` mode!")
        with open(os.path.join(config["model_output_path"], "model.pkl"), "rb") as f:
            model = pickle.load(f)
    else:
        model = RandomForestClassifier()
        LOGGER.info("Training model...")
        model.fit(train_features, train_labels)
    
        # Cache model weights on disk
        os.makedirs(config["model_output_path"], exist_ok=True)
        with open(os.path.join(config["model_output_path"], "model.pkl"), "wb") as f:
            pickle.dump(model, f)
    
    LOGGER.info("Evaluating model...")
    test_output = model.predict(test_features)
    print(accuracy_score(test_labels, test_output))
