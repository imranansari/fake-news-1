import argparse
import json
import logging
import os
import pickle
import random
from typing import Dict
from typing import List
from typing import Optional

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

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
        return json.load(f)


def extract_manual_features(datapoints: List[Dict]) -> List[Dict]:
    all_features = []
    for datapoint in datapoints:
        features = {}
        features["speaker_title"] = datapoint["speaker_title"]
        all_features.append(features)
    return all_features


def extract_statements(datapoints: List[Dict]) -> List[str]:
    return [datapoint["statement"] for datapoint in datapoints]


def set_random_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    # TODO (mihail): Add torch random seeds when we get to those models


# TODO (mihail): Define types for datapoint

def compute_metrics(model: RandomForestClassifier,
                    input: np.array,
                    expected_labels: List[bool],
                    split: Optional[str] = None) -> Dict:
    # TODO (mihail): Consolidate this
    predicted_labels = model.predict(input)
    predicted_proba = model.predict_proba(input)
    accuracy = accuracy_score(expected_labels, predicted_labels)
    f1 = f1_score(expected_labels, predicted_labels)
    auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
    conf_mat = confusion_matrix(expected_labels, predicted_labels)
    tn, fp, fn, tp = conf_mat.ravel()
    split_prefix = "" if split is None else split
    return {
        f"{split_prefix} f1": f1,
        f"{split_prefix} accuracy": accuracy,
        f"{split_prefix} auc": auc,
        f"{split_prefix} true negative": tn,
        f"{split_prefix} false negative": fn,
        f"{split_prefix} false positive": fp,
        f"{split_prefix} true positive": tp,
    }


if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)
    
    set_random_seed()
    mlflow.set_experiment(config["model"])
    
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
    
    with mlflow.start_run() as run:
        with open(os.path.join(config["model_output_path"], "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({
            "evaluate": config["evaluate"]
        })
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
        mlflow.log_params(model.get_params())
        LOGGER.info("Evaluating model...")
        metrics = compute_metrics(model, test_features, test_labels, split="test")
        LOGGER.info(f"Test metrics: {metrics}")
        mlflow.log_metrics(metrics)
