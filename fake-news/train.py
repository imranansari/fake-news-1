import argparse
import csv
import logging
from typing import Dict
from typing import List

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - lineno: %(lineno)d - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument("--test-data-path", type=str)
    return parser.parse_args()


def featurize(datapoints):
    pass


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


class StatementExtractor(TransformerMixin):
    def transform(self, datapoints):
        return [datapoint["statement"] for datapoint in datapoints]
    
    def get_feature_names(self):
        return ["statement"]


if __name__ == "__main__":
    args = read_args()
    
    # Read data
    train_datapoints = read_datapoints(args.train_data_path)
    val_datapoints = read_datapoints(args.val_data_path)
    test_datapoints = read_datapoints(args.test_data_path)
    
    train_labels = [datapoint["label"] for datapoint in train_datapoints]
    val_labels = [datapoint["label"] for datapoint in val_datapoints]
    test_labels = [datapoint["label"] for datapoint in test_datapoints]
    
    dict_featurizer = DictVectorizer()
    tfidf_featurizer = TfidfVectorizer()
    
    model = RandomForestClassifier()
    
    # TODO (mihail): Clean up/normalize data first
    # Featurize
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
    
    model.fit(train_features, train_labels)
    
    test_output = model.predict(test_features)
    print(accuracy_score(test_labels, test_output))
