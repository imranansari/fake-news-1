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
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from torch.utils.data import DataLoader

from fake_news.model.transformer_based import RobertaModel
from fake_news.utils.dataloaders import FakeNewsTorchDataset

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()


def set_random_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    # Torch-specific random-seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def get_all_feature_names(feature_transform: FeatureUnion) -> List[str]:
    all_feature_names = []
    for name, pipeline in feature_transform.transformer_list:
        final_pipe_name, final_pipe_transformer = pipeline.steps[-1]
        all_feature_names.extend(final_pipe_transformer.get_feature_names())
    return all_feature_names


# TODO (mihail): Define types for datapoint

def compute_metrics(model: RobertaModel,
                    dataloader: DataLoader,
                    split: Optional[str] = None) -> Dict:
    expected_labels = []
    for batch in dataloader:
        expected_labels.extend(batch["label"].cpu().numpy())
    predicted_proba = model.predict(dataloader)
    predicted_labels = np.argmax(predicted_proba, axis=1)
    accuracy = accuracy_score(expected_labels, predicted_labels)
    f1 = f1_score(expected_labels, predicted_labels)
    auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
    conf_mat = confusion_matrix(expected_labels, predicted_labels)
    tn, fp, fn, tp = conf_mat.ravel()
    print(f"Accuracy: {accuracy}, F1: {f1}, AUC: {auc}")
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
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_cached_feature_path = os.path.join(base_dir, config["train_cached_features_path"])
    val_cached_feature_path = os.path.join(base_dir, config["val_cached_features_path"])
    test_cached_feature_path = os.path.join(base_dir, config["test_cached_features_path"])
    model_output_path = os.path.join(base_dir, config["model_output_path"])
    os.makedirs(model_output_path, exist_ok=True)
    
    with mlflow.start_run() as run:
        if config["evaluate"] and os.path.exists(train_cached_feature_path) and \
            os.path.exists(val_cached_feature_path) and \
            os.path.exists(test_cached_feature_path):
            pass
        else:
            LOGGER.info("Featurizing data from scratch...")
            train_data_path = os.path.join(base_dir, config["train_data_path"])
            val_data_path = os.path.join(base_dir, config["val_data_path"])
            test_data_path = os.path.join(base_dir, config["test_data_path"])
            # Read data
            train_data = FakeNewsTorchDataset(config, split="train")
            val_data = FakeNewsTorchDataset(config, split="val")
            test_data = FakeNewsTorchDataset(config, split="test")
            
            train_dataloader = DataLoader(train_data,
                                          shuffle=True,
                                          batch_size=config["batch_size"],
                                          pin_memory=True)
            val_dataloader = DataLoader(val_data,
                                        shuffle=False,
                                        batch_size=16,
                                        pin_memory=True)
            test_dataloader = DataLoader(test_data,
                                        shuffle=False,
                                        batch_size=16,
                                        pin_memory=True)

            
            checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                                  mode="min",
                                                  dirpath=model_output_path,
                                                  filename="roberta-model-epoch={epoch}-val_loss={val_loss}")
            model = RobertaModel(config, model_output_path)
            model.train(train_dataloader, val_dataloader)
            
            with open(os.path.join(model_output_path, "meta.json"), "w") as f:
                json.dump({"mlflow_run_id": run.info.run_id}, f)
            mlflow.set_tags({
                "evaluate": config["evaluate"]
            })
            if config["evaluate"]:
                LOGGER.info("Loading up previously saved model...")
                if not os.path.exists(os.path.join(model_output_path)):
                    raise ValueError("Model output path does not exist but in `evaluate` mode!")
                with open(os.path.join(model_output_path, "model.pkl"), "rb") as f:
                    model = pickle.load(f)
            else:
                pass
            # mlflow.log_params(model.get_params())
            LOGGER.info("Evaluating model...")
            metrics = compute_metrics(model, val_dataloader, split="val")
            LOGGER.info(f"Test metrics: {metrics}")
            mlflow.log_metrics(metrics)
