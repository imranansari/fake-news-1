import os
import pickle
from typing import Dict
import numpy as np
import torch
from transformers import RobertaTokenizerFast

from fake_news.utils.reader import read_json_data


class FakeNewsTorchDataset(torch.utils.data.Dataset):
    def __init__(self, config: Dict, split: str = "train"):
        if split == "train":
            data_path = config["train_data_path"]
            cached_features_path = config["train_cached_features_path"]
        elif split == "val":
            data_path = config["val_data_path"]
            cached_features_path = config["val_cached_features_path"]
        else:
            data_path = config["test_data_path"]
            cached_features_path = config["test_cached_features_path"]
        
        self.data = []
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_cached_features_path = os.path.join(base_dir, cached_features_path)
        full_model_output_path = os.path.join(base_dir, config["model_output_path"])
        full_data_path = os.path.join(base_dir, data_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(config["tokenizer_path"],
                                                         cache_dir=full_model_output_path,
                                                         padding_side="right")
        if os.path.exists(full_cached_features_path):
            with open(full_cached_features_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            datapoints = read_json_data(full_data_path)
            for datapoint in datapoints:
                # TODO (mihail): `return_tensors=pt`?
                tokenized = tokenizer(datapoint["statement"],
                                      padding="max_length",
                                      max_length=32,
                                      truncation=True,
                                      return_tensors="np",
                                      return_token_type_ids=True,
                                      return_attention_mask=True,
                                      return_special_tokens_mask=True)
                # Only a single encoding since only a single datapoint tokenized
                self.data.append({
                    "ids": tokenized.data["input_ids"].squeeze(),
                    "type_ids": tokenized.data["token_type_ids"].squeeze(),
                    "attention_mask": tokenized.data["attention_mask"].squeeze(),
                    "special_tokens_mask": tokenized.data["special_tokens_mask"].squeeze(),
                    "label": np.array(int(datapoint["label"]))
                })
            # Cache data
            with open(full_cached_features_path, "wb") as f:
                pickle.dump(self.data, f)
    
    def __getitem__(self, idx: int):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
