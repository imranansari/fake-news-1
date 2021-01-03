import os
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification


class RobertaModule(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        full_model_output_path = os.path.join(base_dir, config["model_output_path"])
        # TODO (mihail): Fix how this `model_type` is defined
        self.config = config
        self.classifier = RobertaForSequenceClassification.from_pretrained(config["model_type"],
                                                                           cache_dir=full_model_output_path)
    
    def forward(self,
                input_ids: np.array,
                attention_mask: np.array,
                token_type_ids: np.array,
                labels: np.array):
        output = self.classifier(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels
                                 )
        return output
    
    def training_step(self, batch, batch_idx):
        output = self(input_ids=batch["ids"],
                      attention_mask=batch["attention_mask"],
                      token_type_ids=batch["type_ids"],
                      labels=batch["label"])
        return output[0]
    
    def validation_step(self, batch, batch_idx):
        output = self(input_ids=batch["ids"],
                      attention_mask=batch["attention_mask"],
                      token_type_ids=batch["type_ids"],
                      labels=batch["label"])
        return output[0]
    
    def test_step(self, batch, batch_idx):
        output = self(input_ids=batch["ids"],
                      attention_mask=batch["attention_mask"],
                      token_type_ids=batch["type_ids"],
                      labels=batch["label"])
        return output[0]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


class RobertaModel(object):
    def __init__(self, config: Dict):
        self.config = config
        self.model = RobertaModule(config)
        self.trainer = Trainer(max_epochs=self.config["num_epochs"],
                               gpus=0 if torch.cuda.is_available() else None)
    
    def train(self, dataloader: DataLoader):
        self.trainer.fit(self.model, dataloader)
    
    def predict(self, dataloader: DataLoader):
        self.trainer.test(self.model, test_dataloaders=dataloader)
