import os
from typing import Dict
from typing import List
from typing import Optional

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
        self.log("train_loss", output[0])
        print(f"Train Loss: {output[0]}")
        return output[0]
    
    def validation_step(self, batch, batch_idx):
        output = self(input_ids=batch["ids"],
                      attention_mask=batch["attention_mask"],
                      token_type_ids=batch["type_ids"],
                      labels=batch["label"])
        self.log("val_loss", output[0])
        return output[0]
    
    def validation_epoch_end(
        self, outputs: List[float]
    ) -> None:
        avg_val_loss = float(sum(outputs) / len(outputs))
        mlflow.log_metric("avg_val_loss", avg_val_loss, self.current_epoch)
        print(f"Avg val loss: {avg_val_loss}")
    
    def test_step(self, batch, batch_idx):
        output = self(input_ids=batch["ids"],
                      attention_mask=batch["attention_mask"],
                      token_type_ids=batch["type_ids"],
                      labels=batch["label"])
        self.log("test_loss", output[0])
        return output[0]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
        return optimizer


class RobertaModel(object):
    # TODO (mihail): Make this config optional
    def __init__(self, config: Dict, model_cache_path: Optional[str] = None, load_from_ckpt: bool = False):
        if load_from_ckpt:
            self.model = RobertaModule.load_from_checkpoint(os.path.join(model_cache_path, "path"))
        else:
            self.config = config
            self.model = RobertaModule(config)
            checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                                  mode="min",
                                                  dirpath=model_cache_path,
                                                  filename="roberta-model-epoch={epoch}-val_loss={val_loss:.4f}")
            
            self.trainer = Trainer(max_epochs=self.config["num_epochs"],
                                   gpus=1 if torch.cuda.is_available() else None,
                                   callbacks=[checkpoint_callback],
                                   logger=False)
    
    def train(self, dataloader: DataLoader, val_dataloader: DataLoader):
        self.trainer.fit(self.model,
                         train_dataloader=dataloader,
                         val_dataloaders=val_dataloader)
    
    def predict(self, dataloader: DataLoader):
        self.model.eval()
        predicted = []
        self.model.cuda()
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                output = self.model(input_ids=batch["ids"].cuda(),
                                    attention_mask=batch["attention_mask"].cuda(),
                                    token_type_ids=batch["type_ids"].cuda(),
                                    labels=batch["label"].cuda())
                predicted.append(output[1])
        return torch.cat(predicted, axis=0).cpu().detach().numpy()
