import os

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset
from model import LogisticRegressionModel
from utils import *

def train(model, dataloader, cfg):
    sw = SummaryWriter(cfg.train.log_dir)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr)

    for epoch in range(cfg.train.epochs):
        for X_batch, y_batch in dataloader:
            model.train()
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        sw.add_scalar('Loss/train', loss.item(), epoch)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {loss.item():.4f}')


@hydra.main(version_base="1.1", config_path="../conf", config_name="logistic_regression")
def main(cfg: DictConfig) -> None:
    dataframe = pd.read_csv(cfg.dataset.path)

    dataframe = ProcessDataframe(dataframe)
    imputer = instantiate(cfg.dataset.impute, dataframe=dataframe)
    impute = getattr(imputer, cfg.dataset.impute_method)
    dataframe = impute()

    normalizer = instantiate(cfg.dataset.normalize, dataframe=dataframe)
    normalize = getattr(normalizer, cfg.dataset.normalize_method)
    dataframe = normalize()

    data_array = dataframe.values
    data_tensor = torch.tensor(data_array, dtype=torch.float32)

    train_tensor = data_tensor[:int(0.9 * len(data_tensor))]
    val_tensor = data_tensor[int(0.9 * len(data_tensor)):]

    train_dataset = MyDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)

    val_dataset = MyDataset(val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_tensor), shuffle=False)
    
    model = LogisticRegressionModel(data_tensor.shape[1] - 1)
    
    train(model, train_dataloader, cfg)

    model.eval()
    with torch.no_grad():
        acc = []
        for X, y in val_dataloader:
            output = model(X).round()
            accuracy=(output == y).sum().item() / len(y)
            print(f'Validation Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()
