import torch
import torch.nn as nn
from omegaconf import OmegaConf

def custom_loss(outputs, labels, inputs):
    loss = 0

    loss_fn = nn.MSELoss()

    for i in range(len(outputs)): 
        if  inputs[i].sum() != labels[i]:
            loss += loss_fn(outputs[i], labels[i])

    return loss

def getLoss():
    cfg = OmegaConf.load("config.yaml")

    if cfg.loss == "MSE":
        loss_fn = nn.MSELoss()
        return loss_fn
    if cfg.loss == "MAE":
        loss_fn = nn.L1Loss()
        return loss_fn 
    if cfg.loss == "custom":   
        return custom_loss  
    if cfg.loss == "BCE":
        loss_fn = nn.BCELoss()
        return loss_fn  
    if cfg.loss == "CE":
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn  
    if cfg.loss == "NLL":
        loss_fn = nn.NLLLoss()
        return loss_fn  
    pass

if __name__ == "__main__":
    getLoss()