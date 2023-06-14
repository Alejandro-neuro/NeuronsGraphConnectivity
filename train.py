import torch
import torch.nn as nn
import torchmetrics
import networkx as nx
import numpy as np
from omegaconf import OmegaConf
import models
import loss_func


def train_epoch(model, loader,loss_fn, optimizer, device='cpu'):
    """
    Trains a neural network model for one epoch using the specified data loader and optimizer.
    Args:
    model (nn.Module): The neural network model to be trained.
    loader (DataLoader): The PyTorch Geometric DataLoader containing the training data.
    optimizer (torch.optim.Optimizer): The PyTorch optimizer used for training the model.
    device (str): The device used for training the model (default: 'cpu').
    Returns:
    float: The mean loss value over all the batches in the DataLoader.
    """
    model.to(device)
    model.train() # specifies that the model is in training mode
    running_loss = 0.
    total_loss = 0.
    
    #loss_fn = nn.MSELoss()
    for data in loader:

        inputs = data
        labels = data['y']

        inputs = inputs.to(device=device)
        labels = labels.to(device = device)
        # Zero gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)


        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    total_loss = running_loss/len(loader)
    return total_loss


#
def evaluate_epoch(model, loader,loss_fn, device='cpu'):
    with torch.no_grad():
        model.to(device)
        model.eval() # specifies that the model is in evaluation mode
        running_loss = 0.
        accuracy=0.
        correct = 0.


        for data in loader:


            inputs = data
            labels = data['y']

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Make predictions for this batch
            outputs = model(inputs)

            running_loss += loss_fn(outputs, labels).item()

            pred = outputs.argmax(dim=1)
            correct += (pred == labels.argmax(dim=-1) ).sum().float() / pred.shape[0] # Check against ground-truth labels.
        loss = running_loss/len(loader)
        accuracy = correct.item() / len(loader)
        
    return loss, accuracy


def train(model, train_loader, val_loader, optimizer, device='cpu'):

    cfg = OmegaConf.load("config.yaml")
    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_epochs = 200

    

    model.to(device)
    #create vectors for the training and validation loss

    train_losses = []
    val_losses = []
    accuracy_list = []
    patience = 15 # patience for early stopping
    print(device)

    for epoch in range(1, num_epochs+1):
        # Model training
        train_loss = train_epoch(model, train_loader, optimizer, device=device)
        # Model validation
        val_loss,accuracy = evaluate_epoch(model, val_loader, device=device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracy_list.append(accuracy)
        # Early stopping
        try:
            if val_losses[-1]>=val_losses[-2]:
                early_stop += 1
            if early_stop == patience:
                print("Early stopping! Epoch:", epoch)
                break
            else:
                early_stop = 0
        except:
            early_stop = 0
        if epoch%(num_epochs /10 )== 0:
            print("epoch:",epoch, "\t training loss:", train_loss,
                  "\t validation loss:",val_loss, 
                  "\t accuracy :", accuracy )
            

if __name__ == '__main__':
    train()