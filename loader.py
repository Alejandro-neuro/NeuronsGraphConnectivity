import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score

from omegaconf import OmegaConf



def gen_Dataset(G,data_original):
    dataset = []
    data1 = from_networkx(G)

    for x,y in zip(data_original['data'][:,:,:], data_original['labels'][:,:,:]):

        data = data1.clone()
        data['x']=torch.from_numpy(x).float()
        data['y']=torch.from_numpy(y).float()
        data['weight']=data['weight'].float()
        dataset.append(data)

    return dataset

def timeSeries2Dataset(timeseries):

    data, labels = timeseries[:,:-2], timeseries[:,-1]
    dataset=[]

    cfg = OmegaConf.load("config.yaml")

    test_size = cfg.dataset.test_size

    X_train, X_test, y_train, y_test  = train_test_split(data, labels, test_size=test_size, random_state=1)

    val_size = cfg.dataset.val_size / cfg.dataset.train_size
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=1) # 0.25 x 0.8 = 0.2

    dataset['trn']['data'] = X_train
    dataset['trn']['labels'] = y_train
    dataset['val']['data'] = X_val
    dataset['val']['labels'] = y_val
    dataset['tst']['data'] = X_test
    dataset['tst']['labels'] = y_test

    return dataset




def create_partitions(G,dataset):
    
    train_loader = DataLoader(gen_Dataset(G,dataset['trn']), batch_size=64, shuffle=True)
    val_loader = DataLoader(gen_Dataset(G,dataset['val']), batch_size=64, shuffle=False)
    test_loader = DataLoader(gen_Dataset(G,dataset['tst']), batch_size=len(dataset['tst']), shuffle=False)
 
    return train_loader,val_loader,test_loader

#create a function that returns adjacency matrix and based on the input string "type" and returns pearson correlation
#or mutual information
def generateAdjacencyMatrix(x,type):
    
    if(type == 'pearson'): 
        adj = np.corrcoef(x)
        adj = np.abs(x)
        return adj
    
    if(type == 'mutual_info'):
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                adj[i,j] = mutual_info_score(x[i,:],x[j,:]) #calculate mutual information between nodes i and j
                adj[j,i] = adj[i,j] #make the matrix symmetric  
        return adj
    
    #calculate adjacency matrix with cosine similarity
    if(type == 'cosine'):
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                adj[i,j] = np.dot(x[i,:],x[j,:]) / (np.linalg.norm(x[i,:]) * np.linalg.norm(x[j,:]))
                adj[j,i] = adj[i,j] #make the matrix symmetric
        return adj   

    return adj
        


def generateGraph(dataset,type):

    x = dataset['trn']['data']

    adj = generateAdjacencyMatrix(x,type)    

    #create a graph from the correlation matrix
    G = nx.from_numpy_matrix(adj)

    return G


def generateLoaders( dataset, type):
    
    G = generateGraph(dataset,type)

    train_loader,val_loader,test_loader = create_partitions(G,dataset)

    return train_loader,val_loader,test_loader

if __name__ == "__main__":
    generateLoaders()

