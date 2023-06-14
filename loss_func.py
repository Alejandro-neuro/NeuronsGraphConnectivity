import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from sklearn.model_selection import train_test_split



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
    data, labels = timeseries[:,], range(5)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, shuffle=False, random_state=42)
    return dataset





def create_partitions(G,dataset):
    
    train_loader = DataLoader(gen_Dataset(G,dataset['trn']), batch_size=64, shuffle=True)
    val_loader = DataLoader(gen_Dataset(G,dataset['val']), batch_size=64, shuffle=False)
    test_loader = DataLoader(gen_Dataset(G,dataset['tst']), batch_size=len(dataset['tst']), shuffle=False)
 
    return train_loader,val_loader,test_loader
