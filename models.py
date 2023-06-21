import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv,GCNConv


class GNN(torch.nn.Module):
    """
    Graph neural network (GNN) model for community detection.

    Args:
    num_communities (int): Number of communities in the graph.
    num_nodes (int): Number of nodes in the graph.
    hid_features (int): Number of hidden features in the GNN layers.
    """
    def __init__(self, nInputs, nOutputs, hid_features=32):
        super(GNN, self).__init__() #Net, self
        self.hid_features = hid_features
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()

        
        self.convs.append(ChebConv(self.nInputs, self.hid_features, K=self.K))
        self.convs.append(ChebConv(self.hid_features, self.hid_features, K=self.K))
        self.convs.append(ChebConv(self.hid_features, self.nOutputs, K=self.K))

        #self.lin = nn.Linear(self.num_nodes*1, self.num_communities)
        #self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
        
            if isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.weight
        batch= data.batch
        
        for i in range(len(self.convs) -1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = torch.relu(x)

        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)       

        return x
    
class MLPGNN(torch.nn.Module):
    """
    Graph neural network (GNN) model for community detection.

    Args:
    num_communities (int): Number of communities in the graph.
    num_nodes (int): Number of nodes in the graph.
    hid_features (int): Number of hidden features in the GNN layers.
    """
    def __init__(self, nInputs, nOutputs, hid_features=32):
        super(GNN, self).__init__() #Net, self
        self.hid_features = hid_features
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()

        
        self.convs.append(ChebConv(self.nInputs, self.hid_features, K=self.K))
        self.convs.append(ChebConv(self.hid_features, self.hid_features, K=self.K))
        self.convs.append(ChebConv(self.hid_features, self.nOutputs, K=self.K))

        #self.lin = nn.Linear(self.num_nodes*1, self.num_communities)
        #self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
        
            if isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.weight
        batch= data.batch
        
        for i in range(len(self.convs) -1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            x = torch.relu(x)

        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)       

        return x
