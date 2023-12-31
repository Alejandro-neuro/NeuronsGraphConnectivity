import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv,GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.dense.linear import Linear
import math


def normalize_adj(adj):
    """
     normalize adjacency matrix.
    """
    max_adj = torch.max(adj)
    a_norm = adj / max_adj

    return a_norm

class GNN(torch.nn.Module):
    """
    Graph neural network (GNN) model for community detection.

    Args:
    num_communities (int): Number of communities in the graph.
    num_nodes (int): Number of nodes in the graph.
    hid_features (int): Number of hidden features in the GNN layers.
    """
    def __init__(self, nInputs, nOutputs, hid_features=32, n_nodes=10):
        super(GNN, self).__init__() #Net, self
        self.hid_features = hid_features
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()

        self.adjMat = None

        self.convs.append(GCNConv(self.nInputs, self.nOutputs, K=self.K))
        
        #self.convs.append(ChebConv(self.nInputs, self.hid_features, K=self.K))
        #self.convs.append(ChebConv(self.hid_features, self.hid_features, K=self.K))
        #self.convs.append(ChebConv(self.hid_features, self.nOutputs, K=self.K))

        #self.lin = nn.Linear(self.num_nodes*1, self.num_communities)
        #self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
        
            if isinstance(m, nn.Linear):
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def get_adjMat(self):

        meanMat = self.adjMat.mean(0)
        return normalize_adj(meanMat)
    
    def reset_adjMat(self):
        self.adjMat = None

    
    
    def add2adjMat(self,x):   
        if self.adjMat is None:
            self.adjMat = x.unsqueeze(0)
        else:
            self.adjMat = torch.cat([self.adjMat, x.unsqueeze(0)], dim=0)
        return self.adjMat
    
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

        x = x
        self.add2adjMat( to_dense_adj(edge_index, batch=batch).mean( 0) )
        

        return x
    
class MLPGNN(torch.nn.Module):
   
    def __init__(self, nInputs, nOutputs, hid_features=32):
        super(MLPGNN, self).__init__() #Net, self
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
    
class GATGCN(torch.nn.Module):

      def __init__(self, nInputs, nOutputs, hid_features=32):
        super(GATGCN, self).__init__() 

        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.hid_features = hid_features
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()

        self.edgeRep = None

        self.GAT = GATv2Conv(self.nInputs, self.nOutputs)

        self.adjMat = None

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        self.W = nn.Parameter(torch.zeros(size=(self.nInputs, self.nOutputs)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*self.nOutputs, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
      
      
      def get_weights(self):
        return self.GAT.get_weights()
      
      def get_adjMat(self):
        return self.adjMat
      
      def reset_adjMat(self):
        self.adjMat = None
          
      def forward(self, data):
    
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.weight
        batch= data.batch
        print(x.shape)
        print(edge_index.shape)
        x, ed  = self.GAT(x=x, edge_index=edge_index, return_attention_weights=True) 

        print(x.shape)
        print(ed[0].shape)
        print(ed[1].shape)
        
        for i in range(len(self.convs) -1):
            x = self.convs[i](x=x, edge_index=ed[0])
            x = torch.relu(x)

        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = self.convs[-1](x=x, edge_index=ed[0])

        self.edgeRep = ed[0]

        return x
    
class GATCustom(torch.nn.Module):

    def __init__(self, nInputs, nOutputs, hid_features=32, threshold=0.5, alpha = 0.2):
        super(GATCustom, self).__init__() 

        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.hid_features = hid_features
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()
        self.threshold = threshold
        self.alpha = alpha 


        self.edgeRep = None


        self.adjMat = None

        self.center = 0.3

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        torch.manual_seed(0)
        self.W = nn.Parameter(torch.ones(size=(self.nInputs, self.nOutputs))*0.5)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.ones(size=(self.nOutputs, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.linear = Linear(self.nInputs*2, self.nOutputs, bias=False, weight_initializer='glorot')

        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.relu = nn.ReLU()

    def get_adjMat(self):
        meanMat = self.adjMat.mean(0)
        return meanMat
        #return normalize_adj(meanMat)
    
    def reset_adjMat(self):
        self.adjMat = None    
    
    def add2adjMat(self,x):   
        if self.adjMat is None:
            self.adjMat = x.unsqueeze(0)
        else:
            self.adjMat = torch.cat([self.adjMat, x.unsqueeze(0)], dim=0)
        return self.adjMat
              
    def forward(self, data):
    
        x = data.x
        batch= data.batch        
        n_batch = batch.max().item() + 1
        n_nodes = x.shape[0] // n_batch

        x2 = torch.reshape(x,(n_batch,n_nodes,self.nInputs) )
           
        # Linear Transformation
        h = torch.mm(x, self.W) 
        N = h.size()[0]
       
        hihj  = torch.cat([x2.repeat(1,1, n_nodes).view(n_batch, -1,self.nInputs), x2.repeat(1,n_nodes, 1)], dim=2).view(-1, 2 * self.nInputs)
       
        A = self.linear(hihj) 
       
        # Attention Mechanism       
       
        #e       = self.leakyrelu(A)
        e = self.relu(A)
        att = torch.matmul(e, self.a)        
        
        att = att.view(-1, n_nodes, n_nodes) # Reshape into n_batch*n_nodes*n_nodes tensor
        
        attention = F.softmax(att, dim=-2) # attention coefficients 

        max_att = torch.max(attention) 
        min_att = torch.min(attention) 

        mid_att = (max_att + min_att)/2

        #attention = F.relu(attention - mid_att) + mid_att
        zero_vec  = -9e15*torch.ones_like(attention)
        #attention = torch.where(attention > self.threshold, attention, zero_vec) # Thresholding attention matrix
        self.add2adjMat( attention.mean( 0) ) # Save attention matrix for analysis
        
        attention_block = torch.block_diag(*attention) # Duplicate atention vector in the diagonal for each batch
        
        h = torch.mm(x, self.W) # matrix multiplication
        h_prime   = torch.mm(attention_block, h)

        return h_prime   

class GAT1Custom(torch.nn.Module):

    def __init__(self, nInputs, nOutputs, hid_features=32, threshold=0.5, alpha = 0.2):
        super(GAT1Custom, self).__init__() 

        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.hid_features = hid_features
        self.dropout_rate=0.0
        self.K = 1
        self.convs = nn.ModuleList()
        self.threshold = threshold
        self.alpha = alpha 


        self.edgeRep = None


        self.adjMat = None

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice 
        torch.manual_seed(0)
        self.W = nn.Parameter(torch.zeros(size=(self.nInputs, self.nOutputs)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(self.nOutputs, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.linear = Linear(self.nInputs*2, self.nOutputs, bias=True, weight_initializer='glorot')

        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def get_adjMat(self):
        meanMat = self.adjMat.mean(0)
        return meanMat
        #return normalize_adj(meanMat)
    
    def reset_adjMat(self):
        self.adjMat = None    
    
    def add2adjMat(self,x):   
        if self.adjMat is None:
            self.adjMat = x.unsqueeze(0)
        else:
            self.adjMat = torch.cat([self.adjMat, x.unsqueeze(0)], dim=0)
        return self.adjMat
              
    def forward(self, data):
    
        x = data.x
        batch= data.batch        
        n_batch = batch.max().item() + 1
        n_nodes = x.shape[0] // n_batch

        x2 = torch.reshape(x,(n_batch,n_nodes,self.nInputs) )
           
        # Linear Transformation
        h = torch.mm(x, self.W) 
        h2 = torch.reshape(h,(n_batch,n_nodes,self.nOutputs) )
        print(h2.shape)
        N = h.size()[0]
        print(torch.cat([h2.repeat(1,1, n_nodes).view(n_batch, -1,self.nOutputs), x2.repeat(1,n_nodes, 1)], dim=2).shape)
        hihj  = torch.cat([h2.repeat(1,1, n_nodes).view(n_batch, -1,self.nOutputs), x2.repeat(1,n_nodes, 1)], dim=2).view(n_batch,n_nodes,n_nodes, 2 * self.nOutputs)
       
        A = self.linear(hihj) 
       
        # Attention Mechanism       
       
        e       = self.leakyrelu(A)
        att = torch.matmul(e, self.a)        
        
        att = att.view(-1, n_nodes, n_nodes) # Reshape into n_batch*n_nodes*n_nodes tensor
        
        attention = F.softmax(att, dim=-1) # attention coefficients        

        zero_vec  = -9e15*torch.ones_like(attention)
        #attention = torch.where(attention > self.threshold, attention, zero_vec) # Thresholding attention matrix
        self.add2adjMat( attention.mean( 0) ) # Save attention matrix for analysis
        
        attention_block = torch.block_diag(*attention) # Duplicate atention vector in the diagonal for each batch
        
        h = torch.mm(x, self.W) # matrix multiplication
        h_prime   = torch.mm(attention_block, h)

        return h_prime   

class GCNlearnable(torch.nn.Module):

    def __init__(self, nInputs, nOutputs, n_nodes, bias = True):
        super(GCNlearnable, self).__init__() 

        self.in_features = nInputs
        self.out_features = nOutputs
        self.n_nodes = n_nodes
        self.threshold = 0.5

        torch.manual_seed(0)

        values = [0.25, 0.5, 1, 0.5]

        self.weight = nn.Parameter(torch.tensor(values).view(self.in_features, self.out_features))
        #nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.adj = nn.Parameter(torch.FloatTensor(self.n_nodes , self.n_nodes))
        nn.init.xavier_uniform_(self.adj.data, gain=1.414)
        self.adjTrue = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_adjMat(self):
        return self.adjTrue
             
    def forward(self, data):
    
        x = data.x
        batch= data.batch
       
        
        n_batch = batch.max().item() + 1   
        
        #support = F.relu(self.adj)
        #zero_vec  = -9e15*torch.ones_like(support)
        #support = torch.where(support > self.threshold, support, zero_vec)
        support = F.softmax(self.adj, dim=0) # attention coefficients

        self.adjTrue = support

        support = torch.block_diag(*self.adjTrue.repeat(n_batch,1, 1))

        h =  torch.mm(x, self.weight)
        h_prime   = torch.mm(support, h)
        h_prime = F.relu(h_prime)

        return h_prime      
   

          
