import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import custom_plots as cp
import Visual_utils as vu
import pandas as pd

def genClusterCycle(adj, start, end):
    for i in range(start, end-1):
        adj[i, i+1] = 1     
    adj[end-1, start] = 1 

    return adj

def genClusterCyclePos(pos, c,r,start,end ):
    n= end- start 
    angle_increment = 2 * math.pi / n
    
    for i in range(n):
        angle = i * angle_increment
        x = c[0] + r * math.cos(angle)
        y = c[1] + r * math.sin(angle)
        pos[i+ start] = (x, y)
    
    return pos

def genSink(adj, node, listInput):
    for i in listInput:
        adj[i, node] = 1
    return adj

def genSource(adj, node, listOutput):

    for i in listOutput:
        adj[node, i] = 1 

    return adj

def postree(pos, origin,start, end):

    x,y = origin
    mid = np.round((end-start)/2 )

    # gerate a random intenger between 0 and 5 with seed 42
    np.random.seed(42)
    

    for i in range(start, end):
        rand = np.random.randint(0, 3)
        inc = -mid + i - start+2
        pos[i] = (x + 1*inc, y-(5+rand))

    return pos  

def posVerticaltree(pos, origin,start, end, dir = 1):

    x,y = origin
    mid = np.round((end-start)/2 )

    # gerate a random intenger between 0 and 5 with seed 42
    np.random.seed(42)    

    for i in range(start, end):
        rand = np.random.randint(0, 3)
        inc = -mid + i - start+2
        pos[i] = (x +  dir*(5+rand) , y + 2*dir*inc )

    return pos  


def genStimulus(A, x0, num_samples):

    nconn= np.sum(A, axis=0)
    nconn[nconn==0] = 1
    samples= num_samples
    timeseries = np.zeros((A.shape[0],samples*2), dtype=float)
    timeseries[:,0] = x0    
    x_inter = np.zeros_like(x0)
    for i in range(1,2*samples-1,2):

        x1 = A.T@x0
        
        x_prop_inter = A.T@x_inter

        x_prop_inter = x_prop_inter/nconn   

        x1 = x1/nconn

        

        indexmulti = np.where(  np.logical_and( x1+x_prop_inter>0.6 , x1+x_prop_inter<1) )
        x1[indexmulti] = 1  

        x1[x1<1] = 0
        
        x_inter = np.zeros_like(x1)
        x_inter[x0==1] = 0.5
        x_inter[x1==1] = 0.5
        #x1[x0==1] = 0.5
        
        timeseries[:,i] = x_inter

        timeseries[:,i+1] = x1

        #i = i+2

        x0 = x1
    
    return timeseries

def genStimulus2(A, x0, num_samples):

    nconn= np.sum(A, axis=0)
    nconn[nconn==0] = 1
    samples= num_samples
    timeseries = np.zeros((A.shape[0],samples), dtype=float)
    timeseries[:,0] = x0    
    x_inter = np.zeros_like(x0)
    for i in range(1,samples,1):

        x1 = A.T@x0
        
 

        x1 = x1/nconn


        x1[x1<0.7] = 0
        x1[x1>=0.7] = 1
        
        
        x1[x0==1] = 0.5
        
        timeseries[:,i] = x1

        #i = i+2

        x0 = x1
    
    return timeseries

def DrawGraph( G,pos = None, styleDark = False ):
    
    plt.figure()
    fig, axarr = plt.subplots(figsize=(20, 10), dpi= 80)

   
    if(styleDark):
        node_color = 'skyblue'
        edge_color='white'
    else:
        node_color = 'skyblue'
        edge_color='black'
    

    # create graph from adjacency matrix  
    if pos is None:
        pos = nx.spring_layout(G) # positions for all nodes
        #pos = nx.spectral_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=1200, edge_color=edge_color) # draw graph
    
    if(styleDark):
        plt.style.use('dark_background')
        fig.set_facecolor("#00000F")
    else:
        plt.style.use('default')

def genData():  

    num_nodes = 50 # number of nodes    0
    num_samples = 100 # number of samples   

    pos = {i: (0, 0) for i in range(num_nodes)} # node positions (x,y)  

    adj = np.zeros((num_nodes, num_nodes)) # adjacency matrix   

    adj = genClusterCycle(adj, 0, 10)  
    pos = genClusterCyclePos(pos, c=(10,10),r = 5 ,start=0,end=10 )
    
    adj = genClusterCycle(adj, 10, 15)  
    pos = genClusterCyclePos(pos, c=(1,10),r = 2 ,start=10,end=15 )  

    adj = genSink(adj, 15, [7,13]) 
    pos[15] = (0, 0)

    adj = genSource(adj, 15, range(16, 20) ) 
    pos = postree(pos, pos[15],16, 20)

    adj = genSink(adj, 20, [8,30]) 
    pos[20] = (11, 0)

    adj = genSource(adj, 20, range(21, 30) ) 
    pos = postree(pos, pos[20],21, 30)

    adj = genSource(adj, 1, range(30, 40) ) 
    pos = posVerticaltree(pos, pos[1],30, 40,  dir = 1)

    adj = genSource(adj, 12, range(40, 50) ) 
    pos = posVerticaltree(pos, pos[12],40, 50, dir = -1)

    

    G = nx.DiGraph(adj) 

    DrawGraph( G,pos = pos, styleDark = True )

    x0 = np.zeros(num_nodes, dtype=float)
    x0[10] = 1
    x0[12] = 1
    x0[6] = 1
    timeseries = genStimulus2(adj, x0,num_samples)
    x0 = np.zeros(num_nodes, dtype=float)
    x0[4] = 1
    x0[7] = 1
    timeseries=np.concatenate((timeseries,genStimulus2(adj, x0,num_samples)) , axis=1)
    x0 = np.zeros(num_nodes, dtype=float)
    x0[10] = 1
    timeseries=np.concatenate((timeseries, genStimulus2(adj, x0,num_samples)) , axis=1)

    x0 = np.zeros(num_nodes, dtype=float)
    x0[7] = 1
    x0[0] = 1
    timeseries=np.concatenate((timeseries, genStimulus2(adj, x0,num_samples)) , axis=1)
    df = pd.DataFrame(timeseries, index = ['Node'+str(i) for i in range(num_nodes)])
    x0 = np.zeros(num_nodes, dtype=float)
    x0[7] = 1
    x0[13] = 1
    x0[0] = 1
    timeseries=np.concatenate((timeseries, genStimulus2(adj, x0,num_samples)) , axis=1)
    df = pd.DataFrame(timeseries, index = ['Node'+str(i) for i in range(num_nodes)])
    
    #print(df)
    cp.plotMatrix(timeseries,'time', 'Node','Timeseries', 'timeseries_plot', styleDark = True)
    return timeseries, adj, pos
    #vu.createImage(pos,x0)
    


