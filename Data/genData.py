import numpy as np
import networkx as nx
import math

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

    for i in range(start, end):
        inc = 2**(i-start)
        pos[i] = (x + inc, y-2)

def genData():  

    num_nodes = 20 # number of nodes    0
    num_samples = 1000 # number of samples   

    pos = {i: (0, 0) for i in range(num_nodes)} # node positions (x,y)  

    adj = np.zeros((num_nodes, num_nodes)) # adjacency matrix   

    adj = genClusterCycle(adj, 0, 10)  
    pos = genClusterCyclePos(pos, c=(10,10),r = 5 ,start=0,end=10 )
    
    adj = genClusterCycle(adj, 10, 15)  
    pos = genClusterCyclePos(pos, c=(1,10),r = 2 ,start=10,end=15 )  

    adj = genSink(adj, 15, [7,13]) 
    pos[15] = (0, 0)

    adj = genSource(adj, 15, range(16, 20) ) 

    

    G = nx.DiGraph(adj) # create graph from adjacency matrix  
    #pos = nx.spectral_layout(G)
    nx.draw(G, pos, with_labels=True) # draw graph


