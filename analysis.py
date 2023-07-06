import matplotlib.pyplot as plt
import networkx as nx  

def compare_Graphs(adj1, adj2, threshold=0, pos=None):
    """
    Compares the two graphs and returns the number of nodes and edges that are
    in both graphs.
    """

    print("adj2 shape: ", adj2.shape)

    adj2[adj2 < threshold] = 0

    real_graph = nx.DiGraph(adj1)
    predicted_graph = nx.DiGraph(adj2)

    sparcity1 = nx.density(real_graph)
    sparcity2 = nx.density(predicted_graph)

    print("sparcity Real Graph: ", sparcity1, " sparcity Predicted Graph: ", sparcity2)  

    intersected_edges = set(real_graph.edges()) & set(predicted_graph.edges())

    # Get the number of intersected values
    num_intersected_edges = len(intersected_edges)

    print("Number of well predicted edges: ", num_intersected_edges , 
          "Number of wrong predicted edges: ", len(predicted_graph.edges()) - num_intersected_edges ,
          "Number of edges in real graph: ", len(real_graph.edges()), 
          "Number of edges in predicted graph: ", len(predicted_graph.edges())
          )
    
    Draw2Graph(real_graph, predicted_graph, pos, styleDark = True ) 
    
    
def Draw2Graph( G1,G2,pos = None, styleDark = False ):
    
    plt.figure()
    fig, axarr = plt.subplots(figsize=(20, 10), dpi= 100)

   
    if(styleDark):
        node_color = 'skyblue'
        edge_color1='lightpink'
        edge_color2='white'
        text_color = 'white'
    else:
        node_color = 'skyblue'
        edge_color1='lightpink'
        edge_color2='black'
        text_color = 'black'
    

    # create graph from adjacency matrix  
    if pos is None:
        pos = nx.spring_layout(G1) # positions for all nodes
        #pos = nx.spectral_layout(G)
    #nx.draw(G1, pos, with_labels=True, node_color=node_color, node_size=1200, edge_color=edge_color1) # draw graph
    #nx.draw(G2, pos, with_labels=True, node_color=node_color, node_size=1200, edge_color=edge_color2) # draw graph
    
    nx.draw_networkx_edges(G1, pos, edge_color=edge_color1, alpha=1.0, width=2.0,connectionstyle="arc3,rad=0.1", ax=axarr)
    nx.draw_networkx_edges(G2, pos, edge_color=edge_color2, alpha=0.5, width=1.0, connectionstyle="arc3,rad=0.1", ax=axarr)
    #nx.draw_networkx_nodes(G1, pos, node_color=node_color, node_size=1200,,ax = axarr)
    nx.draw_networkx_labels(G1, pos, labels=None, font_size=12, font_color= text_color, font_family='Georgia', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=axarr)
    

    plt.text(-10, 25, "$\longrightarrow$ Predicted", size=25, rotation=0.,
         ha="left", va="center", color = edge_color2)
    plt.text(-10, 27, "$\longrightarrow$ Real", size=25, rotation=0.,
         ha="left", va="center", color = edge_color1)
    
    
    if(styleDark):
        plt.style.use('dark_background')
        fig.set_facecolor("#00000F")
    else:
        plt.style.use('default') 