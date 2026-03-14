# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:22:02 2024

@author: ksearle
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Create the delta- function for a single vertex
def delta_(b,data):
    """
    Finds which arcs in A end in vertex b
    
    Args:
    b (int): The vertex that is the endpoint of these arcs
    data (Data): Instance of the Data class containing problem data.


    Returns:
    List of all arcs in A whose endpoint is b
    """
    return [(i,j) for (i,j) in data.A if i != b and j == b]

# Create the delta+ function for a single vertex
def deltap_int(b,data):
    """
    Finds which arcs in A start at vertex b
    
    Args:
    b (int): The vertex that is the starting point of these arcs
    data (Data): Instance of the Data class containing problem data.


    Returns:
    List of all arcs in A whose endpoint is b
    """
    return [(i,j) for (i,j) in data.A if i == b and j != b]

# Create the delta+ function for a set
def deltap(S,data):
    """
    Finds which arcs in A start in set S
    
    Args:
    S (list): The vertices that are the starting points of the arcs
    data (Data): Instance of the Data class containing problem data.
    

    Returns:
    List of all arcs in A that start in S
    """
    return [(i,j) for (i,j) in data.A if i in S and j not in S]

def colour_graph(data, model):
    """
    Colours the solution graph to satisfy the requirements
    
    Args:
    data (Data): Instance of the Data class containing problem data.
    model: the solution found using CPLEX.


    Returns:
    T (list): Set of trees rooted at the roots that together make up the solution graph

    """
    # Create the solution graph made of the solution found with CPLEX
    G = nx.DiGraph()
    for (i,j) in data.A:
        if model.x[i,j].solution_value > 0.9:
            G.add_edge(i,j,capacity=1)
    
    # T is the set of trees for each root
    T = []
    
    # Create a source node with an arc going from it to each root
    source = data.n + data.roots
    source_arcs = [(source,r) for r in data.Ro]
    
    # Include the source arcs in the graph
    for (i,j) in source_arcs: 
        G.add_edge(i,j,capacity=1) 
    
    V_y = set(G.nodes)
    
    # Find the colouring for each root
    for r in data.Ro:
        # List of arcs in the tree starting at root r
        T_r = [(source, r)]
        
        # List of unvisited vertices
        U = [i for i in V_y if len([(a,b) for (a,b) in G.edges if a != i and b == i]) > 0]
            
        # Vertices on the tree we can branch from
        B = [r]
        
        while len(B) > 0:
            l = B[0]
            
            # Loop through arcs starting at l
            outgoing = [(a,b) for (a,b) in G.edges if a == l and b != l]
            for (l,j) in outgoing:
                if j in U and validate_arc(G, T_r, (l,j), source, data):
                    T_r.append((l,j))
                    B.append(j)
                    U.remove(j)
                    
            B.remove(l)
        
        # Add the tree just created to the list of trees    
        T.append(T_r)
        
        # Remove the tree starting at root r from the solution graph
        for arc in T_r:
            G.remove_edge(*arc)
            
    # Remove the source arcs from the list of trees
    for tree in T:
        for arc in tree:
            if arc in source_arcs:
                tree.remove(arc)
    
    return T
                
    

def validate_arc(G, T_r, arc, source, data):
    """
    Checks if the arc (l,j) can be added to tree T_r
    
    Args:
    G (DiGraph): The graph made of the optimal solution
    T_r (list): The tree rooted at root r so far
    arc (tuple): The arc we are checking if it can be added
    source (int): The source node
    data (Data): Instance of the Data class containing problem data.


    Returns:
    True if this arc can be added
    """
    H = G.copy()
    
    # Graph H should be the solution graph without the arcs in T_r and the given arc
    H.remove_edge(*arc)
    for (i,j) in T_r:
        H.remove_edge(i,j)    
    
    Vertices = set(H.nodes)
    
    Vertices.remove(source)
    
    # Check that each vertex is connected by enough paths without this arc
    for i in Vertices:
        cut_value, partition = nx.minimum_cut(H,source,i)
        
        # Find number of all arcs in the Graph ending in vertex i
        incoming = len([(a,b) for (a,b) in G.edges if a != i and b == i])
        if cut_value < incoming - 1:
            return False
        
    return True
        
def plot_solution(data, model):    
    # Colour the solution
    T = colour_graph(data, model)
    
    colours = ['blue', 'purple', 'orange', 'pink', 'black', 'green', 'black', 'brown','cyan','indigo']
    count = 0
    
    plt.figure()
    ax = plt.gca()
    
    # Plot the roots, vertices and tree arcs for each tree
    for tree in T:        
        for r in data.Ro:
            plt.scatter(data.loc[r][0], data.loc[r][1], c='blue')
            plt.annotate(r, (data.loc[r][0]+2, data.loc[r][1]))
        
        for i in data.V:
            plt.scatter(data.loc[i][0], data.loc[i][1], c='black')
            plt.annotate(i, (data.loc[i][0]+2, data.loc[i][1]))
        
        # Plot the arcs based on which tree they are in
        for (i,j) in tree:
            # Draw arrowed arc
            arrow = FancyArrowPatch(
                (data.loc[i][0], data.loc[i][1]),
                (data.loc[j][0], data.loc[j][1]),
                arrowstyle='-|>',
                color= colours[count],
                mutation_scale=10,
                linewidth=1)
            ax.add_patch(arrow)
            
        count += 1
                
    # Plot the customers assigned to each other
    for i in data.V:
        for j in data.V:
            if model.y[i,j].solution_value > 0.9 and i != j:
                arrow = FancyArrowPatch(
                    (data.loc[i][0], data.loc[i][1]),
                    (data.loc[j][0], data.loc[j][1]),
                    arrowstyle='-|>',
                    color= 'yellow',
                    mutation_scale=10,
                    linewidth=1,
                    linestyle="--"
                )
                ax.add_patch(arrow)
                
                
def plot_trees(G, Trees, sat_arcs, pos):

    W_col = ['blue', 'purple', 'orange', 'pink', 'black', 'green', 'black', 'brown','cyan','indigo']

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=150)

    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black')
    
    nx.draw_networkx_edges(sat_arcs, pos, edge_color='grey', style = '--', width=1, label='S edges')
    
    for i in range(len(Trees)):
        nx.draw_networkx_edges(Trees[i], pos, edge_color=W_col[i], width=1, label='T edges')


    plt.show()