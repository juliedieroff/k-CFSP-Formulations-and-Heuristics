# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:22:02 2024

@author: ksearle
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# Create the delta- function for a single vertex
def delta_r(b,r,data):
    """
    Finds which arcs in A^r end in vertex b
    
    Args:
    b (int) : The vertex that is the endpoint of these arcs.
    r (int) : The root to be used in A^r.
    data (Data): Instance of the Data class containing problem data.

    Returns:
    List of all arcs in A^r whose endpoint is b
    """
    return [(i,j) for (i,j) in data.Ar[r] if i != b and j == b]

# Create the delta+ function for a single vertex
def deltapr(b,r,data):
    """
    Finds which arcs in A^r start in vertex b
    
    Args:
    b (int) : The vertex that is the starting point of these arcs.
    r (int) : The root to be used in A^r.
    data (Data): Instance of the Data class containing problem data.

    
    Returns:
    List of all arcs in A that start at S
    """
    return [(i,j) for (i,j) in data.Ar[r] if i == b and j != b]

# Create the delta-r function for a set
def deltaSetMinus(S,r,data):
    """
    Finds which arcs in A end in set S
    
    Args:
    S (list): The vertices that are the endpoints of the arcs
    r (int) : The root to be used in A^r.
    data (Data): Instance of the Data class containing problem data.


    Returns:
    List of all arcs in A that end in S
    """
    return [(i,j) for (i,j) in data.Ar[r] if i not in S and j in S]


def explicit_plot_solution(data, model):
    colours = ['red', 'green', 'orange', 'grey']
    
    plt.figure()
    ax = plt.gca()
    
    # Plot the roots, vertices and tree arcs
    for r in data.Ro:
        plt.scatter(data.loc[r][0], data.loc[r][1], c='blue')
        plt.annotate(r, (data.loc[r][0]+2, data.loc[r][1]))
    
        for i in data.V:
            plt.scatter(data.loc[i][0], data.loc[i][1], c='black')
            plt.annotate(i, (data.loc[i][0]+2, data.loc[i][1]))
        
        for (i,j) in data.Ar[r]:
            if model.x[i,j,r].solution_value > 0.9:
                # Draw arrowed arc
                arrow = FancyArrowPatch(
                    (data.loc[i][0], data.loc[i][1]),
                    (data.loc[j][0], data.loc[j][1]),
                    arrowstyle='-|>',
                    color=colours[r],
                    mutation_scale=10,
                    linewidth=1
                )
                ax.add_patch(arrow)
                
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
    
