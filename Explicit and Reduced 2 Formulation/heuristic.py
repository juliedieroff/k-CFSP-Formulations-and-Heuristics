"""
Created on Thu Oct 30 16:30:25 2025

@author: Julie
"""
import numpy as np
import networkx as nx
from helper import *
from collections import defaultdict
import time

def prim_trees(data):
    """
    Create minimum cost trees one root at a time using Prim's algorithm. 

    Args:
        data (Data): Instance of the Data class containing problem data.

    Returns:
        T (list): List of trees that make up the solution.
    """
    
    # List of all customers
    V = data.V
    
    # List of all roots
    Roots = data.Ro
    
    # List of minimum cost trees and their vertices
    T = []
    Vertices = []
    
    # List of all arcs on the digraph
    all_arcs = []
    
    # Must loop through every root
    for r in Roots:
        
        m = np.inf
        n = 0
        o = 0
        
        # Find the smallest cost arc starting at this root
        for j in V:
            if data.dist[r,j] < m:
                m = data.dist[r,j]
                o = j
        
        # The first arc of this tree will be this minimum cost arc
        T_r = [(r,o)]
        Vertices_r = [r,o]
        all_arcs.append((r,o))
                
        # Choose minimum cost arcs to build the tree, these arcs must be edge-disjoint from the other trees
        V_no_Vertices = [a for a in V if a not in Vertices_r]
        
        while len(V_no_Vertices) > 0:
            m = np.inf
            for k in Vertices_r:
                for j in V_no_Vertices:
                    if (k,j) not in all_arcs and (j,k) not in all_arcs:
                        if data.dist[k,j] < m:
                            m = data.dist[k,j]
                            n = k
                            o = j
                    
            T_r.append((n,o))
            Vertices_r.append(o)
            all_arcs.append((n,o))
            
            V_no_Vertices = [a for a in V if a not in Vertices_r]
             
            
        # Add this minimum cost tree starting at r to the list of all trees
        T.append(T_r)
        Vertices.append(Vertices_r)
        
    return T

def heuristic_sol(weight, data):
    """
    Create a feasible solution composed of edge-disjoint arborescences and satellite vertices. 

    Args:
        weight (array): The weights used for the arcs, usually distance.
        data (Data): Instance of the Data class containing problem data.

    Returns:
        T_old (list): List of trees in the feasible solution.
        S (list): List of arcs connecting the satellite vertices to the tree vertices.
        cost_old (float): The cost of building this feasible solution.
        timeH (float): The time it took to find the solution.
    """
    start = time.time()
    
    # List of all customers
    V = set(data.V)
    
    # List of all roots
    Roots = set(data.Ro)
    
    # List of satellite arcs
    S = []
    
    # Set of connector vertices for satellite vertices
    connectors = set()
    
    # Total weight of vertices assigned to each vertex
    cap = data.d.copy()
    
    # Compute all costs to build arcs from each root for each vertex
    score = {i: sum(data.t[r,i] for r in Roots) for i in V}
    
    # Only check the furthest half of vertices
    ordered_vertices = sorted(score.keys(), key=lambda x: score[x], reverse=True)
    portion = int(len(ordered_vertices) // (3/2 + 1/data.k))
    scored_vertices = ordered_vertices[:portion]
    
    # Build the initial tree
    T_old, cost_old = build_trees(V, Roots, weight, data)
    
    # Check if it would lower the cost by using satellite vertices
    for i in scored_vertices:   
        if i not in connectors:         
            # Only rebuild the whole tree if vertex i has outgoing arcs
            T_new, V_new, cost_new = new_tree(i, T_old, cost_old, V, Roots, weight, data)
            
            # Add satellite arc cost
            
            # Find closest tree vertex to create a satellite vertex
            sat_costs1 = data.ac[i].copy()
            for j in range(len(sat_costs1)):
                if j not in V_new:
                    sat_costs1[j] = np.inf
                if cap[j] > data.q:
                    sat_costs1[j] = np.inf
                    
            o = np.argmin(sat_costs1)
            
            cost_new += data.ac[i,o]
            cost_new += sum(data.ac[a,b] for (a,b) in S)
            
            # If cheaper, update solution
            if cost_new < cost_old:
                T_old = T_new
                cost_old = cost_new
                V = V_new
                
                # Check if we should also make the connector a satellite
                if o not in connectors:
                
                    # Only rebuild the whole tree if vertex o has outgoing arcs
                    T_new1, V_new1, cost_new1 = new_tree(o, T_old, cost_old, V, Roots, weight, data)
                    
                    # Find best satellite endpoints on the tree for both i and o
                    sat_costs2 = data.ac[o].copy()
                    for j in range(len(sat_costs2)):
                        if j not in V_new1:
                            sat_costs2[j] = np.inf
                        if cap[j] > data.q:
                            sat_costs2[j] = np.inf
                            
                    sat_costs3 = data.ac[i].copy()
                    for j in range(len(sat_costs3)):
                        if j not in V_new1:
                            sat_costs3[j] = np.inf
                        if cap[j] > data.q:
                            sat_costs3[j] = np.inf
                            
                    s = np.argmin(sat_costs2)
                    t = np.argmin(sat_costs3)
                    
                    cost_new1 += data.ac[o,s] + data.ac[i,t]
                    cost_new1 += sum(data.ac[a,b] for (a,b) in S)
                    
                    if cost_new1 < cost_old:
                        T_old = T_new1
                        cost_old = cost_new1
                        V = V_new1
                        cap[s] += data.d[o]
                        cap[t] += data.d[i]
                        S.append((o, s))
                        S.append((i, t))
                        connectors.add(s)
                        connectors.add(t)
                
                    else:
                        cap[o] += data.d[i]
                        S.append((i, o))
                        connectors.add(o)
                else:
                    cap[o] += data.d[i]
                    S.append((i, o))
                        
    end = time.time()
    timeH = end - start
    return T_old, S, cost_old, timeH

def new_tree(i, T_old, cost_old, V, Roots, weight, data):
    """
    Only recompute the tree if there are outgoing arcs from vertex i

    """
    # Calculate cost change if we remove vertex i
    V_new = V - {i}
    
    # Check if vertex i has any outgoing arcs in the current trees
    outgoing = False
    for tree in T_old:
        for arc in tree:
            if arc[0] == i:
                outgoing = True
                break
        if outgoing:
            break
    
    if not outgoing:
        # If no arcs are outgoing from i you don't need to rebuild the whole tree
        T_new = [tree.copy() for tree in T_old]
        cost_new = cost_old
        
        for tree in T_new:
            arcs_to_remove = [arc for arc in tree if i in arc]
            for arc in arcs_to_remove:
                tree.remove(arc)
                cost_new -= data.t[arc[0], arc[1]]
    else:
        # Full rebuild needed
        T_new, cost_new = build_trees(V_new, Roots, weight, data)

        
    return T_new, V_new, cost_new

def lp_heuristic(on_tree, sat_verts, weight, data):
    """
    Using information from the problem's LP relaxation create a feasible solution 
    composed of edge-disjoint arborescences and satellite vertices. 

    Args:
        on_tree (list): Which vertices should be on the tree based on the LP relaxation.
        sat_verts (list): Which vertices should be satellite vertices based on the LP relaxation.
        weight (array): The weights used for the arcs, usually distance.
        data (Data): Instance of the Data class containing problem data.

    Returns:
        T (list): List of trees in the feasible solution.
        S (list): List of arcs connecting the satellite vertices to the tree vertices.
        cost (float): The cost of building this feasible solution.
    """   
    # List of customers on the tree
    on_tree = set(on_tree)
    
    # List of customers not on the tree
    sat_verts = set(sat_verts)
    
    # List of all roots
    Roots = set(data.Ro)
    
    # List of satellite arcs
    S = []
    
    # Total weight of vertices assigned to each vertex
    cap = data.d.copy()
    
    # Build the tree
    T, cost = build_trees(on_tree, Roots, weight, data)
    
    # Compute minimum satellite costs for customers not on the tree
    for i in sat_verts:
        sat_costs = data.ac[i].copy()
        for j in range(len(sat_costs)):
            if j not in on_tree:
                sat_costs[j] = np.inf
            if cap[j] > data.q:
                sat_costs[j] = np.inf
                
        o = np.argmin(sat_costs)
    
        cost += data.ac[i,o]
        cap[o] += data.d[i]
        S.append((i,o))
    return T, S, cost
    
    

def build_trees(V, Roots, weight, data):
    """
    Create minimum cost trees simultaneously with improved performance.

    Args:
        V (set): Set of all customers to be included in the trees.
        Roots (list): List of roots the trees start from.
        weight (array): The weights used for the arcs, usually distance.
        data (Data): Instance of the Data class containing problem data.

    Returns:
        T (list): List of trees that make up the solution.
        cost (float): The total cost of building this solution.
    """
    V = set(V)
    
    # Initialize data structures
    T = [[] for r in Roots]
    Vertices = [set() for r in Roots]
    vertex_tree_counts = defaultdict(int)
    all_arcs = set()
    cost = 0
    
    # Set of unassigned vertices
    unassigned = V.copy()
    
    # Start each tree with minimum cost arc from root
    for idx, r in enumerate(Roots):
        feasible_weights = weight[r].copy()
        for j in range(len(feasible_weights)):
            if j not in unassigned:
                feasible_weights[j] = np.inf
                
        j = np.argmin(feasible_weights)
        
        # Create the shortest arcs in the trees
        T[idx].append((r, j))
        Vertices[idx].add(r)
        Vertices[idx].add(j)
        all_arcs.add((r, j))
        cost += data.t[r, j]
        
        vertex_tree_counts[j] += 1
                
        if vertex_tree_counts[j] == data.k:
            unassigned.discard(j)
    
    # Build trees by finding minimum cost arcs
    while unassigned:
        min_cost = np.inf
        best_arc = None
        best_tree_idx = -1
        
        # Find the globally minimum cost arc across all trees
        for tree_idx, vertices in enumerate(Vertices):
            for i in vertices:
                for j in unassigned:
                    if j not in vertices:
                        # Check arc doesn't exist already and isn't reversed
                        if (i, j) not in all_arcs and (j, i) not in all_arcs:
                            arc_cost = weight[i, j]
                            if arc_cost < min_cost:
                                min_cost = arc_cost
                                best_arc = (i, j)
                                best_tree_idx = tree_idx
                
        i, j = best_arc
        T[best_tree_idx].append(best_arc)
        Vertices[best_tree_idx].add(j)
        all_arcs.add(best_arc)
        cost += data.t[i, j]
        
        vertex_tree_counts[j] += 1
        
        if vertex_tree_counts[j] == data.k:
            unassigned.discard(j)
    
    return T, cost

def feasibility_check(T, data):
    """
    Checks if a proposed solution is feasible; formed of edge-disjoint directed arborescences.
    Returns True if feasible, False otherwise.

    Args:
        T (list): List of trees.
        data (Data): Instance of the Data class containing problem data.
    """
    # Count the number of checks passed
    checks_passed = 0
    
    # Check if the given digraph is edge-disjoint
    same_edges = []
    
    for tree in T:
        for other in T:
            if tree != other:
                repeats = [(i,j) for (i,j) in tree if (i,j) in other or (j,i) in other]
                if repeats:
                    same_edges.append(repeats)   
    
    if not same_edges:
        checks_passed += 1
    else:
        print('Not edge-disjoint')
        
    # Check the digraph is made of arborescences (no cycles, every vertex acccessible from each r, 
    # the roots are not accessible from any other vertex)
    
    # Check for cycles
    # Turn the list of edges into digraphs
    Trees = []
    all_cycles = []
    
    for r in data.Ro:
        T_r = nx.DiGraph()
        for (i,j) in T[r]:   
            T_r.add_edge(i,j)
            
        Trees.append(T_r)
    
    # Create list of cycles detected
    for G in Trees:
        cycles = sorted(nx.simple_cycles(G))
        if cycles:
            all_cycles.append(cycles)
            
    if not all_cycles:
        checks_passed += 1
    else:
        print('Contains cycles')

    
    # Check if every vertex is accessible from each root
    disconnected_vertices = 0
    disconnected = []
    
    for r in data.Ro:
        for i in Trees[r].nodes:
            if i not in data.Ro: 
                if nx.has_path(Trees[r], r, i) != True:
                    disconnected_vertices += 1
                    disconnected.append(i)
                
    if disconnected_vertices == 0:
           checks_passed += 1
    else:
        print('Not every vertex is accessible from the roots')
        print(f'Disconnected vertices: {disconnected}')
        
    # Check the roots are not accessible from any other vertex
    connected_roots = 0
    
    for r in data.Ro:
        for i in Trees[r].nodes:
            if i not in data.Ro: 
                if nx.has_path(Trees[r], i, r) == True:
                    connected_roots += 1
                
    if connected_roots == 0:
           checks_passed += 1
    else:
        print('At least one root is accessible from a vertex')
    
    # If all checks have been passed then return true, this is a feasible solution
    if checks_passed == 4:
        return True
    else:
        return False
    