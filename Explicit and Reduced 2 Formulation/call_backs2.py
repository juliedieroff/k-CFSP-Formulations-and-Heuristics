# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:27:14 2024

@author: ksearle
"""
from cplex.callbacks import *
from docplex.mp.callbacks.cb_mixin import *
import networkx as nx
import numpy as np
from helper import *
from heuristic import *

class Callback_lazy2(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        """
        Initializes the Callback_lazy class.

        Args:
            env: CPLEX environment.

        Returns:
            None
        """
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        
    def __call__(self):
        """
        Callback function to be called for lazy constraint callback with more efficient constraints.

        Returns:
            None
        """
        #print('running lazy callback')
        
        # INFORMATION FOR PERFORMANCE SHEET
        gap = self.get_MIP_relative_gap()
        #print(f'gap = {gap}')
        num_nodes = self.get_num_nodes()
        #print(f'num_nodes = {num_nodes}')
        
        upper_bounds = self.get_incumbent_objective_value()
        #print(f'upper bound = {upper_bounds}')
        
        lower_bounds = self.get_objective_value()
        #print(f'lower_bounds = {lower_bounds}')
        
        if num_nodes == 0:
            self.root_MIP_gap = gap
            self.root_UB = upper_bounds
            self.root_LB = lower_bounds
        
        self.num_calls += 1
        
        # Get the current solution
        sol_x = self.make_solution_from_vars(self.model_instance.x.values())
        sol_y = self.make_solution_from_vars(self.model_instance.y.values())
        
        
        # Make a list of all the arcs in this solution
        arcs = []
        
        for (i,j) in (self.model_instance.x_keys):
            if sol_x.get_value(self.model_instance.x[i,j]) > 0:
                arcs.append((i,j,{'capacity':min(1,sol_x.get_value(self.model_instance.x[i,j]))}))
        
        # Create a source and sink node and arcs with capacity 1 between the source and the roots
        source = self.problem_data.n + self.problem_data.roots
        sink = source + 1
        source_arcs = [(source,r,{'capacity':1}) for r in self.problem_data.Ro]
        
        # Include the source arcs in the graph
        for arc in source_arcs: 
            arcs.append(arc) 
        
        # Make a graph of the arcs including capacity
        G = nx.DiGraph()
        for (i,j,cap) in arcs:
            G.add_edge(i,j,capacity=cap['capacity'])
         
            
        Vertices = set(G.nodes)
        
        # Loop over all vertices in this graph except for the roots and the source
        for i in Vertices:
            
            if i != source and i != sink and i not in self.problem_data.Ro:
                
                # Create the sink arcs with capacity k*y_i- between the vertices and the sink
                sink_arcs = [(j,sink,{'capacity':self.problem_data.k*sol_y.get_value(self.model_instance.y[i,j])}) for j in Vertices if j != source and j != sink and j not in self.problem_data.Ro]
                
                # Add the sink arcs to the graph
                for (j,k,cap) in sink_arcs:
                    G.add_edge(j,k,capacity=cap['capacity'])
                
                # Find the mincut separating vertex i from the source
                cut_value, partition = nx.minimum_cut(G,source,sink)
                reachable, non_reachable = partition
                
                if cut_value < self.problem_data.k and i in non_reachable:                 
                    #print("inequality violated, constraint must be added")
                    
                    # Find the number of roots that are not in S
                    R_no_S = len([a for a in self.problem_data.Ro if a not in reachable])
                    
                    # Find the vertices (no roots or source) that are not in S
                    V_no_S = [b for b in self.problem_data.V if b not in reachable]
                    
                    for l in V_no_S:
                        const = self.model_instance.model.sum(self.model_instance.x[i,j] for (i,j) in deltap(reachable,self.problem_data)) + R_no_S + self.problem_data.k*self.model_instance.model.sum(self.model_instance.y[l,m] for m in reachable if m in self.problem_data.V) >= self.problem_data.k 
                        const_cpx = self.linear_ct_to_cplex(const)    
                        self.add(const_cpx[0], const_cpx[1], const_cpx[2])     
                            
                    #print('added lazy constraint')
                
                # Delete the arcs to this sink for the next iteration
                for (j,k,cap) in sink_arcs:
                    G.remove_edge(j,k)

              
class Callback_user2(ConstraintCallbackMixin, UserCutCallback):
    def __init__(self, env):
        """
        Initializes the Callback_user class.

        Args:
            env: CPLEX environment.

        Returns:
            None
        """
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        
    def __call__(self):
        """
        Callback function to be called for user cut callback with more efficient constraints.

        Returns:
            None
        """
        #print('running user callback')
        
        
        # INFORMATION FOR PERFORMANCE SHEET
        gap = self.get_MIP_relative_gap()
        #print(f'gap = {gap}')
        num_nodes = self.get_num_nodes()
        #print(f'num_nodes = {num_nodes}')
        
        upper_bounds = self.get_incumbent_objective_value()
        #print(f'upper bound = {upper_bounds}')
        
        lower_bounds = self.get_objective_value()
        #print(f'lower_bounds = {lower_bounds}')
        
        if num_nodes == 0:
            self.root_MIP_gap = gap
            self.root_UB = upper_bounds
            self.root_LB = lower_bounds
        
        # Get the current solution
        sol_x = self.make_solution_from_vars(self.model_instance.x.values())
        sol_y = self.make_solution_from_vars(self.model_instance.y.values())
        
        # Look for violated cuts and add the constraints

        # Make a list of all the arcs in this solution
        arcs = []
        # #print('fetching arcs')
        for (i,j) in (self.model_instance.x_keys):
            if sol_x.get_value(self.model_instance.x[i,j]) > 0:
                arcs.append((i,j,{'capacity':min(1,round(sol_x.get_value(self.model_instance.x[i,j]), 6))}))
        
        # Create a source and sink node and arcs with capacity 1 between the source and the roots
        source = self.problem_data.n + self.problem_data.roots
        sink = source + 1
        source_arcs = [(source,r,{'capacity':1}) for r in self.problem_data.Ro]
        
        # Include the source arcs in the graph
        for arc in source_arcs: 
            arcs.append(arc) 
        
        # Make a graph of the arcs including capacity
        G = nx.DiGraph()
        for (i,j,cap) in arcs:
            G.add_edge(i,j,capacity=cap['capacity'])
         
            
        Vertices = set(G.nodes)
        
        # Loop over all vertices in this graph except for the roots and the source
        for i in Vertices:
            
            if i != source and i != sink and i not in self.problem_data.Ro:
                
                # Create a sink arcs with capacity k*y_i- between the sink and the vertices
                sink_arcs = [(j,sink,{'capacity':round(self.problem_data.k*sol_y.get_value(self.model_instance.y[i,j]), 6)}) for j in Vertices if j != source and j != sink and j not in self.problem_data.Ro]
                
                # Add the sink arcs to the graph
                for (j,k,cap) in sink_arcs:
                    G.add_edge(j,k,capacity=cap['capacity'])
                
                # Find the mincut separating vertex i from the source
                # #print('computing min cut')
                cut_value, partition = nx.minimum_cut(G,source,sink)
                reachable, non_reachable = partition
                
                # #print(f'cut value = {cut_value}')
                if round(cut_value, 5) < self.problem_data.k - 1e-4 and i in non_reachable and len(reachable) > 2:
                    
                    # Find the number of roots that are not in S
                    R_no_S = len([a for a in self.problem_data.Ro if a not in reachable])
                    
                    # Find the vertices (no roots or source) that are not in S
                    V_no_S = [b for b in self.problem_data.V if b not in reachable]
                    # #print(f'V_no_S = {V_no_S}')
                    for l in V_no_S:
                
                        const = self.model_instance.model.sum(self.model_instance.x[i,j] for (i,j) in deltap(reachable,self.problem_data)) >= self.model_instance.model.sum(self.model_instance.y[l,m] for m in V_no_S)*(self.problem_data.k - R_no_S) 
                        
                        
                        const_cpx = self.linear_ct_to_cplex(const)    

                        self.add(const_cpx[0], const_cpx[1], const_cpx[2])     

                # Delete the arcs to this sink for the next iteration
                for (j,k,cap) in sink_arcs:
                    G.remove_edge(j,k)

class HeuristicsCallback(ConstraintCallbackMixin, HeuristicCallback):
    def __init__(self, env):
       """
       Initializes the HeuristicCallback class.

       Args:
           env: CPLEX environment.

       Returns:
           None
       """ 
       HeuristicCallback.__init__(self, env)
       ConstraintCallbackMixin.__init__(self)
    
    def __call__(self):
       """
       Callback function to be called once the LP relaxation has been found to run the heuristic.
       Uses the y_ij values found.

       Returns:
           None
       """
       
       self.count_heur = 0
       
       # Run heuristic at node 0
       num_nodes = self.get_num_nodes()
        
       #print(self.get_num_remaining_nodes())
       
       if num_nodes == 1 or num_nodes == 2:
           # Get the current solution
           sol_x = self.make_solution_from_vars(self.model_instance.x.values())
           sol_y = self.make_solution_from_vars(self.model_instance.y.values())
            
           # Run the specified heuristic (1 for Y, 2 for X, 3 for X and Y)
           if self.heuristic_choice == 1:                   
                self.count_heur += 1
            
                #print('running Y heuristic callback')
                
                # HEURISTIC BASED ON THE Y VARIABLES IN THE LP RELAXATION
                # Find which customers should be on the tree
                on_tree = []
                for i in self.problem_data.V:
                    if sol_y.get_value(self.model_instance.y[i,i]) > 0.5:
                        on_tree.append(i)
                 
                # All other customers are satellite vertices
                sat_verts = [j for j in self.problem_data.V if j not in on_tree]
                #print('*****************************************')
                #print(on_tree)
                #print(sat_verts)
                #print('*****************************************')
                 
                # Run the heuristic to find a feasible solution
                T, S, cost = lp_heuristic(on_tree, sat_verts, self.problem_data.dist, self.problem_data)
                 
                # Translate this solution into the model's format
                variable = []
                solution = []
                 
                for (i, j) in self.model_instance.y:
                    variable.append(self.model_instance.y[i, j].name)
                     
                    # Find which vertices are on the trees versus assigned to eachother
                    if i == j:
                        if i in on_tree:
                            solution.append(1)
                        else:
                            solution.append(0)
                             
                    if i != j:
                        if (i,j) in S:
                            solution.append(1)
                        else:
                            solution.append(0)
                 
                for (i, j) in self.model_instance.x:
                    variable.append(self.model_instance.x[i, j].name)
                     
                    # Find if this arc is on any tree
                    count = 0
                    for tree in T:
                        if (i,j) in tree:
                            count = 1        
                    if count == 1:
                        solution.append(1)
                    else:
                        solution.append(0)
                 
                # Set the heuristic solution in the solver
                self.set_solution([variable, solution], cost) 
                #print('heuristic solution added')
               
           if self.heuristic_choice == 2:
               #print('running X heuristic callback')
               
               epsilon = 0.0001
               
               # CHANGE THE WEIGHTS OF THE ARCS BASED ON THE LP RELAXATION
               new_weights = self.problem_data.t.copy()
               for (i, j) in self.model_instance.x:
                    new_weights[i,j] = new_weights[i,j]/(sol_x.get_value(self.model_instance.x[i,j]) + epsilon)
                    
                
                
               # Build a solution with these weights
               T, S, cost, timeH = heuristic_sol(new_weights, self.problem_data)
                
               # Vertices on the tree
               tree_verts = list({a for tree in T for pair in tree for a in pair})
                
               # Translate this solution into the model's format
               variable = []
               solution = []
                 
               for (i, j) in self.model_instance.y:
                   variable.append(self.model_instance.y[i, j].name)
                     
                   # Find which vertices are on the trees versus assigned to eachother
                   if i == j:
                       if i in tree_verts:
                           solution.append(1)
                       else:
                           solution.append(0)
                             
                   if i != j:
                       if (i,j) in S:
                           solution.append(1)
                       else:
                           solution.append(0)
                 
               for (i, j) in self.model_instance.x:
                   variable.append(self.model_instance.x[i, j].name)
                     
                   # Find if this arc is on any tree
                   count = 0
                   for tree in T:
                       if (i,j) in tree:
                           count = 1        
                   if count == 1:
                       solution.append(1)
                   else:
                       solution.append(0)
                 
               # Set the heuristic solution in the solver
               self.set_solution([variable, solution], cost) 
               #print('heuristic solution added')  
                   
           if self.heuristic_choice == 3:
               
               #print('running X and Y heuristic callback')
                
               epsilon = 0.0001
                
               # Find which customers should be on the tree
               on_tree = []
               for i in self.problem_data.V:
                   if sol_y.get_value(self.model_instance.y[i,i]) > 0.5:
                       on_tree.append(i)
                 
               # All other customers are satellite vertices
               sat_verts = [j for j in self.problem_data.V if j not in on_tree]
               #print('*****************************************')
               #print(on_tree)
               #print(sat_verts)
               #print('*****************************************')
                
               new_weights = self.problem_data.dist.copy()
               for (i, j) in self.model_instance.x:
                   if i in on_tree and j in on_tree:
                       new_weights[i,j] = new_weights[i,j]/(sol_x.get_value(self.model_instance.x[i,j]) + epsilon)
                 
                  
               # Run the heuristic to find a feasible solution
               T, S, cost = lp_heuristic(on_tree, sat_verts, new_weights, self.problem_data)
               
               # Translate this solution into the model's format
               variable = []
               solution = []
                 
               for (i, j) in self.model_instance.y:
                   variable.append(self.model_instance.y[i, j].name)
                     
                   # Find which vertices are on the trees versus assigned to eachother
                   if i == j:
                       if i in on_tree:
                           solution.append(1)
                       else:
                           solution.append(0)
                            
                   if i != j:
                       if (i,j) in S:
                           solution.append(1)
                       else:
                           solution.append(0)
                 
               for (i, j) in self.model_instance.x:
                   variable.append(self.model_instance.x[i, j].name)
                    
                   # Find if this arc is on any tree
                   count = 0
                   for tree in T:
                       if (i,j) in tree:
                           count = 1        
                   if count == 1:
                       solution.append(1)
                   else:
                       solution.append(0)
                 
               # Set the heuristic solution in the solver
               self.set_solution([variable, solution], cost) 
               #print('*****************************************')
               #print('XY heuristic solution added') 
               #print('*****************************************')
            
        