# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:27:14 2024

@author: ksearle
"""
from cplex.callbacks import *
from docplex.mp.callbacks.cb_mixin import *
import networkx as nx
import numpy as np
from explicit_helper import *

class Callback_lazy(ConstraintCallbackMixin, LazyConstraintCallback):
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
        Callback function to be called for lazy constraint callback.

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
        
        # Compile all the arcs included in this solution
        arcs = []
        for (i,j,r) in (self.model_instance.x_keys):
            if sol_x.get_value(self.model_instance.x[i,j,r]) > -0:
                arcs.append((i,j,r))
        
        # Make a graph for each root r comprised of the arcs in the current solution
        G = {}
        for (i,j,r) in arcs:
            if r not in G:
                G[r] = nx.DiGraph()
                
            G[r].add_edge(i,j)
        
        # Check for violated cycles, add constraints to prevent them
        for r, graph in G.items():   
            # All the strongly connected components of the graph
            components = list(nx.simple_cycles(graph))
            
            if components == []:
                continue
            
            # Find the longest cycle that does not contain the root
            cycles_no_r = []
            for comp in components:
                
                if r not in comp:
                    
                    cycles_no_r.append(comp)
            
            max_cycle = max(cycles_no_r,key=len)
            
            if len(max_cycle) > 2:
                # Add the constraint for each vertex in S
                #print('violated cut, adding cut')
                for l in max_cycle:                
                    const = self.model_instance.model.sum(self.model_instance.x[i,j,r] for (i,j) in deltaSetMinus(max_cycle,r,self.problem_data)) >= self.model_instance.w[l,r]
                    const_cpx = self.linear_ct_to_cplex(const)  
                    ##print(const)
                    self.add(const_cpx[0], const_cpx[1], const_cpx[2])
                    
                #print('added lazy constraint')    

                  
class Callback_user(ConstraintCallbackMixin, UserCutCallback):
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
        Callback function to be called for user cut callback.

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
        sol_w = self.make_solution_from_vars(self.model_instance.w.values())
        
        # Make a list of all the arcs in this solution
        arcs = []
        for (i,j,r) in (self.model_instance.x_keys):
            if sol_x.get_value(self.model_instance.x[i,j,r]) > 0:
                arcs.append((i,j,r,{'capacity':min(1,sol_x.get_value(self.model_instance.x[i,j,r]))}))
                
        # Make a graph of the arcs assigned to each root, one graph per root
        G = {}
        for (i,j,r,cap) in arcs:
            if r not in G:
                G[r] = nx.DiGraph()
            G[r].add_edge(i,j,capacity=cap['capacity'])                
            
  
        for r, graph in G.items():    
            V_r = set(graph.nodes)
            # Loop over all vertices in this graph except for r
            for i in V_r:
                if i !=r:
                    # Find the mincut separating vertex i from root r
                    #print('computing min cut')
                    #print(f'i ={i}')
                    
                    # Check if r is in the graph, if yes then use min-cut from networkX
                    if r in V_r:
                        cut_value, partition = nx.minimum_cut(graph,r,i)
                    # If r is not in the graph then the min-cut is zero and the partition is r and then the nodes in the graph
                    else:
                        cut_value = 0
                        partition = [[r], list(V_r)]
                        
                    if cut_value < sol_w.get_value(self.model_instance.w[i,r]) - 1e-4:   
                        reachable, non_reachable = partition
                        
                        
                        for l in non_reachable:
                            const = self.model_instance.model.sum(self.model_instance.x[i,j,r] for (i,j) in deltaSetMinus(non_reachable,r,self.problem_data)) >= self.model_instance.w[l,r]
                            const_cpx = self.linear_ct_to_cplex(const)    
                            self.add(const_cpx[0], const_cpx[1], const_cpx[2])
                            
                        #print('added mincut constraint')

