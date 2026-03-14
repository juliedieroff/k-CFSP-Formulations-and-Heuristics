# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:17:20 2024

@author: ksearle
"""

from docplex.mp.model import Model
from explicit_helper import *

class explicit_flow_model:
    def __init__(self, name, data):
        """
        Initializes the explicit_flow_model class.

        Args:
            name (str): Name of the model.
            data (Data): Instance of the Data class containing problem data.

        Returns:
            None
        """

        self.data = data  # Instance of Data class containing problem data
        
        
        self.model = Model(name)  # Create a new optimization model with the given name
        
        # Define the decision variables
        # y_ij is 1 if vertex i is assigned to vertex j, 0 otherwise
        self.y_keys = [(i,j) for i in data.V for j in data.V]
        self.y = self.model.binary_var_dict(self.y_keys, name = 'y')
        # x_ij^r is 1 if arc (i,j) is in the tree rooted at r, 0 otherwise
        self.x_keys = [(i,j,r) for r in data.Ro for (i,j) in data.Ar[r]]
        self.x = self.model.binary_var_dict(self.x_keys, name='x')
        # w_i^r is 1 if vertex i is in the tree rooted at r, 0 otherwise
        self.w_keys = [(i,r) for r in data.Ro for i in data.V]
        self.w = self.model.binary_var_dict(self.w_keys, name='w')
        # f_ij^r are the flow variables which are the number of units of flow of r along arc(i,j)
        self.f_keys = [(i,j,r) for r in data.Ro for (i,j) in data.Ar[r]]
        self.f = self.model.integer_var_dict(self.f_keys, name='f')
        
        # Define objective function: minimize total cost for assigning satellite vertices to tree vertices and for constructing arcs
        self.model.minimize(self.model.sum(data.t[i,j]*self.x[i,j,r] for r in data.Ro for (i,j) in data.Ar[r]) + self.model.sum(data.ac[i,j]*self.y[i,j] for i in data.V for j in data.V if i != j))
       
        # Define constraints
        
        # Each vertex is assigned to exactly one other vertex
        self.model.add_constraints(self.model.sum(self.y[i,j] for j in data.V) == 1 for i in data.V)
        # If a vertex has another vertex assigned to it then it is in the tree
        self.model.add_constraints(self.y[i,j] <= self.y[j,j] for i in data.V for j in data.V)
        # Total demand cannot exceed the upper bound
        self.model.add_constraints(self.model.sum(data.d[i]*self.y[i,j] for i in data.V) <= data.q*self.y[j,j] for j in data.V)
        # If a vertex is a tree vertex then it must be included in k trees
        self.model.add_constraints(self.model.sum(self.w[j,r] for r in data.Ro) == data.k*self.y[j,j] for j in data.V)                  
        # If a vertex is on tree r then an arc from tree r must enter that vertex
        self.model.add_constraints(self.model.sum(self.x[i,j,r] for (i,j) in delta_r(j,r,self.data)) == self.w[j,r] for j in data.V for r in data.Ro)      
        # An arc can only be in a tree if there is another arc from that tree leading into it
        self.model.add_constraints(self.x[i,j,r] <= self.model.sum(self.x[l,i,r] for (l,i) in delta_r(i,r,self.data) if l != j) for r in data.Ro for (i,j) in data.Ar[r] if i != r)
        # An arc can be in at most one tree and in one direction
        self.model.add_constraints(self.model.sum(self.x[i,j,r] + self.x[j,i,r] for r in data.Ro) <= 1 for i in data.V for j in data.V if i < j)
        
        # Flow constraints
        # The amount leaving root r must be equal to the number of tree vertices in that tree
        self.model.add_constraints(self.model.sum(self.f[r,j,r] for j in data.V) == self.model.sum(self.w[j,r] for j in data.V) for r in data.Ro)
        # Enforce the flow balance at each vertex
        self.model.add_constraints(self.model.sum(self.f[i,j,r] for (i,j) in delta_r(j,r,self.data)) - self.model.sum(self.f[j,i,r] for (j,i) in deltapr(j,r,self.data)) == self.w[j,r] for j in data.V for r in data.Ro)
        # Flow of r can only be along arcs on tree r
        self.model.add_constraints(self.f[i,j,r] <= data.n*self.x[i,j,r] for r in data.Ro for (i,j) in data.Ar[r])
                          
    
    def solve_flow(self, log=False):
        """
        Solves the optimization model.

        Args:
            log (bool): Whether to print solver log. Default is False.

        Returns:
            None
        """

        self.solution = self.model.solve(log_output=log)  # Solve the model with specified logging options

class explicit_cut_model:
    def __init__(self, name, data):
        """
        Initializes the explicit_cut_model class.

        Args:
            name (str): Name of the model.
            data (Data): Instance of the Data class containing problem data.

        Returns:
            None
        """

        self.data = data  # Instance of Data class containing problem data
        
        self.model = Model(name)  # Create a new optimization model with the given name
        
        # Define the decision variables
        # y_ij is 1 if vertex i is assigned to vertex j, 0 otherwise
        self.y_keys = [(i,j) for (i,j) in data.VxV]
        self.y = self.model.binary_var_dict(self.y_keys, name = 'y')
        # x_ij^r is 1 if arc (i,j) is in the tree rooted at r, 0 otherwise
        self.x_keys = [(i,j,r) for r in data.Ro for (i,j) in data.Ar[r]]
        self.x = self.model.binary_var_dict(self.x_keys, name='x')
        # w_i^r is 1 if vertex i is in the tree rooted at r, 0 otherwise
        self.w_keys = [(i,r) for r in data.Ro for i in data.V]
        self.w = self.model.binary_var_dict(self.w_keys, name='w')

        # Define objective function: minimize total cost for assigning satellite vertices to tree vertices and for constructing arcs
        self.model.minimize(self.model.sum(data.t[i,j]*self.x[i,j,r] for r in data.Ro for (i,j) in data.Ar[r]) + self.model.sum(data.ac[i,j]*self.y[i,j] for i in data.V for j in data.V if i != j))
       
        # Define constraints
        
        # Each vertex is assigned to exactly one other vertex
        self.model.add_constraints(self.model.sum(self.y[i,j] for j in data.V) == 1 for i in data.V)
        # If a vertex has another vertex assigned to it then it is in the tree
        self.model.add_constraints(self.y[i,j] <= self.y[j,j] for i in data.V for j in data.V)
        # Total demand cannot exceed the upper bound
        self.model.add_constraints(self.model.sum(data.d[i]*self.y[i,j] for i in data.V) <= data.q*self.y[j,j] for j in data.V)
        # If a vertex is a tree vertex then it must be included in k trees
        self.model.add_constraints(self.model.sum(self.w[j,r] for r in data.Ro) == data.k*self.y[j,j] for j in data.V)                  
        # If a vertex is on tree r then an arc from tree r must enter that vertex
        self.model.add_constraints(self.model.sum(self.x[i,j,r] for (i,j) in delta_r(j,r,self.data)) == self.w[j,r] for j in data.V for r in data.Ro)      
        # An arc can only be in a tree if there is another arc from that tree leading into it
        self.model.add_constraints(self.x[i,j,r] <= self.model.sum(self.x[l,i,r] for (l,i) in delta_r(i,r,self.data) if l != j) for r in data.Ro for (i,j) in data.Ar[r] if i != r)
        # An arc can be in at most one tree and in one direction
        self.model.add_constraints(self.model.sum(self.x[i,j,r] + self.x[j,i,r] for r in data.Ro) <= 1 for i in data.V for j in data.V if i < j)
        
    def solve_cut(self, log=False):
        """
        Solves the optimization model.

        Args:
            log (bool): Whether to print solver log. Default is False.

        Returns:
            None
        """

        self.solution = self.model.solve(log_output=log)  # Solve the model with specified logging options
