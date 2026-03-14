# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:17:20 2024

@author: ksearle
"""

from docplex.mp.model import Model
from helper import *

class reduced_model:
    def __init__(self, name, data):
        """
        Initializes the reduced_model class.

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
        # x_ij is 1 if arc (i,j) is constructed
        self.x_keys = [(i,j) for (i,j) in data.A]
        self.x = self.model.binary_var_dict(self.x_keys, name = 'x')
        
        # Define objective function: minimize total cost for assigning satellite vertices to tree vertices and for constructing arcs
        self.model.minimize(self.model.sum(data.t[i,j]*self.x[i,j] for (i,j) in data.A) + self.model.sum(data.ac[i,j]*self.y[i,j] for i in data.V for j in data.V if i != j))
       
        # Define constraints
        
        # Each vertex is assigned to exactly one other vertex
        self.model.add_constraints(self.model.sum(self.y[i,j] for j in data.V) == 1 for i in data.V)
        # If a vertex has another vertex assigned to it then it is in the tree
        self.model.add_constraints(self.y[i,j] <= self.y[j,j] for i in data.V for j in data.V)
        # Total demand cannot exceed the upper bound
        self.model.add_constraints(self.model.sum(data.d[i]*self.y[i,j] for i in data.V) <= data.q*self.y[j,j] for j in data.V)
        # If a vertex is a tree vertex then it must have k incoming arcs
        self.model.add_constraints(self.model.sum(self.x[i,j] for (i,j) in delta_(j,self.data)) == data.k*self.y[j,j] for j in data.V)      
        # If an arc is constructed leaving a vertex then at least one must enter that vertex
        self.model.add_constraints(self.x[i,j] <= self.model.sum(self.x[l,i] for (l,i) in delta_(i,self.data) if l != j) for (i,j) in data.VxV)
        # Arcs cannot be connected in both directions
        self.model.add_constraints(self.x[i,j] + self.x[j,i] <= 1 for i in data.V for j in data.V if i < j)
    
    def solve(self, log=False):
        """
        Solves the optimization model.

        Args:
            log (bool): Whether to print solver log. Default is False.

        Returns:
            None
        """

        self.solution = self.model.solve(log_output=log)  # Solve the model with specified logging options
