# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:10:01 2024

@author: ksearle
"""
import numpy as np
import math

class Data:
    def __init__(self, n, roots, k, q, width, seed):
        """
        Initializes the Data class.

        Args:
            n (int): The number of customers.
            roots (int): The number of roots.
            k (int): The survivability of the network.
            q (int): The maximum total weight assigned to a single vertex.
            width (float): The width of the plane for data generation.
            seed (int): The random seed for reproducibility.

        Returns:
            None
        """

        self.n = n  # Number of customers
        self.roots = roots # Number of roots
        self.k = k # The level of survivability
        self.q = q # The maximum total weight assigned to a vertex
        self.seed = seed  # Random seed for reproducibility
        self.width = width  # Width of the plane for data generation
        
        # Define boundaries for random location generation
        self.width_2 = -self.width
        self.width_1 = self.width
        self.length_2 = -self.width
        self.length_1 = self.width
        
        # Define index set for both customers and roots
        self.VR = list(range(self.n + self.roots))
        
        # Define index set for roots
        self.Ro = list(range(self.roots))
        
        # Define index set for customers
        self.V = list(range(self.roots, self.n + self.roots))
        
        # Initialize location, allocation, weight and distance matrix
        self.loc = []  # Locations of customers
        self.d = np.ones(self.n + self.roots) # The weights of each vertex
        self.dist = np.zeros((self.n + self.roots, self.n + self.roots))  # Distance matrix
    
    def create_data(self):
        """
        Creates the data for the problem.

        Returns:
            None
        """

        rnd = np.random.RandomState(self.seed)  # Create random number generator with seed
        self.VxV = [(i,j) for i in self.V for j in self.V]  # Generate set of all possible arcs between customers
        self.A = [(i,j) for i in self.VR for j in self.V] # Generate set of all possible arcs between the roots and customers
        
        # Create the set A^r for each root r
        self.Ar = {}     
        for a in self.Ro:
            rootandvertices = [a] + self.V
            self.Ar[a] = [(i,j) for i in rootandvertices for j in self.V] # Generate set of all arcs going from the customers and a certain root to all customers
        
        # Generate random locations for customers
        self.loc = {i:(self.width_1 + rnd.random()*(self.width_2 -self.width_1),
                       self.length_1 + rnd.random()*(self.length_2 - self.length_1)) for i in self.VR}
        
        # Calculate distances between each pair of customers
        for i in self.VR:
            for j in self.VR:
                self.dist[i,j] = math.hypot(self.loc[i][0]-self.loc[j][0], self.loc[i][1]-self.loc[j][1])
                
        # The allocation costs for customers to eachother
        self.ac = np.copy(self.dist)*25

        # The construction costs for tree vertices
        self.t = np.copy(self.dist)*2