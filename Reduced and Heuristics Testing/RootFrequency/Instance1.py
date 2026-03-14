# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:12:11 2024

@author: ksearle
"""
from data_class import *
from model_class import *
from call_backs2 import *
from helper import *
from heuristic import *
import networkx as nx
from docplex.mp.model import Model
import time
import csv
import os
import pandas as pd


#%%
# TESTING INSTANCES

# Create csv file to record results
csv_filename = 'RootFrequencyV50.csv'
columns = [
    'V', 'R', 'K', 'Frequency', 'Heuristic Used', 'Total Applied',
    'Root_LB', 'Root_UB', 'Root_Gap', 'Heuristic Gap',
    'Objective_Value', 'Final_Gap', 'Solution_Time', 'User_Cuts_Applied'
]

results = pd.DataFrame(columns=columns)

# Paramaters we are testing
roots = [2,4,8,16]
kappa = [2,4,8,16]
frequencies = [1,2,5,10,15,25]

# Run the model and record time, mip gaps, bounds and user cuts
for r in roots:
    for k in kappa:
        if k <= r:
            for f in frequencies:
                # Create the data instance
                p = Data(50,r,k,20,300,10)
                p.create_data()
                                
                # Set up the model REDUCED FORMULATION HEURISTIC Y NO WARMSTART
                c = reduced_model("reduced",p)
                
                c.model.set_time_limit(3600)
                c.model.set_log_output(None)

                cb_lazy = c.model.register_callback(Callback_lazy2)
                cb_lazy.model_instance = c
                cb_lazy.problem_data = p
                cb_lazy.num_calls = 0

                cb_user = c.model.register_callback(Callback_user2)
                cb_user.model_instance = c
                cb_user.problem_data = p
                cb_user.num_calls = 0
                
                # Use heuristic Y
                cb_heur = c.model.register_callback(HeuristicsCallback)
                cb_heur.model_instance = c
                cb_heur.problem_data = p
                cb_heur.heuristic_choice = 1
                cb_heur.node_num = 0
                cb_heur.frequency = f
                cb_heur.cutoff = 0.4
                cb_heur.heuristic_used = 0
                cb_heur.heur_count = 0
                cb_heur.node_zero_count = 0
                
                # Incase callbacks or heuristic isn't called
                cb_lazy.root_LB = None
                cb_lazy.root_UB = None
                cb_lazy.root_MIP_gap = None

                cb_user.root_LB = None
                cb_user.root_UB = None
                cb_user.root_MIP_gap = None
                
                cb_heur.heuristic_gap = None
                
                start = time.time()
                c.solve(True)
                end = time.time()

                # Data from root node
                if cb_user.root_LB is not None:
                    root_LB = cb_user.root_LB
                    root_UB = cb_user.root_UB
                    root_MIP_gap = cb_user.root_MIP_gap
                else:
                    root_LB = cb_lazy.root_LB
                    root_UB = cb_lazy.root_UB
                    root_MIP_gap = cb_lazy.root_MIP_gap
                              
                # Data from final solution, if no feasible solution found, objective value is nan
                sol_time = end - start
                sol_MIP_gap = c.model.solve_details.mip_relative_gap
                cuts_applied = c.model.cplex.solution.MIP.get_num_cuts(
                c.model.cplex.solution.MIP.cut_type.user)
                heuristic_check = cb_heur.heuristic_used
                heur_gap = cb_heur.heuristic_gap
                h_count = cb_heur.heur_count
                
                if c.solution is not None:
                    objective_value = c.solution.get_objective_value()  
                else:
                    objective_value = "-"
                
                
                # Write results in csv file
                results.loc[len(results)] = [
                    50, r, k, f, heuristic_check, h_count,
                    root_LB, root_UB, root_MIP_gap, heur_gap,
                    objective_value, sol_MIP_gap, sol_time, cuts_applied
                ]
                results.to_csv(csv_filename)       
                
                print(f"Completed test: V=50, R={r}, K={k}, Frequency={f}")



