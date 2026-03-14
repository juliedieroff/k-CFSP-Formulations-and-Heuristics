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
csv_filename = 'ModelComparisonsV200p1.csv'
columns = [
    'V', 'R', 'K', 'Type', 'H/W Used', 'Total Applied',
    'Root_LB', 'Root_UB', 'Root_Gap', 'Root Time', 'H/W Gap',
    'Objective_Value', 'Final_Gap', 'Solution_Time', 'Nodes_Processed', 'User_Cuts_Applied'
]

results = pd.DataFrame(columns=columns)

# Paramaters we are testing
roots = [2,4,8]
kappa = [2,4,8]

# Run the model and record time, mip gaps, bounds and user cuts
for r in roots:
    for k in kappa:
        if k <= r:
            # Create the data instance
            p = Data(200,r,k,20,300,10)
            p.create_data()
                    
            # Run model with no warmstart or heuristic
            a = reduced_model("reduced",p)
            
            a.model.set_time_limit(3600)
            a.model.set_log_output(None)

            cb_lazy = a.model.register_callback(Callback_lazy2)
            cb_lazy.model_instance = a
            cb_lazy.problem_data = p
            cb_lazy.num_calls = 0

            cb_user = a.model.register_callback(Callback_user2)
            cb_user.model_instance = a
            cb_user.problem_data = p
            cb_user.num_calls = 0
            
            # Incase callbacks aren't called
            cb_lazy.root_LB = None
            cb_lazy.root_UB = None
            cb_lazy.root_MIP_gap = None

            cb_user.root_LB = None
            cb_user.root_UB = None
            cb_user.root_MIP_gap = None
            
            #cb_heur.heuristic_gap = None
            
            # Time info
            start = time.time()
            cb_user.start_time = start
            cb_lazy.start_time = start
            
            a.solve(True)
            end = time.time()

            # Data from root node
            if cb_user.root_LB is not None:
                root_LB = cb_user.root_LB
                root_UB = cb_user.root_UB
                root_MIP_gap = cb_user.root_MIP_gap
                root_time = cb_user.root_time
            else:
                root_LB = cb_lazy.root_LB
                root_UB = cb_lazy.root_UB
                root_MIP_gap = cb_lazy.root_MIP_gap
                root_time = cb_lazy.root_time
                          
            # Data from final solution, if no feasible solution found, objective value is nan
            sol_time = end - start
            sol_MIP_gap = a.model.solve_details.mip_relative_gap
            cuts_applied = a.model.cplex.solution.MIP.get_num_cuts(
            a.model.cplex.solution.MIP.cut_type.user)
            #heuristic_check = cb_heur.heuristic_used
            #heur_gap = cb_heur.heuristic_gap
            #h_count = cb_heur.heur_count
            nodes_processed = a.model.solve_details.nb_nodes_processed
            
            if a.solution is not None:
                objective_value = a.solution.get_objective_value()  
            else:
                objective_value = "-"
            
            
            # Write results in csv file
            results.loc[len(results)] = [
                200, r, k, 1, 0, None,
                root_LB, root_UB, root_MIP_gap, root_time, None,
                objective_value, sol_MIP_gap, sol_time, nodes_processed, cuts_applied
            ]
            results.to_csv(csv_filename)       
            
            print(f"Completed test: V=200, R={r}, K={k}, Type=1")

            # Run model with warmstart and no heuristic
            b = reduced_model("reduced",p)
            
            b.model.set_time_limit(3600)
            b.model.set_log_output(None)

            cb_lazy = b.model.register_callback(Callback_lazy2)
            cb_lazy.model_instance = b
            cb_lazy.problem_data = p
            cb_lazy.num_calls = 0

            cb_user = a.model.register_callback(Callback_user2)
            cb_user.model_instance = b
            cb_user.problem_data = p
            cb_user.num_calls = 0
            
            # Incase callbacks aren't called
            cb_lazy.root_LB = None
            cb_lazy.root_UB = None
            cb_lazy.root_MIP_gap = None

            cb_user.root_LB = None
            cb_user.root_UB = None
            cb_user.root_MIP_gap = None
            
            #cb_heur.heuristic_gap = None
            
            # Time info
            start = time.time()
            cb_user.start_time = start
            cb_lazy.start_time = start
            
            # Add warmstart
            T, S, cost, timeH = heuristic_sol(p.dist, p)

            # Add the warmstart with this feasible solution
            warmstart = b.model.new_solution()

            for tree in T:
                for (i,j) in tree:
                    warmstart.add_var_value(b.x[i,j],1)
                    
            for (i,j) in S:
                warmstart.add_var_value(b.y[i,j],1)
                
            b.model.add_mip_start(warmstart)
            
            b.solve(True)
            end = time.time()

            # Data from root node
            if cb_user.root_LB is not None:
                root_LB = cb_user.root_LB
                root_UB = cb_user.root_UB
                root_MIP_gap = cb_user.root_MIP_gap
                root_time = cb_user.root_time
            else:
                root_LB = cb_lazy.root_LB
                root_UB = cb_lazy.root_UB
                root_MIP_gap = cb_lazy.root_MIP_gap
                root_time = cb_lazy.root_time
                          
            # Data from final solution, if no feasible solution found, objective value is nan
            sol_time = end - start
            sol_MIP_gap = b.model.solve_details.mip_relative_gap
            cuts_applied = b.model.cplex.solution.MIP.get_num_cuts(
            b.model.cplex.solution.MIP.cut_type.user)
            heuristic_check = b.model.number_of_mip_starts
            heur_gap = (cost - root_LB)/cost
            #h_count = cb_heur.heur_count
            nodes_processed = b.model.solve_details.nb_nodes_processed
            
            if b.solution is not None:
                objective_value = b.solution.get_objective_value()  
            else:
                objective_value = "-"
            
            
            # Write results in csv file
            results.loc[len(results)] = [
                200, r, k, 2, heuristic_check, None,
                root_LB, root_UB, root_MIP_gap, root_time, heur_gap,
                objective_value, sol_MIP_gap, sol_time, nodes_processed, cuts_applied
            ]
            results.to_csv(csv_filename)       
            
            print(f"Completed test: V=200, R={r}, K={k}, Type=2")
            
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
            
            # Use heuristic
            cb_heur = c.model.register_callback(HeuristicsCallback)
            cb_heur.model_instance = c
            cb_heur.problem_data = p
            cb_heur.heuristic_choice = 1
            cb_heur.node_num = 0
            cb_heur.frequency = 5
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
            
            # Time info
            start = time.time()
            cb_user.start_time = start
            cb_lazy.start_time = start
            
            c.solve(True)
            end = time.time()

            # Data from root node
            if cb_user.root_LB is not None:
                root_LB = cb_user.root_LB
                root_UB = cb_user.root_UB
                root_MIP_gap = cb_user.root_MIP_gap
                root_time = cb_user.root_time
            else:
                root_LB = cb_lazy.root_LB
                root_UB = cb_lazy.root_UB
                root_MIP_gap = cb_lazy.root_MIP_gap
                root_time = cb_lazy.root_time
                          
            # Data from final solution, if no feasible solution found, objective value is nan
            sol_time = end - start
            sol_MIP_gap = c.model.solve_details.mip_relative_gap
            cuts_applied = c.model.cplex.solution.MIP.get_num_cuts(
            c.model.cplex.solution.MIP.cut_type.user)
            heuristic_check = cb_heur.heuristic_used
            heur_gap = cb_heur.heuristic_gap
            h_count = cb_heur.heur_count
            nodes_processed = c.model.solve_details.nb_nodes_processed
            
            if c.solution is not None:
                objective_value = c.solution.get_objective_value()  
            else:
                objective_value = "-"
            
            
            # Write results in csv file
            results.loc[len(results)] = [
                200, r, k, 3, heuristic_check, h_count,
                root_LB, root_UB, root_MIP_gap, root_time, heur_gap,
                objective_value, sol_MIP_gap, sol_time, nodes_processed, cuts_applied
            ]
            results.to_csv(csv_filename)       
            
            print(f"Completed test: V=200, R={r}, K={k}, Type=3")
    
            # Set up the model REDUCED FORMULATION HEURISTIC Y WITH WARMSTART
            d = reduced_model("reduced",p)
            
            d.model.set_time_limit(3600)
            d.model.set_log_output(None)

            cb_lazy = d.model.register_callback(Callback_lazy2)
            cb_lazy.model_instance = d
            cb_lazy.problem_data = p
            cb_lazy.num_calls = 0

            cb_user = d.model.register_callback(Callback_user2)
            cb_user.model_instance = d
            cb_user.problem_data = p
            cb_user.num_calls = 0
            
            # Use heuristic
            cb_heur = d.model.register_callback(HeuristicsCallback)
            cb_heur.model_instance = d
            cb_heur.problem_data = p
            cb_heur.heuristic_choice = 1
            cb_heur.node_num = 0
            cb_heur.frequency = 5
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
            
            # Time info
            start = time.time()
            cb_user.start_time = start
            cb_lazy.start_time = start
            
            # Add warmstart
            T, S, cost, timeH = heuristic_sol(p.dist, p)

            # Add the warmstart with this feasible solution
            warmstart = d.model.new_solution()

            for tree in T:
                for (i,j) in tree:
                    warmstart.add_var_value(d.x[i,j],1)
                    
            for (i,j) in S:
                warmstart.add_var_value(d.y[i,j],1)
                
            d.model.add_mip_start(warmstart)
            
            d.solve(True)
            end = time.time()

            # Data from root node
            if cb_user.root_LB is not None:
                root_LB = cb_user.root_LB
                root_UB = cb_user.root_UB
                root_MIP_gap = cb_user.root_MIP_gap
                root_time = cb_user.root_time
            else:
                root_LB = cb_lazy.root_LB
                root_UB = cb_lazy.root_UB
                root_MIP_gap = cb_lazy.root_MIP_gap
                root_time = cb_lazy.root_time
                          
            # Data from final solution, if no feasible solution found, objective value is nan
            sol_time = end - start
            sol_MIP_gap = d.model.solve_details.mip_relative_gap
            cuts_applied = d.model.cplex.solution.MIP.get_num_cuts(
            d.model.cplex.solution.MIP.cut_type.user)
            heuristic_check = cb_heur.heuristic_used
            heur_gap = cb_heur.heuristic_gap
            h_count = cb_heur.heur_count
            nodes_processed = d.model.solve_details.nb_nodes_processed
            
            if d.solution is not None:
                objective_value = d.solution.get_objective_value()  
            else:
                objective_value = "-"
            
            
            # Write results in csv file
            results.loc[len(results)] = [
                200, r, k, 4, heuristic_check, h_count,
                root_LB, root_UB, root_MIP_gap, root_time, heur_gap,
                objective_value, sol_MIP_gap, sol_time, nodes_processed, cuts_applied
            ]
            results.to_csv(csv_filename)       
            
            print(f"Completed test: V=200, R={r}, K={k}, Type=4")
            
