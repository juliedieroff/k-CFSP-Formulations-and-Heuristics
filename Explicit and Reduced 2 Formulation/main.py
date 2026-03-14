# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:12:11 2024

@author: ksearle
"""
from data_class import *
from model_class import *
from explicit_model_class import *
from call_backs2 import *
from explicit_call_backs import *
from helper import *
from explicit_helper import *
from heuristic import *
import networkx as nx
from docplex.mp.model import Model
import time
import csv
import os

p = Data(50,16,8,20,300,10)
p.create_data()

#%%

# Run the explicit model with callbacks
c = explicit_cut_model("cut",p)

cb_lazy = c.model.register_callback(Callback_lazy)
cb_lazy.model_instance = c
cb_lazy.problem_data = p
cb_lazy.num_calls = 0


cb_user = c.model.register_callback(Callback_user)
cb_user.model_instance = c
cb_user.problem_data = p
cb_user.num_calls = 0

c.solve_cut(True)

#%%
# TESTING INSTANCES

# Create csv file to record results
csv_filename = 'ExplicitAndReducedResults.csv'
columns = [
    'V', 'R', 'K',
    'Root_LB1', 'Root_UB1', 'Root_Gap1',
    'Objective_Value1', 'Final_Gap1', 'Solution_Time1', 'User_Cuts_Applied1',
    'Root_LB2', 'Root_UB2', 'Root_Gap2',
    'Objective_Value2', 'Final_Gap2', 'Solution_Time2', 'User_Cuts_Applied2'
]

results = pd.DataFrame(columns=columns)

# Paramaters we are testing
vertices = [50,100,150,200]
roots = [2,4,8,16]
kappa = [2,4,8,16]

# Run the model and record time, mip gaps, bounds and user cuts
for v in vertices:
    for r in roots:
        for k in kappa:
            if k <= r:
                # Create the data instance
                p = Data(v,r,k,20,300,10)
                p.create_data()
                
                # Set up the model EXPLICIT FORMULATION
                m = explicit_cut_model("cut",p)
                
                m.model.set_time_limit(3600)
                m.model.set_log_output(None)

                cb_lazy = m.model.register_callback(Callback_lazy)
                cb_lazy.model_instance = m
                cb_lazy.problem_data = p
                cb_lazy.num_calls = 0

                cb_user = m.model.register_callback(Callback_user)
                cb_user.model_instance = m
                cb_user.problem_data = p
                cb_user.num_calls = 0
                
                # Incase user callback isn't called
                cb_lazy.root_LB = None
                cb_lazy.root_UB = None
                cb_lazy.root_MIP_gap = None

                cb_user.root_LB = None
                cb_user.root_UB = None
                cb_user.root_MIP_gap = None

                start = time.time()
                m.solve_cut(True)
                end = time.time()
                
                # Data from root node
                if cb_user.root_LB is not None:
                    root_LB1 = cb_user.root_LB
                    root_UB1 = cb_user.root_UB
                    root_MIP_gap1 = cb_user.root_MIP_gap
                else:
                    root_LB1 = cb_lazy.root_LB
                    root_UB1 = cb_lazy.root_UB
                    root_MIP_gap1 = cb_lazy.root_MIP_gap              
                
                # Data from final solution, if no feasible solution found, objective value is nan
                sol_time1 = end - start
                sol_MIP_gap1 = m.model.solve_details.mip_relative_gap
                cuts_applied1 = m.model.cplex.solution.MIP.get_num_cuts(
                m.model.cplex.solution.MIP.cut_type.user)
                if m.solution is not None:
                    objective_value1 = m.solution.get_objective_value()  
                else:
                    objective_value1 = "nan"
                
                # Set up the model REDUCED FORMULATION
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
                
                # Incase user callback isn't called
                cb_lazy.root_LB = None
                cb_lazy.root_UB = None
                cb_lazy.root_MIP_gap = None

                cb_user.root_LB = None
                cb_user.root_UB = None
                cb_user.root_MIP_gap = None
                
                start_again = time.time()
                c.solve(True)
                end_again = time.time()

                # Data from root node
                if cb_user.root_LB is not None:
                    root_LB2 = cb_user.root_LB
                    root_UB2 = cb_user.root_UB
                    root_MIP_gap2 = cb_user.root_MIP_gap
                else:
                    root_LB2 = cb_lazy.root_LB
                    root_UB2 = cb_lazy.root_UB
                    root_MIP_gap2 = cb_lazy.root_MIP_gap
                              
                # Data from final solution, if no feasible solution found, objective value is nan
                sol_time2 = end_again - start_again
                sol_MIP_gap2 = c.model.solve_details.mip_relative_gap
                cuts_applied2 = c.model.cplex.solution.MIP.get_num_cuts(
                c.model.cplex.solution.MIP.cut_type.user)
                if c.solution is not None:
                    objective_value2 = c.solution.get_objective_value()  
                else:
                    objective_value2 = "nan"
                
                # Write results in csv file
                results.loc[len(results)] = [
                    v, r, k,
                    root_LB1, root_UB1, root_MIP_gap1,
                    objective_value1, sol_MIP_gap1, sol_time1, cuts_applied1,
                    root_LB2, root_UB2, root_MIP_gap2,
                    objective_value2, sol_MIP_gap2, sol_time2, cuts_applied2
                ]
                results.to_csv(csv_filename)       
                
                print(f"Completed test: V={v}, R={r}, K={k}")
