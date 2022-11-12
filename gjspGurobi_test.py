# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:37:29 2021

@author: liping.huang
"""

from gurobipy import Model, GRB, GurobiError, quicksum, max_
import os 
import datetime
# from datetime import datetime
import time
# import sys
import numpy as np
import pandas as pd
# =============================================================================
# import plotly.figure_factory as ff
# import plotlyli
# plotly.offline.init_notebook_mode(connected=True)
# =============================================================================

StatusDict = {getattr(GRB.Status, s): s for s in dir(GRB.Status) if s.isupper()}

# path
# path = "Z:/Delta/Task Planning/Jun16/"
# path = "Z:/Delta/Task Planning/Jun16/jobshop_8/"
# path = "Z:/Delta/Task Planning/Jun16/jobshop_6/"
# path = "Z:/Delta/Task Planning/Jun16/jobshop_10/"

# path = "E:/Delta_NTU Project/Graph-based Task Planning/coding for gjsp/Jun24/python code_task planning/jobshop_20/"

path = "C:/_Huang Liping/_Manuscript Now/ifac/FJSP/python code_task planning/"

os.chdir(path)
file_workshopInfor = "gjsp_workshopInfor.xlsx"
file_jobInfor = "gjsp_jobInfor.xlsx"


# =============================================================================
# prelimiary (graph defination)----refer to the slides by Prof. Su for details
#   each job is a operation graph denoting  the logial topological sequence, which may be from UI side
#   different jobs may have different graph structures, even for the same workshop
#   such graph G=(O,E) is structured as a matrix for a workshop, where row and column denote the operation_ID o_i. e_j(i,i') = (o_i, o_i')=1 denotes the operation o_i should be completed before starting o_i for job j 
#   O captures the operation set, and E capstures the pairwise sequence of two operations. opertion o_x not appears in E means it does not following the constraint of operational sequencing
#   the graph should be transformed to a pairwise set by a interface that links the UI and the optimization algorithm program. graph G from all jobs utilizes the same matrix structure for a given jobshop
# =============================================================================



class GJSP_Gurobi(object): #   graph-based job scheduling problem with Gurobi as the solver
  """docstring for GJSP_Gurobi"""
  def __init__(self):
    # setting the solver attributes;
    self.schedules = {}   
    self.assign_list = []
    self.order_list = []
    self.start_times = []
    self.complete = []
    self.c_max = []

  def solve(self, jobOperationIndex, job_Index_Operation, jobOperation_pairs, jobOperation_machines, jobOperation_pairs_machine, process_times, jobOperation, operations, resources):
    solved = False
    print('in solve function')
    start = time.time()
    gjsp_model, assign, order, startTime, completeTime = self._create_model(jobOperationIndex, job_Index_Operation, jobOperation_pairs, jobOperation_machines, jobOperation_pairs_machine, process_times, jobOperation, operations, resources)
    # self._set_model_parms(pmsp_model)
    # solve the model
    try:
      # print('try optimizing by calling GUROBI')
      # start = time.time()
      gjsp_model.optimize()
      end = time.time() 
      print("solving time: ", str(end-start))
      
      if gjsp_model.status == GRB.Status.OPTIMAL:
        solved = True
        print('success with the optimal solution') # for testing
        # formulate schedules
        self._formulate_schedules( jobOperationIndex, jobOperation_pairs, resources, jobOperation_machines, jobOperation_pairs_machine, process_times, assign, order, startTime, completeTime)
      else:
        statstr = StatusDict[gjsp_model.status]
        print('Optimization was stopped with status %s' %statstr)
        # formulate schedules
        self._formulate_schedules(jobOperationIndex, job_Index_Operation, jobOperation_pairs, jobOperation_machines, jobOperation_pairs_machine, process_times, jobOperation, operations, resources)
    except GurobiError as e:
      print('Error code '+ str(e.errno)+': '+str(e))

    return solved

  def _set_model_parms(self, m):
    # permittable gap
    m.setParam('MIPGap',0.2)
    # time limit
    m.setParam('TimeLimit',10)
    # percentage of time on heuristics
    m.setParam('Heuristics',0.5)

  def _formulate_schedules(self, jobOperationIndex, jobOperation_pairs, resources, jobOperation_machines, jobOperation_pairs_machine, process_times, assign, order, startTime, completeTime):
    # print("variables: ", startTime.keys) 
    # cur_time = int(time.time())
    # t = datetime.datetime(2021,6,1,0,0,0,0)
    t = datetime.datetime(2021, 6, 1, 0, 0)
    sch_start = time.mktime(t.timetuple())
    # sch_start = (t-datetime.datetime(1970,1,1)).total_seconds()
    assign_list = []
    order_list = []
    start_times = []
    complete_times = []
    Cmax_vector = []
    
    for (j, i, r) in jobOperation_machines:
        if assign[j,i,r].x ==1:
            assign_list.append((j,i,r))
 
    for (j1,i1,j2,i2,r) in jobOperation_pairs_machine:
         # order_list.append((j1, i1, j2, i2, order[j1,i1, j2, i2].x))
          if order[j1,i1, j2, i2, r].x < 0.5:
             order_list.append((j1, i1, j2, i2, r, order[j1,i1,j2,i2, r].x))
                 
    
    for (j, i, r) in jobOperation_machines:
        if assign[j,i,r].x == 1:
            start_times.append((j,i,r,startTime[j,i].x))
            complete_times.append((j,i,r,completeTime[j,i].x))
            Cmax_vector.append(completeTime[j,i].x)
        
    for k in range(len(jobOperationIndex)):
      [j, i] = jobOperationIndex[k]
      self.schedules[k] = {}
      self.schedules[k]['Task'] = 'Job ' + str(j)
      self.schedules[k]['Operation'] = (j,i)
      self.schedules[k]['Start'] = sch_start + startTime[j,i].x * 60
      for r in resources: 
          if (j, i, r) in jobOperation_machines:         
              if assign[j, i, r].x == 1:
                  self.schedules[k]['Finish'] = sch_start + (startTime[j,i].x + process_times[(j,i,r)])*60
                  self.schedules[k]['Machine'] = 'M ' + str(r)                             
        
    self.assign_list = assign_list
    self.order_list = order_list
    self.start_times = start_times
    self.complete = complete_times
    self.c_max = max(Cmax_vector)
    
    
    print("assign_list: ", assign_list)
    print("order_list: ", order_list)
    # print("start_times: ", start_times)
    schedule = self.schedules
    print("schedule: ", schedule)
    

    
    return
    
  def _create_model(self,jobOperationIndex, job_Index_Operation, jobOperation_pairs, jobOperation_machines, jobOperation_pairs_machine, process_times, jobOperation, operations, resources):
      
    # deprived constants from inputs and prepare the index for decision variables
    # Note: job_ids, operation_ids and resource_ids start from 1
    # for describing the resource assignment
            
    #generate operation pariwise seq for each job. note:  the input parameter job_operationIndexPair defines the operations' logical graph for each job    
    job_operationPairwiseSeq = [tuple(xi) for xi in job_operationIndexPair.values]
    # define BigM
    bigM = sum(processIntervals)+sum(changeoverTime['time'])*pow(machine_num, 100)
          
    
    ## create model
    m = Model('GJSP')    
    # create decision variables
    # 1. assignments of job operations on machines: s_jir \in{i,1}
    s = m.addVars(jobOperation_machines, vtype=GRB.BINARY, name='assign')
    theta = m.addVars(jobOperation_pairs_machine, vtype=GRB.BINARY, name='machineShare')    
    # 2. order of executing jobs: y^j'i'_ji
    y = m.addVars(jobOperation_pairs_machine, vtype=GRB.BINARY, name='order')
    # 3. start time of executing each job operation: T_ji
    startTime = m.addVars(jobOperationIndex, name='startTime')
    completeTime = m.addVars(jobOperationIndex, name='completeTime')

    # create objective
    # print('creating objective funtion')
    # m.setObjective(quicksum(startTime)+np.sum(processIntervals), GRB.MINIMIZE) # TOTRY
    m._max_complete = m.addVar(1, name='max_complete_time')
    m.setObjective(m._max_complete, GRB.MINIMIZE)
    # m.addConstr((m._max_complete==max_(startTime)),'minimax')
    m.addConstr((m._max_complete==max_(completeTime)),'minimax')    
    m.addConstrs((completeTime[j,i] == startTime[j,i] + quicksum([s[j,i,r]*process_times[j,i,r] for r in resources if (j,i,r) in jobOperation_machines]) for (j,i) in jobOperationIndex),'start time to complete time')
    
    
    ## create constraints
    # print('creating constraints')
    # 1. operational time sequencing of each job
    # (1-1) resource intersection set： T_j'i' >= T_ji + sum_{r \in R_j(i+1) \cap R_ji}(process_time(j,i,r) + changeoverTime(j,i,r)(s_jir + s_j(i+1)r - 1))
    # m.addConstrs((startTime[j2,i2] >= startTime[j1,i1] + process_times[j1,i1,r] + changeoverTime.at[r-1,'time']*(s[j1,i1,r] + s[j2, i2 ,r]-1) for r in resources for (j1,i1,j2,i2) in jobOperation_pairs if (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'operational time sequencing 1')
                     
    # (1-2) ordinary sequencing： T_ji' >= T_ji + sum_r(o_ji)(S_jir*processtime)
    m.addConstrs((startTime[j,i2] >= startTime[j,i1] + quicksum([s[j,i1,r]*process_times[j,i1,r] for r in resources if (j,i1,r) in jobOperation_machines]) for j in jobs for (j,i2) in jobOperationIndex for (j,i1) in jobOperationIndex if (j,i1,i2) in job_operationPairwiseSeq),'operational time sequencing 2')
    
    # (1-3) materials are availble: T_ji >= T(M_ji)
    m.addConstrs((startTime[j,i] >= materialTime.iat[j-1,o] for (j,i,o) in job_Index_Operation if pd.notnull(materialTime.iat[j-1,o])), 'operational time sequencing 3')
    
    # 2. operational resource assignment for each job: sum(s_jir) = 1
    m.addConstrs((quicksum([s[j,i,r] for r in resources if (j,i,r) in jobOperation_machines])==1 for (j,i) in jobOperationIndex), 'operational resource assignment')
    
# =============================================================================
#     # 3. disjunctive and time seperation
#     # (3-1) dynamic assign
#     m.addConstrs((0-s[j1,i1,r] + theta[j1,i1,j2,i2,r] <= 0 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 1 for the time sepration rule')
#     m.addConstrs((0-s[j2,i2,r] + theta[j1,i1,j2,i2,r] <= 0 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 2 for the time sepration rule')
#     m.addConstrs((s[j1,i1,r] + s[j2,i2,r] - theta[j1,i1,j2,i2,r] <= 1 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 3 for the time separation rule')
#         
#     # (3-3)
#     m.addConstrs((startTime[j2,i2] >= startTime[j1,i1] + process_times[j1,i1,r] + changeoverTime.at[r-1,'time'] + separationTime.at[r-1,'time'] - bigM * (1-y[j1,i1,j2,i2,r] + (1- s[j1,i1,r] + (1 - s[j2,i2,r]))) for r in resources for (j1,i1,j2,i2,r) in jobOperation_pairs_machine), 'disjunctive and time seperation 1')
#     m.addConstrs((startTime[j1,i1] >= startTime[j2,i2] + process_times[j2,i2,r] + changeoverTime.at[r-1,'time'] + separationTime.at[r-1,'time'] - bigM * (y[j1,i1,j2,i2,r] + (1- s[j1,i1,r] + (1 - s[j2,i2,r]))) for r in resources for (j1,i1,j2,i2,r) in jobOperation_pairs_machine), 'disjunctive and time seperation 2')
#     
# =============================================================================
        
    # 3. minimum time separation for each resource: s_jir + s_j'i'r = 2 => abs(T_ji - T_j'i' - process_times[j',i',r]) >= changeovertime(j,i,j',i',r) + q_r
    # (3-1-1) theta_jij'i'r = 1 <=> s_jir = 1 and s_j'i'r = 1
    # <=> the following three constraints
    # -s_jir + theta_jij'i'r <= 0
    # -s_j'i'r + theta_jij'i'r <= 0
    # s_jir + s_j'i'r - theta_jij'i'r <= 1
    m.addConstrs((0-s[j1,i1,r] + theta[j1,i1,j2,i2,r] <= 0 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 1 for the time sepration rule')
    m.addConstrs((0-s[j2,i2,r] + theta[j1,i1,j2,i2,r] <= 0 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 2 for the time sepration rule')
    m.addConstrs((s[j1,i1,r] + s[j2,i2,r] - theta[j1,i1,j2,i2,r] <= 1 for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'antecedent 3 for the time separation rule')
    
    # (3-2-2) the transformed bigM constraint for the time seperation rule
    m.addConstrs((changeoverTime.at[r-1,'time'] + separationTime.at[r-1,'time'] - (startTime[j1,i1] - startTime[j2,i2]) <= bigM * (1 - theta[j1,i1,j2,i2,r]) for r in resources for (j1,i1) in jobOperationIndex for (j2,i2) in jobOperationIndex if (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'transformed bigM 1 for time seperation rule')
    m.addConstrs((startTime[j1,i1] >= startTime[j2,i2] + process_times[j2,i2,r] - bigM * (y[j1,i1,j2,i2,r] + (1- s[j1,i1,r] + (1 - s[j2,i2,r]))) for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'transformed bigM 2 for time seperation rule')
    m.addConstrs((changeoverTime.at[r-1,'time'] + separationTime.at[r-1,'time'] - (startTime[j1,i1] - startTime[j2,i2]) <= bigM * (1-theta[j1,i1,j2,i2,r]) for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'transformed bigM 3 for time seperation rule')
    m.addConstrs((startTime[j2,i2] >= startTime[j1,i1] + process_times[j1,i1,r] - bigM * (1-y[j1,i1,j2,i2,r] + 1-s[j1,i1,r] + 1-s[j2,i2,r]) for (j1,i1,j2,i2,r) in jobOperation_pairs_machine),'transformed bigM 4 for time seperation rule')
    
# =============================================================================
#     3-2. For each resource r, if two operations o_ji and o_j'i' are mutually exclusive
#     3-2-1 is the same to 3-1-1
#     3-2-2: 
# =============================================================================

# =============================================================================
#   4. Job due date and cost constraints:
#    
# =============================================================================

    # return m, s,  startTime
    return m, s, y, startTime, completeTime


if __name__ == '__main__':
  
  
  # load inputs from excel, parameters store in sheets  
  # worshop information
  operation = pd.read_excel(file_workshopInfor, 'operation')
  resource = pd.read_excel(file_workshopInfor,'resource')
  operationResource = pd.read_excel(file_workshopInfor,'operationResource')
  operationDiGraph = pd.read_excel(file_workshopInfor,'operationGraphStruct')
  changeoverTime = pd.read_excel(file_workshopInfor,'changeoverTime')
  separationTime = pd.read_excel(file_workshopInfor,'separationTime')  
  
  # job list information 
  jobOperation = pd.read_excel(file_jobInfor,'jobOperation')
  job_operationIndexPair = pd.read_excel(file_jobInfor,'job_operationIndexPair')
  processTime = pd.read_excel(file_jobInfor,'processTime')
  materialTime = pd.read_excel(file_jobInfor,'materialTime')
   
  machine_num = operationResource.shape[1]-1;
  machine_properties = np.zeros(machine_num,dtype=np.int32) #initial available time
  # machine_properties = np.random.randint(0, 60, size=(machine_num), dtype=np.int32)
  
  # derived from input files
  # workshop operations
  operation_ids = operation['Operation_ID']
  operations = tuple(operation_ids)
  resource_ids = resource.Resource_ID
  machine_num = len(resource_ids)
  resources = tuple(resource_ids)
  
  # jobs
  job_names = jobOperation.columns
  job_ids = np.arange(1, len(job_names)+1, 1, dtype = np.int32)
  jobs = tuple(job_ids)              
  jobOperationNum = np.arange(1,len(job_ids)+1, 1, dtype = np.int32)
  for j in range(len(job_ids)):
      jobOperationNum[j-1] = jobOperation[job_names[j-1]].count()
  jobOperationIndex = [(j,i) for j in jobs for i in range(1,jobOperationNum[j-1]+1,1)]
  job_Index_Operation = [(j,i,o) for (j,i) in jobOperationIndex for o in operations if pd.notnull(jobOperation.iat[i-1,j-1]) and jobOperation.iat[i-1,j-1]==o]         
  # jobOperation pairs for describing the order of excecuting o_ji and o_j'i'
  jobOperation_pairs = [(j1,i1,j2,i2) for (j1,i1) in jobOperationIndex for (j2,i2) in jobOperationIndex if j1<j2]    
  jobOperation_machine = [(j,i,r) for (j,i,o) in job_Index_Operation for r in resources if operationResource.iat[o-1,r]==1]
  jobOperation_machines = tuple(jobOperation_machine) 
  # jobOperation_pairs_machine for describing two operation o_ji and o_j'i' are assinged to the same resource
  jobOperation_pairs_machine = [(j1,i1,j2,i2,r) for (j1,i1,j2,i2) in jobOperation_pairs for r in resources if (j1,i1,r) in jobOperation_machines and (j2,i2,r) in jobOperation_machines]     
  #process time
  processIntervals= [processTime.iat[j-1,r] for (j,i,r)  in jobOperation_machines if pd.notnull(processTime.iat[j-1,r])]
  process_times = dict(zip(jobOperation_machines, tuple(processIntervals)))
    
  #request time for each job: default value is 0
  # request_times = np.zeros(len(job_names),dtype=np.int32)
  # request_times = np.random.randint(0, 60, size=(job_num), dtype=np.int32)
  
  #object instance of the class model
  gjsp_solver = GJSP_Gurobi()
  # solver of gjsp -> call gurobi by the function solve
  # start = time.time()
  solved = gjsp_solver.solve(jobOperationIndex, job_Index_Operation, jobOperation_pairs, jobOperation_machines, jobOperation_pairs_machine, process_times, jobOperation, operations, resources)
  # end = time.time()  
  # print("solving time: ", str(end-start))
  if solved:
    # print("schedules", gjsp_solver.schedules) #display the schedules
    # print("order", gjsp_solver.order) #display the job operation order
    print("schedules: ", gjsp_solver.schedules)
    print("complete time: ", gjsp_solver.complete)
    print("ultimate time for finishing all jobs: ", gjsp_solver.c_max)  
    

# =============================================================================
#   # show the gantt chart
#   sch = gjsp_solver.schedules
#   sch_list = list(sch.values())
#   colors = {'M 1': 'rgb(200, 100, 0)', 'M 2': 'rgb(100, 200, 0)', 'M 3': 'rgb(200, 0, 100)', 'M 4': 'rgb(100, 200, 0)', 'M 5': 'rgb(0,0,200)', 'M 6': 'rgb(0,200,200)', 'M 7':'rgb(150,130,120)'}
# 
#   fig = ff.create_gantt(sch_list, colors=colors, index_col='Machine', show_colorbar=True, group_tasks=True, showgrid_x=True)
#   plotly.offline.plot(fig)
# 
# =============================================================================
   
