from gurobipy import *
import numpy as np
import math
import os
import pandas as pd

def calculate_dvr():    
    
    genres_1m = {'Action':0,    'Adventure':1,    'Animation':2,    "Children's":3,    'Comedy':4,    'Crime':5,    'Documentary':6,    'Drama':7,
    'Fantasy':8,    'Film-Noir':9,    'Horror':10,    'Musical':11,    'Mystery':12,    'Romance':13,    'Sci-Fi':14,    'Thriller':15,    'War':16,    'Western':17}    
    
    file_l = './movies_genre.csv'
    genre_file = pd.read_csv(file_l)

    dataset_ind =  './1m_indices.npy'
    Ind = np.load(dataset_ind)  
    
    genres_indexed = []
    np.random.seed(1)
    for itm in Ind:   ##BINARY VERSION
        x = genre_file[genre_file['original_index'] == itm]['genre'].values[0]
        splitted = x.split('|')
        random_num = np.random.randint(0,len(splitted))
        genres_indexed.append(splitted[random_num])
        
    genres_set = {}
    for i in genres_1m.keys():
        genres_set[i] = []
    for xx in range(len(genres_indexed)):
        genres_set[genres_indexed[xx]].append(xx)
    return genres_set
        
def calculate_pop():
        
    file_l = './1m_ratings.csv'
    ratings = pd.read_csv(file_l)
    ratings.columns = ['user','item','rating']
    
    # # Find the cut_off point for longtail
    abc = ratings.groupby('item').size()>9
    abc_ind = abc.index

    dataset_ind =  './1m_indices.npy'
    Ind = np.load(dataset_ind)  
    
    add_these = []
    
    for ind in abc_ind:
        if abc[ind] == True:
            add_these.append(ind)
    ratings = ratings[ratings['item'].isin(add_these)]
    pops=ratings.groupby('item').size()/len(ratings.user.unique())
    ratings['pop']=ratings['item'].apply(lambda x :pops[x])
  
    ratings2 = ratings[['item','pop']].drop_duplicates()
        

    pops_ind = []
    for j in Ind:
        pops_ind.append(ratings2[ratings2['item'] == j]['pop'].values[0])
        
    return pops_ind
           
def run_model(params): 
    mm = Model('Opt')
    strii = str("mylog_model.txt")
    mm.setParam(GRB.Param.LogFile, str(strii))
    mm.setParam(GRB.Param.Threads, 2)
    mm.setParam(GRB.Param.NodefileStart, 0.1)
    mm.setParam(GRB.Param.Presolve, 0)
    mm.setParam(GRB.Param.Method, 1)
    
    x = [];  I = params["I"];  R_div = params['R_div'] ; U = params["U"]
    x = mm.addVars(U,I, obj=params["u"], vtype=GRB.BINARY)
    y = mm.addVars(U,R_div, vtype=GRB.BINARY)
    print('added_dvars')
    
    num_alpha_fair = int(params["alpha_fair"]*U*params["k"]/(I+0.0))
    
    ####CONSTRAINTS
    mm.addConstrs(quicksum(x[j,i] for i in range(I)) == params["k"] for j in range(U)) ###Top-k list for every user
    mm.addConstr(quicksum(x[j,i]*params['pops'][i] for i in range(I) for j in range(U)) <= params['alpha_pop']*U*params['k'] ) 
    mm.addConstrs(quicksum(y[j,r] for r in range(R_div)) >= params["w"] for j in range(U))
    mm.addConstrs(quicksum(x[j,i] for i in params['dvr_set'][list(params['dvr_set'].keys())[r]]) >= y[j,r] for j in range(U) for r in range(R_div)) 
    
    if params["alpha_fair"] > -0.5 :
        mm.addConstrs(quicksum(x[j,r]  for j in range(U)) >=  num_alpha_fair for r in range(I))
        
#        mm.addConstrs(quicksum(x[j,r]  for j in range(U)) <= int(1+((2-params["alpha_fair"])*max(U,I)*params["k"])/(min(I,U)+0.0)) for r in range(I))
#        mm.addConstrs(quicksum(x[j,r]  for j in range(U)) <= int(1+(max(U,I)*params["k"])/math.pow(min(I,U),params['alpha_fair'])) for r in range(I))

    print('opt start')
    mm.optimize()
       
    return mm, x, float(mm.objVal), float(mm.Runtime)

 
    
def main2(alpha_value, params_k, pops, dvr_set, alpha_parameters):
    
    params = {}    
    dataset= './1m_utility.npy'
    V=np.load(dataset)
      

    params["k"] = params_k ;params["alpha_fair"] =  alpha_parameters[0]
    
    params["U"] =V.shape[0] # number of customers    
    params["I"]=V.shape[1] # number of producers
    
    
    params["R_div"] = len(dvr_set.keys()); params["dvr_set"] = dvr_set
    params['pops'] = pops; params['alpha_pop'] = alpha_parameters[1]
    params['w'] = alpha_parameters[2]
    cutoff = 1 ## this is used for making the data smaller as an approximation measure - set to 0 if you have enough memory
    params["u"] = np.array([0 if V[i,j] < cutoff else -V[i,j] for i in range(len(V)) for j in range(len(V[0])) ]).reshape(params['U'],params['I'])
    del V;  del pops

    model, dvar, objval, soltime = run_model(params)
    return model, dvar, objval, params, soltime

count=0
run_name_file = 'ML1M'
metricx = 'all'

#np.random.seed(1)
pops = calculate_pop()  ### a list of popularity values of each item
dvr_set = calculate_dvr() ### dictionary of item numbers with key values of genres

## In case of infeasibilities, best to penalize the constraints with big-M values


for alpha_dvr in [6]:   ## diversity parameter
    for  params_k in [10]:  ## cardinality of list parameter (top-k)
        for  alpha_value in [0.9]: ## fairness parameter
            for alpha_pop in [0.55]: ## popularity parameter
                alpha_parameters = [alpha_value, alpha_pop, alpha_dvr]
#                np.random.seed(1)
                model, dvar, oo, params,soltime,MAP_val,NDCG_val = main2(alpha_value, params_k, pops, dvr_set, alpha_parameters)



