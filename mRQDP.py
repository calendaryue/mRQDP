import os
import getopt
import math
import sys
import random
import numpy as np
import time
import copy
import glob
import networkx as nx
from truncation import *
def ReadInput():
    global items
    global result
    global num_query
    global connections
    global ds
    global input_file_prefix
    global joincollect

    input_file_names = glob.glob(input_file_prefix+"*")
    num_query = len(input_file_names)

    entities_sensitivity_dic = [{} for i in range(num_query)]
    downward_sensitivity = [0 for i in range(num_query)]

    dic = {}
    result = []
    g_result = []

    items = [[] for i in range(num_query)]
    values = [[] for i in range(num_query)]   
    connections = [[] for i in range(num_query)]
    ds = []

    joincollect = []
    idx = 0
    i = 0

    for input_file_name in input_file_names:
        #For each query
        input_file = open(input_file_name,'r')

        g = nx.MultiGraph(gid=input_file_name)
        node_dict = {}
        vid = 0
        for line in input_file.readlines():
            (contri,v1, v2) = line.split()
            contri = float(contri)
            elements = line.split()
            values[i].append(float(contri))


            if v1 not in node_dict:
                node_dict[v1] = vid
                vid += 1
            if v2 not in node_dict:
                node_dict[v2] = vid
                vid += 1
            g.add_edge(node_dict[v1], node_dict[v2], weight = float(contri))
            
            if elements[1] not in dic:
                dic[elements[1]] = idx
                elements[1] = dic[elements[1]]
                items[i].append(elements[1])
                idx +=1
            else :
                elements[1] = dic[elements[1]]

            if elements[1] in entities_sensitivity_dic[i].keys():#len=107493 [0:14,1:190,2:23...]
                entities_sensitivity_dic[i][elements[1]]+=contri
            else:
                entities_sensitivity_dic[i][elements[1]]=contri
            #Update the DS
            if downward_sensitivity[i]<=float(entities_sensitivity_dic[i][elements[1]]):
                downward_sensitivity[i] = float(entities_sensitivity_dic[i][elements[1]]);                

            if elements[2] not in dic:
                dic[elements[2]] = idx
                elements[2] = dic[elements[2]]
                items[i].append(elements[2])
                idx +=1
            else :
                elements[2] = dic[elements[2]]

            if elements[2] in entities_sensitivity_dic[i].keys():#len=107493 [0:14,1:190,2:23...]
                entities_sensitivity_dic[i][elements[2]]+=contri
            else:
                entities_sensitivity_dic[i][elements[2]]=contri
            # Update the DS
            if downward_sensitivity[i]<=float(entities_sensitivity_dic[i][elements[2]]):
                downward_sensitivity[i] = float(entities_sensitivity_dic[i][elements[2]]);  

            connect = [elements[1],elements[2]]
            connections[i].append(connect)
        # d_sens = max(dict(g.degree()).values())


        # print(max(dict(g.degree()).values()))
        # g= [g.subgraph(c).copy() for c in nx.connected_components(g)]

        # e = []
        # for _, _, data in g.edges(data=True):
        #     e.append(data.get('weight', 0))
        # total = sum(e)
        # g_result.append(total) #true result of graph
        #xiamnx
        max_sum_of_weights = 0
        for node in g.nodes():
            edges = g.edges(node, data=True)
            sum_of_weights = 0
            for _, _, data in edges:
                sum_of_weights += data.get('weight', 0)
                if sum_of_weights > max_sum_of_weights:
                    max_sum_of_weights = sum_of_weights
        result.append(sum(values[i])) #true result
        i+=1
        input_file.close()
        joincollect.append(g)

        if i == 1:
            print('Graph info:')
            print('# of nodes:', g.number_of_nodes())
            print('# of edges:', g.number_of_edges())
            print('ds_g:', max_sum_of_weights)
        
            print("number of items: ",len(items[i-1]))
            print("number of joins: ", len(connections[i-1]))
            print("ds: ", downward_sensitivity[i-1])

        

    # print("g_result: ")
    # print(g_result)
    # print(ds)

    # N = len(items)
    # S = np.zeros((N, num_query))

    # for k in range(num_query):
    #     for j in range(len(connections[k])):
    #         idx = connections[k][j]
    #         S[idx,k]+=values[k][j]


    # final_s = []
    # for i in range(N):
    #     final_s.append(math.sqrt(sum(S[i, : ]**2)))
        
        
def LapNoise():
    a = random.uniform(0,1)
    b = math.log(1/(1-a))
    c = random.uniform(0,1)
    if c>0.5:
        return b
    else:
        return -b

# 计算每个u贡献小于阈值的值数量的过程：对于每个u的贡献，如果超出阈值则调整到阈值并相加得到Qi，循环u次，每次一次性回答所有query
# def calculate_E(threshold):
#     r = threshold
#     count = 0
#     N = len(items)
#     Q = np.zeros(num_query)
#     for i in range(N):
#         if final_s[i]>r:
#             Q += r*S[i]/final_s[i]
#         else:
#             Q +=S[i]
#             count+=1

#这里的count就是count(I,r)            
    # return count, Q
class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)
DIV = Infix(lambda x,y: np.divide(x,y))
Lap = lambda scale: np.random.laplace(0, scale)
def learn_threshold_TSens(tsenses, tsens_limit, eps):
    q_sens = 1
    c = 1
    eps_Qhat = eps |DIV| 10
    eps_tsens = eps - eps_Qhat
    Q_hat = np.sum(tsenses) + Lap(tsens_limit |DIV| eps_Qhat)
    #Q_hat = np.sum(tsenses)
    next_q = next_q_TSens(tsenses, Q_hat)
    res = SVT(next_q(), next_T(), q_sens, eps_tsens, c)
    threshold = len(res)
    return threshold   

def next_q_TSens(tsenses, Q_hat):
    tsenses = tsenses
    def _func():
        nonlocal tsenses
        i = 1
        shift = Q_hat - np.sum(tsenses)
        while True:
            tsenses = tsenses[tsenses > i]
            res = - ((np.sum(tsenses) + shift) |DIV| i)
            yield res
            i += 1
    return _func
def SVT(next_q, next_T, q_sens, eps, c=1):
    res = []
    eps_1 = eps |DIV| 2
    eps_2 = eps - eps_1
    rou = Lap(q_sens |DIV| eps_1)
    count = 0
    while count < c:
        q = next_q.__next__()
        T = next_T.__next__()
        v = Lap((2*c*q_sens) |DIV| eps_2)
        if q + v >= T + rou:
            res.append(True)
            count += 1
        else:
            res.append(False)
    return res
def next_T():
    while True:
        yield - 60/epsilon *math.log(4/beta)
def RunAlgorithm():
    global epsilon 
    global beta
    global delta
    ans = []    
    
    # N = len(items)
    # T = - 60/epsilon *math.log(4/beta)
    # T_hat = T + LapNoise()*20/(1*epsilon)
    # base = 1.3
    # i = 0
    # while(True):
    #     noise = LapNoise()*40/epsilon
    #     E , Q = calculate_E(pow(base,i))
    #     F = E -N
    #     F_hat = F + noise
    #     if F_hat > T_hat:
    #         tau = pow(base,i)
    #         break
    #     i +=1
    tau = 990
    # T = - 60/epsilon *math.log(4/beta)
    # i = 0
    # while(True):
    #     noise = LapNoise()*40/epsilon
    #     E , Q = calculate_E(pow(base,i))
    #     F = E -N
    #     F_hat = F + noise
    #     if F_hat > T_hat:
    #         tau = pow(base,i)
    #         break
    #     i +=1
#这里是SVT算法，算出阈值
    o = 0   
    for j in joincollect:
        e = []
        # if o == 0:
        #     print('# of nodes:', j.number_of_nodes())
        #     print('# of edges:', j.number_of_edges())
        trunc = project_degree_based(j,tau)
        # if o == 0:
        #     print('# of nodes:', trunc.number_of_nodes())
        #     print('# of edges:', trunc.number_of_edges())
        for _, _, data in trunc.edges(data=True):
            e.append(data.get('weight', 0))
        total = sum(e)
        ans.append(total)
        o+=1
    ans = np.array(ans)
    # print(tau)

#截断过程，导入truncation


    # y = np.random.normal(0, 1, num_query)
    # l = np.random.laplace(0, 1, num_query)
    # noises = tau * math.sqrt(2*math.log(1/delta)) * (1+0.9*epsilon/(4*math.log(1/delta))) / (0.9*epsilon) * np.random.normal(0, 1, num_query)
    # noises_y = tau * math.sqrt(2*math.log(1/delta)) * (1+0.9*epsilon/(4*math.log(1/delta))) / (0.9*epsilon) * y
    noises_l = tau  / (0.9*epsilon) * LapNoise()
    # print(noises_l)
    ans = ans + noises_l
    
    return ans
    
    
def main(argv):
    #The input file including the relationships between aggregations and base tuples
    global input_file_prefix
    input_file_prefix = ""
    #Privacy budget
    global epsilon
    epsilon = 1
    global delta
    #Error probablity: with probablity at least 1-beta, the error can be bounded
    global beta
    beta = 0.1
    global num_query

    try:
        opts, args = getopt.getopt(argv,"h:I:e:b:",["Input=","epsilon=","beta="])
    except getopt.GetoptError:
        print("errorPMSJASJF.py -I <input file> -e <epsilon(default 1)> -b <beta(default 0.1)> -d <desired delta(no default)>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("PMSJASJF.py -I <input file> -e <epsilon(default 1)> -b <beta(default 0.1)> -k <desired delta(no default)>")
            sys.exit()
        elif opt in ("-I", "--Input"):
            input_file_prefix = str(arg)
        elif opt in ("-e","--epsilon"):
            epsilon = float(arg)
        elif opt in ("-b","--beta"):
            beta = float(arg)
#../Result/TPCH/sc_0/Q2/eps_1.txt

    start = time.time()
    ReadInput()
    Q = RunAlgorithm()
    end= time.time()


    abs_error = []
    error_rate = []
    total_error = 0.0
    for i in range(num_query):
        total_error+=abs(Q[i]-result[i])**2
        abs_error.append(abs(Q[i]-result[i]))
        error_rate.append(abs(Q[i]-result[i])/result[i])

    total_error = np.sqrt(total_error)

    print("Query Result")
    print(result)

    print("Noised Result")
    print(Q)

    print("abs_error")
    print(abs_error)

    print("error_rate")
    print(error_rate)

    print("total_error")
    print(total_error)


    print("Time")
    print(end-start)


if __name__ == '__main__':
	main(sys.argv[1:])
