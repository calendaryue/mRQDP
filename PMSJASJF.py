import os
import getopt
import math
import sys
import random
import numpy as np
import time
import copy
import glob

def ReadInput():
    global items
    global result
    global num_query
    global final_s
    global S
    global input_file_prefix
    
    input_file_names = glob.glob(input_file_prefix+"*")
    num_query = len(input_file_names)
    items = []
    dic = {}
    result = []
    values = [[] for i in range(num_query)]   
    connections = [[] for i in range(num_query)]

    idx = 0
    i = 0
    for input_file_name in input_file_names:
        #For each query
        input_file = open(input_file_name,'r')

        for line in input_file.readlines():
            elements = line.split()
            values[i].append(float(elements[0]))

            if elements[1] not in dic:
                dic[elements[1]] = idx
                elements[1] = dic[elements[1]]
                items.append(elements[1])
                idx +=1
            else :
                elements[1] = dic[elements[1]]

            connections[i].append(elements[1])

        result.append(sum(values[i]))
        i+=1
        input_file.close()

    N = len(items)
    S = np.zeros((N, num_query))

    for k in range(num_query):
        for j in range(len(connections[k])):
            idx = connections[k][j]
            S[idx,k]+=values[k][j]


    final_s = []
    for i in range(N):
        final_s.append(math.sqrt(sum(S[i, : ]**2)))
        
        
def LapNoise():
    a = random.uniform(0,1)
    b = math.log(1/(1-a))
    c = random.uniform(0,1)
    if c>0.5:
        return b
    else:
        return -b

# 计算每个u贡献小于阈值的值数量的过程：对于每个u的贡献，如果超出阈值则调整到阈值并相加得到Qi，循环u次，每次一次性回答所有query
def calculate_E(threshold):
    r = threshold
    count = 0
    N = len(items)
    Q = np.zeros(num_query)
    for i in range(N):
        if final_s[i]>r:
            Q += r*S[i]/final_s[i]
        else:
            Q +=S[i]
            count+=1

#这里的count就是count(I,r)            
    return count, Q
    
    
def RunAlgorithm():
    global epsilon 
    global beta
    global delta
    
    
    N = len(items)
    T = - 60/epsilon *math.log(4/beta)
    T_hat = T + LapNoise()*20/(1*epsilon)
    base = 1.3
    i = 0
    while(True):
        noise = LapNoise()*40/epsilon
        E , Q = calculate_E(pow(base,i))
        F = E -N
        F_hat = F + noise
        if F_hat > T_hat:
            tau = pow(base,i)
            break
        i +=1
#这里是SVT算法，算出阈值
    y = np.random.normal(0, 1, num_query)
    l = np.random.laplace(0, 1, num_query)
    # noises = tau * math.sqrt(2*math.log(1/delta)) * (1+0.9*epsilon/(4*math.log(1/delta))) / (0.9*epsilon) * np.random.normal(0, 1, num_query)
    # noises_y = tau * math.sqrt(2*math.log(1/delta)) * (1+0.9*epsilon/(4*math.log(1/delta))) / (0.9*epsilon) * y
    noises_l = tau  / (0.9*epsilon) * l
    Q = Q + noises_l
    
    return Q
    
    
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
        print("PMSJASJF.py -I <input file> -e <epsilon(default 1)> -b <beta(default 0.1)> -d <desired delta(no default)>")
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

    # print("Query Result")
    # print(result)
    # print("Noised Result")
    # print(Q)
    print("abs_error")
    print(abs_error)

    print("error_rate")
    print(error_rate)

    print("total_error")
    print(total_error)


    print("Time")
    print(end-start)
        # Define the content to be written to the file
        # content = """
        # Query Result
        # {}
        # Noised Result
        # {}
        # abs_error
        # {}
        # error_rate
        # {}
        # total_error
        # {}
        # time
        # {}
        # """.format(result, Q, abs_error, error_rate, total_error,end-start)

        # # Define the file path
        # outfile = input_file_prefix[-7:]
        # file_path = "../Result/TPCH/"+outfile+"/e_"+str(epsilon)+"_eps"+str(k)+".txt"

        # # Open the file in write mode and write the content
        # with open(file_path, 'w') as file:
        #     file.write(content)


if __name__ == '__main__':
	main(sys.argv[1:])
