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

    global input_file_prefix

    global error_rate_avg 
    global total_error
    global exe_time



    input_file_names = glob.glob(input_file_prefix+"*")

    error_rate_avg = [] #记录20次的平均错误率
    total_error = []
    exe_time = []
    for input_file_name in input_file_names:
        i=0
        #For each query
        input_file = open(input_file_name,'r')
        for line in input_file.readlines():
            # print(i)
            # print(line)
            if i == 45:               
                line = line.strip("[]\n")
                error_rate = [float(x) for x in line.split(",")]
                total = sum(error_rate)
                error_rate_avg.append(total/len(error_rate))

            if i == 47:
                total_error.append(float(line))
            if i == 49:
                exe_time.append(float(line))           
            i += 1
    print("error rate:")
    print(error_rate_avg)
    print("total error:")
    print(total_error)
    print("exe time:")
    print(exe_time)

    
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
        opts, args = getopt.getopt(argv,"h:I:",["Input="])
    except getopt.GetoptError:
        print("PMSJASJF.py -I <input file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("PMSJASJF.py -I <input file>")
            sys.exit()
        elif opt in ("-I", "--Input"):
            input_file_prefix = str(arg)

#../Result/TPCH/sc_0/Q2/eps_1.txt

    ReadInput()

if __name__ == '__main__':
	main(sys.argv[1:])
