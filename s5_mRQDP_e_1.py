import math
import os
import sys
import subprocess
repeat_times = 20

epsilon = 1
queries = ["Q5"]
scales = ["sc_0","sc_1","sc_2","sc_3","sc_4","sc_5","sc_6"]

# epsilon = 1
# queries = ["Q3"]
# scales = ["sc_0"]

def main(argv):
    for query in queries:              
        for scale in scales:
            for i in range(repeat_times):
                output_file = open("../Result/TPCH/"+ scale+"/"+query + "/"+"e_"+str(epsilon)+"_eps"+str(i)+".txt", 'w')       
                cmd = "python3 mRQDP_selfjoin.py -I " + "../Information/TPCH/" +scale+ "/"+query+ " -e "+str(epsilon) +" -b 0.1"
                shell = os.popen(cmd, 'r')
                res = shell.read()

                output_file.write("The result for the "+str(i)+"th execution is"+"\n")
                output_file.write( res + "\n")

if __name__ == "__main__":
	main(sys.argv[1:])

