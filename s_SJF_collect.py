import math
import os
import sys
import subprocess
epsilon = [1,4,8]
queries = ["Q1", "Q2"]
scales = ["sc_0","sc_3","sc_5","sc_6"]
#这个脚本是用来处理前一步得到的输出
# epsilon = [1,4,8]
# queries = ["Q1"]
# scales = ["sc_0"]

def main(argv):
    for e in epsilon:
        for query in queries:              
            for scale in scales:

                # output_file = open("../Result/TPCH/"+ scale+"/"+query + "/"+"e_"+str(epsilon)+"_eps"+str(i)+".txt", 'w')       

                output_file = open("../Result/TPCH/"+ scale+"_"+query + "_e_"+str(e)+".txt", 'w')                   
                cmd = "python3 collect_data.py -I " + "../Result/TPCH/" +scale+ "/"+query+ "/e_"+str(e)

                if e == 2:
                    cmd = "python3 collect_data.py -I " + "../Result/TPCH/" +scale+ "/"+query+ "/eps"
                print(cmd)
                shell = os.popen(cmd, 'r')
                res = shell.read()

                # output_file.write("The result for the "+str(i)+"th execution is"+"\n")
                output_file.write( res + "\n")

if __name__ == "__main__":
	main(sys.argv[1:])

