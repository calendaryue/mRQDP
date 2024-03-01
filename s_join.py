import sys
import os
import time

#这个脚本是用来记录Q1 2 的join关系提取时间，在7个数据集上

def main(argv):
    queries = ["Q1", "Q2"]
    scales = ["sc_0","sc_2","sc_4","sc_1","sc_3","sc_5","sc_6"]
    # queries = ["Q1"]
    # scales = ["sc_1"]
    output_file = open( "../Result/TPCH/Scale_Times_Q12.txt", 'w')
    for query in queries:
        for scale in scales:

            output_file.write(query+"_Scale"+scale+" times: \n")

            start = time.time()
            # if query=="Q3":
            cmd = "python3 ExtractInfoMultiple.py -D "+scale+" -Q ../Query/"+query+".txt -P customer -K ../Query/"+query+"_key.txt -O ../Information/TPCH/"+scale+"/"+query
            #cmd = "python3 ExtractInfoMultiple.py -D " +scale+ " -Q ../Query/"+query+".txt -P ids -K ../Query/"+query+"_key.txt -O ../Information/TPCH/"+query
            #python3 ExtractInfoMultiple.py -D sc_0 -Q ../Query/Q2.txt -P customer -K ../Query/Q2_key.txt -O ../Information/TPCH/Q2
            # else:
            #     cmd = "../../python ../Code/ExtractInfoMultiple.py -D psc"+scale+" -Q ../Query/"+query+".txt -P ids -K ../Query/"+query+"_key.txt -O ../Information/"+query+"_"+scale

            
            shell = os.popen(cmd, 'r')
            shell.read()
            shell.close()
            end= time.time()
            time_used = end-start
            output_file.write("The time for the execution is"+"\n")
            output_file.write(str(time_used)+"\n")
            

    
if __name__ == "__main__":
	main(sys.argv[1:])

