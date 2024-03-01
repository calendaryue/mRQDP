import sys
import os
import time

#这个脚本是用来记录Q3 4 5 的join关系提取时间，在7个数据集上（要补 现有记录是分开的

def main(argv):
    # queries = ["Q3", "Q7"]
    # scales = ["0","1","2","3","4","5","6"]
    queries = ["Q3", "Q4","Q5"]
    scales = ["sc_0","sc_3","sc_5","sc_6"]
    # queries = ["Q3"]
    # scales = ["sc_0"]
    output_file = open( "../Result/TPCH/Scale_TimesQ345.txt", 'w')
    for query in queries:
        for scale in scales:

            output_file.write(query+"_Scale"+scale+" times: \n")

            start = time.time()
            # if query=="Q3":
            cmd = "python3 ExtractInfoMultiple.py -D "+scale+" -Q ../Query/"+query+".txt -P ids -K ../Query/"+query+"_key.txt -O ../Information/TPCH/"+scale+"/"+query
            #cmd = "python3 ExtractInfoMultiple.py -D " +scale+ " -Q ../Query/"+query+".txt -P ids -K ../Query/"+query+"_key.txt -O ../Information/TPCH/"+query
            
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

