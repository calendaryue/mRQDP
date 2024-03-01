Scale = ["sc_0", "sc_1", "sc_2", "sc_3", "sc_4", "sc_5", "sc_6"]
Query = ["Q6"]
Epsilon = [1, 2, 4, 8]
eps_num = 20  # Epsilon 的数量
# Query = ["Q1", "Q2", "Q3", "Q4", "Q5"]
Query = ["Q1"]

def calc_adjusted_avg(values):
    # 去掉最大的 5 个值和最小的 5 个值
    values_sorted = sorted(values)
    # 计算剩余值的平均
    return sum(values_sorted[5:-5]) / 10

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # 初始化变量以存储解析的值
    error_rates = []
    total_error = None
    Time = None
    for i in range(len(content)):
        line = content[i]
        if line.startswith('error_rate'):
            # 提取 error_rate 列表
            error_rates = eval(content[i+1].strip())
        elif line.startswith('total_error'):
            # 提取 total_error 值
            total_error = float(content[i+1].strip())
        elif line.startswith('Time'):
            # 提取 Time 值
            Time = float(content[i+1].strip())

    return error_rates, total_error, Time


# 遍历 Scale、Query 和 Epsilon 生成所有可能的文件路径
for scale in Scale:
    for query in Query:
        for epsilon in Epsilon:
            Input_Files = []

            for eps in range(eps_num):  # 对每个 Epsilon 生成 0 到 19 的序列
                file_path = f"../Result/TPCH/{scale}/{query}/e_{epsilon}_eps{eps}.txt"
                Input_Files.append(file_path)

            # 初始化列表以存储累积数据
            error_rate_totals = []
            total_error_totals = []
            Time_totals = []

            # 遍历每个文件，提取并累积数据
            for file_path in Input_Files:
                error_rates, total_error, Time = parse_file(file_path)
                # 计算当前文件的 error_rate 平均值并累积
                error_rate_avg = sum(error_rates) 
                error_rate_totals.append(error_rate_avg)
                total_error_totals.append(total_error)
                Time_totals.append(Time)

            # 对累积的数据进行处理，去掉最大和最小的 5 个值，计算平均
            adjusted_error_rate_avg = calc_adjusted_avg(error_rate_totals)
            adjusted_total_error_avg = calc_adjusted_avg(total_error_totals)
            adjusted_Time_avg = calc_adjusted_avg(Time_totals)

            # 打印调整后的平均值
            print(f'Error Rate Average: {adjusted_error_rate_avg} [{scale}/{query}/e_{epsilon}]')
            print(f'Total Error Average: {adjusted_total_error_avg} [{scale}/{query}/e_{epsilon}]')
            print(f'Time Average: {adjusted_Time_avg} [{scale}/{query}/e_{epsilon}]')
            print('\n')



