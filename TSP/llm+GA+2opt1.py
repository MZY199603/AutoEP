import numpy as np
import random
from typing import List, Tuple
import requests
import pymysql

chat_id = f"chat_{random.randint(100000, 999999)}"
pop_size=100,
generations=200,
initial_cross_prob=0.8,
initial_mutate_prob=0.1
def read_taillard(file_path: str) -> np.ndarray:
    """读取Taillard数据集并返回处理时间矩阵"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    processing_times = []
    read_data = False
    for line in lines:
        if "processing times" in line:
            read_data = True
            continue
        if read_data and len(line.strip()) > 0:
            processing_times.append([int(x) for x in line.strip().split()])
    return np.array(processing_times).T  # 转换为作业×机器矩阵


def calculate_makespan(sequence: List[int], processing_times: np.ndarray) -> int:
    """计算给定调度序列的总完工时间(makespan)"""
    num_jobs, num_machines = processing_times.shape
    timeline = np.zeros((num_jobs, num_machines))

    # 第一台机器的累积时间
    timeline[0, 0] = processing_times[sequence[0], 0]
    for i in range(1, num_jobs):
        timeline[i, 0] = timeline[i - 1, 0] + processing_times[sequence[i], 0]

    # 后续机器的累积时间
    for m in range(1, num_machines):
        timeline[0, m] = timeline[0, m - 1] + processing_times[sequence[0], m]
        for i in range(1, num_jobs):
            timeline[i, m] = max(timeline[i - 1, m], timeline[i, m - 1]) + processing_times[sequence[i], m]

    return int(timeline[-1, -1])


def two_opt(sequence: List[int], processing_times: np.ndarray) -> List[int]:
    """2-opt局部优化算法"""
    improved = True
    best_sequence = sequence.copy()
    best_makespan = calculate_makespan(best_sequence, processing_times)

    while improved:
        improved = False
        for i in range(1, len(sequence) - 1):
            for j in range(i + 1, len(sequence)):
                new_sequence = sequence[:i] + sequence[i:j + 1][::-1] + sequence[j + 1:]
                new_makespan = calculate_makespan(new_sequence, processing_times)
                if new_makespan < best_makespan:
                    best_sequence = new_sequence
                    best_makespan = new_makespan
                    improved = True
        sequence = best_sequence.copy()
    return best_sequence


def adaptive_mutate(gene: List[int], mutate_prob: float) -> List[int]:
    """自适应变异操作：以给定概率进行子序列逆转"""
    new_gene = gene.copy()
    if random.random() < mutate_prob:
        i, j = random.sample(range(len(gene)), 2)
        if i > j:
            i, j = j, i
        new_gene[i:j + 1] = new_gene[i:j + 1][::-1]
    return new_gene


def parameter_updata(cross_prob,mutate_prob):
    iteration=iteration #现在的迭代次数
    fitness = fitness   #现在的适应度值
    content = content1(iteration,fitness,cross_prob,mutate_prob)
    print(content)
    response = fastgpt(content)
    print(iteration)
    mutate_prob, cross_prob = conmysql(iteration)
    return response


#初始化化的cross_prob，mutate_prob，iteration，然后计算出适应度fitness，然后调用fastgpt，生产出新的cross_prob，mutate_prob，iteration，会直接导入数据库，然后调用conmysql，获取新的cross_prob，mutate_prob，更新参数一直迭代

def fastgpt(content):
    print(content)
    url = 'http://192.168.50.25:3002/api/v1/chat/completions'
    headers = {
        'Authorization': 'Bearer fastgpt-fxw5VXAGwxm9ycMxASetiJ8bJBBUoaUCpvhRwsjyFe2qXo9iKiJXb',
        'Content-Type': 'application/json'
    }
    data = {
        "chatId": chat_id,
        "stream": False,
        "detail": False,
        "messages": [{
            "role": "user",
            "content": content}]}
    response = requests.post(url, headers=headers, json=data)
    text = response.text
    return text

def content1(N, fitness, cross_prob,mutate_prob):
    print(N)
    content = "N:" + str(N) + ",A值:" + str(mutate_prob) + ",B值:" + str(
        cross_prob) + ",效用值:" + str(fitness)
    return content

def conmysql(n):
    # ... existing code ...
    config = {
         'host': '192.168.50.25',
         'port': 33306,
         'user': 'root',
         'password': 'oneapimmysql',
         'database': 'demo',
         'charset': 'utf8mb4'
        }
        # 创建连接
    conn = pymysql.connect(**config)
        # 创建游标
    cursor = conn.cursor()
    # 执行SQL查询
    cursor.execute('SELECT a FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result = cursor.fetchone()
    a = result[0] if result else None
    cursor.execute('SELECT b FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result1 = cursor.fetchone()
    b = result1[0] if result1 else None
    print(a)
    print(b) 
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return a, b

def ga_with_2opt_adaptive(processing_times: np.ndarray,
                          pop_size: int = 100,
                          generations: int = 500,
                          initial_cross_prob: float = 0.8,
                          initial_mutate_prob: float = 0.1) -> Tuple[List[int], int]:
    """带自适应参数调整的遗传算法主函数"""
    num_jobs = processing_times.shape[0]
    population = [random.sample(range(num_jobs), num_jobs) for _ in range(pop_size)]
    global_best = float('inf')
    best_sequence = []

    # 初始化参数
    cross_prob = initial_cross_prob
    mutate_prob = initial_mutate_prob

    for gen in range(generations):
        # 计算当前种群适应度
        scores = [calculate_makespan(ind, processing_times) for ind in population]
        current_fitness = min(scores)  # 当前最佳适应度

        # 调用fastgpt更新参数
        content = content1(gen + 1, current_fitness, cross_prob, mutate_prob)
        response = fastgpt(content)
        
        # 从数据库获取更新后的参数
        mutate_prob, cross_prob = conmysql(gen + 1)
        
        # 参数范围限制
        cross_prob = np.clip(cross_prob, 0.6, 0.9)
        mutate_prob = np.clip(mutate_prob, 0.01, 0.5)

        # 锦标赛选择
        new_population = []
        for _ in range(pop_size):
            candidates = random.sample(population, 3)
            candidates_fitness = [calculate_makespan(c, processing_times) for c in candidates]
            winner = candidates[np.argmin(candidates_fitness)]
            new_population.append(winner)

        # 顺序交叉(OX)
        offspring = []
        for _ in range(pop_size // 2):
            parent1, parent2 = random.sample(new_population, 2)
            if random.random() < cross_prob:
                start = random.randint(0, num_jobs - 1)
                end = random.randint(start + 1, num_jobs)
                child1 = parent1[start:end] + [g for g in parent2 if g not in parent1[start:end]]
                child2 = parent2[start:end] + [g for g in parent1 if g not in parent2[start:end]]
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        # 自适应变异
        offspring = [adaptive_mutate(ind, mutate_prob) for ind in offspring]

        # 2-opt局部优化
        offspring = [two_opt(ind, processing_times) for ind in offspring]

        # 合并种群
        combined_pop = new_population + offspring
        combined_fitness = [calculate_makespan(ind, processing_times) for ind in combined_pop]
        sorted_idx = np.argsort(combined_fitness)
        population = [combined_pop[i] for i in sorted_idx[:pop_size]]

        # 更新全局最优
        current_best = combined_fitness[sorted_idx[0]]
        if current_best < global_best:
            global_best = current_best
            best_sequence = combined_pop[sorted_idx[0]]

        # 迭代信息输出
        print(f"Gen {gen + 1:03d} | Best: {current_best:04d} | Global: {global_best:04d} | "
              f"Crossover: {cross_prob:.2f} | Mutation: {mutate_prob:.2f}")

    return best_sequence, global_best


if __name__ == "__main__":
    # 示例数据路径（需替换为实际路径）
    file_path = "data/tai20_5_0.fsp"
    processing_times = read_taillard(file_path)

    # 运行算法
    best_sequence, best_makespan = ga_with_2opt_adaptive(
        processing_times,
        pop_size=100,
        generations=200,
        initial_cross_prob=0.8,
        initial_mutate_prob=0.1
    )

    print(f"\nOptimal Sequence: {best_sequence}")
    print(f"Minimum Makespan: {best_makespan}")