import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import requests
import pymysql



ininital_w=0.8
ininital_c1=1.5
ininital_c2=1.5
chat_id = f"chat_{random.randint(100000, 999999)}"
# 数据加载函数（保持不变）
def load_data_from_file(file_path):
    data = {'coordinates': [], 'demand': [], 'capacity': 100}
    node_id_map = {}
    depot_index = 0

    with open(file_path, 'r') as f:
        section = None
        node_counter = 0

        for line in f:
            line = line.strip()
            if not line or line.startswith('EOF'):
                continue

            if line.startswith('CAPACITY'):
                data['capacity'] = int(line.split()[-1])
            elif line.startswith('NODE_COORD_SECTION'):
                section = 'COORD'
            elif line.startswith('DEMAND_SECTION'):
                section = 'DEMAND'
            elif line.startswith('DEPOT_SECTION'):
                section = 'DEPOT'
            else:
                if section == 'COORD':
                    parts = list(map(int, line.split()))
                    node_id = parts[0]
                    node_id_map[node_id] = node_counter
                    data['coordinates'].append(parts[1:])
                    node_counter += 1
                elif section == 'DEMAND':
                    parts = list(map(int, line.split()))
                    data['demand'].append(parts[1])
                elif section == 'DEPOT' and line != '-1':
                    depot_id = int(line)
                    depot_index = node_id_map[depot_id]

    data['coordinates'].insert(0, data['coordinates'].pop(depot_index))
    data['demand'].insert(0, data['demand'].pop(depot_index))
    data['coordinates'] = np.array(data['coordinates'], dtype=np.float64)
    data['demand'] = np.array(data['demand'], dtype=np.int32)
    return data


# 路径分割函数（保持不变）
def split_routes(nodes_seq, demand, capacity):
    routes = []
    current_route, current_load = [], 0
    for node in nodes_seq:
        d = demand[node]
        if current_load + d > capacity:
            routes.append(current_route)
            current_route = [node]
            current_load = d
        else:
            current_route.append(node)
            current_load += d
    if current_route:
        routes.append(current_route)
    return routes


# 距离计算函数（保持不变）
def calculate_distance(routes, coordinates):
    total = 0.0
    depot = coordinates[0]
    for route in routes:
        if not route: continue
        path = [depot] + [coordinates[n] for n in route] + [depot]
        for i in range(len(path) - 1):
            total += np.linalg.norm(path[i] - path[i + 1])
    return total


# 2-opt优化函数（保持不变）
def two_opt_improvement(route, data):
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_dist = calculate_distance(
                    split_routes(new_route, data['demand'], data['capacity']),
                    data['coordinates']
                )
                current_dist = calculate_distance(
                    split_routes(best, data['demand'], data['capacity']),
                    data['coordinates']
                )
                if new_dist < current_dist:
                    best = new_route
                    improved = True
        route = best
    return best


class Particle:
    """粒子类（离散PSO实现）"""

    def __init__(self, nodes):
        self.position = nodes.copy()  # 当前解（客户节点排列）
        self.velocity = []  # 交换操作序列
        self.best_position = nodes.copy()
        self.best_fitness = float('inf')


    def update_velocity(self, global_best, w, c1, c2):
        """速度更新（交换操作生成）"""
        w=ininital_w
        c1=ininital_c1
        c2=ininital_c2
        new_velocity = []

        # 保留部分原有速度（惯性）
        if random.random() < w:
            new_velocity.extend(self.velocity)

        # 个体认知（添加向个体最优的交换）
        if random.random() < c1:
            swap = self._get_swap_to_target(self.position, self.best_position)
            new_velocity.extend(swap)

        # 社会认知（添加向全局最优的交换）
        if random.random() < c2:
            swap = self._get_swap_to_target(self.position, global_best)
            new_velocity.extend(swap)

        self.velocity = new_velocity[-50:]  # 限制速度长度

    def _get_swap_to_target(self, current, target):
        """生成使current变为target的最小交换序列"""
        swaps = []
        temp = current.copy()
        for i in range(len(current)):
            if temp[i] != target[i]:
                j = temp.index(target[i])
                swaps.append((i, j))
                temp[i], temp[j] = temp[j], temp[i]
        return swaps

    def update_position(self):
        """应用速度到当前位置"""
        temp = self.position.copy()
        for (i, j) in self.velocity:
            temp[i], temp[j] = temp[j], temp[i]
        self.position = temp

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
        'Authorization': 'Bearer fastgpt-ufq7OOiumpKkAM2yrZDTDi7OaWmG6ZbrBgJfWiLCfGpGOoDlnVVT521r',
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

def content1(N, fitness, w, c1, c2):
    """生成发送给fastgpt的内容
    Args:
        N: 迭代次数
        fitness: 当前适应度值
        w: 惯性权重
        c1: 认知系数
        c2: 社会系数
    Returns:
        str: 格式化后的内容字符串
    """
    print(f"当前迭代次数: {N}")
    content = f"N:{N},w值:{w},c1值:{c1},c2值:{c2},效用值:{fitness}"
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
    cursor.execute('SELECT a FROM pso WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result = cursor.fetchone()
    a = result[0] if result else None
    cursor.execute('SELECT b FROM pso WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result1 = cursor.fetchone()
    b = result1[0] if result1 else None
    cursor.execute('SELECT W FROM pso WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result2 = cursor.fetchone()
    c = result2[0] if result1 else None
    print(a)
    print(b) 
    print(c)
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return a, b, c



def pso_vrp(data, max_iter=200, num_particles=50):
    """改进的PSO主算法"""
    nodes = list(range(1, len(data['coordinates'])))  # 客户节点列表

    # 初始化粒子群
    particles = [Particle(random.sample(nodes, len(nodes))) for _ in range(num_particles)]
    global_best = None
    global_best_fitness = float('inf')
    convergence = []

    # 初始化参数
    w = ininital_w
    c1 = ininital_c1
    c2 = ininital_c2

    for iter in range(max_iter):
        # 评估所有粒子
        current_best_fitness = float('inf')
        for p in particles:
            routes = split_routes(p.position, data['demand'], data['capacity'])
            fitness = calculate_distance(routes, data['coordinates'])

            # 更新个体最优
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.copy()

            # 更新全局最优
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best = p.position.copy()
                current_best_fitness = fitness

        # 记录收敛情况
        convergence.append(global_best_fitness)
        print(f"Iteration {iter + 1:03d}/{max_iter} | Best: {global_best_fitness:.2f}")

        # 调用fastgpt更新参数
        content = content1(iter + 1, fitness, w, c1, c2)
        response = fastgpt(content)
        
        # 从数据库获取更新后的参数
        c1, c2, w = conmysql(iter + 1)
        print(f"w:{w}")
        # 参数范围限制
        w = np.clip(w, 0.4, 0.9)
        c1 = np.clip(c1, 1.0, 2.0)
        c2 = np.clip(c2, 1.0, 2.0)

        # 粒子更新
        for p in particles:
            p.update_velocity(global_best, w, c1, c2)
            p.update_position()

        # 局部搜索（对前20%粒子优化）
        particles.sort(key=lambda x: x.best_fitness)
        for p in particles[:int(0.2 * num_particles)]:
            optimized = two_opt_improvement(p.position, data)
            p.position = optimized

    # 绘制收敛曲线
    plt.figure(figsize=(10, 4))
    plt.plot(convergence, 'b', lw=1)
    plt.title("PSO Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.grid(True)
    plt.show()

    return split_routes(global_best, data['demand'], data['capacity']), global_best_fitness 

    # 绘制收敛曲线
    plt.figure(figsize=(10, 4))
    plt.plot(convergence, 'b', lw=1)
    plt.title("PSO Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.grid(True)
    plt.show()

    return split_routes(global_best, data['demand'], data['capacity']), global_best_fitness


# 可视化函数（保持不变）
def plot_solution(routes, coordinates):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors

    plt.scatter(coordinates[1:, 0], coordinates[1:, 1],
                c='gray', s=50, edgecolors='k', label="Customers")
    plt.scatter(coordinates[0, 0], coordinates[0, 1],
                c='red', s=200, marker='s', edgecolors='k', label="Depot (0)")

    for i, route in enumerate(routes):
        if not route: continue
        path = [coordinates[0]] + [coordinates[n] for n in route] + [coordinates[0]]
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'o-', color=colors[i % 10],
                 lw=2, ms=8, mec='k', label=f"Route {i + 1}")
        for node in route:
            plt.text(coordinates[node][0], coordinates[node][1] + 1.5,
                     str(node + 1), ha='center', fontsize=8)

    plt.title(f"VRP Solution (Distance: {calculate_distance(routes, coordinates):.2f})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data_from_file("E:\zyz-work\FSP\VRP\A-n32-k5.vrp")
    best_routes, best_dist = pso_vrp(data, max_iter=100, num_particles=50)

    print("\n=== Optimal Solution ===")
    print(f"Total Distance: {best_dist:.2f}")
    for i, route in enumerate(best_routes, 1):
        node_ids = [0] + [n + 1 for n in route] + [0]
        print(f"Route {i}: {' → '.join(map(str, node_ids))}")

    plot_solution(best_routes, data['coordinates'])
