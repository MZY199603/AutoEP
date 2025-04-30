import random
import numpy as np
import matplotlib.pyplot as plt
import pymysql

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
        content = content1(iter + 1, current_best_fitness, w, c1, c2)
        response = fastgpt(content)
        
        # 从数据库获取更新后的参数
        w, c1, c2 = conmysql(iter + 1)
        
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
    cursor.execute('SELECT w FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result = cursor.fetchone()
    w = result[0] if result else None
    cursor.execute('SELECT c1 FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result1 = cursor.fetchone()
    c1 = result1[0] if result1 else None
    cursor.execute('SELECT c2 FROM canshu2 WHERE n=%s ORDER BY id DESC LIMIT 1', (n,))
    result2 = cursor.fetchone()
    c2 = result2[0] if result2 else None
    print(w, c1, c2)
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return w, c1, c2 