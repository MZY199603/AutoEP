# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
from numba import jit, prange
import requests
import pymysql


@jit(nopython=True)
def compute_dis_mat_numba(location):
    num_city = location.shape[0]
    dis_mat = np.zeros((num_city, num_city))
    for i in prange(num_city):
        for j in prange(num_city):
            if i == j:
                dis_mat[i, j] = np.inf
                continue
            dx = location[i, 0] - location[j, 0]
            dy = location[i, 1] - location[j, 1]
            dis_mat[i, j] = round(np.sqrt(dx * dx + dy * dy), 12)
    return dis_mat


@jit(nopython=True)
def compute_pathlen_numba(path, dis_mat):
    total = 0.0
    for i in prange(len(path) - 1):
        total += dis_mat[path[i], path[i + 1]]
    total += dis_mat[path[-1], path[0]]
    return round(total, 12)


@jit(nopython=True)
def two_opt_numba(path, dis_mat):
    best_len = compute_pathlen_numba(path, dis_mat)
    improved = True
    while improved:
        improved = False
        for i in prange(1, len(path) - 2):
            for j in prange(i + 2, len(path)):
                new_path = path.copy()
                new_path[i:j] = new_path[i:j][::-1]
                new_len = compute_pathlen_numba(new_path, dis_mat)
                if new_len < best_len:
                    path = new_path
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
    return path


class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.chat_id = f"chat_{random.randint(100000, 999999)}"
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.initial_cross_prob = 0.6
        self.cross_prob = self.initial_cross_prob
        self.initial_mutate_prob = 0.3
        self.mutate_prob = self.initial_mutate_prob
        self.elite_size = 1
        self.dis_mat = compute_dis_mat_numba(data)
        self.fruits = self.diversified_greedy_init(self.dis_mat, num_total, num_city)

        self.iter_x = [0]
        self.iter_y = [compute_pathlen_numba(self.fruits[0], self.dis_mat)]
        self.best_record = [self.iter_y[0]]

    def diversified_greedy_init(self, dis_mat, num_total, num_city):
        result = []
        num_start = min(num_city, 10)
        starts = np.random.choice(num_city, num_start, replace=False)
        for start in starts:
            partial = self.greedy_init_from_point(dis_mat, num_total // num_start, num_city, start)
            result.extend(partial)
        return np.array(result)

    def greedy_init_from_point(self, dis_mat, num_per, num_city, start):
        return np.array([self.greedy_path(dis_mat, num_city, start) for _ in range(num_per)])

    def greedy_path(self, dis_mat, num_city, start):
        path = [start]
        remaining = np.delete(np.arange(num_city), start)
        for _ in range(num_city - 1):
            next_node = np.argmin(dis_mat[path[-1], remaining])
            path.append(remaining[next_node])
            remaining = np.delete(remaining, next_node)
        return path

    def compute_adp(self, fruits):
        lengths = np.array([compute_pathlen_numba(p, self.dis_mat) for p in fruits])
        return np.round(1.0 / lengths, 12)

    def ordered_crossover(self, x, y):
        len_ = len(x)
        start, end = sorted(np.random.choice(len_, 2, replace=False))
        child = np.full(len_, -1)
        child[start:end] = x[start:end]
        mask = np.isin(y, child, invert=True)
        fill = y[mask]
        child[end:] = fill[:len(child) - end]
        child[:start] = fill[len(child) - end:]
        return child

    def adaptive_mutate(self, gene):
        if np.random.rand() < self.mutate_prob:
            i, j = np.random.choice(len(gene), 2, replace=False)
            if i > j:
                i, j = j, i
            gene[i:j + 1] = gene[i:j + 1][::-1]
        return gene


    def fastgpt(self, content):
        print(content)
        url = 'http://192.168.50.25:3002/api/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer fastgpt-udNunmcYyZ2FZMFXsRYiQ8ghoW94vob2j8YHEe5I9pimkttMfHL0U9yP',
            'Content-Type': 'application/json'
        }
        data = {
            "chatId": self.chat_id,
            "stream": False,
            "detail": False,
            "messages": [{
                "role": "user",
                "content": content}]}
        response = requests.post(url, headers=headers, json=data)
        text = response.text
        return text

    def conmysql(self, n):
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


    def content1(self, N, fitness, cross_prob,mutate_prob):
        print(N)
        content = "N:" + str(N) + ",A值:" + str(mutate_prob) + ",B值:" + str(
            cross_prob) + ",效用值:" + str(fitness)
        return content

    def update_probabilities(self, iteration, fitness, cross_prob, mutate_prob):
        """
        根据当前种群适应度情况自适应调整交叉概率和变异概率
        """
        iteration = iteration  # 现在的迭代次数
        fitness = fitness  # 现在的适应度值

        content = self.content1(iteration, fitness, cross_prob, mutate_prob)
        print(content)
        response = self.fastgpt(content)
        print(iteration)
        self.mutate_prob, self.cross_prob = self.conmysql(iteration)




    def ga(self):
        scores = self.compute_adp(self.fruits)
        #self.update_probabilities(scores)

        elite_idx = np.argsort(-scores)[:self.elite_size]
        elite = self.fruits[elite_idx]

        parents = self.fruits[np.argsort(-scores)[:int(self.ga_choose_ratio * self.num_total)]]
        new_fruits = elite.copy()

        with Pool() as pool:
            while len(new_fruits) < self.num_total:
                parent_idx = np.random.choice(len(parents), 2, replace=False)
                x, y = parents[parent_idx]

                if np.random.rand() < self.cross_prob:
                    child1, child2 = self.ordered_crossover(x, y), self.ordered_crossover(y, x)
                else:
                    child1, child2 = x.copy(), y.copy()



                child1 = two_opt_numba(child1, self.dis_mat)
                child2 = two_opt_numba(child2, self.dis_mat)

                child1 = self.adaptive_mutate(child1)
                child2 = self.adaptive_mutate(child2)
                
                len1 = compute_pathlen_numba(child1, self.dis_mat)
                len2 = compute_pathlen_numba(child2, self.dis_mat)
                if len1 < len2 and not any(np.array_equal(child1, f) for f in new_fruits):
                    new_fruits = np.vstack([new_fruits, child1])
                elif not any(np.array_equal(child2, f) for f in new_fruits):
                    new_fruits = np.vstack([new_fruits, child2])

        self.fruits = new_fruits
        best_idx = np.argmin([compute_pathlen_numba(p, self.dis_mat) for p in new_fruits])
        return new_fruits[best_idx], compute_pathlen_numba(new_fruits[best_idx], self.dis_mat)

    def run(self):
        best_len = np.inf
        best_path = None
        for i in range(1, self.iteration + 1):
            current_best, current_len = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(current_len)
            if current_len < best_len:
                best_len = current_len
                best_path = current_best
            self.best_record.append(best_len)
            self.fitness = min(self.best_record)
            self.update_probabilities(i, self.fitness, self.cross_prob, self.mutate_prob)
            print(f"Iteration {i}: Best Length = {best_len:.12f}")
        return best_path, self.location[best_path], best_len


def read_tsp(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    idx = lines.index('NODE_COORD_SECTION\n')
    data = []
    for line in lines[idx + 1:]:
        if line.strip() == 'EOF':
            break
        parts = list(map(float, line.strip().split()))
        data.append(parts[1:])
    return np.array(data)


def main():
    data = read_tsp('data/rat575.tsp')
    model = GA(num_city=data.shape[0], num_total=10, iteration=500, data=data)
    best_path_indices, best_path, best_len = model.run()

    # 确保路径是封闭的，即回到起点
    best_path = np.vstack([best_path, best_path[0]])
    best_path_indices = np.append(best_path_indices, best_path_indices[0])

    print("最优路径遍历城市的序号：", best_path_indices)
    print(f"最优路径长度: {best_len:.12f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.scatter(best_path[:, 0], best_path[:, 1])
    ax1.plot(best_path[:, 0], best_path[:, 1], linewidth=0.5)
    ax1.set_title('Optimized Route')

    ax2.plot(range(model.iteration + 1), model.best_record)
    ax2.set_title('Convergence Curve')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Route Length')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()
