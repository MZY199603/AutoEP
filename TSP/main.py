

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import requests
import pymysql


class GA(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.num_total = num_total
        self.scores = []
        self.fitness=0
        self.chat_id = f"chat_{random.randint(100000, 999999)}"
        self.iteration = iteration
        self.location = data
        self.ga_choose_ratio = 0.2
        self.initial_cross_prob = 0.5
        self.cross_prob = self.initial_cross_prob
        self.initial_mutate_prob = 0.5
        self.mutate_prob = self.initial_mutate_prob
        self.total_fitness_sum = 0
        self.max_fitness = -math.inf
        self.min_fitness = math.inf
        self.elite_size = 1  # 精英个体数量，保留每代中的最优个体直接进入下一代
        # fruits中存每一个个体是下标的list
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.fruits = self.diversified_greedy_init(self.dis_mat, num_total, num_city)  # 使用改进的初始化方法
        # 显示初始化后的最佳路径
        scores = self.compute_adp(self.fruits)
        sort_index = np.argsort(-scores)
        init_best = self.fruits[sort_index[0]]
        init_best = self.location[init_best]

        # 存储每个iteration的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / scores[sort_index[0]]]

    def diversified_greedy_init(self, dis_mat, num_total, num_city):
        """
        改进的多样化贪心初始化方法，从多个随机起始点构建路径
        """
        result = []
        num_start_points = min(num_city, 10)  # 选择10个不同起始点，可调整
        start_points = random.sample(range(num_city), num_start_points)
        for start in start_points:
            partial_result = self.greedy_init_from_point(dis_mat, num_total // num_start_points, num_city, start)
            result.extend(partial_result)
        return result

    def greedy_init_from_point(self, dis_mat, num_per_start, num_city, start_index):
        """
        从指定起始点进行贪心初始化构建部分个体
        """
        result = []
        for _ in range(num_per_start):
            rest = [x for x in range(0, num_city)]
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest)!= 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
        return result

    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest)!= 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        try:
            a = path[0]
            b = path[-1]
        except:
            import pdb
            pdb.set_trace()
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群适应度
    def compute_adp(self, fruits):
        adp = []
        for fruit in fruits:
            if isinstance(fruit, int):
                import pdb
                pdb.set_trace()
            length = self.compute_pathlen(fruit, self.dis_mat)
            adp.append(1.0 / length)
        return np.array(adp)

    def swap_part(self, list1, list2):
        index = len(list1)
        list = list1 + list2
        list = list[::-1]
        return list[:index], list[index:]

    def ordered_crossover(self, x, y):
        """
        顺序交叉（OX）操作，一种常用于TSP的交叉算子
        """
        len_ = len(x)
        assert len(x) == len(y)
        start, end = sorted(random.sample(range(len_), 2))
        child_1 = [-1] * len_
        child_2 = [-1] * len_
        # 复制选定区间
        child_1[start:end] = x[start:end]
        child_2[start:end] = y[start:end]
        # 填充剩余部分
        index_1 = end
        index_2 = end
        for i in range(len_):
            index_y = (start + i) % len_
            index_x = (start + i) % len_
            if y[index_y] not in child_1:
                child_1[index_1 % len_] = y[index_y]
                index_1 += 1
            if x[index_x] not in child_2:
                child_2[index_2 % len_] = x[index_x]
                index_2 += 1
        return child_1, child_2

    def ga_cross(self, x, y):
        if random.random() > self.cross_prob:
            return x, y
        return self.ordered_crossover(x, y)

    def ga_parent(self, scores, ga_choose_ratio):
        sort_index = np.argsort(-scores).copy()
        sort_index = sort_index[0:int(ga_choose_ratio * len(sort_index))]
        parents = []
        parents_score = []
        for index in sort_index:
            parents.append(self.fruits[index])
            parents_score.append(scores[index])
        return parents, parents_score

    def ga_choose(self, genes_score, genes_choose):
        sum_score = sum(genes_score)
        score_ratio = [sub * 1.0 / sum_score for sub in genes_score]
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        for i, sub in enumerate(score_ratio):
            if rand1 >= 0:
                rand1 -= sub
                if rand1 < 0:
                    index1 = i
            if rand2 >= 0:
                rand2 -= sub
                if rand2 < 0:
                    index2 = i
            if rand1 < 0 and rand2 < 0:
                break
        return list(genes_choose[index1]), list(genes_choose[index2])

    def adaptive_mutate(self, gene,mutate_prob):
        """
        根据个体适应度自适应调整变异概率的变异操作
        """
        fitness = 1. / self.compute_pathlen(gene, self.dis_mat)
        #if fitness < self.avg_fitness:
            #mutate_prob = self.initial_mutate_prob * (1 + (self.avg_fitness - fitness) / self.avg_fitness)
        #else:
            #mutate_prob = self.initial_mutate_prob * (1 - (fitness - self.avg_fitness) / self.avg_fitness)
            
        if random.random() < mutate_prob:
            path_list = [t for t in range(len(gene))]
            order = list(random.sample(path_list, 2))
            start, end = min(order), max(order)
            tmp = gene[start:end]
            tmp = tmp[::-1]
            gene[start:end] = tmp
        return gene

    # content = "N:"+str(N)+",A值:"+str(pheromone_factor_alpha)+",B值:"+str(visible_factor_beta)+",效用值:"+str(cycle_distance)
    # request调用fasgptapi接口你每一次执行完算法调用fastgpt接口
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

    # 每跑一个epoch，你调用fastgpt方法，然后你可以判断它的返回值，fastgpt流程走完你就调用 conmysql方法，它会返回a，b
    # 接收fastgpt生成的参数进行储存
    # @app.route('/post_example', methods=['post'])
    # def update_input_canshu(a,b,n):
    #     a = request.form['a']
    #     b = request.form['b']
    #     #这里可以直接执行你的主程序（a,b参数）你执行的轮次可以返回给我
    #     with open('input.txt', 'a') as file:
    #         file.write(f'{a}, {b}\n')
    #     return 'Parameters saved successfully'

    # 配置数据库连接信息
    
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

    def update_probabilities(self,iteration,fitness,cross_prob,mutate_prob):
        """
        根据当前种群适应度情况自适应调整交叉概率和变异概率
        """
        iteration=iteration #现在的迭代次数
        fitness = fitness   #现在的适应度值
    

        content = self.content1(iteration,fitness,cross_prob,mutate_prob)
        print(content)
        response = self.fastgpt(content)
        print(iteration)
        self.mutate_prob, self.cross_prob = self.conmysql(iteration)


    # def ga(self):
    #     # 获得优质父代
    #     scores = self.compute_adp(self.fruits)
    #     self.scores = scores  # 将当前种群适应度赋值给属性，方便后续计算平均等情况
    #     self.total_fitness_sum = sum(scores)  # 更新总适应度和
    #     # 精英保留
    #     elite_inds = np.argsort(-scores)[:self.elite_size]
    #     elite_fruits = [self.fruits[i] for i in elite_inds]
    #
    #     # 选择部分优秀个体作为父代候选集合
    #     parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
    #     # 新的种群fruits
    #     fruits = elite_fruits.copy()
    #     while len(fruits) < self.num_total:
    #         # 轮盘赌方式对父代进行选择
    #         gene_x, gene_y = self.ga_choose(parents_score, parents)
    #         # 交叉
    #         gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
    #         # 变异
    #         gene_x_new = self.adaptive_mutate(gene_x_new)
    #         gene_y_new = self.adaptive_mutate(gene_y_new)
    #         x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
    #         y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
    #         # 将适应度高的放入种群中
    #         if x_adp > y_adp and (not gene_x_new in fruits):
    #             fruits.append(gene_x_new)
    #         elif x_adp <= y_adp and (not gene_y_new in fruits):
    #             fruits.append(gene_y_new)
    #
    #     self.fruits = fruits
    #     self.update_probabilities()  # 调用方法自适应调整概率
    #
    #     return elite_fruits[0], 1. / self.compute_pathlen(elite_fruits[0], self.dis_mat)

    def ga(self):
        # 获得优质父代
        scores = self.compute_adp(self.fruits)
        self.scores = scores
        self.total_fitness_sum = sum(scores)
        self.avg_fitness = self.total_fitness_sum / self.num_total  # 计算平均适应度

        # 先更新概率，确保avg_fitness等属性被正确计算赋值


        # 精英保留
        elite_inds = np.argsort(-scores)[:self.elite_size]
        elite_fruits = [self.fruits[i] for i in elite_inds]

        # 选择部分优秀个体作为父代候选集合
        parents, parents_score = self.ga_parent(scores, self.ga_choose_ratio)
        # 新的种群fruits
        fruits = elite_fruits.copy()
        while len(fruits) < self.num_total:
            # 轮盘赌方式对父代进行选择
            gene_x, gene_y = self.ga_choose(parents_score, parents)
            # 交叉
            gene_x_new, gene_y_new = self.ga_cross(gene_x, gene_y)
            # 变异
            gene_x_new = self.adaptive_mutate(gene_x_new,self.mutate_prob)
            gene_y_new = self.adaptive_mutate(gene_y_new,self.mutate_prob)
            x_adp = 1. / self.compute_pathlen(gene_x_new, self.dis_mat)
            y_adp = 1. / self.compute_pathlen(gene_y_new, self.dis_mat)
            # 将适应度高的放入种群中
            if x_adp > y_adp and (not gene_x_new in fruits):
                fruits.append(gene_x_new)
            elif x_adp <= y_adp and (not gene_y_new in fruits):
                fruits.append(gene_y_new)

        self.fruits = fruits
        return elite_fruits[0], 1. / self.compute_pathlen(elite_fruits[0], self.dis_mat)

    def run(self):
        BEST_LIST = None
        best_score = -math.inf
        self.best_record = []
        for i in range(1, self.iteration + 1):
            tmp_best_one, tmp_best_score = self.ga()
            self.iter_x.append(i)
            self.iter_y.append(1. / tmp_best_score)
            if tmp_best_score > best_score:
                best_score = tmp_best_score
                BEST_LIST = tmp_best_one
            self.best_record.append(1. / best_score)   # 列表的最小值是fitness
            self.fitness = min(self.best_record)
            print(i, 1. / best_score)
            self.update_probabilities(i,self.fitness,self.cross_prob,self.mutate_prob)
        print(1. / best_score)
        return self.location[BEST_LIST], 1. / best_score


# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('kroA150.tsp')

data = np.array(data)
data = data[:, 1:]
Best, Best_path = math.inf, None

model = GA(num_city=data.shape[0], num_total=300, iteration=1000, data=data.copy())
path, path_len = model.run()
if path_len < Best:
    Best = path_len
    Best_path = path
# 加上一行因为会回到起点
fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
axs[0].scatter(Best_path[:, 0], Best_path[:, 1])
Best_path = np.vstack([Best_path, Best_path[0]])
axs[0].plot(Best_path[:, 0], Best_path[:, 1])
axs[0].set_title('规划结果')
iterations = range(model.iteration)
best_record = model.best_record
axs[1].plot(iterations, best_record)
axs[1].set_title('收敛曲线')
plt.show()



