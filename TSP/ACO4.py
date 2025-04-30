"""
author:Xu Zhenxing（NUDT）
data:2023/10/18   20:05
theme:Uav edge computing task assignment and trajectory planning
e-mail：xuzhenxing@nudt.edu.cn
"""



import random
import pickle
from collections import namedtuple
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
import math
import requests
import pymysql



# ***** begin 配置信息 *****
# **** 题目数据配置

generate_new_flag = True
max_car_no = 6
min_car_no = 6  # 如果car_no设置得较小，可能会鉴权失败多次，效率差不能设置得大于 cargo_site_total
cargo_site_total = 20 #访问节点的总数量
#site_location_range = tuple(range(-100, 101, 5))
#car_carrying_range = tuple(range(180, 221, 10))
#car_speed_range = tuple(range(20, 41, 10))

move_cargo_second = 2   # 简化处理的每次装卸货时间, 也可以使用与货物重量相关的变量


time_weight = 0.4 #时间权重
distance_weight = 0.3 #距离权重
cargo_no_weight = 1 - time_weight - distance_weight #用车数量的权重

# **** 算法配置
max_iter = 100

algorithm_GA = False
algorithm_ACO = True


# *** GA配置
population_size = 100
mutate_max_try_times = 20
select_ratio = 0.8


# *** aco 配置

ant_no = 30
pheromone_factor_alpha = 0.7  # 信息素启发式因子
visible_factor_beta = 2 # 能见度启发式因子
volatil_factor = 0.8  # 信息素挥发速度
pheromone_cons = 100  # 信息素强度
N=1#迭代次数


g0 = -20
H = 100
xita = -0.00001
B = 10
P = 1
P0 = 79.85  # 0.012/8*1.225*0.05*0.503*(300**3)*(0.4**3)
Pi = 161.1  # 2*(math.pow(20,3/2))/math.sqrt(2*1.225*0.503)
V1 = 10
V0 = 4.03
U_tip = 120
d0 = 0.6
ro = 1.225
s = 0.05
A = 0.503
V = 10
UAV_num = 5



def rate1(g0, H, xita, B, P):

    """
    :param g0: 表示单位空间距离的信道功率
    :param H:  无人机与节点的高度距离
    :param xita: 通信链路的高斯白噪声功率
    :param B:通信带宽
    :param p: 数据传输的信噪比
    :return: 节点传输信息的速率

    """
    hij = g0 / (H ** 2 + H)  # 无线信道链路功率增益
    rate = (B * math.log2(1 + P * hij / xita)) / 10  # 传输速率
    return rate  # 计算得出为76.36


rate = rate1(g0, H, xita, B, P)


def energy(P0, Pi, V1, V0, U_tip, d0, ro, s, A):
    """
    :param P0:优化后的参数
    :param Pi:优化后的参数
    :param V1:无人机的速度
    :param V0:悬停时平均旋翼诱导速度
    :param U_tip:转子叶片的叶尖速度
    :param d0:机身阻力比
    :param ro:空气密度
    :param s:叶片的面积
    :param A:转子盘半径
    :return:无人机飞行时和悬停时每秒能耗
    """
    P_V = P0 * (1 + (3 * V1 ** 2) / (U_tip ** 2)) + Pi * math.sqrt(
        math.sqrt(1 + V1 ** 4 / (4 * V0 ** 4)) - V1 ** 2 / (2 * V0 ** 2)) \
          + 0.5 * d0 * ro * s * A * V1 ** 3


    P_X = P0 * (1 + (3 * 0 ** 2) / (U_tip ** 2)) + Pi * math.sqrt(
        math.sqrt(1 + 0 ** 4 / (4 * V0 ** 4)) - 0 ** 2 / (2 * V0 ** 2)) \
          + 0.5 * d0 * ro * s * A * 0 ** 3

    return P_V, P_X  # 计算得出为154.86,240.95,P_V是无人机飞行时每秒的能耗，P_X是无人机悬停时每秒的能耗


P_V, P_X = energy(P0, Pi, V1, V0, U_tip, d0, ro, s, A)

#
# Customer = [(50, 50), (96, 24), (40, 5), (49, 8), (13, 7),
#             (29, 89), (48, 30), (84, 39), (14, 47), (2, 24), (3, 82),
#             (65, 10), (98, 52), (84, 25), (41, 69), (1, 65),
#             (51, 71), (75, 83), (29, 32), (83, 3), (50, 93),
#             (80, 94), (5, 42), (62, 70), (31, 62), (19, 97),
#             (91, 75), (27, 49), (23, 15), (20, 70), (85, 60), (98, 85),
#             (62, 89), (97, 38), (2, 33), (12, 77)]

Customer = [(0, 0), (-95, 40), (-40, -50), (95, -55), (40, -15),
                (-10, 95), (45, -50), (-10, 60), (75, 15), (-35, 50), (-85, 20),
                (45, 25), (30, 0), (30, -90), (5, -60), (65, 65),
                (-90, 65), (-80, 90), (-75, 15), (-15, 40),(-15,40)]


Demand = [0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8, 14, 7, 16, 3,
          22, 18, 19, 1, 14, 8]

# Demand = [0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8, 14, 7, 16, 3,
#           22, 18, 19, 1, 14, 8, 12, 4, 8, 24, 24, 2, 10, 15, 2,
#           14, 9, 12, 14, 16, 5, 6]


Demand = [rate * num for num in Demand]

CityCoordinates = Customer


def calDistance(CityCoordinates):
    '''
    计算各节点距离
    输入：CityCoordinates-节点坐标；
    输出：各节点间距离矩阵-dis_matrix
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    # 创建dis_matrix使用 columns 和 index 参数指定了列索引和行索引
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)

    return dis_matrix


def hover_time(Demand):  # 无人机收集信息盘旋的时间
    hover_time = [i / rate for i in Demand]
    return hover_time


hover_time = hover_time(Demand)


def run_time(CityCoordinates):
    '''
    计算城市间距离
    输入：CityCoordinates-城市坐标；
    输出：城市间距离矩阵-dis_matrix
    '''
    dis_matrix_time = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(
        len(CityCoordinates)))
    # 创建dis_matrix使用 columns 和 index 参数指定了列索引和行索引
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix_time.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) / V, 2)

    return dis_matrix_time  # 想要找到i到j的时间，dis_matrix.loc[i,j]


dis_matrix_time = run_time(CityCoordinates)


def energy_paste():  # 无人机运行时消耗的能量

    energy_paste_pv = P_V * dis_matrix_time  # 飞行时消耗的能量
    energy_paste_px = [num * P_X for num in hover_time]  # 盘旋时消耗的能量

    return energy_paste_pv, energy_paste_px


energy_paste_pv, energy_paste_px = energy_paste()    #共用

max_cargo_weight = 100  #车辆的最大载重
max_car_distance = 250* P_V #最大的距离限制


# **** 画图配置
plot_flag = True  # windows下运行 可设为True
# ***** end 配置信息 *****


# 其他定义

cargo_site_info = namedtuple("cargo_site_info", ("site_location_x", "site_location_y", "cargo_weight"))
car_info = namedtuple("car_info", ("speed", "volume"))
time_info = namedtuple("time_info", ("time","UAV"))

class VRPData:#√√√√√√√√√√√√√√
    def __init__(self, cargo_sites, cargo_site_info_dict, car_info_dict):
        self.cargo_sites = cargo_sites          #城市的编号
        self.cargo_site_info_dict = cargo_site_info_dict#每个城市的x，y坐标和单位载重
        self.car_info_dict = car_info_dict#每辆车的编号速度和载重限制
        self.site_location = [(info.site_location_x, info.site_location_y)
                              for key, info in sorted(self.cargo_site_info_dict.items(), key=lambda x: x[0])]
        #获取(site_location_x, site_location_y)到site_location


    def get_a_possible_try(self, cargo_site_list=None):#√√√√√√√√√√√√√√
        if not cargo_site_list:
            cargo_site_list = random.sample(self.cargo_sites, cargo_site_total)#随机生成选择城市的序列1*20
            cargo_site_list1 = random.sample(self.cargo_sites, cargo_site_total)
        try_times = 0
        while True:
            car_list = random.sample(range(1, max_car_no + 1), random.randint(min_car_no, max_car_no))#第一个解
            #不生成在cargo_sites的第一位和最后一位   car_location 从1开始(非从0开始)，生成car的编号序列
            car_location = random.sample(range(2, cargo_site_total - 1), len(car_list) - 1)
            car_location.append(cargo_site_total)  # 最后一辆车
            car_location.sort()#是上述car_list的停靠位置，停靠点的位置，初始解的判断，哪个点返回基地
            car_location2no = {location: car_no for location, car_no in zip(car_location, car_list)}

            # 其实不做强鉴权 或以一定概率 保持不合法的解 也是可以的，后面可能变异成合法解，判断约束后是否满足解的生成
            if self.cargo_site_and_car_location_authok(cargo_site_list, car_location, car_location2no):
                break
            try_times += 1

            #满足约束的生成解，car_list，car_location，car_location2no
        if try_times >= 500:
            print("get_a_possible_try try times is %s:" % try_times)

        cargo_site_car_list, cargo_car_loc2carno = self.get_cargo_site_car_list(car_location, cargo_site_list, car_list)
        return cargo_site_car_list, cargo_car_loc2carno

        #cargo_site_car_list：加入了返回基站的0，完整的访问序列
        #cargo_car_loc2carno：在哪个城市点哪个编号的车返回基地，字典形式



    def get_cargo_site_car_list(self, car_location, cargo_site_list, car_list):#√√√√√√√√√√√√√√
        cargo_site_car_list = cargo_site_list.copy()#城市的序列
        # todo 效率
        for location in car_location[::-1]:
            cargo_site_car_list.insert(location, 0)#在停靠后返回基站起点，让城市序列有返回0
        cargo_car_loc2carno = self.get_cargo_car_loc2carno(cargo_site_car_list, car_list)
        return cargo_site_car_list, cargo_car_loc2carno


    def get_cargo_car_loc2carno(self, cargo_site_car_list, car_list):#√√√√√√√√√√√√√√
        ix = 0
        cargo_car_loc2carno = {}
        for location_ix, location in enumerate(cargo_site_car_list):#使用enumerate函数，location_ix是id，location是具体值
            if location == 0:
                cargo_car_loc2carno[location_ix] = car_list[ix]
                ix += 1
        return cargo_car_loc2carno#在哪个城市点哪个编号的车返回基地，字典形式

    def cargo_site_and_car_location_authok(self, cargo_site_list, car_location, car_location2no):#√√√√√√√√√√√√√√
        # 如果生成连续两个0  两辆车连在一起 后一辆未载重 放弃
        if (np.diff(np.array(car_location)) == 0).any():
            return False
        # if (np.diff(np.array(car_location1)) == 0).any():
        #     return False
        #np.diff(np.array(car_location)) 计算 car_location 数组的连续元素之间的差异。
        # 如果任何两个连续的位置相同（即差异为0，表示有两辆车分配到了同一个位置），any() 函数将返回 True，
        # 因此整个条件为 True，函数返回 False，表示这个车辆位置安排是无效的。-
        # 如果car_no较小，可能会鉴权失败多次，效率差
        last_index = 0
        for location in car_location:
            tmp_cargo_sites = cargo_site_list[last_index: location]#计算每辆车的约束，从基站0开始到第一个停靠点
            # tmp_cargo_sites1 = cargo_site_list1[last_index: location1]

            if sum([self.cargo_site_info_dict[cargo_site_no].cargo_weight for cargo_site_no in tmp_cargo_sites]
                   ) > self.car_info_dict[car_location2no[location]].volume:
                return False

            # if sum([self.cargo_site_info_dict[cargo_site_no].cargo_weight for cargo_site_no in tmp_cargo_sites1]
            #        ) > self.car_info_dict[car_location2no1[location1]].volume:
            #     return False

            total_distance = 0
            prev_site = 0  # 假设所有车辆都从位置0出发
            for site in tmp_cargo_sites:
                total_distance += energy_paste_pv[prev_site][site]
                total_distance += energy_paste_px[site]
                prev_site = site
            # 加上从最后一个站点返回起始点（假设为0）的距离
            total_distance += energy_paste_pv[prev_site][0]

            # 获取当前车辆的最大允许行驶距离
            car_no = car_location2no[location]
            if total_distance > max_car_distance:
                return False
            last_index = location

        #计算生成的策略是否能满足载重的约束，距离约束？？
        return True



    def cargo_site_car_list_authok(self, cargo_site_car_list, cargo_car_loc2carno):  #GA中用到
        last_index = 0
        for location_ix, location in enumerate(cargo_site_car_list):
            if location == 0:
                tmp_cargo_sites = cargo_site_car_list[last_index: location_ix]
                if sum([self.cargo_site_info_dict[cargo_site_no].cargo_weight for cargo_site_no in tmp_cargo_sites]
                       ) > self.car_info_dict[cargo_car_loc2carno[location_ix]].volume:
                    # print("location_index", location_index, car_info_dict[car_ix + 1],  car_ix + 1,)
                    return False
                last_index = location_ix
        return True

    def get_population_by_size(self, size=population_size):#生成初始解，和各个城市返回基站的点信息

        population_info = [self.get_a_possible_try() for _ in range(size)]
        return population_info



class VRPPlot:
    def __init__(self, cargo_sites, cargo_site_info_dict, car_info_dict):
        self.data = VRPData(cargo_sites, cargo_site_info_dict, car_info_dict)

    @classmethod
    def cp_from_ins(cls, vrp_data_ins):
        plt_ins = cls(vrp_data_ins.cargo_sites, vrp_data_ins.cargo_site_info_dict, vrp_data_ins.car_info_dict)
        return plt_ins


    def run(self, cargo_site_car_list):
        plot_data = self.get_plot_data(cargo_site_car_list)
        fig1 = plt.figure(1)
        for a_car_path in plot_data:
            self.plot_path(a_car_path)
        plt.draw()
        plt.pause(10)
        plt.close(fig1)

    def get_plot_data(self, cargo_site_car_list):
        plot_data = []
        a_car_path = []
        for x in cargo_site_car_list:
            if not x:
                plot_data.append([0] + a_car_path + [0])
                a_car_path = []
            else:
                a_car_path.append(x)
        return plot_data

    def plot_path(self, path):
        x_list = []
        y_list = []
        for x in path:
            temp = self.data.site_location[x]
            x_list.append(temp[0])
            y_list.append(temp[1])
        plt.plot(x_list, y_list)

# class TimeInfo:
#     def __init__(self, time, UAV):
#         self.time = time
#         self.UAV = UAV

class VrpHeuristic:


    def __init__(self, cargo_sites, cargo_site_info_dict, car_info_dict):
        self.data = VRPData(cargo_sites, cargo_site_info_dict, car_info_dict)#继承上个类的数据

        self.distance_matrix = squareform(pdist(np.array(self.data.site_location)))

        #计算各个城市之间的距离，distance_matrix的形式为二维列表，各个城市与其他城市的距离在第二维

    def calculate_fitness_info(self, cargo_site_car_list, cargo_car_loc2carno):#计算时间和距离
        distances = []
        car_num=[]
        car_ix_consume_time = {}  # car_ix 不是 car_no， 只为后面变异提供参考
        site_ix_consume_time = {}
        car_ix = 0
        last_site_no = 0  # 其实不会有0号site  这里是为了计算距离
        distance = 0
        calculate_hover_time=0

        car_ix_consume_time_dict={}

        for ix, x in enumerate(cargo_site_car_list):
            tmp_distance = self.distance_matrix[x][last_site_no]
            if x > 0:#计算car的这条路径
                distance += tmp_distance
                calculate_hover_time += hover_time[x]#每个节点收集数据的时间

                site_ix_consume_time.update({x:time_info(time=0,UAV=0)})

                car_num.append(x)#每个无人机收集的节点
                last_site_no = x
            else:#等于0时，返回基站，计算时间，清理数据

                distances.append(distance)
                consume_time = (distance / self.data.car_info_dict[cargo_car_loc2carno[ix]].speed) + calculate_hover_time
                for site in car_num:
                    site_ix_consume_time.update({site: time_info(time=consume_time, UAV=cargo_car_loc2carno[ix])})
                    # site_ix_consume_time[site]["time"]=consume_time
                    # site_ix_consume_time[site]["UAV"] = cargo_car_loc2carno[ix]
                car_ix_consume_time_dict.update({cargo_car_loc2carno[ix]:[consume_time]})#哪一架无人机和飞行的时间

                car_ix += 1
                car_ix_consume_time[car_ix] = consume_time#一个列表，记录每辆车的时间
                last_site_no = 0
                distance = 0
                calculate_hover_time =0
                car_num=[]
        return car_ix_consume_time, distances,site_ix_consume_time , car_ix_consume_time_dict


    def AOI(self,site_ix_consume_time1 ,site_ix_consume_time2):
        average_AOI=[]
        for key in site_ix_consume_time2:
            time1=site_ix_consume_time1[key].time
            UAV1=site_ix_consume_time1[key].UAV
            time2=site_ix_consume_time2[key].time
            UAV2 = site_ix_consume_time2[key].UAV
            if UAV1 == UAV2:
                average_AOI.append(time1+time2)
            else:
                for key, value in site_ix_consume_time1.items():
                    if value.UAV == UAV2:
                        time =site_ix_consume_time1[key].time
                        if time > time1:
                            average_AOI.append(time2 + time)
                            break
                        else:
                            average_AOI.append(time2 + time1)
                            break

        return (-np.mean(average_AOI))


    def data_security(self, path , cargo_site_car_list ):#判断是否窃取无人机的信息
        position=[]
        random.seed(1)  #攻击节点不变化
        position= random.sample(path,10)   #被攻击的节点
        get_data=[]
        get_data_possible =random.uniform(0.4,0.6)  #攻击节点的概率

        for site in position:

            if get_data_possible >= random.uniform(0,1):

                target = site

                # 找到target的索引
                index_of_target = cargo_site_car_list.index(target)

                # 初始化一个变量来存储上一个0的索引
                index_of_prev_zero = -1

                # 从target的索引开始，向前搜索直到找到0
                for i in range(index_of_target, -1, -1):
                    if cargo_site_car_list[i] == 0:
                        index_of_prev_zero = i
                        break
                # 如果target之前没有0，或者target是第一个元素，则从列表开始到target的位置
                # 如果找到了0，则获取从上一个0到target之间的所有元素
                if index_of_prev_zero == -1:
                    elements_before_target_inclusive = cargo_site_car_list[:index_of_target + 1]
                else:
                    elements_before_target_inclusive = cargo_site_car_list[index_of_prev_zero + 1:index_of_target + 1]

                get_data.append(elements_before_target_inclusive)
        get_data=sum(get_data,[])

        return get_data

    def get_data_possible(self, get_data ,get_data1):

        set1=set(get_data)
        set2=set(get_data1)
        get_data_possible_sum =len(set1.intersection(set2))

        return get_data_possible_sum

    # def get_data_possible(self, get_data, get_data1):
    #     set1 = set(tuple(item) for item in get_data)
    #     set2 = set(tuple(item) for item in get_data1)
    #     return len(set1.intersection(set2))

    def calculate_fitness(self,  site_ix_consume_time1 ,  site_ix_consume_time2, path , cargo_site_car_list, path1 , cargo_site_car_list1):
        get_data = self.data_security( path , cargo_site_car_list )
        get_data1 =self.data_security( path1 , cargo_site_car_list1 )
        get_data_possible_sum=self.get_data_possible(get_data,get_data1)

        Fitness= self.AOI(site_ix_consume_time1 ,  site_ix_consume_time2)  +  (-100*get_data_possible_sum)

        return Fitness

    def sum_calculate_fitness(self, path , cargo_site_car_list, path1 , cargo_site_car_list1):
        get_data = self.data_security( path , cargo_site_car_list )
        get_data1 =self.data_security( path1 , cargo_site_car_list1 )
        get_data_possible_sum=self.get_data_possible(get_data,get_data1)


        return  get_data_possible_sum



    def get_fitness_info(self, cargo_site_car_list, cargo_car_loc2carno):  # 适应度值计算
        car_ix_consume_time, distances, site_ix_consume_time , car_ix_consume_time_dict = self.calculate_fitness_info(cargo_site_car_list, cargo_car_loc2carno)

        return -(time_weight * max(car_ix_consume_time.values()) + distance_weight * sum(distances)
                 + cargo_no_weight * len(cargo_car_loc2carno))



    def get_distance_by_path(self, path):  #计算路径距离
        distance = []
        last_index = -1
        for ix, site in enumerate(path):
            if ix != 0:
                distance.append(self.distance_matrix[last_index][site])
            last_index = site
        return distance







class ACO(VrpHeuristic):

    def __init__(self, cargo_sites, cargo_site_info_dict, car_info_dict, phero_by_path_length=False):
        super().__init__(cargo_sites, cargo_site_info_dict, car_info_dict)
        #按一般的蚁群算法介绍，phero_by_path_length 应该设为True;  这里为了解这道VRP题，设为False
        self.phero_by_path_length = phero_by_path_length
        self.phero_mat = np.ones((cargo_site_total + 1, cargo_site_total + 1)) #信息素浓度矩阵
        self.phero_mat1 = np.ones((cargo_site_total + 1, cargo_site_total + 1))
        self.visible_mat = 1 / (self.distance_matrix + np.eye(cargo_site_total + 1))#可见度矩阵
        self.visible_mat[np.diag_indices_from(self.visible_mat)] = 0  # 对角线值置0
        self.visible_mat1 = 1 / (self.distance_matrix + np.eye(cargo_site_total + 1))  # 可见度矩阵
        self.visible_mat1[np.diag_indices_from(self.visible_mat)] = 0  # 对角线值置0
        self.pheromone_factor_alpha = 0.7  # 信息素启发式因子
        self.pheromone_factor_beta = 2 

    def update_input_canshu(self, new_pheromone_factor_alpha,new_pheromone_factor_beta):





        pheromone_factor_alpha = new_pheromone_factor_alpha
        pheromone_factor_beta  = new_pheromone_factor_alpha
        return pheromone_factor_alpha, pheromone_factor_beta

    def update_output_canshu(self,N,pheromone_factor_alpha,pheromone_factor_beta,cycle_distance):

        pheromone_factor_alpha, pheromone_factor_beta




        pass

    def fastgpt(self,content):
        print(content)
        url = 'http://192.168.50.25:3002/api/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer fastgpt-fxw5VXAGwxm9ycMxASetiJ8bJBBUoaUCpvhRwsjyFe2qXo9iKiJXb',
            'Content-Type': 'application/json'
        }
        data = {
            "chatId": f"chat_{random.randint(100000, 999999)}",
            "stream": False,
            "detail": False,
            "messages": [{
                "role": "user",
                "content": content}]}
        response = requests.post(url, headers=headers, json=data)
        text = response.text
        return text

    def content1(self,N, fitness, cross_prob,mutate_prob):
        print(N)
        content = "N:" + str(N) + ",A值:" + str(mutate_prob) + ",B值:" + str(
            cross_prob) + ",效用值:" + str(fitness)
        return content

    def conmysql(self,n):
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



    def __call__(self):
        best_ants_info = None
        best_ants_info1 = None
        for ix in range(max_iter):
            ants_info = []
            ants_info1=[]
            for ant_ix in range(ant_no):#蚁群数量
                path, cargo_site_car_list, cargo_car_loc2carno = self.generate_path_solution(ant_ix)
                path1, cargo_site_car_list1, cargo_car_loc2carno1 = self.generate_path_solution1(ant_ix)

                if not path:
                    continue
                _,_,site_ix_consume_time1,_ = self.calculate_fitness_info(cargo_site_car_list, cargo_car_loc2carno)
                _,_,site_ix_consume_time2,_ = self.calculate_fitness_info(cargo_site_car_list1, cargo_car_loc2carno1)


                score = self.get_fitness_info(cargo_site_car_list, cargo_car_loc2carno)

                score1 = self.calculate_fitness(site_ix_consume_time1, site_ix_consume_time2, path, cargo_site_car_list,
                                                path1, cargo_site_car_list1)

                cycle_distance = sum(self.get_distance_by_path([0] + path + [0]))
                cycle_distance1 = sum(self.get_distance_by_path([0] + path1 + [0]))
                ants_info.append((score, cycle_distance, path, cargo_site_car_list, cargo_car_loc2carno))
                ants_info1.append((score1, cycle_distance1, path1, cargo_site_car_list1, cargo_car_loc2carno1))

            if  ants_info:
                # 一轮蚂蚁完成后更新信息素
                best_ants_info = self.get_best_ant_info(ants_info, best_ants_info)

                best_ants_info1=self.get_best_ant_info1(ants_info1, best_ants_info1)
                self.update_phero_mat(ants_info)
                self.update_phero_mat1(ants_info1)
                content = self.content1(N + 1, cycle_distance, self.pheromone_factor_alpha, self.pheromone_factor_beta)
                response = self.fastgpt(content)
                # pheromone_factor_alpha, pheromone_factor_beta= self.update_output_canshu(N,pheromone_factor_alpha,pheromone_factor_beta,cycle_distance)
                self.pheromone_factor_alpha, self.pheromone_factor_beta = self.conmysql(N + 1)

            if ix % 10 == 0:
                print("run times %s, best score is: %s, car_no is %s, cycle_distance is %s, best score1 is: %s, car_no1 is %s, cycle_distance1 is %s "
                      % (ix, best_ants_info[0], len(best_ants_info[-1]), best_ants_info[1],best_ants_info1[0], len(best_ants_info1[-1]), best_ants_info1[1]))
                sum_get   =   self.sum_calculate_fitness(path, cargo_site_car_list, path1, cargo_site_car_list1)
                print(sum_get)


        self.print_final_result(best_ants_info[0], best_ants_info[-2], best_ants_info[-1],best_ants_info1[-2])
        self.print_final_result(best_ants_info1[0], best_ants_info1[-2], best_ants_info1[-1], best_ants_info1[-2])

        return None


    def generate_path_solution(self, ant_ix):
        path = self.generate_path(ant_ix)
        if path:
            cargo_site_car_list, cargo_car_loc2carno = self.add_car_in_path(path)
            return path, cargo_site_car_list, cargo_car_loc2carno
        return [], [], {}

    def generate_path_solution1(self, ant_ix):
        path = self.generate_path1(ant_ix)
        if path:
            cargo_site_car_list, cargo_car_loc2carno = self.add_car_in_path(path)
            return path, cargo_site_car_list, cargo_car_loc2carno
        return [], [], {}
    def generate_path(self, ant_ix, abnormal_threshold=1e-50, skip_abnormal=False):#根据信息素浓度来选择生成的路径
        path = [self.get_first_cargo_site_no(ant_ix)]
        visited_set = set(path)
        while len(path) < cargo_site_total:
            current_site = path[-1]
            unvisited = [x for x in self.data.cargo_sites if x not in visited_set]
            result = [np.power(self.phero_mat[current_site][site], pheromone_factor_alpha) *
                      np.power(self.visible_mat[current_site][site], visible_factor_beta)
                      for site in unvisited]
            result_sum = sum(result)

            if result_sum < abnormal_threshold:
                # result_sum太小或为0 是因为 current_site 与每一个unvisited的 x y坐标隔得太远了 可能是在反方向上，
                # 此时 前面就错了，不应该留下来类似这种：已经跑到右上很多了，余下只有 在左下很多的货物点；
                # skip_abnormal为True时 放弃这一只蚂蚁 否则 仅放弃当前的路径 重新generate_path。
                if skip_abnormal:
                    # print("break! give up the path!")
                    return []
                path = path[:1]
                visited_set = set(path)
                continue
            probs = np.cumsum(result / result_sum)
            index_need = np.where(probs > np.random.rand())[0][0]
            # 有文章说: 需要一部分蚂蚁遵循信息素最高的分配策略，还需要一部分蚂蚁遵循随机分配的策略，以发现新的局部最优解。
            # 所以这里也可以再加上 随机分配
            next_site = unvisited[index_need]
            path.append(next_site)
            visited_set.add(next_site)
        return path

    def generate_path1(self, ant_ix, abnormal_threshold=1e-50, skip_abnormal=False):#根据信息素浓度来选择生成的路径
        path = [self.get_first_cargo_site_no(ant_ix)]
        visited_set = set(path)
        while len(path) < cargo_site_total:
            current_site = path[-1]
            unvisited = [x for x in self.data.cargo_sites if x not in visited_set]
            result = [np.power(self.phero_mat1[current_site][site], pheromone_factor_alpha)
                      * np.power(self.visible_mat1[current_site][site], visible_factor_beta)
                      for site in unvisited]
            result_sum = sum(result)

            if result_sum < abnormal_threshold:
                # result_sum太小或为0 是因为 current_site 与每一个unvisited的 x y坐标隔得太远了 可能是在反方向上，
                # 此时 前面就错了，不应该留下来类似这种：已经跑到右上很多了，余下只有 在左下很多的货物点；
                # skip_abnormal为True时 放弃这一只蚂蚁 否则 仅放弃当前的路径 重新generate_path。
                if skip_abnormal:
                    # print("break! give up the path!")
                    return []
                path = path[:1]
                visited_set = set(path)
                continue
            probs = np.cumsum(result / result_sum)
            index_need = np.where(probs > np.random.rand())[0][0]
            # 有文章说: 需要一部分蚂蚁遵循信息素最高的分配策略，还需要一部分蚂蚁遵循随机分配的策略，以发现新的局部最优解。
            # 所以这里也可以再加上 随机分配
            next_site = unvisited[index_need]
            path.append(next_site)
            visited_set.add(next_site)
        return path

    def get_first_cargo_site_no(self, ant_ix):# 给蚂蚁随机分配一个初始的起点
        # cargo_site_no 从1开始的
        if ant_no < cargo_site_total or (ant_ix + 1) > int(ant_no / cargo_site_total) * ant_no:
            return random.randint(1, cargo_site_total)# 给蚂蚁随机分配一个初始的起点
        else:
            return ant_ix % cargo_site_total + 1


    def add_car_in_path(self, path):
        cargo_site_car_list, cargo_car_loc2carno = self.data.get_a_possible_try(path)   #
        return cargo_site_car_list, cargo_car_loc2carno

    def get_best_ant_info(self, ants_info, best_ants_info=None):
        # ants_info:   score, cycle_distance, path, cargo_site_car_list, cargo_car_loc2carno
        temp_ants_info = ants_info.copy()
        if best_ants_info:
            temp_ants_info.append(best_ants_info)

        temp_ants_info = sorted(temp_ants_info, key=lambda x: x[0], reverse=True)

        return temp_ants_info[0]

    def get_best_ant_info1(self, ants_info, best_ants_info=None):
        # ants_info:   score, cycle_distance, path, cargo_site_car_list, cargo_car_loc2carno
        temp_ants_info = ants_info.copy()
        if best_ants_info:
            temp_ants_info.append(best_ants_info)

        temp_ants_info = sorted(temp_ants_info, key=lambda x: x[0], reverse=True)

        return temp_ants_info[0]

    def update_phero_mat(self, ants_info):
        # phero_by_path_length时 采用蚁周模型，否则用适应度信息
        temp_phero_mat = np.zeros((cargo_site_total + 1, cargo_site_total + 1))

        for score, cycle_distance, path, cargo_site_car_list, cargo_car_loc2carno in ants_info:
            last_cargo_site = -1
            for cargo_site in path:
                if last_cargo_site != -1:
                    temp = pheromone_cons / cycle_distance if self.phero_by_path_length \
                        else pheromone_cons /  (cycle_distance * - score)
                    temp_phero_mat[last_cargo_site][cargo_site] += temp
                    temp_phero_mat[cargo_site][last_cargo_site] += temp
                last_cargo_site = cargo_site
        self.phero_mat = (1 - volatil_factor) * self.phero_mat + temp_phero_mat



    def update_phero_mat1(self, ants_info):
        # phero_by_path_length时 采用蚁周模型，否则用适应度信息
        temp_phero_mat = np.zeros((cargo_site_total + 1, cargo_site_total + 1))

        for score, cycle_distance, path, cargo_site_car_list, cargo_car_loc2carno in ants_info:
            last_cargo_site = -1
            for cargo_site in path:
                if last_cargo_site != -1:

                    temp = pheromone_cons / (cycle_distance * -score)
                    temp_phero_mat[last_cargo_site][cargo_site] += temp
                    temp_phero_mat[cargo_site][last_cargo_site] += temp
                last_cargo_site = cargo_site
        self.phero_mat1 = (1 - volatil_factor) * self.phero_mat1 + temp_phero_mat




    def print_final_result(self, fitness_score, cargo_site_car_list, cargo_car_loc2carno,cargo_site_car_list1):
        print("\nACO: the final best score is %s cargo_site_car_list is %s, where the 0s represents car_list %s."
              % (round(fitness_score, 2), cargo_site_car_list,
                 [car_no for (_, car_no) in sorted(cargo_car_loc2carno.items(), key=lambda x: x[0])]))
        if plot_flag:
            VRPPlot.cp_from_ins(self.data).run(cargo_site_car_list)
            VRPPlot.cp_from_ins(self.data).run(cargo_site_car_list1)


def run():
    cargo_sites = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    car_info_dict = {1: car_info(speed=40, volume=250), 2: car_info(speed=30, volume=290),
                     3: car_info(speed=40, volume=270), 4: car_info(speed=20, volume=290),
                     5: car_info(speed=20, volume=290), 6: car_info(speed=20, volume=270),
                     7: car_info(speed=30, volume=270), 8: car_info(speed=30, volume=290),
                     9: car_info(speed=30, volume=290), 10: car_info(speed=40, volume=290)}
    cargo_site_info_dict = {1: cargo_site_info(site_location_x=-95, site_location_y=40, cargo_weight=0),
                            2: cargo_site_info(site_location_x=-40, site_location_y=-50, cargo_weight=31),
                            3: cargo_site_info(site_location_x=95, site_location_y=-55, cargo_weight=92),
                            4: cargo_site_info(site_location_x=40, site_location_y=-15, cargo_weight=78),
                            5: cargo_site_info(site_location_x=-10, site_location_y=95, cargo_weight=68),
                            6: cargo_site_info(site_location_x=45, site_location_y=-50, cargo_weight=93),
                            7: cargo_site_info(site_location_x=-10, site_location_y=60, cargo_weight=66),
                            8: cargo_site_info(site_location_x=75, site_location_y=15, cargo_weight=36),
                            9: cargo_site_info(site_location_x=-35, site_location_y=50, cargo_weight=59),
                            10: cargo_site_info(site_location_x=-85, site_location_y=20, cargo_weight=4),
                            11: cargo_site_info(site_location_x=45, site_location_y=25, cargo_weight=84),
                            12: cargo_site_info(site_location_x=30, site_location_y=0, cargo_weight=87),
                            13: cargo_site_info(site_location_x=30, site_location_y=-90, cargo_weight=76),
                            14: cargo_site_info(site_location_x=50, site_location_y=40, cargo_weight=51),
                            15: cargo_site_info(site_location_x=5, site_location_y=-60, cargo_weight=59),
                            16: cargo_site_info(site_location_x=65, site_location_y=65, cargo_weight=97),
                            17: cargo_site_info(site_location_x=-90, site_location_y=65, cargo_weight=89),
                            18: cargo_site_info(site_location_x=-80, site_location_y=90, cargo_weight=5),
                            19: cargo_site_info(site_location_x=-75, site_location_y=15, cargo_weight=64),
                            20: cargo_site_info(site_location_x=-15, site_location_y=40, cargo_weight=99),
                            #21: cargo_site_info(site_location_x=-19, site_location_y=46, cargo_weight=69),
                            0: cargo_site_info(site_location_x=0, site_location_y=0, cargo_weight=0)}

    #cargo_sites, cargo_site_info_dict, car_info_dict = QuestionDataHandle().get_data(generate_new=generate_new_flag)

    ACO(cargo_sites, cargo_site_info_dict, car_info_dict)()


if __name__ == '__main__':
    run()
