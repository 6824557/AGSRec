import os
import math
import pickle
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from datetime import datetime
import pytz
import utils
import heapq
parser = argparse.ArgumentParser()
parser.add_argument('--client_nums', default=35, type=int)
parser.add_argument('--dataset', default='NYCGlo', type=str)
parser.add_argument('--tot_judge_hot', default=0.01, type=float)
parser.add_argument('--graphweight', default=1, type=int)#缺失0，2，3
args = parser.parse_args()

weight=[[0.7,0.9,1.7,1.9],[0.5,0.75,1.25,1.5],[0.3,0.5,1.3,1.5],[0.1,0.3,1.1,1.3]]
weightnow=weight[args.graphweight]
def extract_date_info(date_str):
    """
    从给定的包含日期时间及时区信息的字符串中提取月份、日、星期几和时间信息。

    参数:
    date_str (str): 格式类似 "2012-04-03 18:00:09+00:00" 的日期时间字符串

    返回:
    tuple: 包含月份（int）、日（int）、星期几（str）、时间（str）的元组
    """
    # 将字符串解析为带有时区信息的datetime对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
    # 将其转换为UTC时间（因为原字符串里的时区标识是+00:00，也就是UTC时间）
    utc_date_obj = date_obj.astimezone(pytz.UTC)

    # 提取月份
    month = utc_date_obj.month

    # 提取日
    day = utc_date_obj.day

    # 提取星期几，这里weekday方法返回0表示周一，依次类推
    weekday_num = utc_date_obj.weekday()
    weekdays = ['**','monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    weekday = weekdays[weekday_num]

    # 提取时间，格式化为HH:MM:SS的形式
    clock = utc_date_obj.strftime("%H:%M:%S")
    # 获取小时数，用于时间分类映射
    hour = utc_date_obj.hour
    time_category = ""
    if 0 <= hour < 6:
        time_category = "midnight"
    elif 6 <= hour < 12:
        time_category = "early morning"
    elif hour == 12:
        time_category = "noon"
    elif 12 < hour < 18:
        time_category = "afternoon"
    elif hour >= 18:
        time_category = "night"

    return month, day, weekday, time_category,clock


#data load
def get_data(data_path):
    User, UserCate, UserTime, user_traj, itemUser = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    itemCate = {}
    usernum,itemnum,catenum = 0,0,0
    # assume user/item index starting from 1
    f = open(data_path+'/train_with_time.txt' , 'r')
    for line in f:
        u, i, c, c_name, lan, lon, timestap, time_utc,_,_,zone = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        c = int(float(c))
        lan = float(lan)
        lon = float(lon)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        catenum = max(c, catenum)
        itemCate[i] = c
        User[u].append(i)
        month, day, weekday, time_category, clock = extract_date_info(time_utc)
        user_traj[u].append([u, i, c, c_name,lan, lon,time_utc, month, day, weekday, clock, time_category])

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_traj[user] = user_traj[user]
        else:
            user_traj[user] = user_traj[user][:-2]
    return user_traj, itemCate, itemnum


def build_client_poi_checkin_graph(train_data,tot_pop):
    G = nx.DiGraph()
    G2 = nx.DiGraph()
    users = train_data.keys()
    for user_id in users:
        user_df = train_data[user_id]

        # add node (poi)
        for i in range(len(user_df)):
            node = user_df[i][1]
            if node not in G.nodes():
                G.add_node(user_df[i][1], check_count=1, poi_cat_id=user_df[i][2], poi_cat=user_df[i][3], latitude=user_df[i][4], longitude=user_df[i][5],
                           time_utc=user_df[i][6], time_category = user_df[i][7], poi_pop = tot_pop[user_df[i][1]-1])
                G2.add_node(user_df[i][1], check_count=1, poi_cat_id=user_df[i][2], poi_cat=user_df[i][3], latitude=user_df[i][4], longitude=user_df[i][5],
                           time_utc=user_df[i][6], time_category = user_df[i][7], poi_pop = tot_pop[user_df[i][1]-1])
            else:
                G.nodes[node]['check_count'] += 1
                G2.nodes[node]['check_count'] += 1

        pre_poi_id = user_df[0][1]
        pre_poi_lat = user_df[0][4]
        pre_poi_lon = user_df[0][5]
        pre_poi_time = user_df[0][6]
        pre_poi_pop = tot_pop[user_df[0][1]]
        for i in range(1, len(user_df)):
            poi_id = user_df[i][1]
            lat = user_df[i][4]
            lon = user_df[i][5]
            time = user_df[i][6]

            if G.has_edge(pre_poi_id, poi_id):
                a = (math.sin(math.radians(lon / 2 - pre_poi_lon / 2))) ** 2
                b = math.cos(lon * math.pi / 180) * math.cos(pre_poi_lon * math.pi / 180) * (
                    math.sin((lat / 2 - pre_poi_lat / 2) * math.pi / 180)) ** 2
                L = 2 * 6371.393 * math.asin((a + b) ** 0.5)
                # T = time - pre_poi_time

                if L < 1:
                    Lbonus = 1
                else:
                    Lbonus = 0.76/math.tanh(L)
                #吸引力权重1
                if tot_pop[pre_poi_id-1] >= args.tot_judge_hot and tot_pop[poi_id-1] >= args.tot_judge_hot:
                    Lbonus = weightnow[1]*Lbonus
                elif tot_pop[pre_poi_id-1] < args.tot_judge_hot and tot_pop[poi_id-1] > args.tot_judge_hot:
                    Lbonus = weightnow[0] * Lbonus
                elif tot_pop[pre_poi_id-1] > args.tot_judge_hot and tot_pop[poi_id-1] < args.tot_judge_hot:
                    Lbonus = weightnow[2] * Lbonus
                else:
                    Lbonus = weightnow[3] * Lbonus

                # 吸引力权重2
                Lbonus2 = tot_pop[pre_poi_id-1] + tot_pop[poi_id-1]

                G.edges[pre_poi_id, poi_id]['weight'] += Lbonus
                G2.edges[pre_poi_id, poi_id]['weight'] += Lbonus2
            else:
                a = (math.sin(math.radians(lon / 2 - pre_poi_lon / 2))) ** 2
                b = math.cos(lon * math.pi / 180) * math.cos(pre_poi_lon * math.pi / 180) * (
                    math.sin((lat / 2 - pre_poi_lat / 2) * math.pi / 180)) ** 2
                L = 2 * 6371.393 * math.asin((a + b) ** 0.5)
                # T = time - pre_poi_time

                if L < 1:
                    Lbonus = 1
                else:
                    Lbonus = 0.76 / math.tanh(L)

                # 吸引力权重
                if tot_pop[pre_poi_id-1] >= args.tot_judge_hot and tot_pop[poi_id-1] >= args.tot_judge_hot:
                    Lbonus = weightnow[1] * Lbonus
                elif tot_pop[pre_poi_id-1] < args.tot_judge_hot and tot_pop[poi_id-1] > args.tot_judge_hot:
                    Lbonus = weightnow[0] * Lbonus
                elif tot_pop[pre_poi_id-1] > args.tot_judge_hot and tot_pop[poi_id-1] < args.tot_judge_hot:
                    Lbonus = weightnow[2] * Lbonus
                else:
                    Lbonus = weightnow[3] * Lbonus

                # 吸引力权重2
                Lbonus2 = tot_pop[pre_poi_id - 1] + tot_pop[poi_id - 1]
                G.add_edge(pre_poi_id, poi_id, weight=Lbonus)
                G2.add_edge(pre_poi_id, poi_id, weight=Lbonus2)


            pre_poi_id = poi_id
            pre_poi_lat = lat
            pre_poi_lon = lon
            pre_poi_time = time

    return G,G2


def save_graph_to_pickle(G, file_name):
    pickle.dump(G, open(file_name, 'wb'))


def save_graph_to_csv(G, file_name, flag):
    # 保存图的邻接矩阵
    if flag ==0:
        name1 = "graph_matrixi"+str(args.graphweight)+".csv"
        name2 = "graph_nodesi"+str(args.graphweight)+".csv"
    else:
        name1 = "graph_matrix2.csv"
        name2 = "graph_nodes2.csv"
    nodelist = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodelist)
    np.savetxt(f'{file_name}/{name1}', A.todense(), delimiter=',')


    nodes_data = list(G.nodes.data())
    with open(f'{file_name}/{name2}', 'w') as f:
        print('node_name/poi_id,check_count,poi_cat_id,poi_cat,latitude,longitude,time_utc,time_category,poi_pop', file=f)
        for each in nodes_data:
            node_name = each[0]
            check_count = each[1]['check_count']
            poi_cat_id = each[1]['poi_cat_id']
            poi_cat = each[1]['poi_cat']
            latitude = each[1]['latitude']
            longitude = each[1]['longitude']
            time_utc = each[1]['time_utc']
            time_category = each[1]['time_category']
            poi_pop = each[1]['poi_pop']
            print(f'{node_name},{check_count},{poi_cat_id},{poi_cat},{latitude},{longitude},{time_utc},{time_category},{poi_pop}', file=f)


def save_graph_edgelist(G, file_name,flag):
    nodelist = G.nodes()
    node_id2idx = {k: v for v, k in enumerate(nodelist)}
    if flag ==0:
        name1 = "graph_node_id2idxi"+str(args.graphweight)+".txt"
        name2 = "graph_edgei"+str(args.graphweight)+".edgelist"
    else:
        name1 = "graph_node_id2idx2.txt"
        name2 = "graph_edge.edgelist2"
    with open(f'{file_name}/{name1}', 'w') as f:
        for i, node in enumerate(nodelist):
            print(f'{node},{i}', file=f)

    with open(f'{file_name}/{name2}', 'w') as f:
        for edge in nx.generate_edgelist(G, data=['weight']):
            src_node, dst_node, weight = edge.split(' ')
            print(f'{node_id2idx[int(src_node)]} {node_id2idx[int(dst_node)]} {weight}', file=f)


if __name__ == '__main__':
    print('Build client poi checkin graph -----------------------------------')
    data_path = f"./data/{args.dataset}"
    user_traj, itemCate, itemnum = get_data(data_path)
    pop_item_all, pop_cate_all, pop_space_all, pop_time_all = utils.load_popularity(data_path)
    tot_pop = [0.0] * itemnum
    cate_item_pop = [0.0] * itemnum
    pop_catae_item = [0.0] * itemnum
    tot_pop_num = 0
    for item, pop in enumerate(pop_item_all):
        tot_pop[item] = 0.6 * pop_item_all[item] + 0.2 * pop_cate_all[itemCate[item + 1] - 1] + 0.1 * pop_space_all[
            item] + 0.1 * pop_time_all[item]
        tot_pop_num = tot_pop_num + tot_pop[item]
        cate_item_pop[item] = pop_cate_all[itemCate[item + 1] - 1]
    G,G2 = build_client_poi_checkin_graph(user_traj,tot_pop)
    save_graph_to_pickle(G, f"./data/{args.dataset}/graphi"+str(args.graphweight)+".pkl")
    save_graph_to_csv(G, f"./data/{args.dataset}",0)
    save_graph_edgelist(G, f"./data/{args.dataset}",0)
    save_graph_to_pickle(G2, f"./data/{args.dataset}/graph2.pkl")
    save_graph_to_csv(G2, f"./data/{args.dataset}",1)
    save_graph_edgelist(G2, f"./data/{args.dataset}",1)
