import os
import time
import argparse
import pickle
from utils import *
from collections import Counter
from model import *
from semantic import *
from graph import *
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='NYCGlo')
parser.add_argument('--datadir',  default='./data/')
parser.add_argument('--train_dir', default='default')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=120, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=301, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)

parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda:3', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--tau', default=0.01, type=float)
parser.add_argument('--lmd', default=0.8, type=float)
parser.add_argument('--lmd2', default=0.2, type=float)
parser.add_argument('--similarity_method', default='Item2Vec', type=str)
parser.add_argument('--popularity_exp', default=0.001, type=float)
parser.add_argument('--counter_pro', default=0.7, type=float)
parser.add_argument('--addCL', default=0, type=int)
parser.add_argument('--totCL', default=1, type=int)
parser.add_argument('--lmdcate', default=1, type=float)
parser.add_argument('--lmdpop', default=1, type=float)
parser.add_argument('--methodcate', default=0, type=int)
parser.add_argument('--methodpop', default=0, type=int)
parser.add_argument('--item_judge_hot', default=0.01, type=float)
parser.add_argument('--cate_judge_hot', default=0.01, type=float)
parser.add_argument('--space_judge_hot', default=0.8, type=float)
parser.add_argument('--tot_judge_hot', default=0.01, type=float)
parser.add_argument('--lmdattention1', default=0.5, type=float)
parser.add_argument('--lmdattention2', default=0.5, type=float)
parser.add_argument('--augmethod', default=0, type=int)#0:数据增强方法 所有 1：只替换 2：只删除 3：只增加
parser.add_argument('--alluremethod', default=0, type=int)

parser.add_argument('--withsem', default=1, type=int)
parser.add_argument('--withgraph1', default=1, type=int)
parser.add_argument('--withgraph2', default=1, type=int)
parser.add_argument('--weightsem', default=0.2, type=float)
parser.add_argument('--weightgraph1', default=0.2, type=float)
parser.add_argument('--weightgraph2', default=0.2, type=float)

parser.add_argument('--semlength', default=10, type=int)
parser.add_argument('--buildgraphweight', default=1, type=int)#与构图相同
args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    #load data
    dataset = data_partition(args.datadir+args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, catenum, user_train_cate, user_train_time,itemCate,itemPos,user_traj,loc_sem] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    averagelen = cc / len(user_train)
    averagelen = int(averagelen)
    print('average sequence length: %.2f' % (cc / len(user_train)))
    now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    groupCountFile = open(os.path.join(args.dataset + '_' + args.train_dir, now +"popularity_exp"+str(args.popularity_exp)+ 'groupCount.txt'), 'w')
    f = open(os.path.join(args.dataset + '_' + args.train_dir, now+'withsem'+str(args.withsem)+'withgraph1'+str(args.withgraph1)+'withgraph2'+str(args.withgraph2)+'weightsem'+str(args.weightsem)+'weightgraph1'+str(args.weightgraph1)+'weigraph2'+str(args.weightgraph2)+'semlength'+str(args.semlength)+'buildgraphweight'+str(args.buildgraphweight)+'log.txt'), 'w')

    # load popularity
    pop_item_all, pop_cate_all, pop_space_all,pop_time_all = load_popularity(args.datadir + args.dataset)
    popularity_exp = args.popularity_exp
    tot_pop = [0.0] * itemnum
    cate_item_pop = [0.0] * itemnum
    pop_catae_item = [0.0] * itemnum
    tot_pop_num = 0
    for item, pop in enumerate(pop_item_all):
        tot_pop[item] = 0.6 * pop_item_all[item] + 0.2 * pop_cate_all[itemCate[item + 1] - 1] + 0.1 * pop_space_all[item] + 0.1 * pop_time_all[item]
        tot_pop_num = tot_pop_num + tot_pop[item]
        cate_item_pop[item] = pop_cate_all[itemCate[item + 1] - 1]
    tot_pop = np.array(tot_pop)
    item_pop = np.array(pop_item_all)
    cate_pop = np.array(pop_cate_all)
    space_pop = np.array(pop_space_all)
    time_pop = np.array(pop_time_all)
    # 计算偏差
    if args.alluremethod == 0:
        check_matrix = np.power(tot_pop, popularity_exp)  # 数据从1开始，这里从0开始
    if args.alluremethod == 1:
        check_matrix = np.power(pop_item_all, popularity_exp)  # 数据从1开始，这里从0开始
    if args.alluremethod == 2:
        check_matrix = np.power(cate_item_pop, popularity_exp)  # 数据从1开始，这里从0开始
    if args.alluremethod == 3:
        check_matrix = np.power(pop_space_all, popularity_exp)  # 数据从1开始，这里从0开始
    if args.alluremethod == 4:
        check_matrix = np.power(pop_time_all, popularity_exp)  # 数据从1开始，这里从0开始
    # 判断热度
    tot_pop_judge = tot_pop
    item_pop_judge = item_pop
    cate_pop_judge = cate_pop
    space_pop_judge = space_pop
    time_pop_judge = time_pop
    # intervals = [(0, 0.01), (0.01, 0.015), (0.015, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.1),
    #              (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 1)]
    intervals = [(0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4),
                 (0.4, 0.45), (0.45, 5), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1)]

    #####################根据吸引力分组###############

    file_name = args.dataset + 'Group.pkl'
    if not os.path.isfile(args.dataset + '_' + args.train_dir + '/' + args.dataset + 'Group.pkl'):
        num_groups = 10
        sorted_indices = np.argsort(tot_pop)
        sorted_tot_pop = np.array(tot_pop)[sorted_indices]

        # 计算每组的目标流行度之和
        target_group_pop = tot_pop_num / num_groups

        # 初始化变量
        item_groups = [0] * itemnum
        group_pop = 0.0
        group_num = [0]*100
        gid = 1
        # 遍历排序后的物品
        for item, pop in enumerate(sorted_tot_pop):
            if group_pop + pop <= target_group_pop:
                group_pop += pop
                item_groups[sorted_indices[item]] = gid
                group_num[gid] += 1
            else:

                group_pop = pop
                gid = gid + 1
                item_groups[sorted_indices[item]] = gid
                group_num[gid] += 1


        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'wb') as file:
            pickle.dump(item_groups, file)
        file.close()
    else:
        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'rb') as file:
            item_groups = pickle.load(file)
        file.close()




    #####tot#####
    interval_countstot = {interval: 0 for interval in intervals}
    for pop in tot_pop:
        for interval in intervals:
            if interval[0] <= pop < interval[1]:
                interval_countstot[interval] += 1

    #####item#####
    interval_countsitem = {interval: 0 for interval in intervals}
    for pop in item_pop:
        for interval in intervals:
            if interval[0] <= pop < interval[1]:
                interval_countsitem[interval] += 1

    #####cate#####
    interval_countscate = {interval: 0 for interval in intervals}
    for pop in cate_pop:
        for interval in intervals:
            if interval[0] <= pop < interval[1]:
                interval_countscate[interval] += 1

    #####space#####
    interval_countsspace = {interval: 0 for interval in intervals}
    for pop in space_pop:
        for interval in intervals:
            if interval[0] <= pop < interval[1]:
                interval_countsspace[interval] += 1

    #####

    #calculate similarity
    file_name = args.dataset + 'Sim.pkl'
    if not os.path.isfile(args.dataset + '_' + args.train_dir + '/' +args.dataset + 'Sim.pkl'):
        itemSim = generate_item_similarity(user_train, args.similarity_method, usernum, itemnum)
        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'wb') as file:
            pickle.dump(itemSim, file)
        file.close()
    else:
        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'rb') as file:
            itemSim = pickle.load(file)
        file.close()

    #caculate user intent
    UserIntent = {}
    for key, value_list in user_train.items():
        # 统计该列表中的数字出现的频率
        counts = Counter(value_list)

        # 找到出现频率最高的数字和对应的频率
        most_common = counts.most_common(1)

        # 存储结果
        UserIntent[key] = most_common[0][0]

    ##user senmatic
    # 客户端id列表
    user_list = []
    user_list.extend(user_train.keys())
    # 用户端轨迹语义学�?
    file_name = args.dataset + 'Sem.pkl'
    if not os.path.isfile(args.dataset + '_' + args.train_dir + '/' +args.dataset + 'Sem.pkl'):
        print('sem learning')
        traj_vector = []
        traj_vector_dict = {}
        traj_sem = {}
        semnum = 0
        for u in user_list:
            user_traj_data = user_traj[u]
            user_traj_prompt = get_traj_prompt(user_traj_data, loc_sem,args.semlength)
            user_traj_vector, user_traj_sem = generate_traj_vector(user_traj_prompt, args.device)  # 1, 768
            user_traj_vector = user_traj_vector.detach().numpy()
            semnum = max(semnum, max(user_traj_sem))
            traj_sem[u] = user_traj_sem
            traj_vector.append(list(user_traj_vector))
            traj_vector_dict[u] = list(user_traj_vector)
        # client_traj_vector = np.array(client_traj_vector)
        # print("client_traj_vector：{}".format(client_traj_vector))
        vector_list = []  # 20,768
        for i in range(len(traj_vector)):
            for j in range(len(traj_vector[i])):
                vector_list.append(np.float64(traj_vector[i][j]))
        sem_vector = vector_list
        print('sem learned')
        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'wb') as file:
            pickle.dump(traj_vector_dict, file)
        file.close()
    else:
        with open(os.path.join(args.dataset + '_' + args.train_dir, file_name), 'rb') as file:
            traj_vector_dict = pickle.load(file)
        file.close()

    #########图结构
    ###1
    graph_matrix = {}
    node_features = {}
    graph_node_id2idx = {}
    path = f'./data/{args.dataset}/'
    matrix, node_id2idx = get_graph_matrix(path,1,args.buildgraphweight), get_graph_nodes_id2idx(path,1,args.buildgraphweight)
    node_features = get_graph_node_features(path,1,args.buildgraphweight)
    global_graph_matrix = calculate_laplacian_matrix(matrix, mat_type='hat_rw_normd_lap_mat')
    print(f"global graph matrix is successfully initialized...")
    global_graph_node_features = torch.tensor(node_features)
    global_graph_matrix = torch.tensor(global_graph_matrix)
    global_graph_node_features = global_graph_node_features.to(dtype=torch.float, device=args.device)
    global_graph_matrix = global_graph_matrix.to(dtype=torch.float, device=args.device)
    graph_node_features, graph_adjacent_matrix = global_graph_node_features, global_graph_matrix
    ###2
    graph_matrix2 = {}
    node_features2 = {}
    graph_node_id2idx2 = {}
    path = f'./data/{args.dataset}/'
    matrix2, node_id2idx2 = get_graph_matrix(path,2,args.buildgraphweight), get_graph_nodes_id2idx(path,2,args.buildgraphweight)
    node_features2 = get_graph_node_features(path,2,args.buildgraphweight)
    global_graph_matrix2 = calculate_laplacian_matrix(matrix2, mat_type='hat_rw_normd_lap_mat')
    print(f"global graph matrix is successfully initialized...")
    global_graph_node_features2 = torch.tensor(node_features2)
    global_graph_matrix2 = torch.tensor(global_graph_matrix2)
    global_graph_node_features2 = global_graph_node_features2.to(dtype=torch.float, device=args.device)
    global_graph_matrix2 = global_graph_matrix2.to(dtype=torch.float, device=args.device)
    graph_node_features2, graph_adjacent_matrix2 = global_graph_node_features2, global_graph_matrix2
    #dataLoader
    sampler = WarpSampler(user_train, usernum, itemnum,user_train_cate, user_train_time, check_matrix,traj_vector_dict,
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1 )
    # samplercl = WarpSampler(user_train, usernum, itemnum, user_train_cate, user_train_time, batch_size=args.batch_size,
    #                       maxlen=averagelen, n_workers=1)
    epoch_start_idx = 1
    model = CounterfactualCL(usernum, itemnum,catenum, args).to(args.device)
    model.train()

    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    if args.addCL == 1 :
        weight_decay = 1e-05
        for epoch in range(epoch_start_idx, args.num_epochs + 1):
            if args.inference_only: break  # just to decrease identition
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg, seq_cate, seq_time, pos_pop, neg_pop,sem = sampler.next_batch()  # tuples to ndarray

                seq_pop = seq2pop(seq,tot_pop)
                u, seq, pos, neg, seq_cate, seq_time, pos_pop, neg_pop, seq_pop,sem = np.array(u), np.array(seq), np.array(
                    pos), np.array(neg), np.array(seq_cate), np.array(seq_time), np.array(pos_pop), np.array(
                    neg_pop), np.array(seq_pop),np.array(sem)

                # if args.totCL == 1:
                    # calculate total pop
                aug_item_seq1, aug_item_seq2 = model.augment(seq, itemSim, UserIntent, u, seq_cate, seq_time,
                                                             tot_pop_judge, item_pop_judge, cate_pop_judge,
                                                             space_pop_judge,check_matrix)
                seq_output1 = model.forward(aug_item_seq1)
                seq_output2 = model.forward(aug_item_seq2)
                # seq_output1_change = seq_output1.reshape(seq_output1.size(0), -1)
                # seq_output2_change = seq_output2.reshape(seq_output2.size(0), -1)
                seq_output1_change = seq_output1.sum(-1)
                seq_output2_change = seq_output2.sum(-1)
                nce_logits, nce_labels = model.info_nce(seq_output1_change, seq_output2_change, temp=args.tau,
                                                        batch_size=len(seq_output1))
                nce_loss = model.nce_fct(nce_logits, nce_labels)
                # else:
                #     # calculate item pop
                #     aug_item_seq1, aug_item_seq2 = model.augment(seq, itemSim, UserIntent, u, seq_cate, seq_time,
                #                                                  item_pop_judge, 0)
                #     seq_output1 = model.forward(aug_item_seq1)
                #     seq_output2 = model.forward(aug_item_seq2)
                #     # seq_output1_change = seq_output1.reshape(seq_output1.size(0), -1)
                #     # seq_output2_change = seq_output2.reshape(seq_output2.size(0), -1)
                #     seq_output1_change = seq_output1.sum(-1)
                #     seq_output2_change = seq_output2.sum(-1)
                #     nce_logits, nce_labels = model.info_nce(seq_output1_change, seq_output2_change, temp=args.tau,
                #                                             batch_size=len(seq_output1))
                #     nce_loss_item = model.nce_fct(nce_logits, nce_labels)
                #     # calculate cate pop
                #     aug_item_seq1, aug_item_seq2 = model.augment(seq, itemSim, UserIntent, u, seq_cate, seq_time,
                #                                                  cate_pop_judge, 1)
                #     seq_output1 = model.forward(aug_item_seq1)
                #     seq_output2 = model.forward(aug_item_seq2)
                #     # seq_output1_change = seq_output1.reshape(seq_output1.size(0), -1)
                #     # seq_output2_change = seq_output2.reshape(seq_output2.size(0), -1)
                #     seq_output1_change = seq_output1.sum(-1)
                #     seq_output2_change = seq_output2.sum(-1)
                #     nce_logits, nce_labels = model.info_nce(seq_output1_change, seq_output2_change, temp=args.tau,
                #                                             batch_size=len(seq_output1))
                #     nce_loss_cate = model.nce_fct(nce_logits, nce_labels)
                #     # calculate space pop
                #     aug_item_seq1, aug_item_seq2 = model.augment(seq, itemSim, UserIntent, u, seq_cate, seq_time,
                #                                                  space_pop_judge, 2)
                #     seq_output1 = model.forward(aug_item_seq1)
                #     seq_output2 = model.forward(aug_item_seq2)
                #     # seq_output1_change = seq_output1.reshape(seq_output1.size(0), -1)
                #     # seq_output2_change = seq_output2.reshape(seq_output2.size(0), -1)
                #     seq_output1_change = seq_output1.sum(-1)
                #     seq_output2_change = seq_output2.sum(-1)
                #     nce_logits, nce_labels = model.info_nce(seq_output1_change, seq_output2_change, temp=args.tau,
                #                                             batch_size=len(seq_output1))
                #     nce_loss_space = model.nce_fct(nce_logits, nce_labels)
                #     nce_loss = nce_loss_item + nce_loss_cate+ nce_loss_space


                pos_logits, neg_logits, log_feats, pos_embs, neg_embs = model.trm_encoder(u, seq, pos, neg,seq_cate,seq_pop,sem)
                #CELoss
                pos_pop = torch.tensor(pos_pop, dtype=torch.float, device=args.device)
                neg_pop = torch.tensor(neg_pop, dtype=torch.float, device=args.device)
                pos_logits = pos_logits * pos_pop
                neg_logits = neg_logits * neg_pop
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                       device=args.device)
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in model.trm_encoder.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                #BPRLoss
                # pos_pop = torch.tensor(pos_pop, dtype=torch.float, device=args.device)
                # neg_pop = torch.tensor(neg_pop, dtype=torch.float, device=args.device)
                # pos_scores = torch.nn.functional.elu(pos_logits) + 1
                # neg_scores = torch.nn.functional.elu(neg_logits) + 1
                # pos_scores_with_pop = pos_scores * pos_pop
                # neg_scores_with_pop = neg_scores * neg_pop
                # bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores_with_pop - neg_scores_with_pop)))
                #
                # regularizer = torch.norm(log_feats, p=2) + torch.norm(pos_embs, p=2) + torch.norm(
                #     neg_embs, p=2)
                # reg_loss = weight_decay * regularizer
                # loss = bpr_loss + reg_loss
                # loss = bpr_loss

                #toloss
                totLoss = args.lmd * loss + args.lmd2 * nce_loss

                adam_optimizer.zero_grad()
                totLoss.backward()
                adam_optimizer.step()


            if epoch % 10 == 0 :
                model.eval()
                t1 = time.time() - t0
                T += t1
                print('CL', end='')
                NDCG, HR, group_count, coverage = evaluatefin(model.trm_encoder, dataset, args.maxlen, check_matrix, tot_pop, item_groups, epoch, traj_vector_dict)
                # t_valid = evaluate_valid(model, dataset, args)
                print(
                    '%d,l:%.4f N: %.4f %.4f %.4f %.4f, H: %.4f %.4f %.4f %.4fc, cover %.4f '
                    % (
                        epoch, totLoss.item(), NDCG[0], NDCG[1], NDCG[2], NDCG[3], HR[0], HR[1], HR[2], HR[3], coverage))
                f.write(
                    str(epoch) + ' '+str(T) + ' '+ str(totLoss.item()) + ' ' + str(NDCG[0]) + ' ' + str(NDCG[1]) + ' ' + str(
                        NDCG[2]) + ' ' + str(NDCG[3])
                    + ' ' + str(HR[0]) + ' ' + str(HR[1]) + ' ' + str(HR[2]) + ' ' + str(HR[3])  + ' ' +  str(coverage) + '\n')
                f.flush()
                t0 = time.time()
                model.train()
        for i in range(10):
            groupCountFile.write(str(group_count[i+1]['top1'])+' '+str(group_count[i+1]['top5'])+' '+str(group_count[i+1]['top10'])+' '+str(group_count[i+1]['top20'])+ '\n')

    weight_decay = 1e-05
    rec_model = model.trm_encoder
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    for epoch in range(epoch_start_idx, args.num_epochs ):
        # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg, seq_cate, seq_time, pos_pop, neg_pop,sem = sampler.next_batch()  # tuples to ndarray
            seq_pop = seq2pop(seq, tot_pop)
            u, seq, pos, neg, seq_cate, seq_time, pos_pop, neg_pop, seq_pop, sem = np.array(u), np.array(seq), np.array(
                pos), np.array(neg), np.array(seq_cate), np.array(seq_time), np.array(pos_pop), np.array(
                neg_pop), np.array(seq_pop), np.array(sem)

            pos_logits, neg_logits, log_feats, pos_embs, neg_embs = rec_model(u, seq, pos, neg,seq_cate,seq_pop,sem, graph_node_features, graph_adjacent_matrix,graph_node_features2, graph_adjacent_matrix2)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                         device=args.device)
            pos_pop = torch.tensor(pos_pop, dtype=torch.float, device=args.device)
            neg_pop = torch.tensor(neg_pop, dtype=torch.float, device=args.device)
            #CELoss
            pos_logits = pos_logits * pos_pop
            neg_logits = neg_logits * neg_pop
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0

            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in rec_model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            #BPRloss
            # pos_scores = torch.nn.functional.elu(pos_logits) + 1
            # neg_scores = torch.nn.functional.elu(neg_logits) + 1
            # pos_scores_with_pop = pos_scores * pos_pop
            # neg_scores_with_pop = neg_scores * neg_pop
            #
            # bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores_with_pop - neg_scores_with_pop)))
            #
            # regularizer = torch.norm(log_feats, p=2) + torch.norm(pos_embs, p=2) + torch.norm(
            #     neg_embs, p=2)
            # reg_loss = weight_decay * regularizer
            # loss = bpr_loss + reg_loss
            # loss = bpr_loss
            adam_optimizer.zero_grad()
            loss.backward()
            adam_optimizer.step()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step,
            #                                                  loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 10 == 0 :
            rec_model.eval()
            t1 = time.time() - t0
            T += t1

            print('Rec', end='')
            NDCG, HR, group_count, coverage = evaluatefin(model.trm_encoder, dataset, args.maxlen, check_matrix,
                                                          tot_pop, item_groups, epoch,traj_vector_dict, graph_adjacent_matrix, graph_node_features2, graph_adjacent_matrix2)
            # t_valid = evaluate_valid(model, dataset, args)
            print(
                '%d,l:%.4f N: %.4f %.4f %.4f %.4f, H: %.4f %.4f %.4f %.4f, cover %.4f '
                % (
                    epoch, loss.item(), NDCG[0], NDCG[1], NDCG[2], NDCG[3], HR[0], HR[1], HR[2], HR[3], coverage))
            f.write(
                str(epoch) + ' ' + str(T) + ' ' + str(loss.item()) + ' ' + str(NDCG[0]) + ' ' + str(
                    NDCG[1]) + ' ' + str(
                    NDCG[2]) + ' ' + str(NDCG[3])
                + ' ' + str(HR[0]) + ' ' + str(HR[1]) + ' ' + str(HR[2]) + ' ' + str(HR[3]) + ' ' + str(
                    coverage) + '\n')
            f.flush()
            t0 = time.time()
            rec_model.train()
    for i in range(10):
        groupCountFile.write(str(group_count[i+1]['top1'])+' '+str(group_count[i+1]['top5'])+' '+str(group_count[i+1]['top10'])+' '+str(group_count[i+1]['top20'])+ '\n')

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(rec_model.state_dict(), os.path.join(folder, fname))



    f.close()
    groupCountFile.close()
    sampler.close()
    print("Done")


