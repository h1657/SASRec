import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# 构建用户到物品和物品到用户的索引
def build_index(dataset_name):

    # 从data目录中加载一个名为 dataset_name.txt 的文本文件
    # dtype=np.int32 指定了数据的类型为32位整数
    # ui_mat 是一个二维 NumPy 数组，包含了用户-物品交互数据

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    # 计算用户ID列的最大值
    # 计算物品ID列的最大值
    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()
    
    # 初始化两个列表
    # user to item：每个用户交互过的物品ID
    # item to user：每个物品交互过的用户ID
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# 生成一个不在s的随机数t
# 用于生成不重复的样本，避免在小批次数据中出现重复样本
def random_neq(l, r, s):
    
    # 生成一个在 [l, r) 之间的随机整数
    t = np.random.randint(l, r)
    
    while t in s:
        t = np.random.randint(l, r)
    return t


# user_train：一个字典，键是用户ID，值是该用户的交互序列
# usernum：用户总数
# itemnum：物品总数
# batch_size：每个批次的大小
# maxlen：每个用户交互序列的最大长度
# result_queue：一个队列，用于存储生成的批次数据
# SEED：随机数种子，用于确保随机性的一致性
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        # 这行代码确保所选用户的交互序列长度大于1，如果不满足条件，重新随机选择一个用户ID
        while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

        # 初始化3个数组，初始值为0
        # seq：用户的交互序列
        # pos：正样本
        # neg：负样本
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        # nxt初始化：用户交互序列中的最后一个物品
        nxt = user_train[uid][-1]
        # idx: 序列索引，从 maxlen - 1 开始
        idx = maxlen - 1

        # ts: 用户交互记录的集合
        ts = set(user_train[uid])
        
        # 遍历用户交互的物品ID序列（除了最后一个）[:-1]排除了最后一个
        # reversed翻转物品ID序列，i从倒数第2个开始向前遍历
        # idx从maxlen-1开始
        # 结果为seq中没有最后一个物品；pos中和交互的物品ID序列相同
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    
    # 使用固定的种子（SEED）可以确保每次运行代码时，生成的随机数序列是一致的，这有助于实验的可复现性
    np.random.seed(SEED)
    
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            
            # 对 uids 数组进行原地随机打乱
            # 保证了用户数据的采样是随机的，避免了因固定顺序带来的偏差
            np.random.shuffle(uids)
        
        one_batch = []
        
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        
        # zip(*one_batch) 将批次列表中的数据解包并按数据类型组合
        # 例如，所有用户的 ID 组成一个元组，所有序列组成一个元组
        result_queue.put(zip(*one_batch))


# User：用户数据的字典或列表，每个用户对应一组与之交互的物品
# n_workers：并行处理数据的工作进程数，默认为1
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        
        # 创建一个队列，用于存储生成的数据批次
        # 队列的最大长度是工作进程数的10倍，这是为了避免产生的数据过多时阻塞进程
        self.result_queue = Queue(maxsize=n_workers * 10)
        
        # 初始化一个空列表，用于存储数据生成的进程
        self.processors = []
        
        # 创建指定数量的工作进程
        for i in range(n_workers):
            
            # 创建一个进程，目标函数是 sample_function，这个函数负责生成数据样本
            # args参数对应sample_function函数
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            # 每添加一个进程，[-1]最后一个就是最新添加的进程
            # 后一个进程都是前一个进程的守护子进程
            # 如果主程序执行完毕退出了，它启动的守护子进程也会被自动终止，而不会继续留在后台运行
            self.processors[-1].daemon = True
            
            # self.processors[-1].start()：启动进程
            self.processors[-1].start()

    # 队列中获取下一个数据批次
    def next_batch(self):
        return self.result_queue.get()

    # 关闭所有工作进程
    def close(self):
        for p in self.processors:
            # 终止进程
            p.terminate()
            # 等待进程结束
            p.join()


# 划分训练集、验证集和测试集

# fname：这是要读取的数据文件的名称
def data_partition(fname):
    usernum = 0
    itemnum = 0
    
    # 使用 defaultdict 创建一个默认值为列表的字典
    # 这使得当你向字典中添加一个不存在的键时，它会自动初始化为一个空列表
    User = defaultdict(list)
    
    # 训练数据
    user_train = {}
    # 验证数据
    user_valid = {}
    # 测试数据
    user_test = {}
    
    # 假设用户/物品索引从1开始
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        
        # 去除每行的尾部空白字符，然后按空格分割，得到用户ID (u) 和物品ID (i)
        u, i = line.rstrip().split(' ')
        
        # 将字符串转换成整数
        u = int(u)
        i = int(i)
        
        # 更新数据集中的最大用户ID和物品ID
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        
        # 将物品ID添加到对应用户ID的列表中
        User[u].append(i)

    # 遍历每个用户
    for user in User:
        
        # 获取当前用户的交互物品数
        nfeedback = len(User[user])
        
        # 全部数据用于训练集，而验证集和测试集为空
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        
        # 除了最后两个物品，其余用于训练集
        # 倒数第二个物品用于验证集；最后一个物品用于测试集
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: 合并测试集和验证集的评估函数

# 评估模型在测试集上的表现
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    
    # 有效用户数计数器
    valid_user = 0.0

    # 从所有用户中随机抽取10000个用户
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    # 使用所有用户
    else:
        users = range(1, usernum + 1)
    
    for u in users:

        # 如果用户在训练集或测试集中的交互少于1，跳过此用户
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        
        # idx初始设为最后一个索引
        idx = args.maxlen - 1
        
        # 将验证集中该用户u的第一个物品ID放到seq数组的最后一个位置
        seq[idx] = valid[u][0]
        
        idx -= 1
        
        # 从用户的最近一个交互物品开始向前回溯
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        # 创建一个集合 rated，包含用户 u 在训练集中互动过的所有物品
        rated = set(train[u])
        
        # 添加一个虚拟物品 0
        # 确保在生成负样本时不会选择到这个虚拟编号
        rated.add(0)
        
        # 初始化列表 item_idx ，并将测试集中该用户 u 的第一个物品（正样本）加入列表
        item_idx = [test[u][0]]
        
        # 循环生成100个负样本
        for _ in range(100):
            
            # 随机生成一个介于1和itemnum之间的整数作为候选负样本编号
            t = np.random.randint(1, itemnum + 1)
            
            # 如果生成的物品编号已存在于集合 rated 中（即用户已经评过或是虚拟物品），则重新生成
            while t in rated: t = np.random.randint(1, itemnum + 1)
            
            item_idx.append(t)

        # np.array(l) 将列表转换成NumPy数组，适配模型输入格式
        # def predict(self, user_ids, log_seqs, item_indices)
        
        # '*' 运算符：它用于解包列表，这意味着列表中的每个元素都将作为独立的参数传递给 predict 方法
        # 因此，predict 方法接收的是三个独立的参数：用户ID数组、用户序列数组和物品ID数组
        
        # argsort 方法默认按升序排序，'-'实现降序效果
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
        # predict函数返回的是logits
        # logits的每行代表一个用户的所有物品的评分
        # 选取第一个用户的评分
        predictions = predictions[0] 

        # 两次调用argsort获得每个元素的排名
        # [0]获取第一个元素（正样本）的排名
        # item()：将 NumPy 数值转换成 Python 的标准整数类型
        rank = predictions.argsort().argsort()[0].item()

        # 记录已评估的有效用户数
        valid_user += 1

        # 检查正样本是否排在前10位
        if rank < 10:
            
            # NDCG是一种衡量排名质量的指标，考虑了排名的位置对用户满意度的影响
            # 使用对数函数是为了减少高排名位置的影响，提升低排名位置的权重
            NDCG += 1 / np.log2(rank + 2)
            
            # 中率统计模型能否将正样本推荐到前10的次数
            HT += 1
        
        # 每评估100个用户，打印一个点.作为进度提示
        if valid_user % 100 == 0:
            print('.', end="")
            
            # 确保即时显示，有时在某些环境下输出可能会被缓存
            sys.stdout.flush()

    # 返回平均NDCG和平均HT
    return NDCG / valid_user, HT / valid_user


# 评估模型在验证集上的表现
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        
        # 不同
        item_idx = [valid[u][0]]
        
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
