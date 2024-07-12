import os
import time
import torch
import argparse

from model import SASRec
from utils import *

# 用来将字符串转换成布尔值
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# 使用 argparse 库来定义一个命令行参数解析器
parser = argparse.ArgumentParser()
# 用户必须在命令行中提供文件名
parser.add_argument('--dataset', required=True)
# 用户必须在命令行中提供训练数据存放的目录
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
# 定义 L2 正则化在嵌入层的强度
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
# 指示程序是否仅用于推理，而不进行训练
parser.add_argument('--inference_only', default=False, type=str2bool)
# 允许用户指定一个状态字典的路径，用于加载预训练的模型参数
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

# 检查目录是否存在
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)

# 自动创建args.txt文件
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    
    # 将args参数写入文件
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    # 调用 build_index 函数
    u2i_index, i2u_index = build_index(args.dataset)
    
    # 调用 data_partition 函数
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    # num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    # 向上取整的方法，确保所有用户都能被处理
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    
    # 将每个用户的交互序列长度加到 cc 上
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    
    # 打印出所有用户的平均交互序列长度
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # 创建log.txt文件
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) 
    
    
    # 遍历模型的所有参数
    # named_parameters() 不仅返回参数张量，还返回参数的名称
    for name, param in model.named_parameters():
        try:
            # 使用 Xavier 正态初始化方法初始化参数
            torch.nn.init.xavier_normal_(param.data)
        except:
            # 如果初始化某个参数失败，则忽略错误并继续
            pass 

    
    # model.pos_emb.weight.data：位置嵌入矩阵
    # 第一行即填充值0，需要清0，确保在模型中不会有任何贡献
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() 
    
    # 训练的起始周期。默认值为 1
    epoch_start_idx = 1
    
    # 检查命令行参数中是否提供了状态字典的路径
    if args.state_dict_path is not None:
        
        # 从指定文件加载模型权重
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            
            # '+6'移到数字的起始位置
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            
            # 启用 Python 的调试器 pdb
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        
        # eval() 方法用于通知模型在推理时应禁用某些特定于训练的行为
        model.eval()
        
        # 传入当前模型、数据集和参数
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 存储在验证集上观察到的最高NDCG和HR
    best_val_ndcg, best_val_hr = 0.0, 0.0
    # 存储在测试集上观察到的最高NDCG和HR
    best_test_ndcg, best_test_hr = 0.0, 0.0
    # 计时变量
    T = 0.0
    # time.time() 获取当前时间
    t0 = time.time()
    
    # args.num_epochs：训练的总周期数
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        
        # 如果程序设置为仅推理模式
        if args.inference_only: break # just to decrease identition
        
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            
            # 从数据采样器 sampler 中获取下一批数据
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            
            # 数据转换为 NumPy 数组
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            # 使用模型对正样本和负样本进行预测
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            # 创建正样本标签为1，负样本标签为0的张量
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            
            # 清除优化器的梯度，为新的梯度计算做准备
            adam_optimizer.zero_grad()
            
            # 找出正样本中非零元素的索引，因为零可能表示填充或无效数据
            indices = np.where(pos != 0)
            
            # 计算正样本的损失
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            
            # 计算负样本的损失，并加到总损失中
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            # 添加L2正则化项到损失中，以防止过拟合
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            
            adam_optimizer.step()
           
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        # 每20个训练周期进行一次评估
        if epoch % 20 == 0:
            model.eval()
            
            # 计算自上次记录以来经过的时间
            t1 = time.time() - t0
            
            # 将本次计时累加到总时间 T 上
            T += t1
            
            print('Evaluating', end='')
            
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            
            # 打印当前周期数、总用时、验证集和测试集的NDCG与HR指标
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            # 如果当前周期的验证集或测试集性能超过之前记录的最佳性能，则更新最佳性能记录，并保存当前模型。
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr or t_test[0] > best_test_ndcg or t_test[1] > best_test_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                best_test_ndcg = max(t_test[0], best_test_ndcg)
                best_test_hr = max(t_test[1], best_test_hr)
                
                # 将模型的状态字典保存到指定文件名和目录中
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            # 将当前周期的评估结果写入日志文件
            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            # 确保所有写入的内容都被立即写入磁盘
            f.flush()
            
            # 准备下一周期训练
            t0 = time.time()
            model.train()
    
        # 如果达到了设定的训练总周期数，保存模型
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
