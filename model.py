import numpy as np
import torch

# 点状前馈网络

# hidden_units：每一层的神经元数量
# dropout_rate：Dropout概率


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        # 带有核大小为1的卷积层在本质上与应用于序列中每个位置的全连接层等效
        # 卷积层每个位置权重相同，全连接层每个位置权重不同

        # 尝试Sigmoid？
        
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    # 前向传播

    def forward(self, inputs):

        # transpose(-1, -2)：转置输出
        # inputs:(N, L, C)  //N批量大小，C通道数，L序列长度
        # conv1d需要格式(N, C, L)

        outputs = self.dropout2(self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # 根据作者提示，没有relu效果会更好？
        # https://github.com/pmixer/SASRec.pytorch/issues/7
        
        outputs = outputs.transpose(-1, -2)

        # 残差连接
        outputs += inputs

        return outputs


# 请使用以下自制的多头注意力层
# 如果你的 PyTorch 版本低于 1.16 或出于其他原因
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):

    # user_num：用户数量，item_num：物品数量，args：模型超参数
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # item_emb：物品嵌入层，用于将物品ID映射到嵌入向量
        # pos_emb：位置嵌入层，用于将位置ID映射到嵌入向量

        # args.hidden_units:嵌入向量的维度

        '''
        构建一个物品嵌入层 用来将输入序列转换成项目嵌入矩阵
        物品ID是从1开始的 补齐序列长度需要0 所以item_num+1
        正常ID通过嵌入层转换成嵌入向量 ID=0通过padding_idx=0转换成全零向量
        '''
        # 待办事项：在训练期间通过 loss += args.l2_emb 来正则化嵌入向量
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        # 初始化操作
        self.item_emb = torch.nn.Embedding(
            self.item_num+1, args.hidden_units, padding_idx=0)

        # 初始化操作
        self.pos_emb = torch.nn.Embedding(
            args.maxlen+1, args.hidden_units, padding_idx=0)
        
        self.emb_dropout= torch.nn.Dropout(p=args.dropout_rate)

        # torch.nn.ModuleList 是 PyTorch 提供的一种容器，用于存储多个子模块
        # 多层堆叠子模块    
        # 初始化操作

        # 每一层自注意力层前的层归一化模块
        self.attention_layernorms = torch.nn.ModuleList()  

        # 多层的自注意力模块
        self.attention_layers = torch.nn.ModuleList()

        # 每一层前馈神经网络前的层归一化模块
        self.forward_layernorms = torch.nn.ModuleList()

        # 多层的前馈神经网络模块
        self.forward_layers = torch.nn.ModuleList()

        # 对最后的输出进行归一化
        # args.hidden_units：嵌入向量维度/样本的特征数量
        # eps=1e-8：防止除以零的错误/归一化公式
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 根据指定的层数（args.num_blocks），动态地创建多个注意力层和前馈层
        # 并将它们添加到相应的模块列表中
        for _ in range(args.num_blocks):
            
            # 创建对多头自注意力层的层归一化
            new_attn_layernorm = torch.nn.LayerNorm(
                args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # 创建一个多头自注意力层
            # args.num_heads：注意力头的数量
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                         args.num_heads,
                                                         args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            # 创建对前馈网络层输入的层归一化
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 创建前馈网络层
            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    # 将输入的日志序列转换为特征表示
    # 日志：log
    def log2feats(self, log_seqs):  # TODO: 将默认数据类型设为 fp64 和 int64，是否需要修剪？

        # torch.LongTensor：将 log_seqs 转换为长整型张量
        # 长整型张量在 PyTorch 中的数据类型是 torch.int64
        # log_seqs：一个包含物品ID的序列
        # to(self.dev)：将张量移动到指定的设备（CPU 或 GPU）
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))

        # 对嵌入向量进行缩放，嵌入向量中的每一个元素都乘以嵌入维度的平方根
        # dim = args.hidden_units
        # 位置嵌入的范围较大，物品嵌入的范围较小，位置嵌入可能会压制物品嵌入的信号
        # 通过对物品嵌入向量进行缩放，使得物品嵌入和位置嵌入在数值范围上变得更为一致
        # 理由：https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/88159#88159
        seqs *= self.item_emb.embedding_dim ** 0.5

        # log_seqs.shape[1]：返回序列的长度，举例为5
        # np.arange(1, log_seqs.shape[1] + 1)：生成从1-5的数组
        # log_seqs.shape[0]：返回批次大小，举例为3
        # np.tile(poss, [3, 1]) 将位置索引数组复制3次，以匹配输入序列的批次大小
        # 输入序列批次为3即有三个序列，要为每个序列提供一个位置索引数组【0-4】
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        
        # log_seqs != 0：将输入序列中的填充值0变为布尔值False
        # 位置嵌入中对应的值*False就变为0
        poss *= (log_seqs != 0)

        # 物品嵌入向量和位置嵌入向量相加
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # 为一个序列生成注意力掩码
        # 确保模型在预测时只能看到当前时间步及之前的时间步，而不能看到未来的时间步
        
        # 获取序列长度
        tl = seqs.shape[1]
        
        # torch.ones((tl, tl)创建一个形状为 (tl, tl) 的张量，所有元素都为 1，数据类型为布尔型
        # torch.tril就是将矩阵中上三角部分置为 0，下三角部分保持不变
        # ~是按位取反操作符，会将布尔值取反
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # 多层注意力机制的前向传播过程
        
        # 循环的次数等于注意力层的数量 
        for i in range(len(self.attention_layers)):
            
            # (N,L,C)-->(L,N,C)
            seqs = torch.transpose(seqs, 0, 1)
            
            # 通过第i个注意力层归一化模块
            # 得到查询Q
            Q = self.attention_layernorms[i](seqs)
            
            # 计算多头注意力
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # need_weights=False) 这个参数不起作用？
            
            # 残差连接
            # mha_outputs形状和seqs相同
            seqs = Q + mha_outputs
            
            # (L,N,C)-->(N,L,C)
            seqs = torch.transpose(seqs, 0, 1)

            # 前向层归一化模块
            seqs = self.forward_layernorms[i](seqs)
            
            # 前向层
            seqs = self.forward_layers[i](seqs)

        # 最终归一化
        log_feats = self.last_layernorm(seqs) 

        return log_feats

    # 计算给定用户交互日志的正负样本的预测得分
    
    # log_seqs：用户历史交互的物品ID序列
    # pos_seqs：正样本序列
    # neg_seqs：负样本序列
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # pos_embs：正样本的嵌入向量
        # neg_embs：负样本的嵌入向量
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # 得到每一个物品和正负样本的点积
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    # item_indices：需要进行预测的物品ID列表
    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        # 只要seqs最后一个物品的特征
        # 这个特征与待选项目的嵌入向量进行内积得到每个待选项目的评分
        final_feat = log_feats[:, -1, :] 
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  
