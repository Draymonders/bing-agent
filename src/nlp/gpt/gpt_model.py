import numpy as np
import torch
import torch.nn as nn

d_k = 64 # Q的维度
d_v = 64 # V的维度
d_embedding = 512 # embedding的维度
n_heads = 8 # 多头注意力的个数
batch_size = 10
n_layers = 6 # 解码器的层数
device = "cuda" if torch.cuda.is_available() else "cpu"

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    简单理解 ScaledDotProductAttention，目的是计算Query和Key的相似权重，作用于Value
    结果是
    Query1: {Value1: w11, Value2: w12, Value3: w13}
    Query2: {Value1: w21, Value2: w22, Value3: w23}
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 维度信息
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, len_q, len_k]
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        # scores: [batch_size, n_heads, len_q, len_k]
        # 加上注意力掩码, 将attn_mask中为True的位置的分数设置为极小值
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        # softmax归一化 => 注意力权重
        weights = nn.Softmax(dim=-1)(scores)
        # weights: [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(weights, V) 
        # context: [batch_size, n_heads, len_q, d_v]
        return context, weights # 返回上下文变量 和 注意力分数

class MultiHeadAttention(nn.Module):
    """
    多头注意力
    简单理解，先放大维度，提取Q、K、V的各个维度的信息，再缩小维度，得到最终的结果
    黑盒的看是 (Q、K、V) -> Q
    """
    def __init__(self, d_embedding=d_embedding, n_heads=n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_embedding = d_embedding
        self.n_heads = n_heads

        self.W_Q = nn.Linear(d_embedding, n_heads * d_k)
        self.W_K = nn.Linear(d_embedding, n_heads * d_k)
        self.W_V = nn.Linear(d_embedding, n_heads * d_v)
        self.linear = nn.Linear(n_heads * d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, Q, K, V, attn_mask):
        # 维度信息
        # Q: [batch_size, len_q, d_embedding]
        # K: [batch_size, len_k, d_embedding]
        # V: [batch_size, len_v(=len_k), d_embedding]
        # attn_mask: [batch_size, len_q, len_k]
        
        residual, batch_size = Q, Q.size(0)
        # 线性层，维度提升，为了捕捉更多信息
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2) 
        # q_s: [batch_size, n_heads, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        # k_s: [batch_size, n_heads, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        # v_s: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask: [batch_size, n_heads, len_q, len_k]

        # 点积缩放注意力
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size, n_heads, len_q, d_v]
        # weights: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # context: [batch_size, len_q, n_heads * d_v]

        # 线性层，降维成 Q 原始的维度
        output = self.linear(context) 
        # output: [batch_size, len_q, d_embedding]
        
        # 残差连接，并做归一化（方便将当前Q往下层传递，所以做了残差）
        output = self.layer_norm(output + residual) 
        # output: [batch_size, len_q, d_embedding]
        return output, weights

class PoswiseFeedForwardNet(nn.Module):
    """
    前馈神经网络，目标是优化每个标记（单词）的表征
    对每个位置的d_embedding维度进行升维 => 降维 => 残差归一化
    """
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        # 输入升维
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        # 输入降维
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        # 定义 归一化
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self, inputs):
        # inputs [batch_size, len_q, d_embedding]
        residual = inputs

        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # [batch_size, d_ff, len_q]

        output = self.conv2(output).transpose(1, 2)
        # [batch_size, len_q, d_embedding]
        
        output = self.layer_norm(output + residual)
        # [batch_size, len_q, d_embedding]
        return output

def get_pos_enc_table(n_position, embedding_dim):
    # 位置编码表：目的是让模型知道输入序列中单词的位置信息
    # 也可以用自然序列(1,2,3)作为位置编码，但正余弦能更好表达位置信息
    # 维度信息
    # n_position: 输入序列最大长度
    # embedding_dim: 词向量维度

    pos_table = np.zeros((n_position, embedding_dim), dtype=np.float32)
    for pos_i in range(n_position):
        for idx in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (idx // 2) / embedding_dim)
            pos_table[pos_i, idx] = angle
    
    pos_table[:, 0::2] = np.sin(pos_table[:, 0::2]) # dim 2i偶数维
    pos_table[:, 1::2] = np.cos(pos_table[:,1::2]) # dim 2i+1奇数维
    # pos_table: [n_position, embedding_dim]
    return torch.FloatTensor(pos_table)

def get_attn_pad_mask(seq_q, seq_k):
    # 填充注意力掩码, 补充短句到长度相同的长度，统一用<pad>补充
    # seq_q: [batch_size, len_q]
    # seq_k: [batch_size, len_k]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # =0的位置会变成True,其他是False
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) 
    # [batch_size, 1, len_k]

    pad_aatn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    # [batch_size, len_q, len_k]
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    # 注意力掩码，屏蔽未来的信息
    # seq: [batch_size, seq_len(Q)=seq_len(K)]
    
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, seq_len, seq_len]

    # triu triangle upper
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # subsequent_mask: [batch_size, seq_len, seq_len]

    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    # subsequent_mask: [batch_size, seq_len, seq_len]
    return subsequent_mask

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(d_embedding)
        self.feed_forward = PoswiseFeedForwardNet()
        self.norm2 = nn.LayerNorm(d_embedding)

    def forward(self, dec_inputs, attn_mask):
        # dec_inputs: [batch_size, seq_len, d_embedding]
        attn_outputs, _ = self.self_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        # attn_outputs: [batch_size, seq_len, d_embedding]

        # 残差连接 + 归一化
        norm1_outputs = self.norm1(dec_inputs + attn_outputs)
        # norm1_outputs: [batch_size, seq_len, d_embedding]

        ff_outputs = self.feed_forward(norm1_outputs)
        # ff_outputs: [batch_size, seq_len, d_embedding]
        dec_outputs = self.norm2(norm1_outputs + ff_outputs)
        # dec_outputs: [batch_size, seq_len, d_embedding]
        return dec_outputs

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers=6):
        super(Decoder, self).__init__()
        # 词典维度
        self.src_emb = nn.Embedding(vocab_size, d_embedding)
        # 位置编码
        self.pos_emb = nn.Embedding(max_seq_len, d_embedding)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs):
        # dec_inputs: [batch_size, seq_len]
        # 创建位置编码
        positions = torch.arange(len(dec_inputs), device=dec_inputs.device).unsqueeze(-1)
        # positions: [batch_size, seq_len, 1]
        inputs_embedding = self.src_emb(dec_inputs) + self.pos_emb(positions)
        # inputs_embedding: [batch_size, seq_len, d_embedding]

        # 注意力掩码，屏蔽未来的信息
        attn_mask = get_attn_subsequent_mask(inputs_embedding).to(device)
        attn_mask = torch.gt(attn_mask, 0)
        # print(attn_mask.shape)
        # print(attn_mask.dtype)
        # attn_mask: [batch_size, seq_len, seq_len]

        dec_outputs = inputs_embedding
        for layer in self.layers:
            dec_outputs = layer(dec_outputs, attn_mask)
        return dec_outputs

class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers=6):
        super(GPT, self).__init__()

        self.decoder = Decoder(vocab_size, max_seq_len, n_layers) # 解码器
        self.projection = nn.Linear(d_embedding, vocab_size) # 输出结果

    def forward(self, dec_inputs):
        dec_outputs = self.decoder(dec_inputs)
        # dec_outputs: [batch_size, tgt_len, embedding_dim]
        # 预测结果
        dec_outputs = self.projection(dec_outputs)
        # dec_outputs: [batch_size, tgt_len, vocab_size]
        return dec_outputs