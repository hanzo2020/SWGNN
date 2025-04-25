import torch
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False
import numpy as np
import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_edge
from torch.nn import Dropout, MaxPool1d, AvgPool1d
from torch_geometric.nn import Sequential, GCNConv, MixHopConv
import torch.nn as nn
import utils
HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2


class GTN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual):
        super(GTN, self).__init__()
        self.drop_rate = drop_rate
        self.pooling = pooling
        self.residual = residual
        self.independence = False
        self.convs = t.nn.ModuleList()
        mid_channels = in_channels + hidden_channels if residual else hidden_channels
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                                          concat=False, beta=True))
        self.ln1 = LayerNorm(in_channels=mid_channels)
        if pooling:
            self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                            dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)
            self.pool = MaxPool1d(2, 2) if pooling == 'max' else AvgPool1d(2, 2) if pooling == 'avg' \
                        else Linear(hidden_channels * heads, hidden_channels * heads // 2)
        else:
            self.convs.append(TransformerConv(mid_channels, hidden_channels // 2, heads=heads,
                                            dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True))
            self.ln2 = LayerNorm(in_channels=hidden_channels * heads // 2)
        self.convs_last = GCNConv(48, 1, improved=False)
    def forward(self, data):
        if isinstance(data, list):
            self.independence = True
            data = data[0]
        x = data.x
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        edge_attr = data.edge_attr[edge_mask]

        res = x * self.residual
        x = F.leaky_relu(self.convs[0](x, edge_index, edge_attr), negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)
        edge_index, edge_mask = dropout_edge(data.edge_index, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        edge_attr = data.edge_attr[edge_mask]

        x = self.convs[1](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = t.squeeze(self.pool(t.unsqueeze(x, 1)), dim=1) if self.pooling else x
        x = self.ln2(x)
        if self.independence:
            return t.sigmoid(self.convs_last(x, data.edge_index))[:data.batch_size]

        return x[:data.batch_size]



class Multi_GTN(t.nn.Module):
    def __init__(self, gnn, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, num_ppi, pooling,
                 residual, learnable_weight, ):
        super(Multi_GTN, self).__init__()

        self.convs = t.nn.ModuleList()
        for _ in range(num_ppi):
            if 'GTN' in gnn:
                self.convs.append(
                    GTN(in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, pooling, residual))

        if learnable_weight:
            self.ppi_weight = t.nn.ParameterList([t.nn.Parameter(t.Tensor(1, 1)) for _ in range(num_ppi)])
            for weight in self.ppi_weight:
                t.nn.init.constant_(weight, 1)
        else:
            self.ppi_weight = t.ones(num_ppi, 1)

        self.lins = t.nn.ModuleList()
        self.lins.append(Linear(int(num_ppi * hidden_channels * heads / 2), HIDDEN_DIM,
                                weight_initializer="kaiming_uniform"))
        self.dropout = Dropout(drop_rate)
        self.lins.append(Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform"))

    def forward(self, data_tuple):
        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]  # 这那么多节点出来就128个节点，什么意思？
        x = t.cat(x_list, dim=1)  # 5个128X48变成128X240
        x = self.lins[0](x).relu()  # 先行层，变成32维
        x = self.dropout(x)
        x = self.lins[1](x)  # 32维变成128X1，最终预测值？可是只有128个节点啊

        return t.sigmoid(x), x_list, self.ppi_weight  # x_list是最初输出来的128个节点的特征，weight根本没见用到过



class DSGNNLR(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, residual):
        super(DSGNNLR, self).__init__()
        self.residual = residual
        self.num_hop = 3
        self.hop_weights = self.calculate_weights(self.num_hop)
        self.mid_channels = in_channels + hidden_channels if residual else hidden_channels
        self.mix1 = Sequential('x, edge_index', [
            (MixHopConv(in_channels, hidden_channels), 'x, edge_index -> x'),
            nn.BatchNorm1d(3 * hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

        ])
        self.mix2 = Sequential('x, edge_index', [
            (MixHopConv(hidden_channels, hidden_channels), 'x, edge_index -> x'),
        ])

        self.linear = nn.Sequential(
            nn.Linear(self.mid_channels, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 48)
        )

    def calculate_weights(self, total_jumps):
        weights = torch.tensor(
            [1 / torch.log2(torch.tensor(k + 1, dtype=torch.float32)) for k in range(1, total_jumps + 1)])

        normalized_weights = weights / weights.sum()
        return normalized_weights

    def forward(self, data):
        x = data.x
        res = x * self.residual
        x = self.mix1(x, data.edge_index)
        x = x[:, :64] * self.hop_weights[0] + x[:, 64:128] * self.hop_weights[1] + x[:, 128:] * self.hop_weights[2]
        x = self.mix2(x, data.edge_index)
        x = x[:, :64] * self.hop_weights[0] + x[:, 64:128] * self.hop_weights[1] + x[:, 128:] * self.hop_weights[2]
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.linear(x)[:data.batch_size]

        return t.sigmoid(x)



class DSGNN(t.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_ppi, residual, learnable_weight, args):
        super(DSGNN, self).__init__()
        self.convs = t.nn.ModuleList()
        self.args = args
        self.l2_bias = 0
        self.epsilon = 1e-5
        self.degrees_weights = nn.Parameter(torch.ones(15))
        for _ in range(num_ppi):
            self.convs.append(
                DSGNNLR(in_channels, hidden_channels, residual)
            )

        if learnable_weight:
            self.ppi_weight = t.nn.ParameterList([t.nn.Parameter(t.Tensor(1, 1)) for _ in range(num_ppi)])
            for weight in self.ppi_weight:
                t.nn.init.constant_(weight, 1)
        else:
            self.ppi_weight = t.ones(num_ppi, 1)

        self.lins = t.nn.ModuleList()
        self.layer1 = nn.Sequential(
            nn.Linear(int(48*5 + 15), HIDDEN_DIM),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(HIDDEN_DIM, 1)
        )

        if True:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.constant_(m.bias, 0)

    def contrastive_loss(self, constraint):
        constraint_flattened = constraint.reshape(-1, constraint.size(-1))
        cosine_sim = F.cosine_similarity(constraint_flattened.unsqueeze(1), constraint_flattened.unsqueeze(0),
                                         dim=-1)
        positive_pairs = []
        for i in range(0, constraint_flattened.size(0), 5):
            temp = cosine_sim[i:i + 5, i:i + 5].tril(diagonal=-1).flatten()
            temp = temp[temp != 0]
            positive_pairs.append(temp)

        positive_pairs = torch.cat(positive_pairs, dim=0)
        negative_pairs = []
        num_positive_pairs = positive_pairs.size(0)

        for _ in range(num_positive_pairs):
            i, j = torch.randint(0, constraint.size(0), (2,))
            node_i = constraint[i]
            node_j = constraint[j]
            sim = F.cosine_similarity(node_i.unsqueeze(1), node_j.unsqueeze(0), dim=-1).flatten()
            negative_pairs.append(sim)
        negative_pairs = torch.cat(negative_pairs, dim=0)

        temperature = 0.1
        positive_similarity = positive_pairs / temperature
        negative_similarity = negative_pairs / temperature
        loss = -torch.log(torch.exp(positive_similarity) / (
                    torch.exp(positive_similarity) + torch.sum(torch.exp(negative_similarity), dim=-1)))

        return loss.mean()

    def l2(self):
        """
        模型l2计算，默认是所有参数（除了embedding之外）的平方和，
        Embedding 的 L2是 只计算当前batch用到的
        :return:

        Compute the l2 term of the model, by default it's the square sum of all parameters (except for embedding)
        The l2 norm of embedding only consider those embeddings used in the current batch
        :return:
        """
        l2 = utils.numpy_to_torch(np.array(0.0, dtype='float32'))
        l2 = l2.to(self.args['gpu'])
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if self.l2_bias == 0 and 'bias' in name:
                continue
            l2 += (p ** 2).sum()
        return l2

    def forward(self, data_tuple):
        degrees = [data.degrees[:data_tuple[0].batch_size] for i, data in enumerate(data_tuple)]
        adj_degrees = [data.adj_degrees[:data_tuple[0].batch_size] for i, data in enumerate(data_tuple)]
        ratio = [data.ratio[:data_tuple[0].batch_size] for i, data in enumerate(data_tuple)]
        degrees = torch.cat(degrees, dim=1)
        degrees = torch.nn.functional.normalize(degrees, p=2, dim=0)
        adj_degrees = torch.cat(adj_degrees, dim=1)
        adj_degrees = torch.nn.functional.normalize(adj_degrees, p=2, dim=0)
        ratio = torch.cat(ratio, dim=1)
        ratio = torch.nn.functional.normalize(ratio, p=2, dim=0)
        degrees_f = torch.cat((degrees, adj_degrees, ratio), dim=1)
        degrees_f = degrees_f * self.degrees_weights
        x_list = [self.convs[i](data) for i, data in enumerate(data_tuple)]
        constraint = t.stack(x_list, dim=0).permute(1, 0, 2)
        x = t.cat(x_list, dim=1)
        x = torch.cat((x, degrees_f), dim=1)
        x = self.layer1(x)
        l2_loss = self.l2()
        contrastive_loss = self.contrastive_loss(constraint)
        length_loss = constraint.norm(dim=2).sum()
        constraint_loss = 1e-4 * l2_loss + 1e-5 * length_loss + contrastive_loss * 1e-4
        constraint_dict = {
            'l2_loss' : l2_loss,
            'length_loss': length_loss,
            'contrastive_loss': contrastive_loss,
            'constraint_loss': constraint_loss,
            'result': x
        }
        return t.sigmoid(x), x_list, self.ppi_weight, constraint_dict, self.degrees_weights


