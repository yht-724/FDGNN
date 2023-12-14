import torch.nn as nn
import torch
from torchinfo import summary
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
import os
import numpy as np
import torch
import networkx as nx
from libs.event_list import physical_adj
from libs.utils import Data_Batch
from torch.utils.data import DataLoader
from train_hawkes import get_intensity
from torch.nn import Conv2d, LayerNorm
from dgl.nn.pytorch import SAGEConv
from scipy.sparse import csr_matrix




os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ---------------------GCN-----------------

def numpy_to_graph(A, type_graph='dgl', node_features=None):
    '''Convert numpy arrays to graph
    Parameters:
        A : mxm array, Adjacency matrix
        type_graph : str, 'dgl' or 'nx'
        node_features : dict,  Optional, dictionary with key=feature name, value=list of size m, Allows user to specify node features
    Returns:
        Graph of 'type_graph' specification
    '''
    G = nx.from_numpy_array(A)
    if node_features != None:
        for n in G.nodes():
            for k, v in node_features.items():
                G.nodes[n][k] = v[n]
    if type_graph == 'nx':
        return G
    G = G.to_directed()
    if node_features != None:
        node_attrs = list(node_features.keys())
    else:
        node_attrs = []

    g = dgl.from_networkx(G, node_attrs=node_attrs, edge_attrs=['weight'])
    return g



def generate_intensity(event_16batch_dataset, hawkes_model, p1, p2, nodes, device):
    batch_intensity_list = []
    batch_type_list = []

    for m in range(event_16batch_dataset.shape[0]):
        adj_event_list = []
        event_data = event_16batch_dataset[m]
        # event_data = np.array(event_data)
        for idx, (i, j) in enumerate(zip(p1, p2)):

            if torch.all(event_data[:, i, j] == 0):
                continue
            else:

                events = torch.nonzero(event_data[:, i, j])
                previous = 0
                node_event_list = []

                for k in events[0]:
                    e = {
                        'type_event': event_data[k, i, j] - 1,
                        'time_since_last_event': k - previous,
                        'node_ij': (i, j),
                        'event_time_index': k
                    }
                    previous = k
                    node_event_list.append(e)
                adj_event_list.append(node_event_list)

        if len(adj_event_list) == 0:
            batch_intensity_list.append(None)
            batch_type_list.append(None)
            continue

        time_durations = []
        type_seqs = []
        seq_lens = []
        event_timeIndex = []
        node_ij = []

        for i in range(len(adj_event_list)):
            seq_lens.append(len(adj_event_list[i]))
            type_seqs.append(torch.LongTensor([float(event['type_event']) for event in adj_event_list[i]]).to(device))
            time_durations.append(
                torch.FloatTensor([float(event['time_since_last_event']) for event in adj_event_list[i]]).to(device))
            event_timeIndex.append(
                torch.FloatTensor([float(event['event_time_index']) for event in adj_event_list[i]]).to(device))
            node_ij.append(adj_event_list[i][0]['node_ij'])

        type_size = 2
        max_len = max(seq_lens)
        batch_size = len(time_durations)  # 53
        time_duration_padded = torch.zeros(size=(batch_size, max_len + 1)).to(device)
        type_train_padded = torch.zeros(size=(batch_size, max_len + 1), dtype=torch.long).to(device)
        for idx in range(batch_size):
            time_duration_padded[idx, 1:seq_lens[idx] + 1] = time_durations[idx]
            type_train_padded[idx, 0] = type_size
            type_train_padded[idx, 1:seq_lens[idx] + 1] = type_seqs[idx]

        event_dataset = Data_Batch(time_duration_padded, type_train_padded, seq_lens)
        event_loader = DataLoader(event_dataset, batch_size=8, shuffle=False)

        intensity = get_intensity(event_loader, hawkes_model, device)
        longest_event_timeIndex = max(event_timeIndex, key=lambda x: x.size(0))
        allTimes_intensity = np.zeros((event_data.shape[0], nodes, nodes))
        allTime_type = np.zeros((event_data.shape[0], nodes, nodes))

        for i in range(intensity.shape[0]):
            node_i = node_ij[i][0]
            node_j = node_ij[i][1]
            for j in range(intensity.shape[1]):
                time_index = longest_event_timeIndex[j]
                time_index = time_index.item()
                allTimes_intensity[int(time_index), int(node_i), int(node_j)] = max(intensity[i, j, 0],
                                                                                    intensity[i, j, 1])
                allTime_type[int(time_index), int(node_i), int(node_j)] = np.argmax(intensity[i, j, :]) + 1

        need_intensity = allTimes_intensity[-1]
        need_type = allTime_type[-1]
        batch_intensity_list.append(need_intensity)
        batch_type_list.append(need_type)

    return batch_intensity_list, batch_type_list


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.gcn_msg = fn.copy_u(u='h', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='h')

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(self.gcn_msg, self.gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)

        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, feature):
        z = self.fc(feature)
        g.ndata['z'] = z

        g.apply_edges(self.edge_attention)

        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class GraphSageLayer(nn.Module):
    def __init__(self, num_layer, in_feats, out_feats):
        super(GraphSageLayer, self).__init__()

        self.sage_list = nn.ModuleList()
        for i in range(num_layer):
            self.sage_list.append(SAGEConv(in_feats, out_feats, 'mean').cuda(0))

    def forward(self, g, feature, device):
        feature = feature.permute(0, 3, 2, 1)
        mat = csr_matrix(g)
        g = dgl.from_scipy(mat)
        g = g.to(device)
        res = []
        for i in range(feature.shape[0]):
            x_item = feature[i]
            x_item = x_item.permute(1, 2, 0)
            for layer in self.sage_list:
                x_item = layer(g, x_item)
            res.append(x_item)
        res = torch.stack(res, dim=0)
        res = res.permute(0, 2, 1, 3)

        return res


class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        # self.layer1 = GCNLayer(24, 24)
        self.layer1 = GCNLayer(24, 48)
        self.layer2 = GCNLayer(48, 24)

    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.relu(x)
        x = self.layer2(g, x)
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        x = (x - x_mean) / x_std
        return x


class GATNet(nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.layer1 = GATLayer(24, 24)


    def forward(self, g, features):
        x = self.layer1(g, features)
        x = F.relu(x)

        x_mean = torch.mean(x)
        x_std = torch.std(x)
        x = (x - x_mean) / x_std
        return x


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class FDGNN(nn.Module):

    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=24,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


        self.gcn_layer = GCNNet()
        self.gcn_layer.to('cuda:0')



        self.attention = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )



        self.seta = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)

        self.lin = nn.Linear(self.model_dim,self.model_dim*2)
        self.bn = LayerNorm([self.out_steps, self.num_nodes,self.model_dim])

    def forward(self, x, adj, event, hawkes_model, device, dist):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow)

        x = x.to(device)
        adj = adj.to(device)
        event = event.to(device)

        phy_adj = dist

        non_zero_index = np.nonzero(phy_adj)

        p1, p2 = non_zero_index


        batch_size = x.shape[0]  # 16
        num_nodes = x.shape[2]

        intensity_list, type_list = \
            generate_intensity(event, hawkes_model, p1, p2, num_nodes, device)

        fake_A_list = []

        for i in range(len(intensity_list)):
            if intensity_list[i] is None:
                fake_A_list.append(adj[i])
            else:
                fake_A_list.append(self.seta * torch.Tensor(intensity_list[i] * type_list[i]).to(device) + adj[i])

        fake_A = torch.stack(fake_A_list)
        fake_A = fake_A.to('cpu')

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]



        x = self.input_proj(x)   # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim) tod_emb
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim) dow_emb
            features.append(dow_emb)

        if self.spatial_embedding_dim > 0:

            b, i, n, d = dow_emb.shape

            A = (phy_adj + fake_A[-1][0].detach().numpy())

            A = (A - A.min()) / (A.max() - A.min())

            g = numpy_to_graph(A)
            g = g.to(device)
            x = x.reshape((b * i, n, d))
            spatial_emb = torch.FloatTensor(np.zeros((b * i, n, d)))
            spatial_emb = spatial_emb.to(device)
            for k in range(b * i):
                spatial_emb[k] = self.gcn_layer(g, x[k])
            spatial_emb = spatial_emb.reshape((b, i, n, d))


            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)



        for attn in self.attention:
            x_t = attn(x, dim=1)
        for attn in self.attention:
            x_s = attn(x_t, dim=2)

        x = x_s




        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
            out = out.squeeze()

        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

