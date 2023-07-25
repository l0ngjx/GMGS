import datetime
import numpy as np
from tqdm import tqdm
from aggregator import *
from torch.nn import Module
import torch.nn.functional as F

# device conf
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt
        
        self.dropout_global = opt.dropout_global
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.n_iter = opt.n_iter
        self.hop = opt.g_iter
        self.mu = opt.mu
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()

        # Aggregator
        self.attribute_agg = AttributeAggregator(self.dim, self.opt.alpha, opt, self.opt.dropout_attribute)
        self.local_agg = nn.ModuleList()
        self.mirror_agg = nn.ModuleList()
        self.global_agg = nn.ModuleList()
        for i in range(self.n_iter):
            agg = LocalAggregator(self.dim, self.opt.alpha)
            self.local_agg.append(agg)
            agg = MirrorAggregator(self.dim)
            self.mirror_agg.append(agg)
        for i in range(self.hop):
            agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.tanh)
            self.global_agg.append(agg)

        # high way net
        self.highway = nn.Linear(self.dim * 2, self.dim, bias=False)

        # embeddings
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=False)
        self.glu4 = nn.Linear(self.dim, self.dim)
        self.gate = nn.Linear(self.dim * 2, self.dim, bias=False)

        # loss function
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_score(self, hidden, pos_emb, h_mirror, h_local, h_global, mask, item_weight):
        hm = h_mirror
        hg = h_global
        
#         hidden = hidden + hg
        
        hl = h_local.unsqueeze(1).repeat(1, hidden.size(1), 1)
        hp = hidden + pos_emb

#         hp = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
#         hp = torch.tanh(hp)
        
        
#         hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
#         hs = hs.unsqueeze(-2).repeat(1, hidden.shape[1], 1)
        
        nh = torch.sigmoid(self.glu1(hp) + self.glu2(hg) + self.glu4(hl))
#         nh = torch.sigmoid(self.glu1(hp) + self.glu2(hg))
        beta = torch.matmul(nh, self.w)
        beta = beta * mask
        zg = torch.sum(beta * hp, 1)
        
        gf = torch.sigmoid(self.gate(torch.cat([zg, h_local], dim=-1))) * self.mu
        zh = gf * h_local + (1 - gf) * zg
        zh = F.dropout(zh, self.opt.dropout_score, self.training)
        scores = torch.matmul(zh, item_weight.transpose(1, 0))
        
        return scores

    def similarity_loss(self, hf, hf_SSL, simi_mask):
        h1 = hf
        h2 = hf_SSL
        h1 = h1.unsqueeze(2).repeat(1, 1, h1.size(1), 1)
        h2 = h2.unsqueeze(1).repeat(1, h2.size(1), 1, 1)
        hf_similarity = torch.sum(torch.mul(h1, h2), dim=3) / self.opt.temp
        loss = -torch.log(torch.softmax(hf_similarity, dim=2) + 1e-8)
        simi_mask = simi_mask == 1
        loss = torch.sum(loss * simi_mask, dim=2)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)

        return loss

    def compute_score_and_ssl_loss(self, h, h_local, h_mirror, h_global, mask, hf_SSL1, hf_SSL2, simi_mask):
        mask = mask.float().unsqueeze(-1)
        
        batch_size = h.shape[0]
        len = h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        b = self.embedding.weight[1:]
        
        hs = torch.sum(h_global * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        simi_loss = self.similarity_loss(hf_SSL1, hf_SSL2, simi_mask)
        scores = self.compute_score(h, pos_emb, h_mirror, h_local, hs, mask, b)
        
        return simi_loss, scores

    def forward(self, inputs, adj, last_item_mask, as_items, as_items_SSL, simi_mask, item):
        # preprocess
        mask_item = inputs != 0
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        attribute_num = len(as_items)
        h = self.embedding(inputs)
        h_as = []
        h_as_SSL = []
        as_mask = []
        as_mask_SSL = []
        for k in range(attribute_num):
            nei = as_items[k]
            nei_SSL = as_items_SSL[k]
            nei_emb = self.embedding(nei)
            nei_emb_SSL = self.embedding(nei_SSL)
            h_as.append(nei_emb)
            h_as_SSL.append(nei_emb_SSL)
            as_mask.append(as_items[k] != 0)
            as_mask_SSL.append(as_items_SSL[k] != 0)

        # attribute
        hf_1, hf_2, hf = self.attribute_agg(h, h_as, as_mask, h_as_SSL, as_mask_SSL)

        # global
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                                    masks=None,
                                    batch_size=batch_size,
                                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                                    extra_vector=session_info[hop])
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)

        # GNN
        x = h
        mirror_nodes = hf
        for i in range(self.hop):
            # aggregate neighbor info
            x = self.local_agg[i](x, adj, mask_item)
            x, mirror_nodes = self.mirror_agg[i](mirror_nodes, x, mask_item)

        # highway 
        g = torch.sigmoid(self.highway(torch.cat([h, x], dim=2)))
        x_dot = g * h + (1 - g) * x
#         x_dot = h_global + x_dot
       
        # local representation
        h_local = torch.masked_select(x_dot, last_item_mask.unsqueeze(2).repeat(1, 1, x_dot.size(2))).reshape(mask_item.size(0), -1)
        
        # hidden
        hidden = x_dot
        
        # mirror
        h_mirror = mirror_nodes

        # calculate score
        simi_loss, scores = self.compute_score_and_ssl_loss(hidden, h_local, h_mirror, h_global, mask_item, hf_1, hf_2, simi_mask)

        return simi_loss, scores


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, opt):
    adj, items, targets, last_item_mask, as_items, as_items_SSL, simi_mask, inputs = data
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    inputs = trans_to_cuda(inputs).long()

    last_item_mask = trans_to_cuda(last_item_mask)
    for k in range(opt.attribute_kinds):
        as_items[k] = trans_to_cuda(as_items[k]).long()
        as_items_SSL[k] = trans_to_cuda(as_items_SSL[k]).long()
    targets_cal = trans_to_cuda(targets).long()
    simi_mask = trans_to_cuda(simi_mask).long()

    simi_loss, scores = model(items, adj, last_item_mask, as_items, as_items_SSL, simi_mask, inputs)
    loss = model.loss_function(scores, targets_cal - 1)
    loss = loss + opt.phi * simi_loss

    return targets, scores, loss

def adjust_learning_rate(optimizer, decay_rate, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * decay_rate
    lr * decay_rate

def train_test(model, opt, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    for i, data in enumerate(tqdm(train_loader)):
        targets, scores, loss = forward(model, data, opt)
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    if opt.decay_count < opt.decay_num:
        model.scheduler.step()
        opt.decay_count += 1

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_size=int(opt.batch_size / 8),
                                              shuffle=False, pin_memory=False)
    result_20 = []
    hit_20, mrr_20 = [], []
    result_10 = []
    hit_10, mrr_10 = [], []
    for data in test_loader:
        targets, scores, loss = forward(model, data, opt)
        targets = targets.numpy()
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        for score, target in zip(sub_scores_20, targets):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_scores_10 = scores.topk(10)[1]
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        for score, target in zip(sub_scores_10, targets):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
    result_20.append(np.mean(hit_20) * 100)
    result_20.append(np.mean(mrr_20) * 100)

    result_10.append(np.mean(hit_10) * 100)
    result_10.append(np.mean(mrr_10) * 100)

    return result_10, result_20
