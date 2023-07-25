import pickle
import argparse
import json
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Tmall/30music')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--Omega', type=int, default=1)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num
Omega = opt.Omega

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))
product_attributes = load_json('datasets/' + opt.dataset + '/product_attributes.json')



if dataset == 'diginetica':
    num = 43098
elif dataset == "Tmall":
    num = 40728
elif dataset == "Nowplaying":
    num = 60417
elif dataset == "30music":
    num = 132648
else:
    num = 3

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    # attributes = product_attributes[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1
        
for tup in relation: 
    as_items = product_attributes[f'{tup[0]}'][f'same_{"cate"}'].copy()
    if tup[1] in as_items:
        adj1[tup[0]][tup[1]] += Omega

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]
    # print(weight[i])

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
