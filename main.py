import time
import argparse
import pickle
from model import *
from utils import *
import warnings
import os
warnings.filterwarnings("ignore")

def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/30music/Tmall/')

parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--g_iter', type=int, default=1)
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')

parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=16)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--phi', type=float, default=2.0)
parser.add_argument('--mu', type=float, default=0.01, help='A coefficient to control the impact of the gate mechanism.')
parser.add_argument('--temp', type=float, default=0.1, help='The temperature of the SSL mechanism.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--sequence_len', type=int, default=30)
parser.add_argument('--decay_count', type=int, default=0, help='Once it equals to the decay num, the lr_dc would proceed.')
parser.add_argument('--decay_num', type=int, default=3, help='The epoch needed lr_dc.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1, help='GNN layer num')
parser.add_argument('--dropout_attribute', type=float, default=0.5, help='Dropout rate of the mirror graph generation.')
parser.add_argument('--dropout_score', type=float, default=0.2, help='Dropout rate of the session representation generation.')
parser.add_argument('--validation', default=False, action='store_true', help='when you need a validation dataset.')
parser.add_argument('--sample_num', type=int, default=5, help='the sample num of attribute-same items.')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--model_path', type=str, default='./best_model', help='if you need to store your model.')

opt = parser.parse_args()


def main():
    init_seed(114514)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 4
        opt.g_iter = 1
        opt.dropout_global = 0.2
        opt.dropout_gcn = 0.4
        opt.dropout_score = 0.1
        opt.attribute_kinds = 1
        opt.phi = 1.2
        opt.temp = 0.1
        opt.mu = 0.2
        opt.sample_num = 5
    elif opt.dataset == '30music':
        num_node = 132648
        opt.n_iter = 4
        opt.g_iter = 2 
        opt.dropout_score = 0.5
        opt.attribute_kinds = 1
        opt.hiddenSize = 150
        opt.phi = 1.5
        opt.temp = 0.05
        opt.mu = 1.2
        opt.sample_num = 5
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 6
        opt.g_iter = 1
        opt.dropout_gcn = 0.5
        opt.dropout_attribute = 0.6
        opt.dropout_score = 0.4
        opt.attribute_kinds = 2
        opt.phi = 1.0
        opt.temp = 0.05
        opt.mu = 0.5
        opt.hiddenSize = 200
        opt.sample_num = 5
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    product_attributes = load_json('datasets/' + opt.dataset + '/product_attributes.json')
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)


    train_data = Data(train_data, product_attributes, opt)
    test_data = Data(test_data, product_attributes, opt)

    model = CombineGraph(opt, num_node,adj, num)
    model = trans_to_cuda(model)
    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train_data.sample_num = opt.sample_num
        [hit_10, mrr_10], [hit_20, mrr_20] = train_test(model, opt, train_data, test_data)
        flag = 0
        if hit_20 >= best_result[0]:
#             dir_name = os.path.dirname(f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
#             if not os.path.isdir(dir_name):
#                 os.makedirs(dir_name)
#             torch.save(model.state_dict(), f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
#             dir_name = os.path.dirname(f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
#             if not os.path.isdir(dir_name):
#                 os.makedirs(dir_name)
#             torch.save(model.state_dict(), f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tPrecision@10:\t%.4f\tMRR@10:\t%.4f\tPrecision@20:\t%.4f\tMRR@20:\t%.4f' % (
            hit_10, mrr_10, hit_20, mrr_20))
        print('Best Result:')
        print('\tPrecision@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
