import numpy as np
import scipy.sparse as sp
import os
from time import time


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_item_adj = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_item_adj

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../Dataset/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    '''
    *****************************************
    Rating Matrix for NGCF
    '''
    R = sp.dok_matrix((n_user, n_item), dtype=np.float32)
    for i in range(train_data.shape[0]):
        if train_data[i, 2]>0.5:
            R[train_data[i, 0], train_data[i, 1]] = 1

    plain_adj, norm_adj, mean_adj = get_adj_mat(args, n_user, n_item, R)
    if args.adj_type == 'plain':
        user_item_adj = plain_adj
    elif args.adj_type == 'norm':
        user_item_adj = norm_adj
    elif args.adj_type == 'gcmc':
        user_item_adj = mean_adj
    else:
        user_item_adj = mean_adj + sp.eye(mean_adj.shape[0])
    '''
    *****************************************
    '''
    return n_user, n_item, train_data, eval_data, test_data, user_item_adj

'''
*****************************************
From NGCF
'''
def get_adj_mat(args, n_user, n_item, R):
    path = '../Dataset/{}'.format(args.dataset)
    print('Creating UI Graph ...')
    adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_user, n_item, R)
    print('Finish Creating Adjacency Matrix of UI Graph.')

    return adj_mat, norm_adj_mat, mean_adj_mat

# def get_adj_mat(args, n_user, n_item, R):
#     path = '../Dataset/{}'.format(args.dataset)
#     try:
#         t1 = time()
#         adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
#         norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
#         mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')
#         print('already load adj matrix', adj_mat.shape, time() - t1)

#     except Exception:
#         adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_user, n_item, R)
#         sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
#         sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
#         sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
#     return adj_mat, norm_adj_mat, mean_adj_mat

def create_adj_mat(n_user, n_item, R):
    t1 = time()
    adj_mat = sp.dok_matrix((n_user + n_item, n_user + n_item), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R.tolil()

    adj_mat[:n_user, n_user:] = R
    adj_mat[n_user:, :n_user] = R.T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def normalized_adj_single(adj):
        '''
        D^(-1)*A
        '''
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)

        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        print('check normalized adjacency matrix whether equal to this laplacian matrix.')
        return temp

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix', time() - t2)
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
'''
*****************************************
'''

def dataset_split(rating_np, args):
    print('---------------')
    print(args.seed)
    print('---------------')
    np.random.seed(args.seed)

    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../Dataset/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
