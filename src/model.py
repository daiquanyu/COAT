import tensorflow as tf
import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, pretrain=None):
        self.pretrain = pretrain
        self._parse_args(args, n_user, n_item, adj_entity, adj_relation, user_item_adj)
        self._build_inputs()
        self._build_model(n_user, n_item, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, n_user, n_item, adj_entity, adj_relation, user_item_adj):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

        '''
        *****************************************
        CF parameters
        '''
        self.n_user = n_user
        self.n_item = n_item
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.model_type = args.model_type
        self.smoothing_steps = args.smoothing_steps

        self.norm_adj = user_item_adj  # To do
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag

        # Create Model Parameters (i.e., Initialize Weights).
        if self.model_type in ['KGCN_NGCF', 'KGCN_GCMC', 'KGCN_GCN', 'NGCF', 'GCN', 'GCMC']:
            self.weights_ngcf = self._init_weights_ngcf()

        self.att = args.att # or 'u_r'
        if self.att in ['h_ur_t', 'h_r_t', 'hrt_add', 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add', \
                        'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp']:
            self.weights_att = self._init_weights_att()
            self.weights_att['att'] = self.att
        else:
            self.weights_att = dict()
            self.weights_att['att'] = self.att

        # item embeddings aggregation from KGCN and CF
        self.agg_type = args.agg_type
        self.alpha = args.alpha

        if self.agg_type in ['gcn', 'graphsage', 'bi']:
            self.weights_agg = self._init_weights_agg()
        '''
        *****************************************
        '''

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices_{}'.format(self.model_type))
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices_{}'.format(self.model_type))
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels_{}'.format(self.model_type))

        '''
        *****************************************
        NGCF dropout
        '''
        self.node_dropout = tf.placeholder(tf.float32, shape=[None], name='node_dropout_{}'.format(self.model_type))
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None], name='mess_dropout_{}'.format(self.model_type))
        '''
        *****************************************
        '''

    def _build_model(self, n_user, n_item, n_entity, n_relation):
        with tf.variable_scope('{}'.format(self.model_type)):
            if self.pretrain is not None:
                self.user_emb_matrix = tf.get_variable(initializer=self.pretrain['user'], \
                    name='user_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.get_variable(initializer=self.pretrain['entity'], \
                    name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.get_variable(initializer=self.pretrain['relation'], \
                    name='relation_emb_matrix_{}'.format(self.model_type))
            else:
                self.user_emb_matrix = tf.get_variable(
                    shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix_{}'.format(self.model_type))
                # self.item_emb_matrix = tf.get_variable(
                #     shape=[n_item, self.dim], initializer=KGCN.get_initializer(), name='item_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.get_variable(
                    shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.get_variable(
                    shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix_{}'.format(self.model_type))

        '''
        *****************************************
        User Embedding
        '''
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
        
        if self.model_type == 'KGCN':
            self.user_embeddings_final = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
            # KGCN for item embeddings
            # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
            # dimensions of entities:
            # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
            entities, relations = self.get_neighbors(self.item_indices)
            # [batch_size, dim]
            self.item_embeddings_final, self.aggregators = self.aggregate(entities, relations)

        elif self.model_type in ['GCMC', 'NGCF', 'GCN']:
            # ngcf, gcn or gcmc for user and item embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'gcn':
                self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
            elif self.alg_type == 'gcmc':
                self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)

            self.user_embeddings_final = tf.nn.embedding_lookup(self.ua_embeddings, self.user_indices)
            self.item_embeddings_final = tf.nn.embedding_lookup(self.ia_embeddings, self.item_indices)

        elif self.model_type in ['KGCN_NGCF', 'KGCN_GCMC', 'KGCN_GCN']:
            # ngcf, gcn or gcmc for user embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'gcn':
                self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()
            elif self.alg_type == 'gcmc':
                self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)
            # [batch_size, dim]
            self.user_embeddings_final = tf.nn.embedding_lookup(self.ua_embeddings, self.user_indices)
            self.item_embeddings_cf = tf.nn.embedding_lookup(self.ia_embeddings, self.item_indices)

            # KGCN for item embeddings
            entities, relations = self.get_neighbors(self.item_indices)
            self.item_embeddings_kg, self.aggregators = self.aggregate(entities, relations)

            # combine item embeddings from CF & KG
            if self.agg_type == 'weighted_avg':
                self.item_embeddings_final = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
            elif self.agg_type == 'gcn':
                # item_embeddings = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings = self.item_embeddings_cf + self.item_embeddings_kg
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'graphsage':
                item_embeddings = tf.concat([self.item_embeddings_cf, self.item_embeddings_kg], axis=-1)
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'bi':
                # item_embeddings_1 = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings_1 = self.item_embeddings_cf + self.item_embeddings_kg
                item_embeddings_2 = tf.multiply(self.item_embeddings_cf, self.item_embeddings_kg)
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings_1, self.weights_agg['agg_w_1'])) + \
                    tf.nn.leaky_relu(tf.matmul(item_embeddings_2, self.weights_agg['agg_w_2']))

        else:
            raise Exception("Unknown model_type: " + self.model_type)
        '''
        *****************************************
        '''

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings_final * self.item_embeddings_final, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    '''
    *****************************************
    weights initialization for attention
    '''
    def _init_weights_att(self):
        all_weights = dict()

        if self.att == 'h_ur_t':
            shape_w_1 = [3*self.dim, 1]

        elif self.att == 'h_r_t':
            shape_w_1 = [3*self.dim, 1]

        elif self.att == 'hrt_add':
            shape_w_1 = [self.dim, 1]

        elif self.att == 'ur_ht_mlp':
            shape_w_1 = [2*self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_r_mlp']:
            shape_w_1 = [2*self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_h_r_t_mlp']:
            shape_w_1 = [2*self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

            shape_w_1_2 = [2*self.dim, self.dim]
            shape_w_2_2 = [self.dim, 1]

        elif self.att in ['uhrt_concat']:
            shape_w_1 = [4*self.dim, 1]

        elif self.att in ['uhrt_add']:
            shape_w_1 = [self.dim, 1]

        elif self.att in ['uhrt_add_2']:
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att == 'uhrt_bi':
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, self.dim]
            shape_w_3 = [self.dim, 1]

        with tf.variable_scope('{}'.format(self.model_type)):

            if self.att in ['h_r_t', 'h_ur_t', 'uhrt_concat', 'uhrt_add']:
                all_weights['att_w_1'] = tf.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))


            if self.att in ['uhrt_bi']:
                all_weights['att_w_1'] = tf.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_3'] = tf.get_variable(
                    shape=shape_w_3, initializer=KGCN.get_initializer(), name='att_w_3_{}'.format(self.model_type))
                all_weights['att_b_3'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_3_{}'.format(self.model_type))   


            if self.att in ['ur_ht_mlp', 'u_r_mlp', 'uhrt_add_2']:
                all_weights['att_w_1'] = tf.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))


            if self.att in ['u_h_r_t_mlp']:
                all_weights['att_w_1'] = tf.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_1_2'] = tf.get_variable(
                    shape=shape_w_1_2, initializer=KGCN.get_initializer(), name='att_w_1_2_{}'.format(self.model_type))
                all_weights['att_b_1_2'] = tf.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_2{}'.format(self.model_type))

                all_weights['att_w_2_2'] = tf.get_variable(
                    shape=shape_w_2_2, initializer=KGCN.get_initializer(), name='att_w_2_2_{}'.format(self.model_type))
                all_weights['att_b_2_2'] = tf.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_2_{}'.format(self.model_type))

        return all_weights
    '''
    *****************************************
    '''  

    '''
    *****************************************
    weights initialization for combining item embeddings
    '''
    def _init_weights_agg(self):
        all_weights = dict()

        if self.agg_type in ['gcn', 'bi']:
            shape_w = [self.dim, self.dim]
            shape_b = [1, self.dim]
        elif self.agg_type == 'graphsage':
            shape_w = [2*self.dim, self.dim]
            shape_b = [1, self.dim]
        else:
            raise Exception('Unknow  agg_type: {}'.format(self.agg_type))

        with tf.variable_scope('{}'.format(self.model_type)):
            all_weights['agg_w_1'] = tf.get_variable(
                shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_1_{}'.format(self.model_type))
            all_weights['agg_b_1'] = tf.get_variable(
                shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_1_{}'.format(self.model_type))

            if self.agg_type == 'bi':
                all_weights['agg_w_2'] = tf.get_variable(
                    shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_2_{}'.format(self.model_type))
                all_weights['agg_b_2'] = tf.get_variable(
                    shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_2_{}'.format(self.model_type))

        return all_weights
    '''
    *****************************************
    '''  

    '''
    *****************************************
    user and item embedding with NGCF
    '''
    def _init_weights_ngcf(self):
        all_weights = dict()

        self.weight_size_list = [self.dim] + self.weight_size

        for k in range(self.n_layers):

            with tf.variable_scope('{}'.format(self.model_type)):

                all_weights['W_gc_%d' %k] = tf.get_variable(
                    shape=[self.weight_size_list[k], self.weight_size_list[k+1]], initializer=KGCN.get_initializer(), \
                    name='W_gc_{}_{}'.format(k, self.model_type))
                all_weights['b_gc_%d' %k] = tf.get_variable(
                    shape=[1, self.weight_size_list[k+1]], initializer=KGCN.get_initializer(), \
                    name='b_gc_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'ngcf':
                    all_weights['W_bi_%d' % k] = tf.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k + 1]], \
                        initializer=KGCN.get_initializer(), name='W_bi_{}_{}'.format(k, self.model_type))
                    all_weights['b_bi_%d' % k] = tf.get_variable(
                        shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(), \
                        name='b_bi_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'gcmc':
                    all_weights['W_mlp_%d' % k] = tf.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k+1]], \
                        initializer=KGCN.get_initializer(), name='W_mlp_{}_{}'.format(k, self.model_type))
                    all_weights['b_mlp_%d' % k] = tf.get_variable(
                        shape=[1, self.weight_size_list[k+1]], initializer=KGCN.get_initializer(), \
                        name='b_mlp_{}_{}'.format(k, self.model_type))

        return all_weights

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            A = self._dropout_sparse(temp, 1 - self.node_dropout[0], self.n_nonzero_elems)
        else:
            A = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        # ego_embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)  # E

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            side_embeddings = tf.sparse_tensor_dense_matmul(A, ego_embeddings)  # Eq. (7) L*E
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights_ngcf['W_gc_%d' % k]) + self.weights_ngcf['b_gc_%d' % k])  # Eq. (7) L*E*W

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)  # Eq. (7) LE element_wise_dot E
            # transformed bi messages of neighbors.
            # Eq. (7) LE element_wise_dot E*W
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights_ngcf['W_bi_%d' % k]) + self.weights_ngcf['b_bi_%d' % k])  

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.mess_dropout[k])

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]
        # Method 1
        all_embeddings = tf.concat(all_embeddings, 1)  # Eq. (9)
        # # Method 2
        # all_embeddings = all_embeddings[-1]
        # # Method 3
        # all_embeddings = tf.add_n(all_embeddings)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings

    def _create_gcn_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            A = self._dropout_sparse(temp, 1 - self.node_dropout[0], self.n_nonzero_elems)
        else:
            A = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        # embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)  # E

        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            for _ in range(self.smoothing_steps):
                embeddings = tf.sparse_tensor_dense_matmul(A, embeddings)

            embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_ngcf['W_gc_%d' %k]) + self.weights_ngcf['b_gc_%d' %k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [embeddings]

        # # Method 1
        # all_embeddings = tf.concat(all_embeddings, 1)  # Eq. (9)
        # # Method 2
        # all_embeddings = all_embeddings[-1]
        # Method 3
        all_embeddings = tf.add_n(all_embeddings)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings 

    def _create_gcmc_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            A = self._dropout_sparse(temp, 1 - self.node_dropout[0], self.n_nonzero_elems)
        else:
            A = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        # embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)  # E

        all_embeddings = []

        for k in range(0, self.n_layers):
            for _ in range(self.smoothing_steps):
                embeddings = tf.sparse_tensor_dense_matmul(A, embeddings)
            # convolutional layer.
            embeddings = tf.nn.relu(tf.matmul(embeddings, self.weights_ngcf['W_gc_%d' %k]) + self.weights_ngcf['b_gc_%d' % k])
            # dense layer.
            mlp_embeddings = tf.nn.relu(tf.matmul(embeddings, self.weights_ngcf['W_mlp_%d' %k]) + self.weights_ngcf['b_mlp_%d' %k])
            # mlp_embeddings = tf.matmul(embeddings, self.weights_ngcf['W_mlp_%d' %k]) + self.weights_ngcf['b_mlp_%d' %k]
            # mlp_embeddings = tf.nn.dropout(mlp_embeddings, 1 - self.mess_dropout[k])

            all_embeddings += [mlp_embeddings]

        # # Method 1
        # all_embeddings = tf.concat(all_embeddings, 1)   # Eq. (9)
        # # Method 2
        # all_embeddings = all_embeddings[-1]
        # Method 3
        all_embeddings = tf.add_n(all_embeddings)
        
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings
    '''
    *****************************************
    '''  

    '''
    ******************************************
    item embedding in knowledge graph with KGCN
    '''
    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        res = [tf.reshape(entity_vectors[0], [self.batch_size, self.dim])]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim, \
                    self.n_neighbor, act=tf.nn.tanh, name=self.model_type+'_{}'.format(i))
            else:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim, \
                    self.n_neighbor, name=self.model_type+'_{}'.format(i))
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None,
                                    hops=hop+1)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
            res.append(tf.reshape(entity_vectors[0], [self.batch_size, self.dim]))

        # res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        # # Method 1
        # res = tf.concat(res, 1)

        # Method 2
        res = tf.add_n(res)

        # # Method 3
        # res = res[-1]

        return res, aggregators
    '''
    *****************************************
    '''

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
        #         tf.nn.l2_loss(self.relation_emb_matrix) + tf.nn.l2_loss(self.item_emb_matrix)

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
                tf.nn.l2_loss(self.relation_emb_matrix)

        if self.model_type in ['KGCN', 'KGCN_NGCF', 'KGCN_GCMC', 'KGCN_GCN']:
            for aggregator in self.aggregators:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

            if self.weights_att['att'] in ['h_ur_t', 'h_r_t' 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add', 'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp']:
                for weights in self.weights_att.keys():
                    if weights.startswith('att_w'):
                        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_att[weights])

            if self.agg_type in ['gcn', 'graphsage']:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1'])

            if self.agg_type == 'bi':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1']) + \
                               tf.nn.l2_loss(self.weights_agg['agg_w_2'])

        '''
        ******************************************
        L2 loss in NGCF
        '''
        # l2_loss_ngcf = []
        # for key in self.weights_ngcf:
        #     if 'W' in key:
        #         l2_loss_ngcf.append(tf.nn.l2_loss(self.weights_ngcf[key]))
        # l2_loss_ngcf = tf.add_n(l2_loss_ngcf)

        # self.l2_loss = self.l2_loss + l2_loss_ngcf
        '''
        *****************************************
        '''

        self.loss = self.base_loss + self.l2_weight * self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

    def get_embeddings(self, sess):
        user_emb, entity_emb, relation_emb = sess.run([self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix])

        return user_emb, entity_emb, relation_emb
