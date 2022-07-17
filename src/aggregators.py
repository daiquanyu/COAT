import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, weights_att, batch_size, dim, n_neighbor, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
        self.weights_att = weights_att
        self.n_neighbor = n_neighbor

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops):
        if self.weights_att['att'] == 'avg':
            avg = True
        else:
            avg = False

        # avg = True  # no personalized transformation of knowledge graph
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # neighbor_vectors: [self.batch_size, -1, self.n_neighbor, self.dim]

            # [batch_size, -1, n_neighbor]
            if self.weights_att['att'] == 'u_r':
                print('Attention: u_r')
                user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

            if self.weights_att['att'] == 'h_r_t':
                print('Attention: h_r_t')
                shape_1 = [-1, self.dim*3]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_e = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_e = tf.tile(h_e, [1, 1, self.n_neighbor, 1])

                embeddings = tf.concat([h_e, neighbor_vectors, neighbor_relations], -1)
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'hrt_add':
                print('Attention: hrt_add')
                shape_1 = [-1, self.dim]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_e = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_e = tf.tile(h_e, [1, 1, self.n_neighbor, 1])

                embeddings = tf.add_n([h_e, neighbor_vectors, neighbor_relations])
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'u_h_r_t':
                print('Attention: u_h_r_t')

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                # h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
                target_neigh_scores = tf.reduce_mean(h_embeddings * neighbor_vectors, axis=-1)

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'h_ur_t':
                print('Attention: h_ur_t')
                shape_1 = [-1, self.dim*3]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                user_relation_embeddings = user_embeddings * neighbor_relations
                embeddings = tf.concat([h_embeddings, neighbor_vectors, user_relation_embeddings], -1)
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'ur_ht_mlp':
                # u-r inner product, h-t MLP
                print('Attention: ur_ht_mlp')

                # user-relation
                user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)

                # target-neighbor
                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_t = tf.concat([h_e, neighbor_vectors], axis=-1)
                h_t = tf.reshape(h_t, [-1, 2*self.dim])
                
                h_t = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                target_neigh_scores = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                target_neigh_scores = tf.reshape(target_neigh_scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_concat':
                print('Attention: uhrt_concat')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.concat([u_e, r_e, h_e, t_e], axis=-1)
                embeddings = tf.reshape(embeddings, [-1, 4*self.dim])
                
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_add':
                print('Attention: uhrt_add')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings = tf.reshape(embeddings, [-1, self.dim])
                
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_add_2':
                # 2-layer MLP
                print('Attention: uhrt_add_2')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings = tf.reshape(embeddings, [-1, self.dim])
                
                embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_bi':
                print('Attention: uhrt_bi')
                shape = [-1, self.dim]

                # u_e = tf.reshape(tf.concat([user_embeddings]*(self.n_neighbor**hops), axis=2), shape)
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_e = tf.reshape(h_e, shape)
                t_e = tf.reshape(neighbor_vectors, shape)

                embeddings_1 = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings_2 = u_e * r_e * h_e * t_e

                # embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                # user_relation_scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                embeddings_1 = tf.nn.leaky_relu(tf.matmul(embeddings_1, self.weights_att['att_w_1']))
                embeddings_2 = tf.nn.leaky_relu(tf.matmul(embeddings_2, self.weights_att['att_w_2']))

                user_relation_scores = tf.matmul(embeddings_1 + embeddings_2, self.weights_att['att_w_3'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

            if self.weights_att['att'] == 'u_h_r_t_mlp':
                print('Attention: u_h_r_t_mlp')

                shape = [-1, self.dim]

                # user-relation
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                u_r = tf.concat([u_e, r_e], axis=-1)
                u_r = tf.nn.leaky_relu(tf.matmul(u_r, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                user_relation_scores = tf.nn.leaky_relu(tf.matmul(u_r, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])

                # target-neighbor
                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_e = tf.reshape(h_e, shape)
                t_e = tf.reshape(neighbor_vectors, shape)

                h_t = tf.concat([h_e, t_e], axis=-1)
                h_t = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_1_2'])+self.weights_att['att_b_1_2'])
                target_neigh_scores = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_2_2'])+self.weights_att['att_b_2_2'])
                target_neigh_scores = tf.reshape(target_neigh_scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'u_r_mlp':
                print('Attention: u_r_mlp')
                shape = [-1, self.dim]

                # u_e = tf.reshape(tf.concat([user_embeddings]*(self.n_neighbor**hops), axis=2), shape)
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                embeddings = tf.concat([u_e, r_e], axis=-1)
                embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                user_relation_scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

