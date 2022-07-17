import tensorflow as tf
import numpy as np
import scipy.io as sio
from model import KGCN
import os


def train_n_runs(args, data, show_loss, show_topk, show_ctr):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_item_adj = data[9]

    '''
    ***********************************************
    best epoch recording for different runs
    '''
    auc = []
    f1_score = []
    precision = []
    recall = []
    epochs_auc = []
    epochs_recall = []

    RUNS = args.runs
    '''
    ***********************************************
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    '''
    ***********************************************
    logging and loading paths
    '''
    ckpt_save_path_1 = '../Pretrain/{}/{}/ctr/'.format(args.model_type, args.dataset)
    ckpt_save_path_2 = '../Pretrain/{}/{}/topk/'.format(args.model_type, args.dataset)
    ckpt_restore_path_1 = '../Pretrain/{}/{}/ctr/'.format(args.model_type, args.dataset)
    ckpt_restore_path_2 = '../Pretrain/{}/{}/topk/'.format(args.model_type, args.dataset)

    if not os.path.exists(ckpt_save_path_1):
        os.makedirs(ckpt_save_path_1)
    if not os.path.exists(ckpt_save_path_2):
        os.makedirs(ckpt_save_path_2)

    if args.pretrain == 1:
        user_emb = np.load('../output/KGCN_{}_user_emb.npy'.format(args.dataset))
        entity_emb = np.load('../output/KGCN_{}_entity_emb.npy'.format(args.dataset))
        relation_emb = np.load('../output/KGCN_{}_relation_emb.npy'.format(args.dataset))
        pretrain = {'user': user_emb, 'entity': entity_emb, 'relation': relation_emb}
    else:
        pretrain = None
    '''
    ***********************************************
    '''
    args.seed += 1
    print('----------')
    print(args.seed)
    print('----------')
    for r in range(RUNS):
        tf.reset_default_graph()

        model = KGCN(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, pretrain=pretrain)

        '''
        ***********************************************
        save
        '''
        saver_ckpt = tf.train.Saver()
        '''
        ***********************************************
        '''

        '''
        ***********************************************
        best epoch recording current run
        '''
        best_auc = 0
        best_f1 = 0
        best_recall = None
        best_precision = None

        best_auc_epoch = 0
        best_recall_epoch = 0

        best_auc_eval = 0
        best_r10_eval = 0
        '''
        ***********************************************
        '''

        # top-K evaluation settings
        user_list_eval, train_record_eval, test_record_eval, \
            item_set_eval, k_list_eval = topk_settings(show_topk, train_data, eval_data, n_item, seed=args.seed)
        args.seed += 1
        print('----------')
        print(args.seed)
        print('----------')

        user_list_test, train_record_test, test_record_test, \
            item_set_test, k_list_test = topk_settings(show_topk, train_data, test_data, n_item, seed=args.seed)
        args.seed += 1
        print('----------')
        print(args.seed)
        print('----------')

        print('Users for evaluation:')
        print(user_list_eval)
        print('Users for testing:')
        print(user_list_test)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            '''
            ***********************************************
            loading
            '''
            if not (args.logging == 'save'):
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
                print(ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
                    print('================Done loading======================')         
            else:
                print('Initialized from scratch')
            '''
            ***********************************************
            '''

            for step in range(args.n_epochs):
                # training
                np.random.shuffle(train_data)
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(args, model, train_data, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(start, loss)

                '''
                ***********************************************
                result logging
                '''
                if show_ctr:
                    
                    # CTR evaluation
                    train_auc, train_f1 = ctr_eval(args, sess, model, train_data, args.batch_size)
                    eval_auc, eval_f1 = ctr_eval(args, sess, model, eval_data, args.batch_size)
                    test_auc, test_f1 = ctr_eval(args, sess, model, test_data, args.batch_size)

                    print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                          % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

                    eval_logging_ctr(args, step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1)
                    if eval_auc>best_auc_eval:
                        best_auc_epoch = step
                        best_auc_eval = eval_auc
                        best_auc = test_auc
                        best_f1 = test_f1

                        user_embedding, entity_embedding, relation_embedding = model.get_embeddings(sess)
                        sio.savemat('../output/{}_{}_emb.mat'.format(args.model_type, args.dataset), \
                            {'user': user_embedding, 'entity': entity_embedding, 'relation': relation_embedding})
                        np.save('../output/{}_{}_user_emb.npy'.format(args.model_type, args.dataset), user_embedding)
                        np.save('../output/{}_{}_entity_emb.npy'.format(args.model_type, args.dataset), entity_embedding)
                        np.save('../output/{}_{}_relation_emb.npy'.format(args.model_type, args.dataset), relation_embedding)

                        # saver_ckpt.save(sess, ckpt_save_path_1+'weights', global_step=step)
                '''
                ***********************************************
                '''

                # top-K evaluation
                if show_topk:
                    # validation
                    eval_precision, eval_recall = topk_eval(
                        args, sess, model, user_list_eval, train_record_eval, test_record_eval, item_set_eval, k_list_eval, args.batch_size)
                    print('eval precision: ', end='')
                    for i in eval_precision:
                        print('%.4f\t' % i, end='')
                    print()
                    print('eval recall   : ', end='')
                    for i in eval_recall:
                        print('%.4f\t' % i, end='')
                    print('')

                    # testing
                    test_precision, test_recall = topk_eval(
                        args, sess, model, user_list_test, train_record_test, test_record_test, item_set_test, k_list_test, args.batch_size)
                    print('test precision: ', end='')
                    for i in test_precision:
                        print('%.4f\t' % i, end='')
                    print()
                    print('test recall   : ', end='')
                    for i in test_recall:
                        print('%.4f\t' % i, end='')
                    print('\n')


                    '''
                    ***********************************************
                    result logging
                    '''
                    eval_logging_topk(args, step, eval_precision, eval_recall, test_precision, test_recall)
                    if eval_recall[3]>best_r10_eval:
                        best_recall_epoch = step
                        best_r10_eval = eval_recall[3]
                        best_recall = test_recall
                        best_precision = test_precision

                        # saver_ckpt.save(sess, ckpt_save_path_2+'weights', global_step=step)
                    '''
                    ***********************************************
                    '''

        '''
        ***********************************************
        best result recording
        '''
        if show_ctr:
            auc.append(best_auc)
            f1_score.append(best_f1)
            recall.append(best_recall)
            precision.append(best_precision)

        if show_topk:
            epochs_recall.append(best_recall_epoch)
            epochs_auc.append(best_auc_epoch)
        '''
        ***********************************************
        '''

    '''
    ***********************************************
    average performance
    '''
    if show_ctr and show_topk:
        auc = np.array(auc)
        f1_score = np.array(f1_score)
        recall = np.array(recall)
        precision = np.array(precision)

        print(auc)
        print(f1_score)
        print(recall)
        print(precision)

        printings_auc = 'average auc: {:.4f}+\\-{:.4f}'.format(np.mean(auc), np.std(auc))
        printings_f1 = 'average f1 : {:.4f}+\\-{:.4f}'.format(np.mean(f1_score), np.std(f1_score))

        printings_recall = 'average recall   : '
        for i in range(len(k_list_test)):
            printings_recall = printings_recall + '{:.4f}+\\-{:.4f} '.format(np.mean(recall[:, i]), np.std(recall[:, i]))
        printings_precision = 'average precision: '
        for i in range(len(k_list_test)):
            printings_precision = printings_precision + '{:.4f}+\\-{:.4f} '.format(np.mean(precision[:, i]), np.std(precision[:, i]))

        print(printings_auc)
        print(printings_f1)
        print(printings_recall)
        print(printings_precision)

        logging_final(args, printings_auc, printings_f1, printings_recall, printings_precision)
    '''
    ***********************************************
    '''

'''
***********************************************
logging functions
'''
def logging_final(args, auc, f1_score, recall, precision):
    fid = open('../result/{}_{}_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

    fid.write('\n******************************************************************************\n')
    params = eval(args.params)
    settings = ''
    for i in range(len(params)):
        settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
    fid.write(settings+'\n')
    fid.write('******************************************************************************\n')

    fid.write(auc+'\n')
    fid.write(f1_score+'\n')
    fid.write(recall+'\n')
    fid.write(precision+'\n')

    fid.close()


def eval_logging_topk(args, epoch, eval_pr, eval_re, test_pr, test_re):
    fid = open('../result/{}_{}_topk_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

    if epoch == 0:
        fid.write('\n******************************************************************************\n')
        params = eval(args.params)
        settings = ''
        for i in range(len(params)):
            settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
        fid.write(settings+'\n')
        fid.write('******************************************************************************\n')

    result_type = ['epoch:{:2d} eval precision: '.format(epoch), 'eval recall: '.format(epoch), \
                   'epoch:{:2d} test precision: '.format(epoch), 'test recall: '.format(epoch)]
    results = [eval_pr, eval_re, test_pr, test_re]
    for i, result in enumerate(results):    
        printings = result_type[i]
        for j in result:
            printings = printings + '{:.4f}\t'.format(j)
        fid.write(printings)
        if i == 1 or i == 3:
            fid.write('\n')
    fid.write('\n')
    fid.close()

def eval_logging_ctr(args, epoch, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1):
    fid = open('../result/{}_{}_ctr_{}.txt'.format(args.model_type, args.dataset, args.att), 'a')

    if epoch == 0:
        fid.write('\n******************************************************************************\n')
        params = eval(args.params)
        settings = ''
        for i in range(len(params)):
            settings = settings + params[i] + ':{} '.format(eval('args.'+params[i]))
        fid.write(settings+'\n')
        fid.write('******************************************************************************\n')

    fid.write('epoch:{:2d} train_auc:{:.4f} train_f1:{:.4f} '.format(epoch, train_auc, train_f1) +
              'eval_auc:{:.4f} eval_f1:{:.4f} '.format(eval_auc, eval_f1) +
              'test_auc:{:.4f} test_f1:{:.4f}\n'.format(test_auc, test_f1))

    fid.close()
'''
***********************************************
'''

def topk_settings(show_topk, train_data, test_data, n_item, seed=555):
    np.random.seed(seed)
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(args, model, data, start, end, flag=True):
    if flag:
        feed_dict = {model.user_indices: data[start:end, 0],
                     model.item_indices: data[start:end, 1],
                     model.labels: data[start:end, 2],
                     model.node_dropout: eval(args.node_dropout),
                     model.mess_dropout: eval(args.mess_dropout)}
    else:
        feed_dict = {model.user_indices: data[start:end, 0],
                     model.item_indices: data[start:end, 1],
                     model.labels: data[start:end, 2],
                     model.node_dropout: [0.]*len(eval(args.layer_size)),
                     model.mess_dropout: [0.]*len(eval(args.layer_size))}
    return feed_dict


def ctr_eval(args, sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(args, model, data, start, start + batch_size, flag=False))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def topk_eval(args, sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size],
                                                    model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start),
                       model.node_dropout: [0.]*len(eval(args.layer_size)),
                       model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
