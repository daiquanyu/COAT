
# Personalized Knowledge-Aware Recommendation with Collaborative and Attentive Graph Convolutional Networks

This is our implementation for the following paper:

>Quanyu Dai, Xiao-Ming Wu, Lu Fan, Qimai Li, Han Liu, Xiaotong Zhang, Dan Wang, Guli Lin, Keping Yang, Personalized knowledge-aware recommendation with collaborative and attentive graph convolutional networks,
Pattern Recognition, Volume 128, 2022, 108628, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2022.108628. (https://www.sciencedirect.com/science/article/pii/S0031320322001091)

Author: Quanyu Dai (quanyu.dai at connect.polyu.hk)

## Introduction
Knowledge graphs (KGs) are increasingly used to solve the data sparsity and cold start problems of collaborative filtering. Recently, graph neural networks (GNNs) have been applied to build KG-based recommender systems and achieved competitive performance. However, existing GNN-based methods are either limited in their ability to capture fine-grained semantics in a KG, or insufficient in effectively modeling user-item interactions. To address these issues, we propose a novel framework with collaborative and attentive graph convolutional networks for personalized knowledge-aware recommendation. Particularly, we model the user-item graph and the KG separately and simultaneously with an efficient graph convolutional network and a personalized knowledge graph attention network, where the former aims to extract informative collaborative signals, while the latter is designed to capture fine-grained semantics. Collectively, they are able to learn meaningful node representations for predicting user-item interactions. Extensive experiments on benchmark datasets demonstrate the effectiveness of the proposed method compared with state-of-the-arts.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* python == 3.6.10
* tensorflow-gpu == 1.15.2
* numpy == 1.19.1
* scipy == 1.1.0
* sklearn == 0.23.1

## Examples to Run the code
The instruction of commands has been clearly stated in the code (see src/main.py).

* Movie
```
python main.py --dataset movie --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type gcmc --model_type KGCN_NGCF --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --alpha 0 --smoothing_steps 1 --pretrain 0 --att 'u_r' --runs 3 --gpu_id 0
```

* book
```
python main.py --dataset book --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 5e-5 --layer_size [64] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att 'uhrt_bi' --runs 3 --gpu_id 0
```

* Music
```
python main.py --dataset music --aggregator 'sum' --n_epochs 10 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att 'uhrt_bi' --runs 3 --gpu_id 0
```


* Restaurant
```
python main.py --dataset restaurant --aggregator 'sum' --n_epochs 20 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 65536 --l2_weight 1e-7 --lr 2e-2 --layer_size [8] --adj_type norm --alg_type gcn --model_type KGCN_GCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att 'uhrt_bi' --runs 3 --gpu_id 0
```

## About implementation

We build our model based on the implementations of KGCN (https://github.com/hwwang55/KGCN) and NGCF (https://github.com/delldu/NGCF).

## About Datasets
Hyperlink: https://pan.baidu.com/s/1As8RVt-yfA0qnl9VorfLmA 
Code:h96j

## Citation 
If you would like to use our code, please cite:
```
@article{DAI2022108628,
title = {Personalized knowledge-aware recommendation with collaborative and attentive graph convolutional networks},
journal = {Pattern Recognition},
volume = {128},
pages = {108628},
year = {2022},
issn = {0031-3203},
author = {Quanyu Dai and Xiao-Ming Wu and Lu Fan and Qimai Li and Han Liu and Xiaotong Zhang and Dan Wang and Guli Lin and Keping Yang}
}
```
